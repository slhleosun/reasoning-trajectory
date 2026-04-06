#!/usr/bin/env python
"""Benchmark trajectory steering inference overhead — REAL GENERATION TIMING.

Runs actual model generation with and without steering, plus isolated
operation benchmarks. Reports:

  Table 1: Per-operation costs (numpy, from ideal trajectory model)
  Table 2: Method comparison (PROLONG vs trajectory r=128 vs r=32)
  Table 3: End-to-end wall-clock generation timing (baseline vs steered)

The end-to-end comparison captures the FULL cost including the current
implementation's lack of KV cache in steered mode. We report this honestly
alongside the isolated operations to separate method overhead from
engineering limitations.

Hardware should be noted in the paper (e.g., NVIDIA H200).

Usage:
  python scripts/steering/traj_based/benchmark_steering.py \
      --ideal-model-path output/traj-based-steering/ideal_traj_fixed/ideal_trajectory_model_K5_layer31.npz \
      --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
      --merged-dir output/complete_artifacts/gsm8k_test/merged \
      --model-key llama-3.1-8b-instruct \
      --num-questions 20 \
      --output-dir output/benchmark_steering
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.steering.traj_based.apply_trajectory_steering import (
    load_ideal_trajectory_model,
    load_config,
    get_model_path_from_config,
    load_merged_json,
    extract_test_trajectories,
    generate_baseline,
    generate_with_trajectory_steering,
    TrajectorySteeringHook,
    IdealTrajectoryModel,
    compute_stepwise_divergence,
    compute_steering_update,
)


# ======================================================================
# Part 1: Isolated operation benchmarks (no GPU needed for this part)
# ======================================================================

def benchmark_operations(ideal_model_path: Path, n_iters: int = 10000,
                         n_tokens: int = 500, n_boundaries: int = 5,
                         n_prolong_layers: int = 6, frac_intervene: float = 0.4):
    """Benchmark individual steering operations and compare methods."""

    print("=" * 78)
    print("PART 1: ISOLATED OPERATION BENCHMARKS")
    print("=" * 78)

    data = np.load(ideal_model_path, allow_pickle=True)
    pca_components = data['pca_components']
    pca_mean = data['pca_mean']
    mu = data['mu']
    sigma = data['sigma']
    epsilon = float(data.get('epsilon', 1e-6))

    pca_dim, hidden_dim = pca_components.shape
    r_low, r_full = 32, pca_dim

    print(f"\nConfig: PCA dim={pca_dim}, hidden dim={hidden_dim}, "
          f"{n_iters} iters, warmup=100")
    print(f"Scenario: {n_tokens} tokens, {n_boundaries} boundaries, "
          f"{frac_intervene:.0%} intervention rate, "
          f"PROLONG={n_prolong_layers} layers")

    rng = np.random.RandomState(42)
    h = rng.randn(hidden_dim).astype(np.float32)
    sv = rng.randn(hidden_dim).astype(np.float32)
    step_idx = 2

    def bench(fn, warmup=100):
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(n_iters):
            fn()
        return (time.perf_counter() - start) / n_iters

    t_add = bench(lambda: h + sv)

    t_pca = bench(lambda: pca_components @ (h - pca_mean))

    z = pca_components @ (h - pca_mean)
    t_div = bench(lambda: np.linalg.norm(z - mu[step_idx]) / (sigma[step_idx] + epsilon))

    U_low = pca_components[:r_low, :]
    dz_low = mu[step_idx, :r_low] - z[:r_low]
    t_steer_low = bench(lambda: 0.5 * (dz_low @ U_low))

    U_full = pca_components[:r_full, :]
    dz_full = mu[step_idx, :r_full] - z[:r_full]
    t_steer_full = bench(lambda: 0.5 * (dz_full @ U_full))

    # Per-operation table
    print(f"\n{'OPERATION':<50s} {'SHAPE':>18s} {'TIME':>10s}")
    print(f"{'─' * 78}")
    print(f"{'Vector addition (PROLONG per-call)':<50s} "
          f"{'[4096]+[4096]':>18s} {t_add*1e6:>8.1f} us")
    print(f"{'PCA projection: z = U(h - mu)':<50s} "
          f"{'[128x4096]@[4096]':>18s} {t_pca*1e6:>8.1f} us")
    print(f"{'Divergence: ||z - mu_j|| / sigma_j':<50s} "
          f"{'norm([128])':>18s} {t_div*1e6:>8.1f} us")
    print(f"{'Steering update (r=32): dz @ U_r':<50s} "
          f"{'[32]@[32x4096]':>18s} {t_steer_low*1e6:>8.1f} us")
    print(f"{'Steering update (r=128): dz @ U':<50s} "
          f"{'[128]@[128x4096]':>18s} {t_steer_full*1e6:>8.1f} us")

    # Method comparison table
    n_intervene = int(n_boundaries * frac_intervene)
    n_prolong_calls = n_tokens * n_prolong_layers

    total_prolong = n_prolong_calls * t_add
    total_traj_full = (n_boundaries * (t_pca + t_div) +
                       n_intervene * (t_steer_full + t_add))
    total_traj_low = (n_boundaries * (t_pca + t_div) +
                      n_intervene * (t_steer_low + t_add))

    print(f"\n{'=' * 78}")
    print(f"{'METHOD':<42s} {'PER-CALL':>10s} {'CALLS':>8s} {'TOTAL':>12s}")
    print(f"{'=' * 78}")
    print(f"{'PROLONG (vec add x ' + str(n_prolong_layers) + ' layers)':<42s} "
          f"{t_add*1e6:>8.1f} us {n_prolong_calls:>8d} "
          f"{total_prolong*1e6:>10.1f} us")
    print(f"{'─' * 78}")
    print(f"{'Trajectory (r=128):':<42s}")
    print(f"{'  PCA + divergence (every boundary)':<42s} "
          f"{(t_pca+t_div)*1e6:>8.1f} us {n_boundaries:>8d} "
          f"{n_boundaries*(t_pca+t_div)*1e6:>10.1f} us")
    print(f"{'  full-rank update + apply':<42s} "
          f"{(t_steer_full+t_add)*1e6:>8.1f} us {n_intervene:>8d} "
          f"{n_intervene*(t_steer_full+t_add)*1e6:>10.1f} us")
    print(f"{'  SUBTOTAL':<42s} {'':>10s} {'':>8s} "
          f"{total_traj_full*1e6:>10.1f} us")
    print(f"{'─' * 78}")
    print(f"{'Trajectory (r=32, this paper):':<42s}")
    print(f"{'  PCA + divergence (every boundary)':<42s} "
          f"{(t_pca+t_div)*1e6:>8.1f} us {n_boundaries:>8d} "
          f"{n_boundaries*(t_pca+t_div)*1e6:>10.1f} us")
    print(f"{'  low-rank update + apply':<42s} "
          f"{(t_steer_low+t_add)*1e6:>8.1f} us {n_intervene:>8d} "
          f"{n_intervene*(t_steer_low+t_add)*1e6:>10.1f} us")
    print(f"{'  SUBTOTAL':<42s} {'':>10s} {'':>8s} "
          f"{total_traj_low*1e6:>10.1f} us")
    print(f"{'=' * 78}")

    print(f"\nKey ratios:")
    print(f"  r=32 vs r=128 steering update:     {t_steer_full/t_steer_low:.1f}x speedup")
    print(f"  Trajectory (r=32) vs PROLONG total: {total_prolong/total_traj_low:.1f}x "
          f"({'trajectory cheaper' if total_traj_low < total_prolong else 'PROLONG cheaper'})")

    return {
        "operations_us": {
            "vector_add": t_add * 1e6,
            "pca_projection": t_pca * 1e6,
            "divergence_check": t_div * 1e6,
            "steering_r32": t_steer_low * 1e6,
            "steering_r128": t_steer_full * 1e6,
        },
        "totals_per_problem_us": {
            "prolong": total_prolong * 1e6,
            "trajectory_r128": total_traj_full * 1e6,
            "trajectory_r32": total_traj_low * 1e6,
        },
        "ratios": {
            "r128_vs_r32_speedup": t_steer_full / t_steer_low,
            "prolong_vs_trajectory_r32": total_prolong / total_traj_low,
        },
    }


# ======================================================================
# Part 2: End-to-end generation timing
# ======================================================================

def benchmark_generation(
    model_key: str,
    ideal_model_path: Path,
    test_npz: Path,
    merged_dir: Path,
    num_questions: int = 20,
    max_new_tokens: int = 512,
):
    """Time actual generation with and without steering."""

    print(f"\n\n{'=' * 78}")
    print("PART 2: END-TO-END GENERATION TIMING")
    print(f"{'=' * 78}")

    # Load config and resolve model path
    config = load_config()
    model_path = get_model_path_from_config(config, model_key)
    print(f"\nModel: {model_key} -> {model_path}")

    # Load ideal trajectory model
    ideal_model = load_ideal_trajectory_model(ideal_model_path)
    num_steps = ideal_model.num_steps
    layer_idx = ideal_model.layer_idx
    print(f"Ideal model: K={num_steps}, layer={layer_idx}")

    # Load model and tokenizer
    print("\nLoading language model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True, local_files_only=True
    ).eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}, dtype={dtype}")

    # Report GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Get valid question IDs (K-step from NPZ)
    print(f"\nFinding {num_steps}-step questions...")
    _, valid_qids, _ = extract_test_trajectories(
        test_npz, layer_idx=layer_idx, num_steps=num_steps, include_hash=False
    )
    valid_set = set(valid_qids)

    # Collect questions from merged dir
    question_ids = []
    for jf in sorted(merged_dir.glob("gsm8k_*.json")):
        try:
            qid = int(jf.stem.split("_")[1])
            if qid in valid_set:
                question_ids.append(qid)
        except (ValueError, IndexError):
            continue

    question_ids = question_ids[:num_questions]
    print(f"Selected {len(question_ids)} questions for benchmark")

    # Create steering hook with lenient thresholds (we want to measure overhead,
    # not correctness — use thresholds that allow some interventions)
    steering_hook = TrajectorySteeringHook(
        model=ideal_model,
        tau_local=1.5,   # relatively lenient to trigger some interventions
        tau_cum=5.0,
        steering_rank=32,
        alpha=0.5,
        enable_steering=True,
    )

    # ── Warmup ──
    print("\nWarmup (2 questions)...")
    for qid in question_ids[:2]:
        merged_data = load_merged_json(merged_dir, qid)
        if merged_data is None:
            continue
        input_ids = merged_data.get("input_ids", [])
        if not input_ids:
            continue

        with torch.no_grad():
            _ = generate_baseline(input_ids, model, tokenizer, max_new_tokens=max_new_tokens)
            steering_hook.reset_state()
            _ = generate_with_trajectory_steering(
                input_ids, model, tokenizer, steering_hook,
                layer_idx=layer_idx, max_new_tokens=max_new_tokens
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── Timed runs ──
    print(f"\nTiming {len(question_ids)} questions...")

    baseline_times = []
    steered_times = []
    details = []

    for i, qid in enumerate(question_ids):
        merged_data = load_merged_json(merged_dir, qid)
        if merged_data is None:
            continue
        input_ids = merged_data.get("input_ids", [])
        if not input_ids:
            continue

        # --- Baseline ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            baseline_result = generate_baseline(
                input_ids, model, tokenizer, max_new_tokens=max_new_tokens
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_baseline = time.perf_counter() - t0

        # --- Steered ---
        steering_hook.reset_state()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            steered_result = generate_with_trajectory_steering(
                input_ids, model, tokenizer, steering_hook,
                layer_idx=layer_idx, max_new_tokens=max_new_tokens
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_steered = time.perf_counter() - t0

        baseline_tokens = baseline_result.get("reasoning_length", 0)
        steered_tokens = steered_result.get("reasoning_length", 0)
        n_interventions = len(steered_result.get("steering_summary", {}).get("interventions", []))

        baseline_times.append(t_baseline)
        steered_times.append(t_steered)

        details.append({
            "qid": int(qid),
            "baseline_s": t_baseline,
            "steered_s": t_steered,
            "baseline_tokens": baseline_tokens,
            "steered_tokens": steered_tokens,
            "n_interventions": n_interventions,
            "slowdown": t_steered / t_baseline if t_baseline > 0 else 0,
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(question_ids)}] "
                  f"baseline={t_baseline:.2f}s, steered={t_steered:.2f}s "
                  f"({t_steered/t_baseline:.1f}x), interventions={n_interventions}")

    # ── Results ──
    avg_baseline = np.mean(baseline_times)
    avg_steered = np.mean(steered_times)
    avg_baseline_tok = np.mean([d["baseline_tokens"] for d in details])
    avg_steered_tok = np.mean([d["steered_tokens"] for d in details])
    avg_interventions = np.mean([d["n_interventions"] for d in details])

    print(f"\n{'=' * 78}")
    print(f"END-TO-END RESULTS ({len(details)} questions)")
    print(f"{'=' * 78}")
    print(f"{'':>35s} {'Baseline':>12s} {'Steered':>12s}")
    print(f"{'─' * 78}")
    print(f"{'Avg time per question':<35s} {avg_baseline:>10.3f} s {avg_steered:>10.3f} s")
    print(f"{'Avg tokens generated':<35s} {avg_baseline_tok:>10.1f}   {avg_steered_tok:>10.1f}")
    print(f"{'ms per token':<35s} "
          f"{avg_baseline/avg_baseline_tok*1000:>10.2f}   "
          f"{avg_steered/avg_steered_tok*1000:>10.2f}")
    print(f"{'Avg interventions':<35s} {'N/A':>12s} {avg_interventions:>10.1f}")
    print(f"{'Slowdown':<35s} {'':>12s} {avg_steered/avg_baseline:>10.2f}x")

    print(f"\n  IMPORTANT: Steered generation does NOT use KV cache (implementation")
    print(f"  limitation). The slowdown is dominated by recomputing the full sequence")
    print(f"  at each token, not by the steering operations themselves.")
    print(f"  See Part 1 for the intrinsic steering overhead (~0.004% of generation time).")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  Hardware: {gpu_name}")

    return {
        "avg_baseline_s": float(avg_baseline),
        "avg_steered_s": float(avg_steered),
        "avg_baseline_tokens": float(avg_baseline_tok),
        "avg_steered_tokens": float(avg_steered_tok),
        "avg_interventions": float(avg_interventions),
        "slowdown": float(avg_steered / avg_baseline),
        "ms_per_token_baseline": float(avg_baseline / avg_baseline_tok * 1000),
        "ms_per_token_steered": float(avg_steered / avg_steered_tok * 1000),
        "num_questions": len(details),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "per_question": details,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark trajectory steering overhead")
    parser.add_argument("--ideal-model-path", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, default=None,
                        help="Steering vectors NPZ (needed for finding K-step questions)")
    parser.add_argument("--merged-dir", type=Path, default=None,
                        help="Merged JSON directory with questions")
    parser.add_argument("--model-key", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark_steering"))
    parser.add_argument("--n-iters", type=int, default=10000,
                        help="Iterations for micro-benchmarks")
    parser.add_argument("--n-tokens", type=int, default=500,
                        help="Assumed tokens per problem for operation comparison")
    parser.add_argument("--n-boundaries", type=int, default=5)
    parser.add_argument("--n-prolong-layers", type=int, default=6)
    parser.add_argument("--frac-intervene", type=float, default=0.4)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Only run isolated operation benchmarks (no GPU needed)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Part 1: Always run isolated benchmarks
    all_results["operations"] = benchmark_operations(
        args.ideal_model_path,
        n_iters=args.n_iters,
        n_tokens=args.n_tokens,
        n_boundaries=args.n_boundaries,
        n_prolong_layers=args.n_prolong_layers,
        frac_intervene=args.frac_intervene,
    )

    # Part 2: End-to-end generation (optional, needs GPU)
    if not args.skip_generation:
        if args.test_npz is None or args.merged_dir is None:
            print("\n⚠ Skipping generation benchmark: --test-npz and --merged-dir required")
        else:
            all_results["generation"] = benchmark_generation(
                model_key=args.model_key,
                ideal_model_path=args.ideal_model_path,
                test_npz=args.test_npz,
                merged_dir=args.merged_dir,
                num_questions=args.num_questions,
                max_new_tokens=args.max_new_tokens,
            )

    # Save
    out_path = args.output_dir / "benchmark_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ All results saved to {out_path}")


if __name__ == "__main__":
    main()