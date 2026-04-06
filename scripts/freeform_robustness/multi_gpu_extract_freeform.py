#!/usr/bin/env python
"""Multi-GPU launcher for freeform structure activation extraction.

Distributes questions across GPUs via sharding, then performs a
streaming merge that processes one shard at a time to keep peak
memory bounded:

  1. Load shard_0 → initialize master arrays
  2. Delete shard_0.npz from disk
  3. Load shard_1 → concatenate into master
  4. Delete shard_1.npz
  5. ... repeat until all shards merged
  6. Save master NPZ

Each shard runs extract_freeform_activations.py which does a SINGLE
forward pass per question (no generation needed — see that script's
docstring for the justification).

Usage
-----
python scripts/freeform_robustness/multi_gpu_extract_freeform.py \
    --master-json output/freeform/gsm8k_freeform.json \
    --num-gpus 8 \
    --output output/freeform/freeform_structure_activations.npz
"""

import os
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np


# Structure code mapping (must match extract_freeform_activations.py)
STRUCTURE_CODES = {
    "step_x": 0, "numbered_list": 1, "double_newline": 2,
    "single_newline": 3, "single_block": 4,
}
STRUCTURE_NAMES = {v: k for k, v in STRUCTURE_CODES.items()}


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except ImportError:
        return []


def launch_shard(gpu_id, shard_id, num_shards, script_args, log_dir):
    """Launch a single shard process on a specific GPU."""
    script_path = Path(__file__).parent / "extract_freeform_activations.py"

    cmd = [
        sys.executable, str(script_path),
        "--shard-id", str(shard_id),
        "--num-shards", str(num_shards),
    ]
    for key, value in script_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_HUB_OFFLINE"]       = "1"
    env["HF_DATASETS_OFFLINE"]  = "1"
    env["TRANSFORMERS_OFFLINE"]  = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    log_file = log_dir / f"shard_{shard_id}_gpu_{gpu_id}.log"
    log_f = open(log_file, "w")

    print(f"  [GPU {gpu_id}] Launching shard {shard_id}/{num_shards}")

    process = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    return process, log_f


def monitor(processes, log_files, num_shards):
    """Wait for all shard processes to complete."""
    print(f"\n{'─'*80}")
    print("Monitoring shard processes...")
    print(f"{'─'*80}")

    t0 = time.time()
    while True:
        time.sleep(15)
        running   = [(s, g, p) for s, g, p in processes if p.poll() is None]
        completed = [(s, g, p) for s, g, p in processes if p.poll() is not None and p.returncode == 0]
        failed    = [(s, g, p) for s, g, p in processes if p.poll() is not None and p.returncode != 0]

        elapsed = time.time() - t0
        print(f"\r  {elapsed/60:.1f}m | Done {len(completed)}/{num_shards} | "
              f"Running {len(running)} | Failed {len(failed)}", end="", flush=True)

        if not running:
            print()
            break

    for lf in log_files:
        lf.close()

    total = time.time() - t0
    print(f"\n  Total shard time: {total/60:.1f} min")

    if failed:
        for s, g, p in failed:
            print(f"  ⚠ Shard {s} (GPU {g}) exit code {p.returncode}")

    return len(failed) == 0


def streaming_merge(output_dir: Path, num_shards: int, shard_filename: str, final_path: Path):
    """Merge shards one at a time, deleting each after incorporation.

    Memory profile: at most 2 shards worth of data in memory at once
    (the accumulated master + the current shard being loaded).
    """
    print(f"\n{'='*80}")
    print("STREAMING MERGE (shard-by-shard)")
    print(f"{'='*80}")

    num_layers = None
    hidden_dim = None

    # Master arrays — initialized from shard_0, then extended
    master_step_acts    = None
    master_hash_acts    = None
    master_step_nums    = None
    master_qids_step    = None
    master_qids_hash    = None
    master_correct_step = None
    master_correct_hash = None
    master_struct_step  = None
    master_struct_hash  = None

    merged_stats = {}
    shards_merged = 0

    for sid in range(num_shards):
        shard_path = output_dir / f"shard_{sid}" / shard_filename
        if not shard_path.exists():
            print(f"  ⚠ Shard {sid} missing: {shard_path}")
            continue

        fsize = shard_path.stat().st_size / 1024 / 1024
        print(f"  Loading shard {sid} ({fsize:.1f} MB)...")

        data = np.load(shard_path, allow_pickle=True)

        if num_layers is None:
            # Initialize master from first shard
            num_layers = int(data["num_layers"])
            hidden_dim = int(data["hidden_dim"])

            master_step_acts    = list(data["step_activations"])
            master_hash_acts    = list(data["hash_activations"])
            master_step_nums    = list(data["step_numbers"])
            master_qids_step    = list(data["question_ids_step"])
            master_qids_hash    = list(data["question_ids_hash"])
            master_correct_step = list(data["is_correct_step"])
            master_correct_hash = list(data["is_correct_hash"])
            master_struct_step  = list(data.get("structure_types_step", [np.array([], dtype=np.int32)] * num_layers))
            master_struct_hash  = list(data.get("structure_types_hash", [np.array([], dtype=np.int32)] * num_layers))

            n0 = len(master_step_acts[0]) if len(master_step_acts[0]) > 0 else 0
            print(f"    Initialized master: {n0} step acts (layer 0)")
        else:
            # Merge into master
            shard_step_acts   = data["step_activations"]
            shard_hash_acts   = data["hash_activations"]
            shard_step_nums   = data["step_numbers"]
            shard_qids_step   = data["question_ids_step"]
            shard_qids_hash   = data["question_ids_hash"]
            shard_correct_step = data["is_correct_step"]
            shard_correct_hash = data["is_correct_hash"]
            shard_struct_step  = data.get("structure_types_step", [np.array([], dtype=np.int32)] * num_layers)
            shard_struct_hash  = data.get("structure_types_hash", [np.array([], dtype=np.int32)] * num_layers)

            for layer_idx in range(num_layers):
                # Concatenate step arrays
                if len(shard_step_acts[layer_idx]) > 0:
                    if len(master_step_acts[layer_idx]) > 0:
                        master_step_acts[layer_idx] = np.concatenate(
                            [master_step_acts[layer_idx], shard_step_acts[layer_idx]], axis=0)
                        master_step_nums[layer_idx] = np.concatenate(
                            [master_step_nums[layer_idx], shard_step_nums[layer_idx]])
                        master_qids_step[layer_idx] = np.concatenate(
                            [master_qids_step[layer_idx], shard_qids_step[layer_idx]])
                        master_correct_step[layer_idx] = np.concatenate(
                            [master_correct_step[layer_idx], shard_correct_step[layer_idx]])
                        master_struct_step[layer_idx] = np.concatenate(
                            [master_struct_step[layer_idx], shard_struct_step[layer_idx]])
                    else:
                        master_step_acts[layer_idx] = shard_step_acts[layer_idx]
                        master_step_nums[layer_idx] = shard_step_nums[layer_idx]
                        master_qids_step[layer_idx] = shard_qids_step[layer_idx]
                        master_correct_step[layer_idx] = shard_correct_step[layer_idx]
                        master_struct_step[layer_idx] = shard_struct_step[layer_idx]

                # Concatenate hash arrays
                if len(shard_hash_acts[layer_idx]) > 0:
                    if len(master_hash_acts[layer_idx]) > 0:
                        master_hash_acts[layer_idx] = np.concatenate(
                            [master_hash_acts[layer_idx], shard_hash_acts[layer_idx]], axis=0)
                        master_qids_hash[layer_idx] = np.concatenate(
                            [master_qids_hash[layer_idx], shard_qids_hash[layer_idx]])
                        master_correct_hash[layer_idx] = np.concatenate(
                            [master_correct_hash[layer_idx], shard_correct_hash[layer_idx]])
                        master_struct_hash[layer_idx] = np.concatenate(
                            [master_struct_hash[layer_idx], shard_struct_hash[layer_idx]])
                    else:
                        master_hash_acts[layer_idx] = shard_hash_acts[layer_idx]
                        master_qids_hash[layer_idx] = shard_qids_hash[layer_idx]
                        master_correct_hash[layer_idx] = shard_correct_hash[layer_idx]
                        master_struct_hash[layer_idx] = shard_struct_hash[layer_idx]

            n_new = len(shard_step_acts[0]) if len(shard_step_acts[0]) > 0 else 0
            n_total = len(master_step_acts[0]) if len(master_step_acts[0]) > 0 else 0
            print(f"    +{n_new} step acts → total {n_total} (layer 0)")

        # Merge stats
        if "stats" in data:
            try:
                shard_stats = json.loads(str(data["stats"]))
                for k, v in shard_stats.items():
                    if isinstance(v, (int, float)):
                        merged_stats[k] = merged_stats.get(k, 0) + v
            except (json.JSONDecodeError, ValueError):
                pass

        # Free shard data
        del data
        shards_merged += 1

        # DELETE the shard NPZ to free disk space
        shard_path.unlink()
        print(f"    ✓ Deleted shard {sid} NPZ")

        # Also delete shard stats JSON if present
        stats_path = output_dir / f"shard_{sid}" / f"{Path(shard_filename).stem}_stats.json"
        if stats_path.exists():
            stats_path.unlink()

        # Try to remove the empty shard directory
        shard_dir = output_dir / f"shard_{sid}"
        try:
            shard_dir.rmdir()
        except OSError:
            pass  # directory not empty (might have other files)

    if num_layers is None:
        print("\n  ✗ No shards found!")
        return False

    # ── Save master NPZ ──────────────────────────────────────────────
    print(f"\n  Saving master NPZ: {final_path}")
    steering_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        final_path,
        steering_vectors=steering_vectors,
        step_activations=np.array(master_step_acts, dtype=object),
        hash_activations=np.array(master_hash_acts, dtype=object),
        step_numbers=np.array(master_step_nums, dtype=object),
        question_ids_step=np.array(master_qids_step, dtype=object),
        question_ids_hash=np.array(master_qids_hash, dtype=object),
        is_correct_step=np.array(master_correct_step, dtype=object),
        is_correct_hash=np.array(master_correct_hash, dtype=object),
        structure_types_step=np.array(master_struct_step, dtype=object),
        structure_types_hash=np.array(master_struct_hash, dtype=object),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        stats=json.dumps(merged_stats),
        structure_codes=json.dumps(STRUCTURE_CODES),
    )

    fsize = final_path.stat().st_size / 1024 / 1024
    n_step_l0 = len(master_step_acts[0]) if hasattr(master_step_acts[0], '__len__') and len(master_step_acts[0]) > 0 else 0
    n_hash_l0 = len(master_hash_acts[0]) if hasattr(master_hash_acts[0], '__len__') and len(master_hash_acts[0]) > 0 else 0

    print(f"  ✓ Master NPZ saved ({fsize:.1f} MB)")
    print(f"    Shards merged:     {shards_merged}")
    print(f"    Step acts (L0):    {n_step_l0}")
    print(f"    Hash acts (L0):    {n_hash_l0}")

    if n_step_l0 > 0:
        unique_steps, counts = np.unique(master_step_nums[0], return_counts=True)
        print(f"    Step distribution: {dict(zip(unique_steps.tolist(), counts.tolist()))}")

    if n_hash_l0 > 0:
        unique_structs, counts = np.unique(master_struct_hash[0], return_counts=True)
        dist = {STRUCTURE_NAMES.get(int(s), str(s)): int(c) for s, c in zip(unique_structs, counts)}
        print(f"    Structure dist:    {dist}")

    if merged_stats:
        total_eval = merged_stats.get("correct", 0) + merged_stats.get("incorrect", 0)
        if total_eval > 0:
            print(f"    Accuracy:          {merged_stats['correct']}/{total_eval} "
                  f"({merged_stats['correct']/total_eval:.1%})")

    # Save merged stats
    stats_path = final_path.parent / "merged_stats.json"
    with open(stats_path, "w") as f:
        json.dump(merged_stats, f, indent=2)
    print(f"    Stats:             {stats_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU freeform structure activation extraction + streaming merge")
    parser.add_argument("--master-json", type=Path, required=True,
                        help="Master JSON of freeform responses")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (default: all available)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path or config key (default: from config)")
    parser.add_argument("--output", type=Path,
                        default=Path("output/freeform/freeform_structure_activations.npz"))
    parser.add_argument("--min-positions", type=int, default=2)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--min-reasoning-chars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    available_gpus = get_available_gpus()
    if not available_gpus:
        print("No GPUs detected!")
        sys.exit(1)

    num_gpus = args.num_gpus or len(available_gpus)
    gpu_ids = available_gpus[:num_gpus]
    num_shards = num_gpus

    print(f"\n{'='*100}")
    print(f"MULTI-GPU FREEFORM STRUCTURE EXTRACTION — {num_gpus} GPUs")
    print(f"{'='*100}")
    print(f"GPUs:         {gpu_ids}")
    print(f"Master JSON:  {args.master_json}")
    print(f"Output:       {args.output}")
    print(f"Shards:       {num_shards}")
    print(f"{'='*100}\n")

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.output.parent / "logs" / f"extract_freeform_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build script args
    script_args = {
        "master-json":          str(args.master_json),
        "output":               str(args.output),
        "min-positions":        args.min_positions,
        "max-positions":        args.max_positions,
        "min-reasoning-chars":  args.min_reasoning_chars,
        "seed":                 args.seed,
    }
    if args.model:
        script_args["model"] = args.model
    if args.verbose:
        script_args["verbose"] = True

    # ── Launch all shards ─────────────────────────────────────────────
    processes = []
    log_files = []
    for i, gpu_id in enumerate(gpu_ids):
        proc, log_f = launch_shard(gpu_id, i, num_shards, script_args, log_dir)
        processes.append((i, gpu_id, proc))
        log_files.append(log_f)

    # ── Monitor ───────────────────────────────────────────────────────
    success = monitor(processes, log_files, num_shards)

    if not success:
        print("\n⚠ Some shards failed. Check logs:")
        print(f"  {log_dir}")
        print("Continuing with available shards...\n")

    # ── Streaming merge ───────────────────────────────────────────────
    output_dir = args.output.parent
    merge_ok = streaming_merge(
        output_dir, num_shards, args.output.name, args.output)

    if not merge_ok:
        print("\n✗ Merge failed!")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("✓ ALL DONE")
    print(f"{'='*100}")
    print(f"Master NPZ:  {args.output}")
    print(f"Logs:        {log_dir}")
    print(f"\nNext steps:")
    print(f"  # Train step-wise probes on structure-aware activations")
    print(f"  python scripts/predictors/train_stepwise_probes.py \\")
    print(f"      --data {args.output} --output output/freeform/probes_structure")
    print(f"")
    print(f"  # Train correctness predictors")
    print(f"  python scripts/predictors/train_predictors.py \\")
    print(f"      --data {args.output} --output output/freeform/predictors_structure")
    print(f"")
    print(f"  # Compare with formatted results")
    print(f"  python scripts/freeform_robustness/compare_formatted_vs_freeform.py \\")
    print(f"      --formatted-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \\")
    print(f"      --equidistant-npz {args.output} \\")
    print(f"      --output-dir output/freeform/comparison_structure")
    print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())