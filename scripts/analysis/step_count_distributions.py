#!/usr/bin/env python
"""Compute per-model step-count distributions from existing steering vector NPZs.

Reports:
  - Step-count distribution per model (how many questions have K=2,3,4,...,12 steps)
  - Mean/median/std step count per model
  - Overlap statistics (what fraction of questions have the same step count across models)

This addresses jqhQ W2: "Distillation comparison not well isolated — models may
differ in typical step counts."

Usage:
  python scripts/analysis/step_count_distributions.py \
      --npz-paths \
          output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
          output/steering_vectors/steering_vectors_base_8000_llama.npz \
          output/steering_vectors/steering_vectors_r1-distill-llama-8b_8000_deepseek.npz \
      --model-names Instruct Base R1-Distill \
      --output-dir output/step_count_distributions
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np


def get_step_counts(npz_path: Path) -> dict:
    """Extract per-question step counts from NPZ."""
    data = np.load(npz_path, allow_pickle=True)

    # Use layer 0 (step counts are the same across layers)
    step_nums = data['step_numbers']
    qids_step = data['question_ids_step']

    # Handle object arrays vs stacked arrays
    if step_nums.dtype == object:
        sn = step_nums[0]
        qi = qids_step[0]
    elif step_nums.ndim >= 2:
        sn = step_nums[0]
        qi = qids_step[0]
    else:
        sn = step_nums
        qi = qids_step

    # Group by question ID, count steps per question
    question_steps = {}
    for qid, step in zip(qi, sn):
        qid = int(qid)
        if qid not in question_steps:
            question_steps[qid] = set()
        question_steps[qid].add(int(step))

    # Step count = max step number per question
    step_counts = {qid: max(steps) for qid, steps in question_steps.items()}
    return step_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--model-names", nargs="+", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/step_count_distributions"))
    args = parser.parse_args()

    assert len(args.npz_paths) == len(args.model_names), \
        "Must provide same number of NPZ paths and model names"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load step counts per model
    all_counts = {}
    for name, path in zip(args.model_names, args.npz_paths):
        print(f"\nLoading {name}: {path}")
        all_counts[name] = get_step_counts(path)
        print(f"  {len(all_counts[name])} questions")

    # Per-model distributions
    print(f"\n{'=' * 80}")
    print(f"STEP-COUNT DISTRIBUTIONS")
    print(f"{'=' * 80}")

    all_K_values = set()
    for name, counts in all_counts.items():
        all_K_values.update(counts.values())
    K_range = sorted(all_K_values)

    # Header
    header = f"{'K':>4s}"
    for name in args.model_names:
        header += f"  {name:>12s}"
    print(f"\n{header}")
    print(f"{'─' * (4 + 14 * len(args.model_names))}")

    for K in K_range:
        row = f"{K:>4d}"
        for name in args.model_names:
            n = sum(1 for v in all_counts[name].values() if v == K)
            pct = n / len(all_counts[name]) * 100
            row += f"  {n:>5d} ({pct:>4.1f}%)"
        print(row)

    # Summary statistics
    print(f"\n{'─' * 80}")
    print(f"{'Stat':>12s}", end="")
    for name in args.model_names:
        print(f"  {name:>12s}", end="")
    print()
    print(f"{'─' * (12 + 14 * len(args.model_names))}")

    for stat_name, stat_fn in [("N questions", lambda v: len(v)),
                                ("Mean K", lambda v: np.mean(list(v.values()))),
                                ("Median K", lambda v: np.median(list(v.values()))),
                                ("Std K", lambda v: np.std(list(v.values()))),
                                ("Min K", lambda v: min(v.values())),
                                ("Max K", lambda v: max(v.values()))]:
        row = f"{stat_name:>12s}"
        for name in args.model_names:
            val = stat_fn(all_counts[name])
            if isinstance(val, float):
                row += f"  {val:>12.2f}"
            else:
                row += f"  {val:>12d}"
        print(row)

    # Overlap analysis: for questions present in all models
    if len(args.model_names) >= 2:
        common_qids = set.intersection(*[set(c.keys()) for c in all_counts.values()])
        print(f"\n{'=' * 80}")
        print(f"OVERLAP ANALYSIS ({len(common_qids)} questions common to all models)")
        print(f"{'=' * 80}")

        if common_qids:
            # How often do models agree on step count?
            names = args.model_names
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    agree = sum(1 for qid in common_qids
                                if all_counts[names[i]][qid] == all_counts[names[j]][qid])
                    pct = agree / len(common_qids) * 100
                    print(f"  {names[i]} vs {names[j]}: "
                          f"{agree}/{len(common_qids)} agree ({pct:.1f}%)")

            # Distribution of step-count differences
            if len(names) == 3:
                print(f"\n  Step-count difference distribution (absolute):")
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        diffs = [abs(all_counts[names[i]][qid] - all_counts[names[j]][qid])
                                 for qid in common_qids]
                        print(f"    {names[i]} vs {names[j]}: "
                              f"mean={np.mean(diffs):.2f}, "
                              f"median={np.median(diffs):.1f}, "
                              f"max={max(diffs)}, "
                              f"|diff|≤1: {sum(1 for d in diffs if d <= 1)/len(diffs)*100:.1f}%")

    # Save
    results = {
        "per_model": {
            name: {
                "n_questions": len(counts),
                "mean_K": float(np.mean(list(counts.values()))),
                "median_K": float(np.median(list(counts.values()))),
                "std_K": float(np.std(list(counts.values()))),
                "distribution": dict(Counter(counts.values())),
            }
            for name, counts in all_counts.items()
        }
    }

    out_path = args.output_dir / "step_count_distributions.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()