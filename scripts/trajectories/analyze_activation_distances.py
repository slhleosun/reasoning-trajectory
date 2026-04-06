#!/usr/bin/env python
"""Analyze activation distances between key reasoning points

Computes averaged cosine distances and Euclidean distances between:
- Step 1 and Step 2
- Step 2 and Last Step (before ####)
- Last Step and ####
- Second-to-last Step and Last Step

Includes 95% confidence intervals for all measurements.

Usage:
    python scripts/trajectories/analyze_activation_distances.py \
        --input output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --layer 31
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_steering_data(npz_path: Path) -> Dict:
    """Load activation data from steering vectors NPZ file"""
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Extract metadata
    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])
    stats_str = str(data["stats"])
    stats_dict = json.loads(stats_str)

    # Load activations (arrays per layer)
    step_activations = data["step_activations"]  # List of arrays per layer
    hash_activations = data["hash_activations"]  # List of arrays per layer
    step_numbers = data["step_numbers"]  # List of arrays per layer
    question_ids_step = data["question_ids_step"]  # List of arrays per layer
    question_ids_hash = data["question_ids_hash"]  # List of arrays per layer
    is_correct_step = data["is_correct_step"]  # List of arrays per layer
    is_correct_hash = data["is_correct_hash"]  # List of arrays per layer

    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Successful questions: {stats_dict.get('successful_questions', 'N/A')}")

    return {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "step_activations": step_activations,
        "hash_activations": hash_activations,
        "step_numbers": step_numbers,
        "question_ids_step": question_ids_step,
        "question_ids_hash": question_ids_hash,
        "is_correct_step": is_correct_step,
        "is_correct_hash": is_correct_hash,
        "stats": stats_dict
    }


def build_question_trajectories(data: Dict, layer_idx: int) -> Dict[int, Dict]:
    """Build trajectories for each question from the specified layer

    Returns:
        Dict mapping question_id -> {
            "steps": {step_num: activation},
            "hash": activation,
            "correct": bool
        }
    """
    print(f"\nBuilding trajectories from layer {layer_idx}...")

    # Get data for this layer
    step_acts = data["step_activations"][layer_idx]  # [n_samples, hidden_dim]
    hash_acts = data["hash_activations"][layer_idx]
    step_nums = data["step_numbers"][layer_idx]
    q_ids_step = data["question_ids_step"][layer_idx]
    q_ids_hash = data["question_ids_hash"][layer_idx]
    is_correct_step = data["is_correct_step"][layer_idx]
    is_correct_hash = data["is_correct_hash"][layer_idx]

    # Build mapping: question_id -> (steps, hash_activation, correctness)
    question_data = {}

    # Collect steps for each question
    for i in range(len(step_acts)):
        q_id = int(q_ids_step[i])
        step_num = int(step_nums[i])
        step_act = step_acts[i]
        correct = bool(is_correct_step[i])

        if q_id not in question_data:
            question_data[q_id] = {"steps": {}, "hash": None, "correct": correct}

        question_data[q_id]["steps"][step_num] = step_act
        question_data[q_id]["correct"] = correct

    # Add hash activations
    for i in range(len(hash_acts)):
        q_id = int(q_ids_hash[i])
        hash_act = hash_acts[i]
        correct = bool(is_correct_hash[i])

        if q_id in question_data:
            question_data[q_id]["hash"] = hash_act
            question_data[q_id]["correct"] = correct

    # Filter to only questions with hash and at least 1 step
    filtered_data = {}
    for q_id, q_data in question_data.items():
        if q_data["hash"] is not None and len(q_data["steps"]) > 0:
            filtered_data[q_id] = q_data

    print(f"  Built trajectories for {len(filtered_data)} questions")
    correct_count = sum(1 for q in filtered_data.values() if q["correct"])
    print(f"    Correct: {correct_count}")
    print(f"    Incorrect: {len(filtered_data) - correct_count}")

    return filtered_data


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine_similarity)"""
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0  # Max distance if either vector is zero
    cosine_sim = dot / (norm_v1 * norm_v2)
    return 1.0 - cosine_sim


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance"""
    return np.linalg.norm(v1 - v2)


def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    values_array = np.array(values)
    mean = np.mean(values_array)

    if len(values) == 1:
        return mean, mean, mean

    # Compute standard error
    sem = stats.sem(values_array)

    # Compute confidence interval using t-distribution
    ci = stats.t.interval(confidence, len(values_array) - 1, loc=mean, scale=sem)

    return mean, ci[0], ci[1]


def analyze_distances(question_data: Dict[int, Dict]) -> Dict:
    """Compute distance statistics for different trajectory segments, grouped by correctness

    Returns dict with:
        - correct: distances for correct questions
        - incorrect: distances for incorrect questions
        Each containing:
            - step1_to_step2: distances from Step 1 to Step 2
            - step2_to_last: distances from Step 2 to Last Step
            - last_to_hash: distances from Last Step to ####
            - second_last_to_last: distances from Second-to-last to Last Step
    """
    print("\nComputing distances between trajectory points...")

    results = {
        "correct": {
            "step1_to_step2": {"cosine": [], "euclidean": []},
            "step2_to_last": {"cosine": [], "euclidean": []},
            "last_to_hash": {"cosine": [], "euclidean": []},
            "second_last_to_last": {"cosine": [], "euclidean": []},
        },
        "incorrect": {
            "step1_to_step2": {"cosine": [], "euclidean": []},
            "step2_to_last": {"cosine": [], "euclidean": []},
            "last_to_hash": {"cosine": [], "euclidean": []},
            "second_last_to_last": {"cosine": [], "euclidean": []},
        }
    }

    counts = {
        "correct": {
            "step1_to_step2": 0,
            "step2_to_last": 0,
            "last_to_hash": 0,
            "second_last_to_last": 0,
        },
        "incorrect": {
            "step1_to_step2": 0,
            "step2_to_last": 0,
            "last_to_hash": 0,
            "second_last_to_last": 0,
        }
    }

    for q_id, q_data in question_data.items():
        steps = q_data["steps"]
        hash_act = q_data["hash"]
        correct = q_data["correct"]

        # Determine which group to add to
        group = "correct" if correct else "incorrect"

        # Get sorted step numbers
        sorted_step_nums = sorted(steps.keys())
        num_steps = len(sorted_step_nums)

        if num_steps == 0:
            continue

        # Step 1 to Step 2
        if 1 in steps and 2 in steps:
            step1_act = steps[1]
            step2_act = steps[2]
            results[group]["step1_to_step2"]["cosine"].append(cosine_distance(step1_act, step2_act))
            results[group]["step1_to_step2"]["euclidean"].append(euclidean_distance(step1_act, step2_act))
            counts[group]["step1_to_step2"] += 1

        # Step 2 to Last Step (only if there are 3+ steps)
        if 2 in steps and num_steps >= 3:
            step2_act = steps[2]
            last_step_num = sorted_step_nums[-1]
            last_step_act = steps[last_step_num]
            results[group]["step2_to_last"]["cosine"].append(cosine_distance(step2_act, last_step_act))
            results[group]["step2_to_last"]["euclidean"].append(euclidean_distance(step2_act, last_step_act))
            counts[group]["step2_to_last"] += 1

        # Last Step to ####
        if num_steps >= 1 and hash_act is not None:
            last_step_num = sorted_step_nums[-1]
            last_step_act = steps[last_step_num]
            results[group]["last_to_hash"]["cosine"].append(cosine_distance(last_step_act, hash_act))
            results[group]["last_to_hash"]["euclidean"].append(euclidean_distance(last_step_act, hash_act))
            counts[group]["last_to_hash"] += 1

        # Second-to-last Step to Last Step (only if there are 2+ steps)
        if num_steps >= 2:
            second_last_num = sorted_step_nums[-2]
            last_step_num = sorted_step_nums[-1]
            second_last_act = steps[second_last_num]
            last_step_act = steps[last_step_num]
            results[group]["second_last_to_last"]["cosine"].append(cosine_distance(second_last_act, last_step_act))
            results[group]["second_last_to_last"]["euclidean"].append(euclidean_distance(second_last_act, last_step_act))
            counts[group]["second_last_to_last"] += 1

    print(f"  Computed distances:")
    print(f"    Correct questions:")
    for key, count in counts["correct"].items():
        print(f"      {key}: {count} pairs")
    print(f"    Incorrect questions:")
    for key, count in counts["incorrect"].items():
        print(f"      {key}: {count} pairs")

    return results, counts


def print_summary_table(results: Dict, counts: Dict):
    """Print a summary table with means and 95% CIs, grouped by correctness"""
    print("\n" + "=" * 120)
    print("ACTIVATION DISTANCE SUMMARY (Grouped by Correctness)")
    print("=" * 120)
    print()

    segment_labels = {
        "step1_to_step2": "Step 1 → Step 2",
        "step2_to_last": "Step 2 → Last Step",
        "last_to_hash": "Last Step → ####",
        "second_last_to_last": "Second-last → Last Step",
    }

    # Process each group (correct and incorrect)
    for group in ["correct", "incorrect"]:
        print(f"\n{'CORRECT QUESTIONS' if group == 'correct' else 'INCORRECT QUESTIONS'}")
        print("-" * 120)

        # Compute statistics for this group
        stats_table = []

        for segment_name, segment_data in results[group].items():
            cosine_vals = segment_data["cosine"]
            euclidean_vals = segment_data["euclidean"]

            # Compute means and CIs
            cos_mean, cos_lower, cos_upper = compute_ci(cosine_vals)
            euc_mean, euc_lower, euc_upper = compute_ci(euclidean_vals)

            n = counts[group][segment_name]

            stats_table.append({
                "segment": segment_name,
                "n": n,
                "cosine_mean": cos_mean,
                "cosine_ci": (cos_lower, cos_upper),
                "euclidean_mean": euc_mean,
                "euclidean_ci": (euc_lower, euc_upper),
            })

        # Print header
        print(f"{'Segment':<30} {'N':<8} {'Cosine Distance':<40} {'Euclidean Distance':<40}")
        print(f"{'':30} {'':8} {'Mean (95% CI)':<40} {'Mean (95% CI)':<40}")
        print("-" * 120)

        # Print rows
        for stat in stats_table:
            segment = segment_labels[stat["segment"]]
            n = stat["n"]

            cos_str = f"{stat['cosine_mean']:.4f} ({stat['cosine_ci'][0]:.4f}, {stat['cosine_ci'][1]:.4f})"
            euc_str = f"{stat['euclidean_mean']:.2f} ({stat['euclidean_ci'][0]:.2f}, {stat['euclidean_ci'][1]:.2f})"

            print(f"{segment:<30} {n:<8} {cos_str:<40} {euc_str:<40}")

    print("\n" + "=" * 120)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation distances between reasoning trajectory points"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input NPZ file from collect_steering_vectors.py")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer index to analyze (-1 for last layer, default: -1)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output JSON file for results")

    args = parser.parse_args()

    print("=" * 100)
    print("ANALYZE ACTIVATION DISTANCES")
    print("=" * 100)
    print(f"Input: {args.input}")
    print(f"Layer: {args.layer} (-1 = last layer)")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 100)

    # Load data
    data = load_steering_data(args.input)

    # Handle layer index
    layer_idx = args.layer
    if layer_idx == -1:
        layer_idx = data["num_layers"] - 1

    if layer_idx < 0 or layer_idx >= data["num_layers"]:
        print(f"\n❌ Error: Layer index {layer_idx} out of range [0, {data['num_layers']-1}]")
        return 1

    print(f"\nAnalyzing layer {layer_idx}...")

    # Build trajectories
    question_data = build_question_trajectories(data, layer_idx)

    if len(question_data) == 0:
        print("\n❌ Error: No trajectories found!")
        return 1

    # Compute distances
    results, counts = analyze_distances(question_data)

    # Print summary table
    print_summary_table(results, counts)

    # Save to JSON if requested
    if args.output:
        print(f"Saving results to {args.output}...")

        # Convert results to serializable format
        output_data = {
            "layer": layer_idx,
            "correct": {},
            "incorrect": {}
        }

        for group in ["correct", "incorrect"]:
            for segment_name, segment_data in results[group].items():
                cosine_vals = segment_data["cosine"]
                euclidean_vals = segment_data["euclidean"]

                cos_mean, cos_lower, cos_upper = compute_ci(cosine_vals)
                euc_mean, euc_lower, euc_upper = compute_ci(euclidean_vals)

                output_data[group][segment_name] = {
                    "n": counts[group][segment_name],
                    "cosine": {
                        "mean": float(cos_mean),
                        "ci_lower": float(cos_lower),
                        "ci_upper": float(cos_upper),
                        "values": [float(v) for v in cosine_vals]
                    },
                    "euclidean": {
                        "mean": float(euc_mean),
                        "ci_lower": float(euc_lower),
                        "ci_upper": float(euc_upper),
                        "values": [float(v) for v in euclidean_vals]
                    }
                }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Results saved to {args.output}")

    print("\n" + "=" * 100)
    print("COMPLETE!")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
