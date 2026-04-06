#!/usr/bin/env python
"""Analyze step-by-step trajectory distances

This script computes:
1. Training set: Average distances between consecutive steps (1→2, 2→3, ..., last→hash)
   grouped by number of steps and correctness
2. Test set: Difference between each question's step distances and training averages
   (matched by step count)

Usage:
    python scripts/trajectories/analyze_step_distances.py \
        --train-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
        --output-dir output/trajectory_distance_analysis
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_steering_vectors(npz_path: Path) -> Dict:
    """Load steering vectors NPZ file"""
    print(f"\nLoading steering vectors from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    result = {
        'step_activations': data['step_activations'],
        'hash_activations': data['hash_activations'],
        'step_numbers': data['step_numbers'],
        'question_ids_step': data['question_ids_step'],
        'question_ids_hash': data['question_ids_hash'],
        'is_correct_step': data['is_correct_step'],
        'is_correct_hash': data['is_correct_hash'],
        'num_layers': int(data['num_layers']),
        'hidden_dim': int(data['hidden_dim']),
    }

    print(f"  Num layers: {result['num_layers']}")
    print(f"  Hidden dim: {result['hidden_dim']}")

    return result


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors"""
    return np.linalg.norm(v1 - v2)


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine distance between two vectors"""
    return cosine(v1, v2)


def extract_question_trajectories(
    steering_data: Dict,
    layer_idx: int
) -> Dict[int, Dict]:
    """Extract full trajectories for each question

    Returns:
        Dict mapping question_id -> {
            'steps': List of (step_num, activation) tuples sorted by step_num,
            'hash': hash activation,
            'is_correct': bool,
            'num_steps': int
        }
    """
    print(f"\nExtracting trajectories for layer {layer_idx}...")

    # Get step activations
    step_acts = steering_data['step_activations'][layer_idx]
    step_nums = steering_data['step_numbers'][layer_idx]
    question_ids_step = steering_data['question_ids_step'][layer_idx]
    is_correct_step = steering_data['is_correct_step'][layer_idx]

    # Get hash activations
    hash_acts = steering_data['hash_activations'][layer_idx]
    question_ids_hash = steering_data['question_ids_hash'][layer_idx]
    is_correct_hash = steering_data['is_correct_hash'][layer_idx]

    # Build trajectories
    trajectories = {}

    # First, collect all step activations per question
    for i, qid in enumerate(question_ids_step):
        if qid not in trajectories:
            trajectories[qid] = {
                'steps': [],
                'hash': None,
                'is_correct': bool(is_correct_step[i]),
                'num_steps': 0
            }

        trajectories[qid]['steps'].append((int(step_nums[i]), step_acts[i]))

    # Add hash activations
    for i, qid in enumerate(question_ids_hash):
        if qid in trajectories:
            trajectories[qid]['hash'] = hash_acts[i]

    # Sort steps and count
    for qid in trajectories:
        trajectories[qid]['steps'].sort(key=lambda x: x[0])
        trajectories[qid]['num_steps'] = len(trajectories[qid]['steps'])

    # Filter out questions without hash activations
    trajectories = {qid: traj for qid, traj in trajectories.items() if traj['hash'] is not None}

    print(f"  Found {len(trajectories)} complete trajectories")

    # Count by num_steps
    num_steps_counts = defaultdict(int)
    for traj in trajectories.values():
        num_steps_counts[traj['num_steps']] += 1

    print(f"  Step count distribution:")
    for num_steps in sorted(num_steps_counts.keys()):
        print(f"    {num_steps} steps: {num_steps_counts[num_steps]} questions")

    return trajectories


def compute_trajectory_distances(
    trajectories: Dict[int, Dict],
    distance_fn
) -> Dict[int, Dict]:
    """Compute step-by-step distances for each trajectory

    Args:
        trajectories: Dict mapping question_id -> trajectory info
        distance_fn: Function to compute distance between two vectors

    Returns:
        Dict mapping question_id -> {
            'is_correct': bool,
            'num_steps': int,
            'distances': List[float] for each transition (step1→step2, ..., last→hash)
        }
    """
    results = {}

    for qid, traj in trajectories.items():
        distances = []

        # Compute distances between consecutive steps
        for i in range(len(traj['steps']) - 1):
            _, act1 = traj['steps'][i]
            _, act2 = traj['steps'][i + 1]
            dist = distance_fn(act1, act2)
            distances.append(dist)

        # Distance from last step to hash
        _, last_act = traj['steps'][-1]
        dist = distance_fn(last_act, traj['hash'])
        distances.append(dist)

        results[qid] = {
            'is_correct': traj['is_correct'],
            'num_steps': traj['num_steps'],
            'distances': distances
        }

    return results


def compute_training_statistics(
    distance_results: Dict[int, Dict]
) -> Dict[int, Dict]:
    """Compute average distances and CIs for training set, grouped by num_steps and correctness

    Returns:
        Dict mapping num_steps -> {
            'correct': {
                'count': int,
                'mean_distances': List[float] for each transition,
                'std_distances': List[float],
                'ci_lower': List[float],
                'ci_upper': List[float]
            },
            'incorrect': {...}
        }
    """
    # Group by num_steps and correctness
    grouped = defaultdict(lambda: {'correct': [], 'incorrect': []})

    for qid, result in distance_results.items():
        num_steps = result['num_steps']
        is_correct = result['is_correct']

        key = 'correct' if is_correct else 'incorrect'
        grouped[num_steps][key].append(result['distances'])

    # Compute statistics
    statistics = {}

    for num_steps in sorted(grouped.keys()):
        statistics[num_steps] = {}

        for correctness in ['correct', 'incorrect']:
            distances_list = grouped[num_steps][correctness]

            if not distances_list:
                statistics[num_steps][correctness] = {
                    'count': 0,
                    'mean_distances': [],
                    'std_distances': [],
                    'ci_lower': [],
                    'ci_upper': []
                }
                continue

            # Convert to array: [num_questions, num_transitions]
            distances_array = np.array(distances_list)
            count = len(distances_list)

            # Compute mean and std for each transition
            mean_distances = np.mean(distances_array, axis=0)
            std_distances = np.std(distances_array, axis=0, ddof=1)

            # Compute 95% confidence intervals
            # CI = mean ± 1.96 * (std / sqrt(n))
            margin = 1.96 * (std_distances / np.sqrt(count))
            ci_lower = mean_distances - margin
            ci_upper = mean_distances + margin

            statistics[num_steps][correctness] = {
                'count': count,
                'mean_distances': mean_distances.tolist(),
                'std_distances': std_distances.tolist(),
                'ci_lower': ci_lower.tolist(),
                'ci_upper': ci_upper.tolist()
            }

    return statistics


def compute_test_differences(
    test_distance_results: Dict[int, Dict],
    train_statistics: Dict[int, Dict]
) -> Dict:
    """Compute differences between test distances and training averages

    For each test question:
    - Find its num_steps
    - Compute difference from training correct average
    - Compute difference from training incorrect average
    - Compute accumulated path difference
    - Compute acceleration (differences between step differences)

    Returns:
        Dict with detailed differences for correct and incorrect test questions
    """
    # Filter to only num_steps 2-8
    valid_num_steps = set(range(2, 9))

    # Collect differences grouped by num_steps
    # Structure: {num_steps: {'correct': {'vs_correct_train': [...], ...}, 'incorrect': {...}}}
    per_step_diffs = defaultdict(lambda: {
        'correct': {'vs_correct_train': [], 'vs_incorrect_train': []},
        'incorrect': {'vs_correct_train': [], 'vs_incorrect_train': []}
    })

    # Accumulated path differences
    accumulated_diffs = {
        'correct': {'vs_correct_train': [], 'vs_incorrect_train': []},
        'incorrect': {'vs_correct_train': [], 'vs_incorrect_train': []}
    }

    # Acceleration differences
    acceleration_diffs = {
        'correct': {'vs_correct_train': [], 'vs_incorrect_train': []},
        'incorrect': {'vs_correct_train': [], 'vs_incorrect_train': []}
    }

    # Track which num_steps are missing or filtered
    missing_num_steps = set()
    filtered_num_steps = set()

    for qid, result in test_distance_results.items():
        num_steps = result['num_steps']
        is_correct = result['is_correct']
        test_distances = np.array(result['distances'])

        # Filter to num_steps 2-9
        if num_steps not in valid_num_steps:
            filtered_num_steps.add(num_steps)
            continue

        # Check if we have training statistics for this num_steps
        if num_steps not in train_statistics:
            missing_num_steps.add(num_steps)
            continue

        # Get training averages
        train_correct_mean = np.array(train_statistics[num_steps]['correct']['mean_distances'])
        train_incorrect_mean = np.array(train_statistics[num_steps]['incorrect']['mean_distances'])

        # Skip if no training data for a particular correctness group
        if len(train_correct_mean) == 0 or len(train_incorrect_mean) == 0:
            continue

        # Compute differences (test - train_average)
        diff_vs_correct_train = test_distances - train_correct_mean
        diff_vs_incorrect_train = test_distances - train_incorrect_mean

        # Accumulated path difference (sum of all step differences)
        accumulated_vs_correct = np.sum(diff_vs_correct_train)
        accumulated_vs_incorrect = np.sum(diff_vs_incorrect_train)

        # Acceleration (differences between consecutive step differences)
        # acceleration[i] = diff[i+1] - diff[i]
        accel_vs_correct = np.diff(diff_vs_correct_train)
        accel_vs_incorrect = np.diff(diff_vs_incorrect_train)

        # Store based on test question correctness
        correctness_key = 'correct' if is_correct else 'incorrect'

        per_step_diffs[num_steps][correctness_key]['vs_correct_train'].append(diff_vs_correct_train)
        per_step_diffs[num_steps][correctness_key]['vs_incorrect_train'].append(diff_vs_incorrect_train)

        accumulated_diffs[correctness_key]['vs_correct_train'].append(accumulated_vs_correct)
        accumulated_diffs[correctness_key]['vs_incorrect_train'].append(accumulated_vs_incorrect)

        acceleration_diffs[correctness_key]['vs_correct_train'].append(accel_vs_correct)
        acceleration_diffs[correctness_key]['vs_incorrect_train'].append(accel_vs_incorrect)

    if missing_num_steps:
        print(f"\n  ⚠ Warning: No training data for num_steps: {sorted(missing_num_steps)}")
    if filtered_num_steps:
        print(f"  ℹ Filtered out num_steps (outside 2-8): {sorted(filtered_num_steps)}")

    # Compute statistics for each num_steps
    per_step_statistics = {}
    for num_steps in sorted(per_step_diffs.keys()):
        per_step_statistics[num_steps] = {}

        for correctness in ['correct', 'incorrect']:
            per_step_statistics[num_steps][correctness] = {}

            for train_type in ['vs_correct_train', 'vs_incorrect_train']:
                diffs_list = per_step_diffs[num_steps][correctness][train_type]

                if not diffs_list:
                    per_step_statistics[num_steps][correctness][train_type] = {
                        'count': 0,
                        'mean_per_step': [],
                        'std_per_step': [],
                        'ci_lower_per_step': [],
                        'ci_upper_per_step': []
                    }
                    continue

                # Convert to array: [num_questions, num_transitions]
                diffs_array = np.array(diffs_list)
                count = len(diffs_list)

                # Compute mean and std for each step
                mean_per_step = np.mean(diffs_array, axis=0)
                std_per_step = np.std(diffs_array, axis=0, ddof=1 if count > 1 else 0)

                # Compute 95% CI
                margin = 1.96 * (std_per_step / np.sqrt(count))
                ci_lower = mean_per_step - margin
                ci_upper = mean_per_step + margin

                per_step_statistics[num_steps][correctness][train_type] = {
                    'count': count,
                    'mean_per_step': mean_per_step.tolist(),
                    'std_per_step': std_per_step.tolist(),
                    'ci_lower_per_step': ci_lower.tolist(),
                    'ci_upper_per_step': ci_upper.tolist()
                }

    # Compute accumulated statistics
    def compute_stats(values_list):
        if not values_list:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        values = np.array(values_list)
        count = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1 if count > 1 else 0)
        margin = 1.96 * (std / np.sqrt(count))
        return {
            'count': count,
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(mean - margin),
            'ci_upper': float(mean + margin)
        }

    accumulated_statistics = {
        'correct': {
            'vs_correct_train': compute_stats(accumulated_diffs['correct']['vs_correct_train']),
            'vs_incorrect_train': compute_stats(accumulated_diffs['correct']['vs_incorrect_train'])
        },
        'incorrect': {
            'vs_correct_train': compute_stats(accumulated_diffs['incorrect']['vs_correct_train']),
            'vs_incorrect_train': compute_stats(accumulated_diffs['incorrect']['vs_incorrect_train'])
        }
    }

    # Compute acceleration statistics
    def compute_accel_stats(accel_list):
        if not accel_list:
            return {'count': 0, 'num_questions': 0, 'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        all_accels = np.concatenate(accel_list)
        count = len(all_accels)
        num_questions = len(accel_list)
        mean = np.mean(all_accels)
        std = np.std(all_accels, ddof=1 if count > 1 else 0)
        margin = 1.96 * (std / np.sqrt(count))
        return {
            'count': count,
            'num_questions': num_questions,
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(mean - margin),
            'ci_upper': float(mean + margin)
        }

    acceleration_statistics = {
        'correct': {
            'vs_correct_train': compute_accel_stats(acceleration_diffs['correct']['vs_correct_train']),
            'vs_incorrect_train': compute_accel_stats(acceleration_diffs['correct']['vs_incorrect_train'])
        },
        'incorrect': {
            'vs_correct_train': compute_accel_stats(acceleration_diffs['incorrect']['vs_correct_train']),
            'vs_incorrect_train': compute_accel_stats(acceleration_diffs['incorrect']['vs_incorrect_train'])
        }
    }

    results = {
        'per_step_statistics': per_step_statistics,
        'accumulated_statistics': accumulated_statistics,
        'acceleration_statistics': acceleration_statistics
    }

    return results


def print_training_statistics(statistics: Dict[int, Dict], distance_type: str):
    """Print training statistics (filtered to num_steps 2-8)"""
    print(f"\n{'='*100}")
    print(f"TRAINING SET STATISTICS - {distance_type.upper()} DISTANCE (num_steps 2-8)")
    print(f"{'='*100}")

    # Filter to num_steps 2-8
    valid_num_steps = set(range(2, 9))
    filtered_statistics = {k: v for k, v in statistics.items() if k in valid_num_steps}

    for num_steps in sorted(filtered_statistics.keys()):
        print(f"\n--- {num_steps} STEPS ---")

        for correctness in ['correct', 'incorrect']:
            stats = statistics[num_steps][correctness]
            count = stats['count']

            if count == 0:
                print(f"  {correctness.upper()}: No data")
                continue

            print(f"\n  {correctness.upper()} ({count} questions):")

            mean_dists = stats['mean_distances']
            ci_lower = stats['ci_lower']
            ci_upper = stats['ci_upper']

            # Print each transition
            for i, (mean, lower, upper) in enumerate(zip(mean_dists, ci_lower, ci_upper)):
                if i < len(mean_dists) - 1:
                    transition = f"Step {i+1} → Step {i+2}"
                else:
                    transition = f"Last → Hash"

                print(f"    {transition:20s}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")


def print_test_differences(test_diffs: Dict, distance_type: str):
    """Print test set differences with detailed per-step analysis"""
    print(f"\n{'='*100}")
    print(f"TEST SET DISTANCE DIFFERENCES - {distance_type.upper()} DISTANCE")
    print(f"{'='*100}")

    per_step_stats = test_diffs['per_step_statistics']
    accumulated_stats = test_diffs['accumulated_statistics']
    acceleration_stats = test_diffs['acceleration_statistics']

    # Print per-step differences for each num_steps
    print(f"\n--- PER-STEP DIFFERENCES (num_steps 2-8) ---")

    for num_steps in sorted(per_step_stats.keys()):
        print(f"\n### {num_steps} STEPS ###")

        for correctness in ['correct', 'incorrect']:
            for train_type in ['vs_correct_train', 'vs_incorrect_train']:
                stats = per_step_stats[num_steps][correctness][train_type]
                count = stats['count']

                if count == 0:
                    continue

                train_label = "CORRECT" if train_type == 'vs_correct_train' else "INCORRECT"
                test_label = correctness.upper()

                print(f"\n  {test_label} test questions ({count} questions) vs. {train_label} training avg:")

                mean_per_step = stats['mean_per_step']
                ci_lower = stats['ci_lower_per_step']
                ci_upper = stats['ci_upper_per_step']

                # Print each transition
                for i, (mean, lower, upper) in enumerate(zip(mean_per_step, ci_lower, ci_upper)):
                    if i < len(mean_per_step) - 1:
                        transition = f"Step {i+1} → Step {i+2}"
                    else:
                        transition = f"Last → Hash"

                    print(f"    {transition:20s}: {mean:+.4f} (95% CI: [{lower:+.4f}, {upper:+.4f}])")

    # Print accumulated path differences
    print(f"\n{'='*100}")
    print(f"ACCUMULATED PATH DIFFERENCES (sum of all step differences)")
    print(f"{'='*100}")
    print(f"Note: For each question, sums all step-wise distance differences into a single value.")

    print(f"\nCORRECT Test Questions:")
    print(f"  vs. CORRECT training average:")
    stats = accumulated_stats['correct']['vs_correct_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N={stats['count']}")

    print(f"  vs. INCORRECT training average:")
    stats = accumulated_stats['correct']['vs_incorrect_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N={stats['count']}")

    print(f"\nINCORRECT Test Questions:")
    print(f"  vs. CORRECT training average:")
    stats = accumulated_stats['incorrect']['vs_correct_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N={stats['count']}")

    print(f"  vs. INCORRECT training average:")
    stats = accumulated_stats['incorrect']['vs_incorrect_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N={stats['count']}")

    # Print acceleration differences
    print(f"\n{'='*100}")
    print(f"ACCELERATION DIFFERENCES (differences between consecutive step differences)")
    print(f"{'='*100}")
    print(f"Note: Acceleration measures how the distance difference changes between consecutive steps.")
    print(f"      For a question with k steps: k distances → k differences → (k-1) acceleration values.")
    print(f"      N_values = total acceleration values across all steps and questions")
    print(f"      N_questions = number of questions contributing to these statistics")

    print(f"\nCORRECT Test Questions:")
    print(f"  vs. CORRECT training average:")
    stats = acceleration_stats['correct']['vs_correct_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N_values={stats['count']} ({stats['num_questions']} questions)")

    print(f"  vs. INCORRECT training average:")
    stats = acceleration_stats['correct']['vs_incorrect_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N_values={stats['count']} ({stats['num_questions']} questions)")

    print(f"\nINCORRECT Test Questions:")
    print(f"  vs. CORRECT training average:")
    stats = acceleration_stats['incorrect']['vs_correct_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N_values={stats['count']} ({stats['num_questions']} questions)")

    print(f"  vs. INCORRECT training average:")
    stats = acceleration_stats['incorrect']['vs_incorrect_train']
    print(f"    Mean: {stats['mean']:+.4f} (95% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}])")
    print(f"    Std:  {stats['std']:.4f} | N_values={stats['count']} ({stats['num_questions']} questions)")


class DualOutput:
    """Context manager to print to both console and file"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.original_print = None

    def __enter__(self):
        self.file = open(self.filepath, 'w')
        self.original_print = print

        # Create new print function that writes to both
        def dual_print(*args, **kwargs):
            # Print to console
            self.original_print(*args, **kwargs)
            # Print to file
            kwargs_copy = kwargs.copy()
            kwargs_copy['file'] = self.file
            kwargs_copy['flush'] = True
            self.original_print(*args, **kwargs_copy)

        # Replace global print
        import builtins
        builtins.print = dual_print
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original print
        import builtins
        builtins.print = self.original_print
        if self.file:
            self.file.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze step-by-step trajectory distances",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--train-npz",
        type=Path,
        default=Path("output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz"),
        help="Path to training steering vectors NPZ"
    )
    parser.add_argument(
        "--test-npz",
        type=Path,
        default=Path("output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz"),
        help="Path to test steering vectors NPZ"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        help="Layer to analyze (default: 31)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/trajectory_distance_analysis"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up dual output (console + file)
    output_txt = args.output_dir / f"step_distance_analysis_layer{args.layer}.txt"

    with DualOutput(output_txt):
        print(f"\n{'='*100}")
        print(f"STEP-BY-STEP TRAJECTORY DISTANCE ANALYSIS")
        print(f"{'='*100}")
        print(f"Training NPZ: {args.train_npz}")
        print(f"Test NPZ: {args.test_npz}")
        print(f"Layer: {args.layer}")
        print(f"Output dir: {args.output_dir}")
        print(f"\nAnalysis scope: num_steps 2-8 only")
        print(f"Distance metrics: Euclidean and Cosine")

        # Load data
        train_data = load_steering_vectors(args.train_npz)
        test_data = load_steering_vectors(args.test_npz)

        # Extract trajectories
        train_trajectories = extract_question_trajectories(train_data, args.layer)
        test_trajectories = extract_question_trajectories(test_data, args.layer)

        # Analyze for both distance types
        all_results = {}

        for distance_type, distance_fn in [('euclidean', euclidean_distance), ('cosine', cosine_distance)]:
            print(f"\n{'#'*100}")
            print(f"# ANALYZING {distance_type.upper()} DISTANCE")
            print(f"{'#'*100}")

            # Compute distances for all trajectories
            print(f"\nComputing {distance_type} distances for training set...")
            train_distances = compute_trajectory_distances(train_trajectories, distance_fn)

            print(f"Computing {distance_type} distances for test set...")
            test_distances = compute_trajectory_distances(test_trajectories, distance_fn)

            # Compute training statistics
            print(f"\nComputing training statistics...")
            train_stats = compute_training_statistics(train_distances)

            # Compute test differences
            print(f"Computing test set differences...")
            test_diffs = compute_test_differences(test_distances, train_stats)

            # Print training statistics first
            print_training_statistics(train_stats, distance_type)

            # Store results
            all_results[distance_type] = {
                'training_statistics': train_stats,
                'test_differences': test_diffs
            }

        # Print all test differences at the end (after both distance types)
        print(f"\n\n{'#'*100}")
        print(f"# TEST SET DISTANCE DIFFERENCES - ALL DISTANCE TYPES")
        print(f"{'#'*100}")

        for distance_type in ['euclidean', 'cosine']:
            test_diffs = all_results[distance_type]['test_differences']
            print_test_differences(test_diffs, distance_type)

        # Save results to JSON
        output_json = args.output_dir / f"step_distance_analysis_layer{args.layer}.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        all_results_native = convert_to_native(all_results)

        with open(output_json, 'w') as f:
            json.dump(all_results_native, f, indent=2)

        print(f"\n{'='*100}")
        print(f"✓ Results saved to:")
        print(f"   JSON: {output_json}")
        print(f"   TXT:  {output_txt}")
        print(f"{'='*100}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
