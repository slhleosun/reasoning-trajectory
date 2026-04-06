#!/usr/bin/env python
"""Plot reasoning trajectories in activation space using PCA

For each question, visualizes the trajectory of activations through reasoning steps:
Step 1 → Step 2 → Step 3 → ... → Hash

Creates two plots:
1. Individual trajectories for n_samples correct and incorrect questions
2. Averaged trajectories for correct vs incorrect groups

Publication-ready mode adds:
- 95% confidence interval error bars (separate x and y SE) for each step
- Smart labeling (merged labels when points are close)
- Uses "Final Answer Marker" instead of "Hash"
- Removes "(avg of xx)" from legend
- Drops master title
- Font sizes matching visualize_linear_probe_results.py

Usage:
    # Standard mode
    python scripts/trajectories/plot_reasoning_trajectories.py \
        --input output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --output output/trajectories/step4_traj.png \
        --n-samples 8000 \
        --filter-steps 4

    # Publication-ready mode
    python scripts/trajectories/plot_reasoning_trajectories.py \
        --input output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --output output/trajectories/step4_traj_pub.png \
        --n-samples 8000 \
        --filter-steps 4 \
        --publication-ready
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_steering_data(npz_path: Path) -> Dict:
    """Load activation data from steering vectors NPZ file

    Returns:
        Dict with activations, step numbers, question IDs, and correctness labels
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Extract metadata
    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])
    stats_str = str(data["stats"])
    stats = json.loads(stats_str)

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
    print(f"  Successful questions: {stats.get('successful_questions', 'N/A')}")

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
        "stats": stats
    }


def build_trajectories(data: Dict, layer_idx: int = -1) -> Tuple[List[np.ndarray], List[bool], List[int]]:
    """Build trajectories for each question from the specified layer

    Args:
        data: Loaded steering data
        layer_idx: Layer index (-1 for last layer)

    Returns:
        trajectories: List of trajectories, each is [num_steps+1, hidden_dim] (steps + hash)
        correctness: List of bool indicating if question was answered correctly
        question_ids: List of question IDs
    """
    num_layers = data["num_layers"]
    if layer_idx == -1:
        layer_idx = num_layers - 1

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

    # Build trajectories
    trajectories = []
    correctness = []
    question_ids = []

    for q_id, q_data in question_data.items():
        if q_data["hash"] is None:
            continue  # Skip if no hash

        if len(q_data["steps"]) == 0:
            continue  # Skip if no steps

        # Sort steps by step number
        sorted_steps = sorted(q_data["steps"].items())

        # Build trajectory: [Step 1, Step 2, ..., Hash]
        trajectory = []
        for step_num, step_act in sorted_steps:
            trajectory.append(step_act)
        trajectory.append(q_data["hash"])

        trajectory = np.array(trajectory)  # [num_steps+1, hidden_dim]

        trajectories.append(trajectory)
        correctness.append(q_data["correct"])
        question_ids.append(q_id)

    print(f"  Built {len(trajectories)} trajectories")
    print(f"  Correct: {sum(correctness)}, Incorrect: {len(correctness) - sum(correctness)}")

    return trajectories, correctness, question_ids


def filter_trajectories_by_steps(
    trajectories: List[np.ndarray],
    correctness: List[bool],
    question_ids: List[int],
    filter_steps: int = None
) -> Tuple[List[np.ndarray], List[bool], List[int]]:
    """Filter trajectories to only include those with specified number of steps

    Args:
        trajectories: List of trajectories
        correctness: List of correctness labels
        question_ids: List of question IDs
        filter_steps: Number of steps to filter for (None = no filtering)

    Returns:
        Filtered trajectories, correctness, question_ids
    """
    if filter_steps is None:
        return trajectories, correctness, question_ids

    print(f"\nFiltering trajectories with {filter_steps} steps...")

    filtered_trajs = []
    filtered_correct = []
    filtered_qids = []

    for traj, correct, qid in zip(trajectories, correctness, question_ids):
        # Number of steps = trajectory length - 1 (excluding hash)
        num_steps = len(traj) - 1
        if num_steps == filter_steps:
            filtered_trajs.append(traj)
            filtered_correct.append(correct)
            filtered_qids.append(qid)

    correct_count = sum(filtered_correct)
    incorrect_count = len(filtered_correct) - correct_count

    print(f"  After filtering: {len(filtered_trajs)} trajectories")
    print(f"    Correct: {correct_count}")
    print(f"    Incorrect: {incorrect_count}")

    return filtered_trajs, filtered_correct, filtered_qids


def sample_trajectories(
    trajectories: List[np.ndarray],
    correctness: List[bool],
    question_ids: List[int],
    n_samples: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    """Sample n_samples correct and incorrect trajectories

    Returns:
        correct_trajs: List of correct trajectories
        incorrect_trajs: List of incorrect trajectories
        correct_qids: List of correct question IDs
        incorrect_qids: List of incorrect question IDs
    """
    correct_trajs = []
    incorrect_trajs = []
    correct_qids = []
    incorrect_qids = []

    for traj, correct, qid in zip(trajectories, correctness, question_ids):
        if correct:
            correct_trajs.append(traj)
            correct_qids.append(qid)
        else:
            incorrect_trajs.append(traj)
            incorrect_qids.append(qid)

    # Sample
    if len(correct_trajs) > n_samples:
        indices = np.random.choice(len(correct_trajs), n_samples, replace=False)
        correct_trajs = [correct_trajs[i] for i in indices]
        correct_qids = [correct_qids[i] for i in indices]

    if len(incorrect_trajs) > n_samples:
        indices = np.random.choice(len(incorrect_trajs), n_samples, replace=False)
        incorrect_trajs = [incorrect_trajs[i] for i in indices]
        incorrect_qids = [incorrect_qids[i] for i in indices]

    print(f"\nSampled trajectories:")
    print(f"  Correct: {len(correct_trajs)}")
    print(f"  Incorrect: {len(incorrect_trajs)}")

    return correct_trajs, incorrect_trajs, correct_qids, incorrect_qids


def fit_pca_on_all_activations(
    correct_trajs: List[np.ndarray],
    incorrect_trajs: List[np.ndarray]
) -> PCA:
    """Fit PCA on all activations from all trajectories

    Returns:
        Fitted PCA object
    """
    print("\nFitting PCA on all activations...")

    # Collect all activations
    all_activations = []

    for traj in correct_trajs:
        for act in traj:
            all_activations.append(act)

    for traj in incorrect_trajs:
        for act in traj:
            all_activations.append(act)

    all_activations = np.array(all_activations)  # [n_total_points, hidden_dim]

    print(f"  Total activations: {len(all_activations)}")
    print(f"  Shape: {all_activations.shape}")

    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(all_activations)

    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    print(f"  Total: {sum(explained_var):.3f}")

    return pca


def plot_individual_trajectories(
    correct_trajs: List[np.ndarray],
    incorrect_trajs: List[np.ndarray],
    correct_qids: List[int],
    incorrect_qids: List[int],
    pca: PCA,
    output_path: Path
):
    """Plot individual trajectories on 2D PCA space"""
    print("\nPlotting individual trajectories...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Collect all points to determine plot bounds
    all_points = []
    for traj in correct_trajs + incorrect_trajs:
        traj_pca = pca.transform(traj)
        all_points.extend(traj_pca)
    all_points = np.array(all_points)

    # Plot correct trajectories (green)
    for traj, qid in zip(correct_trajs, correct_qids):
        # Transform to PCA space
        traj_pca = pca.transform(traj)  # [num_steps+1, 2]

        # Plot trajectory as connected line segments with arrow markers
        for i in range(len(traj_pca) - 1):
            # Draw line segment
            ax.plot([traj_pca[i, 0], traj_pca[i+1, 0]],
                   [traj_pca[i, 1], traj_pca[i+1, 1]],
                   color='green', alpha=0.4, lw=1.5, zorder=1)

            # Add arrow at midpoint using quiver (more controlled)
            dx = traj_pca[i+1, 0] - traj_pca[i, 0]
            dy = traj_pca[i+1, 1] - traj_pca[i, 1]
            mid_x = traj_pca[i, 0] + 0.5 * dx
            mid_y = traj_pca[i, 1] + 0.5 * dy

            # Very thin arrow with tiny head
            arrow_scale = 0.3
            ax.quiver(mid_x, mid_y, dx, dy,
                     angles='xy', scale_units='xy', scale=1/arrow_scale,
                     color='green', alpha=0.6, width=0.002, headwidth=2, headlength=2,
                     zorder=2)

        # Mark all step positions with small dots
        for i in range(len(traj_pca)):
            if i == 0:
                # Step 1 - darker green circle
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='darkgreen', s=50, marker='o',
                          alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)
            elif i == len(traj_pca) - 1:
                # Hash - green star
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='green', s=100, marker='*',
                          alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)
            else:
                # Intermediate steps - small dots
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='green', s=20, marker='o',
                          alpha=0.6, edgecolors='darkgreen', linewidths=0.3, zorder=4)

    # Plot incorrect trajectories (red)
    for traj, qid in zip(incorrect_trajs, incorrect_qids):
        # Transform to PCA space
        traj_pca = pca.transform(traj)  # [num_steps+1, 2]

        # Plot trajectory as connected line segments with arrow markers
        for i in range(len(traj_pca) - 1):
            # Draw line segment
            ax.plot([traj_pca[i, 0], traj_pca[i+1, 0]],
                   [traj_pca[i, 1], traj_pca[i+1, 1]],
                   color='red', alpha=0.4, lw=1.5, zorder=1)

            # Add arrow at midpoint using quiver (more controlled)
            dx = traj_pca[i+1, 0] - traj_pca[i, 0]
            dy = traj_pca[i+1, 1] - traj_pca[i, 1]
            mid_x = traj_pca[i, 0] + 0.5 * dx
            mid_y = traj_pca[i, 1] + 0.5 * dy

            # Very thin arrow with tiny head
            arrow_scale = 0.3
            ax.quiver(mid_x, mid_y, dx, dy,
                     angles='xy', scale_units='xy', scale=1/arrow_scale,
                     color='red', alpha=0.6, width=0.002, headwidth=2, headlength=2,
                     zorder=2)

        # Mark all step positions with small dots
        for i in range(len(traj_pca)):
            if i == 0:
                # Step 1 - darker red circle
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='darkred', s=50, marker='o',
                          alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)
            elif i == len(traj_pca) - 1:
                # Hash - red star
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='red', s=100, marker='*',
                          alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)
            else:
                # Intermediate steps - small dots
                ax.scatter(traj_pca[i, 0], traj_pca[i, 1], c='red', s=20, marker='o',
                          alpha=0.6, edgecolors='darkred', linewidths=0.3, zorder=4)

    # Set plot limits with padding
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label=f'Correct (n={len(correct_trajs)})'),
        Line2D([0], [0], color='red', lw=2, label=f'Incorrect (n={len(incorrect_trajs)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Step 1', linestyle=''),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=12, label='Hash', linestyle=''),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)

    # Add filter info to title if filtering is applied
    title = 'Individual Reasoning Trajectories in Activation Space'
    if len(correct_trajs) > 0:
        num_steps = len(correct_trajs[0]) - 1
        title += f'\n(Filtered: {num_steps} reasoning steps)'
    elif len(incorrect_trajs) > 0:
        num_steps = len(incorrect_trajs[0]) - 1
        title += f'\n(Filtered: {num_steps} reasoning steps)'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved individual trajectories plot to {output_path}")
    plt.close()


def compute_error_bars(points: np.ndarray, n_std=1.96):
    """Compute error bars (separate x and y) for a set of 2D points

    Args:
        points: [n_samples, 2] array of points
        n_std: Number of standard deviations for the error bars (1.96 for 95% CI)

    Returns:
        mean: [2] mean position
        x_err: float, x-direction error (standard error * n_std)
        y_err: float, y-direction error (standard error * n_std)
    """
    if len(points) < 2:
        return None, None, None

    mean = np.mean(points, axis=0)

    # Compute standard error for x and y separately
    se_x = np.std(points[:, 0], ddof=1) / np.sqrt(len(points))
    se_y = np.std(points[:, 1], ddof=1) / np.sqrt(len(points))

    # Multiply by n_std for confidence interval
    x_err = n_std * se_x
    y_err = n_std * se_y

    return mean, x_err, y_err


def points_are_close(point1: np.ndarray, point2: np.ndarray, threshold: float = 0.15) -> bool:
    """Check if two points are close enough to merge labels

    Args:
        point1: First point [2]
        point2: Second point [2]
        threshold: Distance threshold as fraction of plot range

    Returns:
        True if points are close
    """
    distance = np.linalg.norm(point1 - point2)
    return distance < threshold


def compute_averaged_trajectory(trajectories: List[np.ndarray]) -> np.ndarray:
    """Compute averaged trajectory by averaging at each step position

    Returns:
        averaged_traj: [max_steps+1, hidden_dim] averaged trajectory
    """
    if len(trajectories) == 0:
        return np.array([])

    # Find max trajectory length
    max_len = max(len(traj) for traj in trajectories)

    # For each position, average all trajectories that have that position
    averaged = []
    for i in range(max_len):
        acts_at_i = []
        for traj in trajectories:
            if i < len(traj):
                acts_at_i.append(traj[i])

        if len(acts_at_i) > 0:
            averaged.append(np.mean(acts_at_i, axis=0))

    return np.array(averaged)


def plot_averaged_trajectories(
    correct_trajs: List[np.ndarray],
    incorrect_trajs: List[np.ndarray],
    pca: PCA,
    output_path: Path
):
    """Plot averaged trajectories for correct vs incorrect groups"""
    print("\nPlotting averaged trajectories...")

    # Compute averaged trajectories
    avg_correct = compute_averaged_trajectory(correct_trajs)
    avg_incorrect = compute_averaged_trajectory(incorrect_trajs)

    print(f"  Average correct trajectory length: {len(avg_correct)}")
    print(f"  Average incorrect trajectory length: {len(avg_incorrect)}")

    # Transform to PCA space
    avg_correct_pca = pca.transform(avg_correct)
    avg_incorrect_pca = pca.transform(avg_incorrect)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot correct averaged trajectory (thick green line with arrows)
    for i in range(len(avg_correct_pca) - 1):
        ax.annotate(
            '',
            xy=avg_correct_pca[i+1],
            xytext=avg_correct_pca[i],
            arrowprops=dict(
                arrowstyle='->',
                color='green',
                lw=3
            )
        )

    # Plot incorrect averaged trajectory (thick red line with arrows)
    for i in range(len(avg_incorrect_pca) - 1):
        ax.annotate(
            '',
            xy=avg_incorrect_pca[i+1],
            xytext=avg_incorrect_pca[i],
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=3
            )
        )

    # Mark waypoints with step numbers
    for i, point in enumerate(avg_correct_pca):
        if i == len(avg_correct_pca) - 1:
            label = 'Hash'
            marker = '*'
            size = 200
        else:
            label = f'S{i+1}'
            marker = 'o'
            size = 100

        ax.scatter(point[0], point[1], c='green', s=size, marker=marker,
                  edgecolors='darkgreen', linewidths=2, zorder=5)
        ax.annotate(label, (point[0], point[1]), fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

    for i, point in enumerate(avg_incorrect_pca):
        if i == len(avg_incorrect_pca) - 1:
            label = 'Hash'
            marker = '*'
            size = 200
        else:
            label = f'S{i+1}'
            marker = 'o'
            size = 100

        ax.scatter(point[0], point[1], c='red', s=size, marker=marker,
                  edgecolors='darkred', linewidths=2, zorder=5)
        ax.annotate(label, (point[0], point[1]), fontsize=10, fontweight='bold',
                   xytext=(5, -15), textcoords='offset points')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label=f'Correct (avg of {len(correct_trajs)})'),
        Line2D([0], [0], color='red', lw=3, label=f'Incorrect (avg of {len(incorrect_trajs)})'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)

    # Add filter info to title if filtering is applied
    title = 'Averaged Reasoning Trajectories: Correct vs Incorrect'
    if len(correct_trajs) > 0:
        num_steps = len(avg_correct) - 1
        title += f'\n(Filtered: {num_steps} reasoning steps)'
    elif len(incorrect_trajs) > 0:
        num_steps = len(avg_incorrect) - 1
        title += f'\n(Filtered: {num_steps} reasoning steps)'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved averaged trajectories plot to {output_path}")
    plt.close()


def plot_segmented_trajectories_publication(
    correct_trajs: List[np.ndarray],
    incorrect_trajs: List[np.ndarray],
    pca: PCA,
    output_path: Path,
    limit_steps: int = 4
):
    """Plot tiny subplots for each step-to-step segment with CI bars

    Creates 3 subplots in 1×3 grid, showing:
    - Subplot 1: Step 1 → 2
    - Subplot 2: Step 3 → 4
    - Subplot 3: Step 4 → Ans. Marker

    Each subplot shows correct vs incorrect with 95% CI.
    """
    print(f"\nPlotting segmented trajectories (publication mode, 3 segments)...")

    # Transform all trajectories to PCA space
    correct_trajs_pca = [pca.transform(traj) for traj in correct_trajs]
    incorrect_trajs_pca = [pca.transform(traj) for traj in incorrect_trajs]

    # Create subplots in 1×3 grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define which segments to plot: (start_step_idx, end_step_idx, display_name)
    segments_to_plot = [
        (0, 1, 'Step 1 → 2'),      # S1 → S2
        (2, 3, 'Step 3 → 4'),      # S3 → S4
        (3, 4, 'Step 4 → Ans. Marker')  # S4 → S5 (or Ans. Marker)
    ]

    # For each segment
    for plot_idx, (start_step, end_step, title_text) in enumerate(segments_to_plot):
        ax = axes[plot_idx]

        # Determine if end_step is the final answer marker
        max_traj_len = max(
            max(len(t) for t in correct_trajs_pca) if correct_trajs_pca else 0,
            max(len(t) for t in incorrect_trajs_pca) if incorrect_trajs_pca else 0
        )
        is_final_marker = (end_step == max_traj_len - 1)

        # Collect start and end points for this segment
        correct_starts = []
        correct_ends = []
        for traj_pca in correct_trajs_pca:
            if start_step < len(traj_pca) and end_step < len(traj_pca):
                correct_starts.append(traj_pca[start_step])
                correct_ends.append(traj_pca[end_step])

        incorrect_starts = []
        incorrect_ends = []
        for traj_pca in incorrect_trajs_pca:
            if start_step < len(traj_pca) and end_step < len(traj_pca):
                incorrect_starts.append(traj_pca[start_step])
                incorrect_ends.append(traj_pca[end_step])

        if len(correct_starts) == 0 and len(incorrect_starts) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('PC1', fontsize=13)
            ax.set_ylabel('PC2', fontsize=13)
            continue

        # Convert to arrays
        correct_starts = np.array(correct_starts) if correct_starts else None
        correct_ends = np.array(correct_ends) if correct_ends else None
        incorrect_starts = np.array(incorrect_starts) if incorrect_starts else None
        incorrect_ends = np.array(incorrect_ends) if incorrect_ends else None

        # Compute means and error bars
        if correct_starts is not None and len(correct_starts) >= 2:
            correct_start_mean, correct_start_x_err, correct_start_y_err = compute_error_bars(correct_starts)
            correct_end_mean, correct_end_x_err, correct_end_y_err = compute_error_bars(correct_ends)
        else:
            correct_start_mean = correct_starts[0] if correct_starts is not None and len(correct_starts) > 0 else None
            correct_end_mean = correct_ends[0] if correct_ends is not None and len(correct_ends) > 0 else None
            correct_start_x_err = correct_start_y_err = 0
            correct_end_x_err = correct_end_y_err = 0

        if incorrect_starts is not None and len(incorrect_starts) >= 2:
            incorrect_start_mean, incorrect_start_x_err, incorrect_start_y_err = compute_error_bars(incorrect_starts)
            incorrect_end_mean, incorrect_end_x_err, incorrect_end_y_err = compute_error_bars(incorrect_ends)
        else:
            incorrect_start_mean = incorrect_starts[0] if incorrect_starts is not None and len(incorrect_starts) > 0 else None
            incorrect_end_mean = incorrect_ends[0] if incorrect_ends is not None and len(incorrect_ends) > 0 else None
            incorrect_start_x_err = incorrect_start_y_err = 0
            incorrect_end_x_err = incorrect_end_y_err = 0

        # Plot correct trajectory segment
        if correct_start_mean is not None and correct_end_mean is not None:
            # Arrow from start to end
            ax.annotate(
                '',
                xy=correct_end_mean,
                xytext=correct_start_mean,
                arrowprops=dict(
                    arrowstyle='->',
                    color='green',
                    lw=2.5
                ),
                zorder=3
            )

            # Start point with error bars
            ax.scatter(correct_start_mean[0], correct_start_mean[1],
                      c='green', s=100, marker='o',
                      edgecolors='darkgreen', linewidths=2, zorder=5)
            if correct_start_x_err > 0:
                ax.errorbar(correct_start_mean[0], correct_start_mean[1],
                           xerr=correct_start_x_err, yerr=correct_start_y_err,
                           fmt='none', ecolor='darkgreen', alpha=0.4,
                           capsize=3, capthick=1.0, linewidth=0.8, zorder=4)

            # End point with error bars
            end_marker = '*' if is_final_marker else 'o'
            end_size = 180 if is_final_marker else 100
            ax.scatter(correct_end_mean[0], correct_end_mean[1],
                      c='green', s=end_size, marker=end_marker,
                      edgecolors='darkgreen', linewidths=2, zorder=5)
            if correct_end_x_err > 0:
                ax.errorbar(correct_end_mean[0], correct_end_mean[1],
                           xerr=correct_end_x_err, yerr=correct_end_y_err,
                           fmt='none', ecolor='darkgreen', alpha=0.4,
                           capsize=3, capthick=1.0, linewidth=0.8, zorder=4)

        # Plot incorrect trajectory segment
        if incorrect_start_mean is not None and incorrect_end_mean is not None:
            # Arrow from start to end
            ax.annotate(
                '',
                xy=incorrect_end_mean,
                xytext=incorrect_start_mean,
                arrowprops=dict(
                    arrowstyle='->',
                    color='red',
                    lw=2.5
                ),
                zorder=3
            )

            # Start point with error bars
            ax.scatter(incorrect_start_mean[0], incorrect_start_mean[1],
                      c='red', s=100, marker='o',
                      edgecolors='darkred', linewidths=2, zorder=5)
            if incorrect_start_x_err > 0:
                ax.errorbar(incorrect_start_mean[0], incorrect_start_mean[1],
                           xerr=incorrect_start_x_err, yerr=incorrect_start_y_err,
                           fmt='none', ecolor='darkred', alpha=0.4,
                           capsize=3, capthick=1.0, linewidth=0.8, zorder=4)

            # End point with error bars
            end_marker = '*' if is_final_marker else 'o'
            end_size = 180 if is_final_marker else 100
            ax.scatter(incorrect_end_mean[0], incorrect_end_mean[1],
                      c='red', s=end_size, marker=end_marker,
                      edgecolors='darkred', linewidths=2, zorder=5)
            if incorrect_end_x_err > 0:
                ax.errorbar(incorrect_end_mean[0], incorrect_end_mean[1],
                           xerr=incorrect_end_x_err, yerr=incorrect_end_y_err,
                           fmt='none', ecolor='darkred', alpha=0.4,
                           capsize=3, capthick=1.0, linewidth=0.8, zorder=4)

        # Formatting for this subplot
        ax.set_xlabel('PC1', fontsize=13)
        # Show Y-axis label only on leftmost subplot
        if plot_idx == 0:
            ax.set_ylabel('PC2', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(False)  # No gridlines
        ax.set_facecolor('#f9f9f9')

        # Title for this segment - use predefined title
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)

        # Only show legend on first subplot (lower left corner)
        if plot_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=2.5, label='Correct'),
                Line2D([0], [0], color='red', lw=2.5, label='Incorrect'),
                Line2D([0], [0], marker='|', color='darkgreen', linestyle='none',
                       markersize=10, markeredgewidth=1.5, label='95% CI'),
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved segmented trajectories plot to {output_path}")
    plt.close()


def plot_averaged_trajectories_publication(
    correct_trajs: List[np.ndarray],
    incorrect_trajs: List[np.ndarray],
    pca: PCA,
    output_path: Path
):
    """Plot averaged trajectories for correct vs incorrect groups with publication-ready formatting"""
    print("\nPlotting averaged trajectories (publication mode)...")

    # Compute averaged trajectories
    avg_correct = compute_averaged_trajectory(correct_trajs)
    avg_incorrect = compute_averaged_trajectory(incorrect_trajs)

    print(f"  Average correct trajectory length: {len(avg_correct)}")
    print(f"  Average incorrect trajectory length: {len(avg_incorrect)}")

    # Transform to PCA space
    avg_correct_pca = pca.transform(avg_correct)
    avg_incorrect_pca = pca.transform(avg_incorrect)

    # Also collect all individual trajectories in PCA space for ellipses
    correct_trajs_pca = []
    for traj in correct_trajs:
        traj_pca = pca.transform(traj)
        correct_trajs_pca.append(traj_pca)

    incorrect_trajs_pca = []
    for traj in incorrect_trajs:
        traj_pca = pca.transform(traj)
        incorrect_trajs_pca.append(traj_pca)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot correct averaged trajectory (thick green line with arrows)
    for i in range(len(avg_correct_pca) - 1):
        ax.annotate(
            '',
            xy=avg_correct_pca[i+1],
            xytext=avg_correct_pca[i],
            arrowprops=dict(
                arrowstyle='->',
                color='green',
                lw=3
            )
        )

    # Plot incorrect averaged trajectory (thick red line with arrows)
    for i in range(len(avg_incorrect_pca) - 1):
        ax.annotate(
            '',
            xy=avg_incorrect_pca[i+1],
            xytext=avg_incorrect_pca[i],
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=3
            )
        )

    # Add 95% CI error bars for each step (separate x and y)
    max_len = max(len(avg_correct_pca), len(avg_incorrect_pca))

    # Store error bars for plotting later
    correct_errors = []
    incorrect_errors = []

    for step_idx in range(max_len):
        # Collect all points at this step for correct trajectories
        correct_points_at_step = []
        for traj_pca in correct_trajs_pca:
            if step_idx < len(traj_pca):
                correct_points_at_step.append(traj_pca[step_idx])

        if len(correct_points_at_step) >= 2:
            correct_points_at_step = np.array(correct_points_at_step)
            mean, x_err, y_err = compute_error_bars(correct_points_at_step, n_std=1.96)
            correct_errors.append((mean, x_err, y_err))
        else:
            correct_errors.append((None, None, None))

        # Collect all points at this step for incorrect trajectories
        incorrect_points_at_step = []
        for traj_pca in incorrect_trajs_pca:
            if step_idx < len(traj_pca):
                incorrect_points_at_step.append(traj_pca[step_idx])

        if len(incorrect_points_at_step) >= 2:
            incorrect_points_at_step = np.array(incorrect_points_at_step)
            mean, x_err, y_err = compute_error_bars(incorrect_points_at_step, n_std=1.96)
            incorrect_errors.append((mean, x_err, y_err))
        else:
            incorrect_errors.append((None, None, None))

    # Mark waypoints with step numbers (smart labeling)
    # Determine plot range for threshold calculation
    all_points = np.vstack([avg_correct_pca, avg_incorrect_pca])
    x_range = all_points[:, 0].max() - all_points[:, 0].min()
    y_range = all_points[:, 1].max() - all_points[:, 1].min()
    plot_range = max(x_range, y_range)
    label_threshold = plot_range * 0.08  # 8% of plot range

    for i in range(max_len):
        # Get points (if they exist)
        correct_point = avg_correct_pca[i] if i < len(avg_correct_pca) else None
        incorrect_point = avg_incorrect_pca[i] if i < len(avg_incorrect_pca) else None

        # Determine label
        if i == max_len - 1 or (i == len(avg_correct_pca) - 1 and i == len(avg_incorrect_pca) - 1):
            label = 'Final Answer\nMarker'
            marker = '*'
            size = 200
        else:
            label = f'S{i+1}'
            marker = 'o'
            size = 100

        # Check if points are close
        if correct_point is not None and incorrect_point is not None:
            distance = np.linalg.norm(correct_point - incorrect_point)
            points_close = distance < label_threshold
        else:
            points_close = False

        # Plot correct point
        if correct_point is not None:
            ax.scatter(correct_point[0], correct_point[1], c='green', s=size, marker=marker,
                      edgecolors='darkgreen', linewidths=2, zorder=5)

            if points_close:
                # Label at midpoint if close
                if incorrect_point is not None:
                    mid_point = (correct_point + incorrect_point) / 2
                    ax.annotate(label, (mid_point[0], mid_point[1]), fontsize=10, fontweight='bold',
                               xytext=(5, 5), textcoords='offset points', ha='left')
            else:
                # Label separately if far
                ax.annotate(label, (correct_point[0], correct_point[1]), fontsize=10, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points', ha='left')

        # Plot incorrect point
        if incorrect_point is not None:
            ax.scatter(incorrect_point[0], incorrect_point[1], c='red', s=size, marker=marker,
                      edgecolors='darkred', linewidths=2, zorder=5)

            if not points_close:
                # Only label if not close (already labeled at midpoint above)
                ax.annotate(label, (incorrect_point[0], incorrect_point[1]), fontsize=10, fontweight='bold',
                           xytext=(5, -15), textcoords='offset points', ha='left')

    # Plot error bars for correct trajectories
    for i, (mean, x_err, y_err) in enumerate(correct_errors):
        if mean is not None and x_err is not None and y_err is not None:
            correct_point = avg_correct_pca[i] if i < len(avg_correct_pca) else None
            if correct_point is not None:
                ax.errorbar(correct_point[0], correct_point[1],
                           xerr=x_err, yerr=y_err,
                           fmt='none', ecolor='darkgreen', alpha=0.6,
                           capsize=4, capthick=1.5, zorder=3)

    # Plot error bars for incorrect trajectories
    for i, (mean, x_err, y_err) in enumerate(incorrect_errors):
        if mean is not None and x_err is not None and y_err is not None:
            incorrect_point = avg_incorrect_pca[i] if i < len(avg_incorrect_pca) else None
            if incorrect_point is not None:
                ax.errorbar(incorrect_point[0], incorrect_point[1],
                           xerr=x_err, yerr=y_err,
                           fmt='none', ecolor='darkred', alpha=0.6,
                           capsize=4, capthick=1.5, zorder=3)

    # Legend (remove "avg of xx")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label='Correct'),
        Line2D([0], [0], color='red', lw=3, label='Incorrect'),
        Line2D([0], [0], marker='|', color='darkgreen', linestyle='none',
               markersize=10, markeredgewidth=1.5, label='95% CI'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.95)

    # Font sizes matching visualize_linear_probe_results.py
    ax.set_xlabel('PC1', fontsize=13)
    ax.set_ylabel('PC2', fontsize=13)
    ax.tick_params(labelsize=11)

    # No master title in publication mode
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_facecolor('#f9f9f9')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved averaged trajectories plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot reasoning trajectories in activation space using PCA"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input NPZ file from collect_steering_vectors.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output plot path (e.g., trajectories.png)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of correct/incorrect samples to plot (default: 10)")
    parser.add_argument("--filter-steps", type=int, default=4,
                        help="Filter to only show questions with this many steps (default: 4, use 0 for no filtering)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer index to use (-1 for last layer, default: -1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--publication-ready", action="store_true",
                        help="Publication-ready mode: Creates tiny subplots for each segment (S1→S2, S2→S3, etc.); "
                             "Shows correct vs incorrect with 95%% CI error bars; "
                             "Uses clean visual style matching other publication plots")
    parser.add_argument("--publication-segments", action="store_true",
                        help="Create segmented subplot layout in publication mode (default: averaged plot)")

    args = parser.parse_args()

    # Handle filter_steps argument
    filter_steps = args.filter_steps if args.filter_steps > 0 else None

    print("=" * 100)
    print("PLOT REASONING TRAJECTORIES IN ACTIVATION SPACE")
    print("=" * 100)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"N samples: {args.n_samples}")
    print(f"Filter steps: {filter_steps if filter_steps is not None else 'None (no filtering)'}")
    print(f"Layer: {args.layer} (-1 = last layer)")
    print(f"Seed: {args.seed}")
    print("=" * 100)

    np.random.seed(args.seed)

    # Load data
    data = load_steering_data(args.input)

    # Build trajectories
    trajectories, correctness, question_ids = build_trajectories(data, layer_idx=args.layer)

    if len(trajectories) == 0:
        print("\n❌ Error: No trajectories found!")
        return 1

    # Filter by number of steps if specified
    trajectories, correctness, question_ids = filter_trajectories_by_steps(
        trajectories, correctness, question_ids, filter_steps
    )

    if len(trajectories) == 0:
        print(f"\n❌ Error: No trajectories found with {filter_steps} steps!")
        return 1

    # Sample trajectories
    correct_trajs, incorrect_trajs, correct_qids, incorrect_qids = sample_trajectories(
        trajectories, correctness, question_ids, args.n_samples
    )

    if len(correct_trajs) == 0 and len(incorrect_trajs) == 0:
        print("\n❌ Error: No trajectories to plot after sampling!")
        return 1

    # Fit PCA on all activations
    pca = fit_pca_on_all_activations(correct_trajs, incorrect_trajs)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Plot individual trajectories
    individual_output = args.output.parent / (args.output.stem + "_individual" + args.output.suffix)
    plot_individual_trajectories(
        correct_trajs, incorrect_trajs,
        correct_qids, incorrect_qids,
        pca, individual_output
    )

    # Plot averaged trajectories
    averaged_output = args.output.parent / (args.output.stem + "_averaged" + args.output.suffix)

    if args.publication_ready and args.publication_segments:
        # Use segmented subplot layout for publication
        segmented_output = args.output.parent / (args.output.stem + "_segments" + args.output.suffix)
        plot_segmented_trajectories_publication(
            correct_trajs, incorrect_trajs,
            pca, segmented_output,
            limit_steps=filter_steps if filter_steps else 4
        )
        print(f"Segmented plot: {segmented_output}")
    elif args.publication_ready:
        # Use publication-ready plotting function (averaged)
        plot_averaged_trajectories_publication(
            correct_trajs, incorrect_trajs,
            pca, averaged_output
        )
    else:
        # Use standard plotting function
        plot_averaged_trajectories(
            correct_trajs, incorrect_trajs,
            pca, averaged_output
        )

    print("\n" + "=" * 100)
    print("COMPLETE!")
    print("=" * 100)
    print(f"Individual trajectories: {individual_output}")
    print(f"Averaged trajectories: {averaged_output}")
    if args.publication_ready:
        print("  (Publication-ready mode enabled)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
