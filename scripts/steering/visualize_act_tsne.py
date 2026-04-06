#!/usr/bin/env python
"""Visualize steering vector activations using t-SNE

Single file mode:
- Step activations: light blue
- Hash activations: light red

Dual file mode (compare final vs hash):
- Group 1 (final): Blue family for Steps, Purple for Final Answer
- Group 2 (hash): Orange/Red family for Steps, Dark red for ####

Outputs 32 plots (one per layer) in a grid layout and individual files.

Usage:
    # Single file
    python scripts/steering/visualize_act_tsne.py \
        --data output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --output output/act_tsne_plots

    # Dual file comparison
    python scripts/steering/visualize_act_tsne.py \
        --data output/steering_vectors_final.npz \
        --data2 output/steering_vectors.npz \
        --output output/tsne_plots_comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from tqdm import tqdm


# Define distinct colors for steps 1-10 (highly distinguishable) - SINGLE FILE MODE
STEP_COLORS = {
    1: '#1f77b4',  # Blue
    2: '#2ca02c',  # Green
    3: '#ff7f0e',  # Orange
    4: '#9467bd',  # Purple
    5: '#8c564b',  # Brown
    6: '#e377c2',  # Pink
    7: '#17becf',  # Cyan
    8: '#bcbd22',  # Yellow-green
    9: '#d62728',  # Red (darker than hash)
    10: '#7f7f7f', # Gray
}

HASH_COLOR = '#ffb3ba'  # Light red/pink for Hash

# DUAL FILE MODE COLORS - Group 1 (final): Blue family, Group 2 (hash): Orange/Red family
# Raw mode colors
DUAL_RAW_COLORS = {
    'group1_step': '#87ceeb',      # Sky blue (for Final Answer file Steps)
    'group1_final': '#4b0082',     # Indigo/Dark purple (for Final Answer marker)
    'group2_step': '#ffcc99',      # Light orange (for Hash file Steps)
    'group2_hash': '#cc0000',      # Dark red (for #### marker)
}

# Stepwise mode colors - Group 1 (Blue family)
DUAL_GROUP1_STEP_COLORS = {
    1: '#1e90ff',  # Dodger blue
    2: '#4169e1',  # Royal blue
    3: '#0000cd',  # Medium blue
    4: '#000080',  # Navy
    5: '#6495ed',  # Cornflower blue
    6: '#4682b4',  # Steel blue
    7: '#5f9ea0',  # Cadet blue
    8: '#00bfff',  # Deep sky blue
    9: '#1c86ee',  # Dodger blue 2
    10: '#104e8b', # Dodger blue 4
}
DUAL_GROUP1_FINAL_COLOR = '#4b0082'  # Indigo (for Final Answer)

# Stepwise mode colors - Group 2 (Orange/Red family)
DUAL_GROUP2_STEP_COLORS = {
    1: '#ff8c00',  # Dark orange
    2: '#ff7f50',  # Coral
    3: '#ff6347',  # Tomato
    4: '#ff4500',  # Orange red
    5: '#ff8c69',  # Salmon
    6: '#cd5c5c',  # Indian red
    7: '#dc143c',  # Crimson
    8: '#b22222',  # Fire brick
    9: '#8b0000',  # Dark red
    10: '#a52a2a', # Brown
}
DUAL_GROUP2_HASH_COLOR = '#cc0000'  # Dark red (for ####)


def plot_tsne_layer(
    step_acts: np.ndarray,
    hash_acts: np.ndarray,
    step_numbers: np.ndarray,
    layer_idx: int,
    output_path: Path = None,
    ax=None,
    raw_mode: bool = False,
    max_step: int = 6,
    color_by_correctness: bool = False,
    is_correct_step: np.ndarray = None,
    is_correct_hash: np.ndarray = None,
    publication_ready: bool = False,
    is_first_subplot: bool = False
):
    """Create t-SNE plot for a single layer

    Args:
        step_acts: [n_step, hidden_dim]
        hash_acts: [n_hash, hidden_dim]
        step_numbers: [n_step] - step number for each activation (1, 2, 3, ...)
        layer_idx: Layer index for title
        output_path: If provided, save individual plot
        ax: If provided, plot on this axis (for grid)
        raw_mode: If True, plot all steps as single light blue color
        max_step: Maximum step number to include (filter out steps > max_step)
    """
    # Filter step activations by max_step
    if len(step_numbers) > 0 and max_step is not None:
        step_mask = step_numbers <= max_step
        step_acts = step_acts[step_mask]
        step_numbers = step_numbers[step_mask]
        if is_correct_step is not None and len(is_correct_step) > 0:
            is_correct_step = is_correct_step[step_mask]

    if len(step_acts) == 0 or len(hash_acts) == 0:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    # Combine activations
    all_acts = np.vstack([step_acts, hash_acts])
    labels = np.array([0] * len(step_acts) + [1] * len(hash_acts))

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_acts) - 1))
    embeddings = tsne.fit_transform(all_acts)

    # Split back into step and hash
    step_embeddings = embeddings[labels == 0]
    hash_embeddings = embeddings[labels == 1]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone = True
    else:
        standalone = False

    # Plot Step activations
    if raw_mode and color_by_correctness and is_correct_step is not None and len(is_correct_step) > 0:
        # Raw mode with correctness: darker blue=correct, lighter blue=incorrect
        correct_mask = is_correct_step
        incorrect_mask = ~is_correct_step

        if np.any(correct_mask):
            ax.scatter(step_embeddings[correct_mask, 0], step_embeddings[correct_mask, 1],
                      c='#0066cc', alpha=0.8, s=35, label='Step (Correct)', edgecolors='darkblue', linewidths=0.3)
        if np.any(incorrect_mask):
            ax.scatter(step_embeddings[incorrect_mask, 0], step_embeddings[incorrect_mask, 1],
                      c='#add8e6', alpha=0.6, s=35, label='Step (Incorrect)', edgecolors='blue', linewidths=0.3)
    elif color_by_correctness and is_correct_step is not None and len(is_correct_step) > 0:
        # Color by correctness: green=correct, red=incorrect
        correct_mask = is_correct_step
        incorrect_mask = ~is_correct_step

        if np.any(correct_mask):
            ax.scatter(step_embeddings[correct_mask, 0], step_embeddings[correct_mask, 1],
                      c='green', alpha=0.7, s=35, label='Step (Correct)', edgecolors='darkgreen', linewidths=0.3)
        if np.any(incorrect_mask):
            ax.scatter(step_embeddings[incorrect_mask, 0], step_embeddings[incorrect_mask, 1],
                      c='red', alpha=0.7, s=35, label='Step (Incorrect)', edgecolors='darkred', linewidths=0.3)
    elif raw_mode:
        # Raw mode: plot all steps as single light blue color
        ax.scatter(step_embeddings[:, 0], step_embeddings[:, 1],
                  c='lightblue', alpha=0.6, s=35, label='Step', edgecolors='blue', linewidths=0.3)
    elif len(step_numbers) > 0:
        # Plot with different colors for each step number
        unique_steps = np.unique(step_numbers)

        for step_num in unique_steps:
            mask = step_numbers == step_num
            step_embeddings_subset = step_embeddings[mask]

            # Select color from global color map (cycle through if > 10)
            step_num_int = int(step_num)
            if step_num_int in STEP_COLORS:
                color = STEP_COLORS[step_num_int]
            else:
                # Cycle through colors for steps > 10
                color_idx = ((step_num_int - 1) % len(STEP_COLORS)) + 1
                color = STEP_COLORS[color_idx]

            ax.scatter(
                step_embeddings_subset[:, 0],
                step_embeddings_subset[:, 1],
                c=color,
                alpha=0.8,
                s=35,
                label=f'Step {step_num}',
                edgecolors='black',
                linewidths=0.3
            )
    else:
        # Fallback: plot all steps as light blue
        ax.scatter(step_embeddings[:, 0], step_embeddings[:, 1],
                  c='lightblue', alpha=0.6, s=30, label='Step', edgecolors='blue', linewidths=0.5)

    # Plot Hash activations
    if raw_mode and color_by_correctness and is_correct_hash is not None and len(is_correct_hash) > 0:
        # Raw mode with correctness: darker red=correct, lighter red=incorrect
        correct_mask = is_correct_hash
        incorrect_mask = ~is_correct_hash

        if np.any(correct_mask):
            ax.scatter(hash_embeddings[correct_mask, 0], hash_embeddings[correct_mask, 1],
                      c='#cc0000', alpha=0.8, s=35, label='#### (Correct)', edgecolors='darkred', linewidths=0.3)
        if np.any(incorrect_mask):
            ax.scatter(hash_embeddings[incorrect_mask, 0], hash_embeddings[incorrect_mask, 1],
                      c='#ffcccc', alpha=0.6, s=35, label='#### (Incorrect)', edgecolors='red', linewidths=0.3)
    elif color_by_correctness and is_correct_hash is not None and len(is_correct_hash) > 0:
        # Color by correctness for hash too
        correct_mask = is_correct_hash
        incorrect_mask = ~is_correct_hash

        if np.any(correct_mask):
            ax.scatter(hash_embeddings[correct_mask, 0], hash_embeddings[correct_mask, 1],
                      c='darkgreen', alpha=0.8, s=35, label='#### (Correct)', edgecolors='black', linewidths=0.3)
        if np.any(incorrect_mask):
            ax.scatter(hash_embeddings[incorrect_mask, 0], hash_embeddings[incorrect_mask, 1],
                      c='darkred', alpha=0.8, s=35, label='#### (Incorrect)', edgecolors='black', linewidths=0.3)
    else:
        # Default: light red/pink for all hash
        hash_label = 'Final Answer Marker' if publication_ready else '####'
        ax.scatter(hash_embeddings[:, 0], hash_embeddings[:, 1],
                  c=HASH_COLOR, alpha=0.8, s=35, label=hash_label, edgecolors='darkred', linewidths=0.3)

    if publication_ready:
        # Simplified title for publication (match linear probe fontsize)
        ax.set_title(f"Layer {layer_idx}", fontsize=16, fontweight='bold')
        ax.set_xlabel("t-SNE Dim 1", fontsize=13)
        ax.set_ylabel("t-SNE Dim 2", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_title(f"Layer {layer_idx} (n_step={len(step_acts)}, n_hash={len(hash_acts)})", fontsize=10)
        ax.set_xlabel("t-SNE Dim 1", fontsize=8)
        ax.set_ylabel("t-SNE Dim 2", fontsize=8)
        ax.tick_params(labelsize=6)

    if not publication_ready:
        ax.grid(True, alpha=0.3)

    # Add small legend in leftmost subplot for publication mode (no box/frame)
    if publication_ready and is_first_subplot and len(step_numbers) > 0:
        ax.legend(fontsize=9, loc='lower left', frameon=False)

    if standalone:
        # Add legend for standalone plots
        if len(step_numbers) > 0:
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_tsne_layer_dual(
    g1_step_acts: np.ndarray,
    g1_final_acts: np.ndarray,
    g1_step_numbers: np.ndarray,
    g2_step_acts: np.ndarray,
    g2_hash_acts: np.ndarray,
    g2_step_numbers: np.ndarray,
    layer_idx: int,
    output_path: Path = None,
    ax=None,
    raw_mode: bool = False,
    max_step: int = 6,
):
    """Create t-SNE plot for a single layer comparing two groups (dual file mode)

    Args:
        g1_step_acts: [n_step, hidden_dim] - Group 1 (final) step activations
        g1_final_acts: [n_final, hidden_dim] - Group 1 "Final Answer:" activations
        g1_step_numbers: [n_step] - step numbers for group 1
        g2_step_acts: [n_step, hidden_dim] - Group 2 (hash) step activations
        g2_hash_acts: [n_hash, hidden_dim] - Group 2 "####" activations
        g2_step_numbers: [n_step] - step numbers for group 2
        layer_idx: Layer index for title
        output_path: If provided, save individual plot
        ax: If provided, plot on this axis (for grid)
        raw_mode: If True, use simple coloring (one for Step, one for Final/####)
        max_step: Maximum step number to include
    """
    # Filter by max_step
    if max_step is not None:
        if len(g1_step_numbers) > 0:
            mask = g1_step_numbers <= max_step
            g1_step_acts = g1_step_acts[mask]
            g1_step_numbers = g1_step_numbers[mask]
        if len(g2_step_numbers) > 0:
            mask = g2_step_numbers <= max_step
            g2_step_acts = g2_step_acts[mask]
            g2_step_numbers = g2_step_numbers[mask]

    # Check if we have data
    total_points = (len(g1_step_acts) + len(g1_final_acts) +
                   len(g2_step_acts) + len(g2_hash_acts))

    if total_points == 0:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    # Combine all activations for t-SNE
    all_acts = []
    labels = []  # To track which group/type each point belongs to

    # Group 1 - Steps (label: 0)
    if len(g1_step_acts) > 0:
        all_acts.append(g1_step_acts)
        labels.extend([0] * len(g1_step_acts))

    # Group 1 - Final Answer (label: 1)
    if len(g1_final_acts) > 0:
        all_acts.append(g1_final_acts)
        labels.extend([1] * len(g1_final_acts))

    # Group 2 - Steps (label: 2)
    if len(g2_step_acts) > 0:
        all_acts.append(g2_step_acts)
        labels.extend([2] * len(g2_step_acts))

    # Group 2 - Hash (label: 3)
    if len(g2_hash_acts) > 0:
        all_acts.append(g2_hash_acts)
        labels.extend([3] * len(g2_hash_acts))

    if not all_acts:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    all_acts = np.vstack(all_acts)
    labels = np.array(labels)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_acts) - 1))
    embeddings = tsne.fit_transform(all_acts)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone = True
    else:
        standalone = False

    # RAW MODE: Simple coloring
    if raw_mode:
        # Group 1 - Steps (light blue)
        mask = labels == 0
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_RAW_COLORS['group1_step'], alpha=0.7, s=35,
                      label='Final file - Step', edgecolors='#4169e1', linewidths=0.3)

        # Group 1 - Final Answer (dark purple)
        mask = labels == 1
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_RAW_COLORS['group1_final'], alpha=0.9, s=40,
                      label='Final file - Final Answer', edgecolors='black', linewidths=0.5)

        # Group 2 - Steps (light orange)
        mask = labels == 2
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_RAW_COLORS['group2_step'], alpha=0.7, s=35,
                      label='Hash file - Step', edgecolors='#ff8c00', linewidths=0.3)

        # Group 2 - Hash (dark red)
        mask = labels == 3
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_RAW_COLORS['group2_hash'], alpha=0.9, s=40,
                      label='Hash file - ####', edgecolors='black', linewidths=0.5)

    # STEPWISE MODE: Color each step differently
    else:
        # Group 1 - Steps (blue family, by step number)
        g1_step_mask = labels == 0
        if np.any(g1_step_mask):
            g1_step_embeddings = embeddings[g1_step_mask]
            unique_steps = np.unique(g1_step_numbers)

            for step_num in unique_steps:
                step_num_int = int(step_num)
                mask_in_g1 = g1_step_numbers == step_num

                # Get color
                if step_num_int in DUAL_GROUP1_STEP_COLORS:
                    color = DUAL_GROUP1_STEP_COLORS[step_num_int]
                else:
                    color_idx = ((step_num_int - 1) % len(DUAL_GROUP1_STEP_COLORS)) + 1
                    color = DUAL_GROUP1_STEP_COLORS[color_idx]

                ax.scatter(
                    g1_step_embeddings[mask_in_g1, 0],
                    g1_step_embeddings[mask_in_g1, 1],
                    c=color, alpha=0.8, s=35,
                    label=f'Final - Step {step_num}',
                    edgecolors='#000080', linewidths=0.3
                )

        # Group 1 - Final Answer (dark purple)
        mask = labels == 1
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_GROUP1_FINAL_COLOR, alpha=0.9, s=40,
                      label='Final - Answer', edgecolors='black', linewidths=0.5)

        # Group 2 - Steps (orange/red family, by step number)
        g2_step_mask = labels == 2
        if np.any(g2_step_mask):
            g2_step_embeddings = embeddings[g2_step_mask]
            unique_steps = np.unique(g2_step_numbers)

            for step_num in unique_steps:
                step_num_int = int(step_num)
                mask_in_g2 = g2_step_numbers == step_num

                # Get color
                if step_num_int in DUAL_GROUP2_STEP_COLORS:
                    color = DUAL_GROUP2_STEP_COLORS[step_num_int]
                else:
                    color_idx = ((step_num_int - 1) % len(DUAL_GROUP2_STEP_COLORS)) + 1
                    color = DUAL_GROUP2_STEP_COLORS[color_idx]

                ax.scatter(
                    g2_step_embeddings[mask_in_g2, 0],
                    g2_step_embeddings[mask_in_g2, 1],
                    c=color, alpha=0.8, s=35,
                    label=f'Hash - Step {step_num}',
                    edgecolors='#8b0000', linewidths=0.3
                )

        # Group 2 - Hash (dark red)
        mask = labels == 3
        if np.any(mask):
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=DUAL_GROUP2_HASH_COLOR, alpha=0.9, s=40,
                      label='Hash - ####', edgecolors='black', linewidths=0.5)

    # Set title with counts
    n_g1 = len(g1_step_acts) + len(g1_final_acts)
    n_g2 = len(g2_step_acts) + len(g2_hash_acts)
    ax.set_title(f"Layer {layer_idx} (Final file: {n_g1}, Hash file: {n_g2})", fontsize=10)
    ax.set_xlabel("t-SNE Dim 1", fontsize=8)
    ax.set_ylabel("t-SNE Dim 2", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    if standalone:
        ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_per_step_correctness(
    step_acts_all_layers,
    hash_acts_all_layers,
    step_numbers_all_layers,
    is_correct_step_all_layers,
    is_correct_hash_all_layers,
    num_layers: int,
    output_dir: Path
):
    """Create separate plots for each step (1-5) and ####, showing correct vs incorrect

    Args:
        step_acts_all_layers: List of [n_step, hidden_dim] for each layer
        hash_acts_all_layers: List of [n_hash, hidden_dim] for each layer
        step_numbers_all_layers: List of [n_step] for each layer
        is_correct_step_all_layers: List of [n_step] for each layer
        is_correct_hash_all_layers: List of [n_hash] for each layer
        num_layers: Number of layers
        output_dir: Directory to save plots
    """
    print(f"\nCreating per-step correctness plots...")

    # Steps to visualize: 1-5 and "####"
    steps_to_plot = [1, 2, 3, 4, 5, "####"]

    for step_target in steps_to_plot:
        print(f"\n  Processing {'Step ' + str(step_target) if step_target != '####' else '####'}...")

        # Create figure with 32 subplots (4x8 grid)
        n_cols = 8
        n_rows = (num_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for layer_idx in tqdm(range(num_layers), desc=f"  Layers", ncols=100):
            ax = axes[layer_idx]

            if step_target == "####":
                # Use hash activations
                acts = hash_acts_all_layers[layer_idx]
                is_correct = is_correct_hash_all_layers[layer_idx] if is_correct_hash_all_layers is not None else None

                if len(acts) == 0:
                    ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo #### data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Run t-SNE on hash activations only
                if len(acts) >= 2:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(acts) - 1))
                    embeddings = tsne.fit_transform(acts)

                    # Plot by correctness
                    if is_correct is not None and len(is_correct) > 0:
                        correct_mask = is_correct
                        incorrect_mask = ~is_correct

                        if np.any(correct_mask):
                            ax.scatter(embeddings[correct_mask, 0], embeddings[correct_mask, 1],
                                      c='darkgreen', alpha=0.7, s=35, label='Correct',
                                      edgecolors='black', linewidths=0.3)
                        if np.any(incorrect_mask):
                            ax.scatter(embeddings[incorrect_mask, 0], embeddings[incorrect_mask, 1],
                                      c='darkred', alpha=0.7, s=35, label='Incorrect',
                                      edgecolors='black', linewidths=0.3)
                    else:
                        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                  c='purple', alpha=0.7, s=35, label='####')
                else:
                    ax.text(0.5, 0.5, f"Layer {layer_idx}\nNot enough data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

            else:
                # Use step activations for this specific step
                step_acts = step_acts_all_layers[layer_idx]
                step_nums = step_numbers_all_layers[layer_idx]
                is_correct = is_correct_step_all_layers[layer_idx] if is_correct_step_all_layers is not None else None

                # Filter to only this step
                if len(step_nums) > 0:
                    mask = step_nums == step_target
                    acts = step_acts[mask]
                    if is_correct is not None and len(is_correct) > 0:
                        is_correct_filtered = is_correct[mask]
                    else:
                        is_correct_filtered = None
                else:
                    acts = np.array([])
                    is_correct_filtered = None

                if len(acts) == 0:
                    ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo Step {step_target} data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Run t-SNE on this step's activations only
                if len(acts) >= 2:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(acts) - 1))
                    embeddings = tsne.fit_transform(acts)

                    # Plot by correctness
                    if is_correct_filtered is not None and len(is_correct_filtered) > 0:
                        correct_mask = is_correct_filtered
                        incorrect_mask = ~is_correct_filtered

                        if np.any(correct_mask):
                            ax.scatter(embeddings[correct_mask, 0], embeddings[correct_mask, 1],
                                      c='darkgreen', alpha=0.7, s=35, label='Correct',
                                      edgecolors='black', linewidths=0.3)
                        if np.any(incorrect_mask):
                            ax.scatter(embeddings[incorrect_mask, 0], embeddings[incorrect_mask, 1],
                                      c='darkred', alpha=0.7, s=35, label='Incorrect',
                                      edgecolors='black', linewidths=0.3)
                    else:
                        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                  c='blue', alpha=0.7, s=35, label=f'Step {step_target}')
                else:
                    ax.text(0.5, 0.5, f"Layer {layer_idx}\nNot enough data",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

            ax.set_title(f"Layer {layer_idx} (n={len(acts)})", fontsize=10)
            ax.set_xlabel("t-SNE Dim 1", fontsize=8)
            ax.set_ylabel("t-SNE Dim 2", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

        # Add master legend
        legend_handles = [
            Patch(facecolor='darkgreen', edgecolor='black', label='Correct'),
            Patch(facecolor='darkred', edgecolor='black', label='Incorrect')
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=2,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=12,
            frameon=True,
            title="Baseline Correctness",
            title_fontsize=14
        )

        # Set title
        step_name = f"Step {step_target}" if step_target != "####" else "####"
        plt.suptitle(f"t-SNE: {step_name} Activations - Correct vs Incorrect Across Layers",
                    fontsize=16, y=0.998)
        plt.tight_layout(rect=[0, 0.02, 1, 0.995])

        # Save
        output_filename = f"tsne_{'step_' + str(step_target) if step_target != '####' else 'hash'}_correctness.png"
        output_path = output_dir / output_filename
        print(f"  Saving to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {step_name} plot saved!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize steering vector activations with t-SNE"
    )
    parser.add_argument("--data", type=Path,
                        default=Path("output/steering_vectors.npz"),
                        help="Path to steering vectors NPZ file (Group 1)")
    parser.add_argument("--data2", type=Path, default=None,
                        help="Path to second steering vectors NPZ file (Group 2). "
                             "If provided, enables dual-file comparison mode.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/tsne_plots"),
                        help="Directory to save plots")
    parser.add_argument("--individual", action="store_true",
                        help="Save individual plots for each layer")
    parser.add_argument("--raw-mode", action="store_true",
                        help="Plot all steps as single color (no per-step coloring). "
                             "In dual mode: Group1=blue/purple, Group2=orange/red")
    parser.add_argument("--max-step", type=int, default=6,
                        help="Maximum step number to include (default: 6, ignores step 7+)")
    parser.add_argument("--color-by-correctness", action="store_true",
                        help="[Single file mode only] Color points by baseline correctness. "
                             "If used with --raw-mode: darker shades=correct, lighter shades=incorrect. "
                             "Otherwise: green=correct, red=incorrect")
    parser.add_argument("--per-step-correctness", action="store_true",
                        help="[Single file mode only] Create separate plots for each step (1-5) and ####, "
                             "showing correct vs incorrect across all 32 layers. Generates 6 plots total.")
    parser.add_argument("--publication-ready", action="store_true",
                        help="Publication-ready mode: Only plot layers 0, 11, 21, 31; Steps 1-5 only; "
                             "Per-panel titles only ('Layer X'); No master title; Rename #### to 'Final Answer Marker'; "
                             "Remove legend border. Overrides other visualization parameters.")
    parser.add_argument("--publication-appendix", action="store_true",
                        help="Publication appendix mode: Same clean formatting as --publication-ready, "
                             "but plots ALL 32 layers in an 8x4 grid. Steps 1-5 only; Per-panel titles; "
                             "No master title; 'Final Answer Marker' label; No legend border.")

    args = parser.parse_args()

    # Publication-ready mode: override parameters
    if args.publication_ready:
        print("\n🎨 PUBLICATION-READY MODE ACTIVATED")
        print("   Overriding parameters:")
        print("   - Layers: Only 0, 11, 21, 31")
        print("   - Steps: 1-5 only (max_step=5)")
        print("   - Titles: Per-panel only")
        print("   - Legend: No border, '####' → 'Final Answer Marker'")
        args.max_step = 5

    # Publication-appendix mode: override parameters
    if args.publication_appendix:
        print("\n📑 PUBLICATION-APPENDIX MODE ACTIVATED")
        print("   Overriding parameters:")
        print("   - Layers: ALL 32 layers (8x4 grid)")
        print("   - Steps: 1-5 only (max_step=5)")
        print("   - Titles: Per-panel only")
        print("   - Legend: No border, '####' → 'Final Answer Marker'")
        args.max_step = 5

    # Check if dual-file mode
    is_dual_mode = args.data2 is not None

    print(f"\n{'='*100}")
    if is_dual_mode:
        print("VISUALIZE STEERING ACTIVATIONS (t-SNE) - DUAL FILE COMPARISON")
    else:
        print("VISUALIZE STEERING ACTIVATIONS (t-SNE)")
    print(f"{'='*100}")
    print(f"Data (Group 1): {args.data}")
    if is_dual_mode:
        print(f"Data (Group 2): {args.data2}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'Dual-file comparison' if is_dual_mode else 'Single file'}")
    print(f"{'='*100}\n")

    # Validate arguments for dual mode
    if is_dual_mode:
        if args.color_by_correctness:
            print("⚠️  Warning: --color-by-correctness is ignored in dual-file mode")
            args.color_by_correctness = False
        if args.per_step_correctness:
            print("❌ Error: --per-step-correctness is not supported in dual-file mode")
            return 1

    # Load data
    print("Loading data...")
    data = np.load(args.data, allow_pickle=True)

    num_layers = int(data["num_layers"])
    step_acts = data["step_activations"]
    hash_acts = data["hash_activations"]
    step_numbers = data.get("step_numbers", None)  # May not exist in old files
    is_correct_step = data.get("is_correct_step", None)  # May not exist in old files
    is_correct_hash = data.get("is_correct_hash", None)  # May not exist in old files

    # Load stats
    if "stats" in data:
        stats = json.loads(str(data["stats"]))
        print(f"\nCollection statistics:")
        print(f"  Successful questions: {stats.get('successful_questions', 'N/A')}")
        print(f"  Total differences: {stats.get('total_differences', 'N/A')}")

    print(f"\nData loaded:")
    print(f"  Num layers: {num_layers}")
    print(f"  Step activations: {[len(step_acts[i]) for i in range(min(5, num_layers))]}... (first 5 layers)")
    print(f"  Hash activations: {[len(hash_acts[i]) for i in range(min(5, num_layers))]}... (first 5 layers)")

    if step_numbers is not None:
        print(f"  Step numbers available: Yes")
        # Show step distribution for first layer
        if len(step_numbers[0]) > 0:
            unique, counts = np.unique(step_numbers[0], return_counts=True)
            print(f"  Layer 0 step distribution: {dict(zip(unique, counts))}")
    else:
        print(f"  Step numbers available: No (old format)")
        # Create dummy step numbers (all 1)
        step_numbers = [np.ones(len(step_acts[i]), dtype=np.int32) if len(step_acts[i]) > 0 else np.array([], dtype=np.int32)
                       for i in range(num_layers)]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # DUAL-FILE MODE: Load second file and create comparison plots
    # ===================================================================
    if is_dual_mode:
        print("\n" + "="*100)
        print("DUAL-FILE MODE: Loading second file...")
        print("="*100)

        # Load second file
        data2 = np.load(args.data2, allow_pickle=True)

        num_layers2 = int(data2["num_layers"])
        step_acts2 = data2["step_activations"]
        hash_acts2 = data2["hash_activations"]
        step_numbers2 = data2.get("step_numbers", None)

        # Validate compatibility
        if num_layers != num_layers2:
            print(f"❌ Error: Number of layers mismatch!")
            print(f"  Group 1: {num_layers} layers")
            print(f"  Group 2: {num_layers2} layers")
            return 1

        print(f"Group 2 data loaded:")
        print(f"  Num layers: {num_layers2}")
        print(f"  Step activations: {[len(step_acts2[i]) for i in range(min(5, num_layers2))]}... (first 5 layers)")
        print(f"  Hash activations: {[len(hash_acts2[i]) for i in range(min(5, num_layers2))]}... (first 5 layers)")

        # Handle missing step numbers
        if step_numbers2 is None:
            print(f"  Step numbers available: No (old format) - creating dummy")
            step_numbers2 = [np.ones(len(step_acts2[i]), dtype=np.int32) if len(step_acts2[i]) > 0 else np.array([], dtype=np.int32)
                           for i in range(num_layers2)]
        else:
            print(f"  Step numbers available: Yes")

        # Create grid plot (all layers) - DUAL MODE
        print(f"\nCreating DUAL-FILE comparison grid plot with all {num_layers} layers...")

        n_cols = 8
        n_rows = (num_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows + 0.8))
        axes = axes.flatten()

        for layer_idx in tqdm(range(num_layers), desc="Processing layers", ncols=100):
            plot_tsne_layer_dual(
                g1_step_acts=step_acts[layer_idx],
                g1_final_acts=hash_acts[layer_idx],  # "Final Answer:" from Group 1
                g1_step_numbers=step_numbers[layer_idx],
                g2_step_acts=step_acts2[layer_idx],
                g2_hash_acts=hash_acts2[layer_idx],  # "####" from Group 2
                g2_step_numbers=step_numbers2[layer_idx],
                layer_idx=layer_idx,
                ax=axes[layer_idx],
                raw_mode=args.raw_mode,
                max_step=args.max_step,
            )

        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

        # Create master legend
        legend_handles = []
        if args.raw_mode:
            # Raw mode - 4 categories
            legend_handles.append(Patch(facecolor=DUAL_RAW_COLORS['group1_step'],
                                      edgecolor='#4169e1', label='Final file - Step'))
            legend_handles.append(Patch(facecolor=DUAL_RAW_COLORS['group1_final'],
                                      edgecolor='black', label='Final file - Final Answer'))
            legend_handles.append(Patch(facecolor=DUAL_RAW_COLORS['group2_step'],
                                      edgecolor='#ff8c00', label='Hash file - Step'))
            legend_handles.append(Patch(facecolor=DUAL_RAW_COLORS['group2_hash'],
                                      edgecolor='black', label='Hash file - ####'))
        else:
            # Stepwise mode - show steps for both groups
            # Group 1 steps
            all_unique_steps = set()
            for layer_idx in range(num_layers):
                if len(step_numbers[layer_idx]) > 0:
                    # Apply max_step filter
                    layer_steps = step_numbers[layer_idx]
                    if args.max_step is not None:
                        layer_steps = layer_steps[layer_steps <= args.max_step]
                    all_unique_steps.update(layer_steps)
                if len(step_numbers2[layer_idx]) > 0:
                    # Apply max_step filter
                    layer_steps2 = step_numbers2[layer_idx]
                    if args.max_step is not None:
                        layer_steps2 = layer_steps2[layer_steps2 <= args.max_step]
                    all_unique_steps.update(layer_steps2)
            all_unique_steps = sorted(all_unique_steps)

            for step_num in all_unique_steps:
                step_num_int = int(step_num)
                if step_num_int in DUAL_GROUP1_STEP_COLORS:
                    color = DUAL_GROUP1_STEP_COLORS[step_num_int]
                else:
                    color_idx = ((step_num_int - 1) % len(DUAL_GROUP1_STEP_COLORS)) + 1
                    color = DUAL_GROUP1_STEP_COLORS[color_idx]
                legend_handles.append(Patch(facecolor=color, edgecolor='#000080',
                                          label=f'Final - Step {step_num}'))

            legend_handles.append(Patch(facecolor=DUAL_GROUP1_FINAL_COLOR, edgecolor='black',
                                      label='Final - Answer'))

            for step_num in all_unique_steps:
                step_num_int = int(step_num)
                if step_num_int in DUAL_GROUP2_STEP_COLORS:
                    color = DUAL_GROUP2_STEP_COLORS[step_num_int]
                else:
                    color_idx = ((step_num_int - 1) % len(DUAL_GROUP2_STEP_COLORS)) + 1
                    color = DUAL_GROUP2_STEP_COLORS[color_idx]
                legend_handles.append(Patch(facecolor=color, edgecolor='#8b0000',
                                          label=f'Hash - Step {step_num}'))

            legend_handles.append(Patch(facecolor=DUAL_GROUP2_HASH_COLOR, edgecolor='black',
                                      label='Hash - ####'))

        # Add legend
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(len(legend_handles), 8),
            bbox_to_anchor=(0.5, -0.01),
            fontsize=11,
            frameon=True,
            title="Groups & Step Numbers" if not args.raw_mode else "Groups",
            title_fontsize=13
        )

        plt.suptitle("t-SNE Visualization: Final Answer File vs Hash File Comparison",
                    fontsize=16, y=0.998)
        plt.tight_layout(rect=[0, 0.02, 1, 0.995])

        grid_output = args.output_dir / "tsne_all_layers_dual_comparison.png"
        print(f"\nSaving dual-file comparison grid plot to {grid_output}...")
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Dual-file comparison grid plot saved!")

        # Individual plots if requested
        if args.individual:
            print(f"\nCreating individual dual-comparison plots for each layer...")

            individual_dir = args.output_dir / "individual_dual"
            individual_dir.mkdir(exist_ok=True)

            for layer_idx in tqdm(range(num_layers), desc="Saving individual plots", ncols=100):
                output_path = individual_dir / f"layer_{layer_idx:02d}_tsne_dual.png"
                plot_tsne_layer_dual(
                    g1_step_acts=step_acts[layer_idx],
                    g1_final_acts=hash_acts[layer_idx],
                    g1_step_numbers=step_numbers[layer_idx],
                    g2_step_acts=step_acts2[layer_idx],
                    g2_hash_acts=hash_acts2[layer_idx],
                    g2_step_numbers=step_numbers2[layer_idx],
                    layer_idx=layer_idx,
                    output_path=output_path,
                    raw_mode=args.raw_mode,
                    max_step=args.max_step,
                )

            print(f"✓ Individual dual-comparison plots saved to {individual_dir}/")

        print(f"\n{'='*100}")
        print("DUAL-FILE COMPARISON VISUALIZATION COMPLETE!")
        print(f"{'='*100}")
        print(f"Outputs:")
        print(f"  Grid plot: {grid_output}")
        if args.individual:
            print(f"  Individual plots: {individual_dir}/")
        print()

        return 0

    # ===================================================================
    # SINGLE-FILE MODE (original logic)
    # ===================================================================

    # If per-step correctness mode, create those plots and exit
    if args.per_step_correctness:
        if is_correct_step is None or is_correct_hash is None:
            print("\n❌ Error: --per-step-correctness requires correctness data in NPZ file!")
            print("   Please re-run steering vector collection to include correctness data.")
            return 1

        plot_per_step_correctness(
            step_acts,
            hash_acts,
            step_numbers,
            is_correct_step,
            is_correct_hash,
            num_layers,
            args.output_dir
        )

        print(f"\n{'='*100}")
        print("PER-STEP CORRECTNESS VISUALIZATION COMPLETE!")
        print(f"{'='*100}")
        print(f"Outputs (6 plots total):")
        for step in [1, 2, 3, 4, 5]:
            print(f"  {args.output_dir / f'tsne_step_{step}_correctness.png'}")
        print(f"  {args.output_dir / 'tsne_hash_correctness.png'}")
        print()
        return 0

    # Collect all unique step numbers across all layers for master legend
    # IMPORTANT: Filter by max_step to only show steps that will actually appear in plots
    all_unique_steps = set()
    for layer_idx in range(num_layers):
        if len(step_numbers[layer_idx]) > 0:
            # Apply max_step filter
            layer_steps = step_numbers[layer_idx]
            if args.max_step is not None:
                layer_steps = layer_steps[layer_steps <= args.max_step]
            all_unique_steps.update(layer_steps)
    all_unique_steps = sorted(all_unique_steps)

    print(f"  Unique steps found: {all_unique_steps}")

    # Create grid plot (all layers or selected layers for publication)
    if args.publication_ready:
        selected_layers = [0, 11, 21, 31]
        print(f"\nCreating publication-ready grid plot with layers {selected_layers}...")

        # 1x4 grid for publication
        n_cols = 4
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows + 0.5))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for plot_idx, layer_idx in enumerate(tqdm(selected_layers, desc="Processing layers", ncols=100)):
            plot_tsne_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
                layer_idx,
                ax=axes[plot_idx],
                raw_mode=args.raw_mode,
                max_step=args.max_step,
                color_by_correctness=args.color_by_correctness,
                is_correct_step=is_correct_step[layer_idx] if is_correct_step is not None else None,
                is_correct_hash=is_correct_hash[layer_idx] if is_correct_hash is not None else None,
                publication_ready=True,
                is_first_subplot=(plot_idx == 0)  # Mark first subplot for legend
            )

        # No unused subplots in publication mode (exact 4 layers, 4 subplots)

    elif args.publication_appendix:
        print(f"\nCreating publication-appendix grid plot with all {num_layers} layers...")

        # 8x4 grid for appendix (all 32 layers)
        n_cols = 8
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows + 0.8))
        axes = axes.flatten()

        for layer_idx in tqdm(range(num_layers), desc="Processing layers", ncols=100):
            plot_tsne_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
                layer_idx,
                ax=axes[layer_idx],
                raw_mode=args.raw_mode,
                max_step=args.max_step,
                color_by_correctness=args.color_by_correctness,
                is_correct_step=is_correct_step[layer_idx] if is_correct_step is not None else None,
                is_correct_hash=is_correct_hash[layer_idx] if is_correct_hash is not None else None,
                publication_ready=True,  # Use same clean formatting
                is_first_subplot=False  # No legend in first subplot for appendix mode
            )

        # Hide unused subplots (if any)
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

    else:
        print(f"\nCreating grid plot with all {num_layers} layers...")

        n_cols = 8
        n_rows = (num_layers + n_cols - 1) // n_cols
        # Add extra space at bottom for legend
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows + 0.8))
        axes = axes.flatten()

        for layer_idx in tqdm(range(num_layers), desc="Processing layers", ncols=100):
            plot_tsne_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
                layer_idx,
                ax=axes[layer_idx],
                raw_mode=args.raw_mode,
                max_step=args.max_step,
                color_by_correctness=args.color_by_correctness,
                is_correct_step=is_correct_step[layer_idx] if is_correct_step is not None else None,
                is_correct_hash=is_correct_hash[layer_idx] if is_correct_hash is not None else None
            )

        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

    # Create master legend at the bottom
    legend_handles = []
    is_publication_mode = args.publication_ready or args.publication_appendix

    if args.raw_mode and args.color_by_correctness:
        # Raw mode with correctness: darker/lighter shades
        legend_handles.append(Patch(facecolor='#0066cc', edgecolor='darkblue', label='Step (Correct)'))
        legend_handles.append(Patch(facecolor='#add8e6', edgecolor='blue', label='Step (Incorrect)'))
        if is_publication_mode:
            legend_handles.append(Patch(facecolor='#cc0000', edgecolor='darkred', label='Final Answer Marker (Correct)'))
            legend_handles.append(Patch(facecolor='#ffcccc', edgecolor='red', label='Final Answer Marker (Incorrect)'))
        else:
            legend_handles.append(Patch(facecolor='#cc0000', edgecolor='darkred', label='#### (Correct)'))
            legend_handles.append(Patch(facecolor='#ffcccc', edgecolor='red', label='#### (Incorrect)'))
    elif args.color_by_correctness:
        # Correctness mode: show correct/incorrect for both step and hash
        legend_handles.append(Patch(facecolor='green', edgecolor='darkgreen', label='Step (Correct)'))
        legend_handles.append(Patch(facecolor='red', edgecolor='darkred', label='Step (Incorrect)'))
        if is_publication_mode:
            legend_handles.append(Patch(facecolor='darkgreen', edgecolor='black', label='Final Answer Marker (Correct)'))
            legend_handles.append(Patch(facecolor='darkred', edgecolor='black', label='Final Answer Marker (Incorrect)'))
        else:
            legend_handles.append(Patch(facecolor='darkgreen', edgecolor='black', label='#### (Correct)'))
            legend_handles.append(Patch(facecolor='darkred', edgecolor='black', label='#### (Incorrect)'))
    elif args.raw_mode:
        # Raw mode: only show "Step" and "####"
        legend_handles.append(Patch(facecolor='lightblue', edgecolor='blue', label='Step'))
        if is_publication_mode:
            legend_handles.append(Patch(facecolor=HASH_COLOR, edgecolor='darkred', label='Final Answer Marker'))
        else:
            legend_handles.append(Patch(facecolor=HASH_COLOR, edgecolor='darkred', label='####'))
    else:
        # Show individual step numbers
        for step_num in all_unique_steps:
            step_num_int = int(step_num)
            if step_num_int in STEP_COLORS:
                color = STEP_COLORS[step_num_int]
            else:
                color_idx = ((step_num_int - 1) % len(STEP_COLORS)) + 1
                color = STEP_COLORS[color_idx]
            legend_handles.append(Patch(facecolor=color, edgecolor='black', label=f'Step {step_num}'))
        # Add Hash to legend
        if is_publication_mode:
            legend_handles.append(Patch(facecolor=HASH_COLOR, edgecolor='darkred', label='Final Answer Marker'))
        else:
            legend_handles.append(Patch(facecolor=HASH_COLOR, edgecolor='darkred', label='#### (Hash)'))

    # Add legend below the plots
    # Skip only for publication-ready mode (as it's added to first subplot)
    # Show master legend for publication-appendix mode
    if not args.publication_ready:
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(len(legend_handles), 11),  # Max 11 columns
            bbox_to_anchor=(0.5, -0.01),
            fontsize=12,
            frameon=True if not args.publication_appendix else False,  # No frame for appendix mode
            title="Step Numbers" if not is_publication_mode else None,
            title_fontsize=14 if not is_publication_mode else None
        )

    # Add suptitle only if not publication mode
    if not is_publication_mode:
        plt.suptitle("t-SNE Visualization of Step vs #### Activations Across Layers",
                    fontsize=16, y=0.998)

    # Adjust layout rect based on mode
    if args.publication_ready:
        # Publication-ready: legend in first subplot, tight layout, no bottom space
        plt.tight_layout(rect=[0, 0, 1, 1.0])
    elif args.publication_appendix:
        # Publication-appendix: master legend at bottom, need bottom space
        plt.tight_layout(rect=[0, 0.02, 1, 1.0])
    else:
        # Regular mode: master legend at bottom with title
        plt.tight_layout(rect=[0, 0.02, 1, 0.995])

    if args.publication_ready:
        grid_output = args.output_dir / "tsne_publication_ready.png"
    elif args.publication_appendix:
        grid_output = args.output_dir / "tsne_publication_appendix.png"
    else:
        grid_output = args.output_dir / "tsne_all_layers.png"

    print(f"\nSaving grid plot to {grid_output}...")
    is_publication_mode = args.publication_ready or args.publication_appendix
    plt.savefig(grid_output, dpi=300 if is_publication_mode else 150, bbox_inches='tight')
    plt.close()

    print(f"✓ Grid plot saved!")

    # Create individual plots if requested
    if args.individual:
        print(f"\nCreating individual plots for each layer...")

        individual_dir = args.output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

        for layer_idx in tqdm(range(num_layers), desc="Saving individual plots", ncols=100):
            output_path = individual_dir / f"layer_{layer_idx:02d}_tsne.png"
            plot_tsne_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
                layer_idx,
                output_path=output_path,
                raw_mode=args.raw_mode,
                max_step=args.max_step,
                color_by_correctness=args.color_by_correctness,
                is_correct_step=is_correct_step[layer_idx] if is_correct_step is not None else None,
                is_correct_hash=is_correct_hash[layer_idx] if is_correct_hash is not None else None
            )

        print(f"✓ Individual plots saved to {individual_dir}/")

    print(f"\n{'='*100}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*100}")
    print(f"Outputs:")
    print(f"  Grid plot: {grid_output}")
    if args.individual:
        print(f"  Individual plots: {individual_dir}/")
    print()


if __name__ == "__main__":
    main()
