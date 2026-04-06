#!/usr/bin/env python
"""Visualize activation distance tables

Creates publication-quality visualizations of activation distances between
trajectory segments, comparing correct vs incorrect questions.

Usage:
    python scripts/trajectories/visualize_distances.py \
        --input output/trajectories/activation_distances.json \
        --output output/trajectories/plots/

    # Custom style
    python scripts/trajectories/visualize_distances.py \
        --input output/trajectories/activation_distances.json \
        --output output/trajectories/plots/ \
        --style seaborn-v0_8-darkgrid \
        --dpi 300
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_distance_data(json_path: Path) -> Dict:
    """Load activation distance data from JSON file

    Args:
        json_path: Path to activation_distances.json

    Returns:
        Dictionary with distance data
    """
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"  ✓ Loaded data for {len(data)} correctness groups")
    return data


def extract_plot_data(data: Dict) -> Tuple[List[str], Dict[str, Dict[str, Tuple]]]:
    """Extract plotting data from distance dictionary

    Args:
        data: Distance data dictionary

    Returns:
        segments: List of segment names (in order)
        plot_data: Dict[correctness][metric][segment] = (mean, ci_lower, ci_upper, N)
    """
    # Define segment order
    segment_order = [
        "Step 1 → Step 2",
        "Step 2 → Last Step",
        "Second-last → Last Step",
        "Last Step → ####"
    ]

    plot_data = {}

    for correctness, correctness_data in data.items():
        plot_data[correctness] = {
            'cosine': {},
            'euclidean': {}
        }

        for segment, segment_data in correctness_data.items():
            # Cosine distance
            cos_mean = segment_data['cosine_distance']['mean']
            cos_ci = segment_data['cosine_distance']['ci_95']
            cos_n = segment_data['n_samples']

            plot_data[correctness]['cosine'][segment] = (
                cos_mean, cos_ci[0], cos_ci[1], cos_n
            )

            # Euclidean distance
            euc_mean = segment_data['euclidean_distance']['mean']
            euc_ci = segment_data['euclidean_distance']['ci_95']

            plot_data[correctness]['euclidean'][segment] = (
                euc_mean, euc_ci[0], euc_ci[1], cos_n
            )

    return segment_order, plot_data


def plot_distance_comparison(
    segments: List[str],
    plot_data: Dict[str, Dict[str, Tuple]],
    metric: str,
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 6)
):
    """Create bar plot comparing distances for correct vs incorrect questions

    Args:
        segments: List of segment names
        plot_data: Plotting data dictionary
        metric: 'cosine' or 'euclidean'
        output_path: Output file path
        dpi: DPI for saved figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for both groups
    correct_means = []
    correct_cis = []
    incorrect_means = []
    incorrect_cis = []

    for segment in segments:
        # Correct
        if segment in plot_data['correct'][metric]:
            mean, ci_low, ci_high, n = plot_data['correct'][metric][segment]
            correct_means.append(mean)
            correct_cis.append([mean - ci_low, ci_high - mean])
        else:
            correct_means.append(0)
            correct_cis.append([0, 0])

        # Incorrect
        if segment in plot_data['incorrect'][metric]:
            mean, ci_low, ci_high, n = plot_data['incorrect'][metric][segment]
            incorrect_means.append(mean)
            incorrect_cis.append([mean - ci_low, ci_high - mean])
        else:
            incorrect_means.append(0)
            incorrect_cis.append([0, 0])

    # Convert to numpy arrays
    correct_means = np.array(correct_means)
    incorrect_means = np.array(incorrect_means)
    correct_cis = np.array(correct_cis).T
    incorrect_cis = np.array(incorrect_cis).T

    # Set up bar positions
    x = np.arange(len(segments))
    width = 0.35

    # Define colors
    color_correct = '#2ecc71'  # Green
    color_incorrect = '#e74c3c'  # Red

    # Create bars
    bars1 = ax.bar(x - width/2, correct_means, width,
                   label='Correct', color=color_correct, alpha=0.8,
                   edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, incorrect_means, width,
                   label='Incorrect', color=color_incorrect, alpha=0.8,
                   edgecolor='black', linewidth=1)

    # Add error bars (95% CI)
    ax.errorbar(x - width/2, correct_means, yerr=correct_cis,
                fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.7)
    ax.errorbar(x + width/2, incorrect_means, yerr=incorrect_cis,
                fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.7)

    # Customize plot
    if metric == 'cosine':
        ax.set_ylabel('Cosine Distance', fontsize=14, fontweight='bold')
        ax.set_title('Activation Cosine Distance: Correct vs Incorrect Questions',
                     fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_ylabel('Euclidean Distance', fontsize=14, fontweight='bold')
        ax.set_title('Activation Euclidean Distance: Correct vs Incorrect Questions',
                     fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel('Trajectory Segment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bars, means in [(bars1, correct_means), (bars2, incorrect_means)]:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved {metric} distance plot to {output_path}")
    plt.close()


def plot_combined_comparison(
    segments: List[str],
    plot_data: Dict[str, Dict[str, Tuple]],
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (14, 10)
):
    """Create combined plot with both cosine and euclidean distances

    Args:
        segments: List of segment names
        plot_data: Plotting data dictionary
        output_path: Output file path
        dpi: DPI for saved figure
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    metrics = ['cosine', 'euclidean']
    titles = [
        'Cosine Distance: Correct vs Incorrect Questions',
        'Euclidean Distance: Correct vs Incorrect Questions'
    ]
    ylabels = ['Cosine Distance', 'Euclidean Distance']

    # Define colors
    color_correct = '#2ecc71'  # Green
    color_incorrect = '#e74c3c'  # Red

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]

        # Extract data
        correct_means = []
        correct_cis = []
        incorrect_means = []
        incorrect_cis = []

        for segment in segments:
            # Correct
            if segment in plot_data['correct'][metric]:
                mean, ci_low, ci_high, n = plot_data['correct'][metric][segment]
                correct_means.append(mean)
                correct_cis.append([mean - ci_low, ci_high - mean])
            else:
                correct_means.append(0)
                correct_cis.append([0, 0])

            # Incorrect
            if segment in plot_data['incorrect'][metric]:
                mean, ci_low, ci_high, n = plot_data['incorrect'][metric][segment]
                incorrect_means.append(mean)
                incorrect_cis.append([mean - ci_low, ci_high - mean])
            else:
                incorrect_means.append(0)
                incorrect_cis.append([0, 0])

        # Convert to numpy
        correct_means = np.array(correct_means)
        incorrect_means = np.array(incorrect_means)
        correct_cis = np.array(correct_cis).T
        incorrect_cis = np.array(incorrect_cis).T

        # Set up bars
        x = np.arange(len(segments))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width/2, correct_means, width,
                      label='Correct', color=color_correct, alpha=0.8,
                      edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, incorrect_means, width,
                      label='Incorrect', color=color_incorrect, alpha=0.8,
                      edgecolor='black', linewidth=1)

        # Error bars
        ax.errorbar(x - width/2, correct_means, yerr=correct_cis,
                   fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.7)
        ax.errorbar(x + width/2, incorrect_means, yerr=incorrect_cis,
                   fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.7)

        # Customize
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(segments, rotation=15, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Add value labels
        for bars, means in [(bars1, correct_means), (bars2, incorrect_means)]:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.2f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add overall title
    fig.suptitle('Activation Distance Analysis: Trajectory Segments',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved combined plot to {output_path}")
    plt.close()


def plot_difference_heatmap(
    segments: List[str],
    plot_data: Dict[str, Dict[str, Tuple]],
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10, 6)
):
    """Create heatmap showing difference (Incorrect - Correct) for each segment

    Args:
        segments: List of segment names
        plot_data: Plotting data dictionary
        output_path: Output file path
        dpi: DPI for saved figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate differences
    metrics = ['cosine', 'euclidean']
    metric_labels = ['Cosine Distance', 'Euclidean Distance']

    diff_matrix = []

    for metric in metrics:
        row = []
        for segment in segments:
            correct_mean = plot_data['correct'][metric].get(segment, (0, 0, 0, 0))[0]
            incorrect_mean = plot_data['incorrect'][metric].get(segment, (0, 0, 0, 0))[0]
            diff = incorrect_mean - correct_mean
            row.append(diff)
        diff_matrix.append(row)

    diff_matrix = np.array(diff_matrix)

    # Create heatmap
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)

    # Set ticks
    ax.set_xticks(np.arange(len(segments)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(segments, rotation=15, ha='right', fontsize=11)
    ax.set_yticklabels(metric_labels, fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(segments)):
            text = ax.text(j, i, f'{diff_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=11, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difference (Incorrect - Correct)', rotation=270, labelpad=20,
                   fontsize=12, fontweight='bold')

    ax.set_title('Distance Difference: Incorrect minus Correct Questions',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved difference heatmap to {output_path}")
    plt.close()


def plot_sample_sizes(
    segments: List[str],
    plot_data: Dict[str, Dict[str, Tuple]],
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10, 6)
):
    """Create bar plot showing sample sizes for each segment

    Args:
        segments: List of segment names
        plot_data: Plotting data dictionary
        output_path: Output file path
        dpi: DPI for saved figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract sample sizes (same for both metrics)
    correct_ns = []
    incorrect_ns = []

    for segment in segments:
        # Use cosine metric to get sample sizes
        correct_n = plot_data['correct']['cosine'].get(segment, (0, 0, 0, 0))[3]
        incorrect_n = plot_data['incorrect']['cosine'].get(segment, (0, 0, 0, 0))[3]
        correct_ns.append(correct_n)
        incorrect_ns.append(incorrect_n)

    # Set up bars
    x = np.arange(len(segments))
    width = 0.35

    # Define colors
    color_correct = '#2ecc71'
    color_incorrect = '#e74c3c'

    # Create bars
    bars1 = ax.bar(x - width/2, correct_ns, width,
                  label='Correct', color=color_correct, alpha=0.8,
                  edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, incorrect_ns, width,
                  label='Incorrect', color=color_incorrect, alpha=0.8,
                  edgecolor='black', linewidth=1)

    # Customize
    ax.set_ylabel('Number of Samples (N)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Trajectory Segment', fontsize=13, fontweight='bold')
    ax.set_title('Sample Sizes by Trajectory Segment and Correctness',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ Saved sample size plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize activation distance tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/trajectories/visualize_distances.py \\
      --input output/trajectories/activation_distances.json \\
      --output output/trajectories/plots/

  # High-resolution plots
  python scripts/trajectories/visualize_distances.py \\
      --input output/trajectories/activation_distances.json \\
      --output output/trajectories/plots/ \\
      --dpi 600

  # Custom matplotlib style
  python scripts/trajectories/visualize_distances.py \\
      --input output/trajectories/activation_distances.json \\
      --output output/trajectories/plots/ \\
      --style seaborn-v0_8-whitegrid
        """
    )

    parser.add_argument("--input", type=Path, required=True,
                       help="Path to activation_distances.json file")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures (default: 300)")
    parser.add_argument("--style", type=str, default=None,
                       help="Matplotlib style (e.g., seaborn-v0_8-darkgrid)")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Output format (default: png)")

    args = parser.parse_args()

    # Check input file exists
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*100}")
    print("ACTIVATION DISTANCE VISUALIZATION")
    print(f"{'='*100}")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"DPI: {args.dpi}")
    print(f"Format: {args.format}")
    if args.style:
        print(f"Style: {args.style}")
    print(f"{'='*100}\n")

    # Set matplotlib style
    if args.style:
        try:
            plt.style.use(args.style)
            print(f"✓ Using matplotlib style: {args.style}")
        except Exception as e:
            print(f"⚠ Warning: Could not use style '{args.style}': {e}")
            print("  Using default style instead")

    # Load data
    data = load_distance_data(args.input)

    # Extract plotting data
    segments, plot_data = extract_plot_data(data)
    print(f"\n✓ Extracted data for {len(segments)} segments\n")

    # Generate plots
    print("Generating plots...")
    print(f"{'-'*100}\n")

    # 1. Cosine distance comparison
    plot_distance_comparison(
        segments, plot_data, 'cosine',
        args.output / f"cosine_distance_comparison.{args.format}",
        dpi=args.dpi
    )

    # 2. Euclidean distance comparison
    plot_distance_comparison(
        segments, plot_data, 'euclidean',
        args.output / f"euclidean_distance_comparison.{args.format}",
        dpi=args.dpi
    )

    # 3. Combined comparison
    plot_combined_comparison(
        segments, plot_data,
        args.output / f"combined_distance_comparison.{args.format}",
        dpi=args.dpi
    )

    # 4. Difference heatmap
    plot_difference_heatmap(
        segments, plot_data,
        args.output / f"distance_difference_heatmap.{args.format}",
        dpi=args.dpi
    )

    # 5. Sample sizes
    plot_sample_sizes(
        segments, plot_data,
        args.output / f"sample_sizes.{args.format}",
        dpi=args.dpi
    )

    print(f"\n{'-'*100}")
    print(f"\n{'='*100}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*100}")
    print(f"\nGenerated 5 plots in {args.output}/:")
    print(f"  1. cosine_distance_comparison.{args.format}")
    print(f"  2. euclidean_distance_comparison.{args.format}")
    print(f"  3. combined_distance_comparison.{args.format}")
    print(f"  4. distance_difference_heatmap.{args.format}")
    print(f"  5. sample_sizes.{args.format}")
    print(f"{'='*100}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
