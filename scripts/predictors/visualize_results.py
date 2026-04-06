#!/usr/bin/env python
"""Visualize predictor training results

Reads summary JSON file(s) and creates:
1. Test AUC plots by layer for each feature×label combination
2. Test Accuracy + F1 plots by layer for each feature×label combination

Usage:
    # Single file
    python scripts/predictors/visualize_results.py --input output/predictors/summary.json

    # Directory with multiple summary files
    python scripts/predictors/visualize_results.py --input output/predictor_results/

    # Publication-ready mode (only AUC plots, no baseline, no train curves, no title)
    python scripts/predictors/visualize_results.py --input output/predictors/summary.json --publication-ready
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def load_summary_from_file(json_path: Path) -> List[Dict]:
    """Load a single summary JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_summaries_from_directory(directory: Path, pattern: str = "*summary*.json") -> List[Dict]:
    """Load all matching JSON files from a directory

    Args:
        directory: Directory to search
        pattern: Glob pattern for matching JSON files (default: *summary*.json)

    Returns:
        Combined list of all predictor results from all files
    """
    print(f"\nScanning directory: {directory}")
    print(f"  Pattern: {pattern}")

    # Find all matching JSON files
    json_files = sorted(directory.glob(pattern))

    if len(json_files) == 0:
        # Try broader pattern
        print(f"  No files found with pattern '{pattern}', trying '*.json'...")
        json_files = sorted(directory.glob("*.json"))

    if len(json_files) == 0:
        print(f"  ERROR: No JSON files found in {directory}")
        return []

    print(f"  Found {len(json_files)} JSON file(s):")
    for f in json_files:
        print(f"    - {f.name}")

    # Load and combine all files
    all_data = []
    for json_path in json_files:
        try:
            data = load_summary_from_file(json_path)
            print(f"    ✓ {json_path.name}: {len(data)} results")
            all_data.extend(data)
        except Exception as e:
            print(f"    ✗ {json_path.name}: Failed to load ({e})")

    return all_data


def load_summary(input_path: Path, pattern: str = "*summary*.json") -> List[Dict]:
    """Load summary data from either a file or directory

    Args:
        input_path: Path to JSON file or directory
        pattern: Glob pattern if input_path is a directory

    Returns:
        List of predictor results
    """
    if input_path.is_file():
        print(f"\nLoading summary from file: {input_path}")
        data = load_summary_from_file(input_path)
        print(f"  Loaded {len(data)} classifier results")
        return data
    elif input_path.is_dir():
        data = load_summaries_from_directory(input_path, pattern)
        print(f"\n  Total loaded: {len(data)} classifier results")
        return data
    else:
        print(f"ERROR: Input path does not exist or is not a file/directory: {input_path}")
        return []


def organize_by_feature_label(data: List[Dict]) -> Dict:
    """Organize data by feature_set and label_type

    Returns:
        Dict[tuple(feature_set, label_type)] -> List[Dict]
    """
    organized = defaultdict(list)

    for entry in data:
        key = (entry['feature_set'], entry['label_type'])
        organized[key].append(entry)

    # Sort each group by layer_idx
    for key in organized:
        organized[key] = sorted(organized[key], key=lambda x: x['layer_idx'])

    return dict(organized)


def plot_auc_by_layer(data: List[Dict], feature_set: str, label_type: str, output_dir: Path, publication_ready: bool = False):
    """Plot test AUC across layers

    Args:
        data: List of result dictionaries
        feature_set: Feature set name
        label_type: Label type ('correctness' or 'error_type')
        output_dir: Output directory for plots
        publication_ready: If True, use publication-ready formatting
    """
    if len(data) == 0:
        return

    layers = [entry['layer_idx'] for entry in data]
    test_auc = [entry['test_roc_auc'] for entry in data]
    train_auc = [entry['train_roc_auc'] for entry in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    if publication_ready:
        # Publication mode: only test AUC, no baseline, no title
        # Match style from visualize_linear_probe_results.py
        ax.plot(layers, test_auc, 'o-', linewidth=2.5, markersize=7,
                color='#2E86AB', alpha=0.9)

        # Formatting matching visualize_linear_probe_results.py
        ax.set_xlabel('Layer', fontsize=13)
        ax.set_ylabel('ROC-AUC', fontsize=13)

        # No title in publication mode
        # No legend needed when only one curve

        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_facecolor('#f9f9f9')
        ax.tick_params(labelsize=11)

        # Set y-axis limits
        y_min = max(0.3, min(test_auc) - 0.05)
        y_max = min(1.0, max(test_auc) + 0.05)
        ax.set_ylim(y_min, y_max)

        # Highlight best test AUC with a star marker
        best_idx = np.argmax(test_auc)
        best_layer = layers[best_idx]
        best_auc = test_auc[best_idx]
        ax.plot(best_layer, best_auc, marker='*', markersize=15,
                color='#2E86AB', markeredgecolor='black',
                markeredgewidth=1.5, zorder=10)

    else:
        # Standard mode: test and train AUC
        ax.plot(layers, test_auc, 'o-', linewidth=2, markersize=6, label='Test AUC', color='#2E86AB')
        ax.plot(layers, train_auc, 's--', linewidth=1.5, markersize=4, label='Train AUC', color='#A23B72', alpha=0.6)

        # Add horizontal line at 0.5 (random performance)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')

        # Formatting
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')

        # Create title based on label type
        if label_type == 'correctness':
            title = f'Test AUC: {feature_set} (Predicting Errors)'
        else:
            title = f'Test AUC: {feature_set} (Predicting Error Type)'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        all_aucs = test_auc + train_auc
        y_min = max(0.3, min(all_aucs) - 0.05)
        y_max = min(1.0, max(all_aucs) + 0.05)
        ax.set_ylim(y_min, y_max)

        # Highlight best test AUC
        best_idx = np.argmax(test_auc)
        best_layer = layers[best_idx]
        best_auc = test_auc[best_idx]
        ax.plot(best_layer, best_auc, 'r*', markersize=20, label=f'Best: Layer {best_layer} (AUC={best_auc:.3f})')
        ax.legend(loc='best', fontsize=10)

        # Add text box with statistics
        stats_text = f"Best Test AUC: {best_auc:.4f} (Layer {best_layer})\n"
        stats_text += f"Mean Test AUC: {np.mean(test_auc):.4f}\n"
        stats_text += f"Std Test AUC: {np.std(test_auc):.4f}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    if publication_ready:
        filename = f"auc_{feature_set}_{label_type}_pub.png"
    else:
        filename = f"auc_{feature_set}_{label_type}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300 if publication_ready else 150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def plot_accuracy_f1_by_layer(data: List[Dict], feature_set: str, label_type: str, output_dir: Path):
    """Plot test accuracy and F1 across layers"""
    if len(data) == 0:
        return

    layers = [entry['layer_idx'] for entry in data]
    test_acc = [entry['test_accuracy'] for entry in data]
    test_f1 = [entry['test_f1'] for entry in data]
    train_acc = [entry['train_accuracy'] for entry in data]
    train_f1 = [entry['train_f1'] for entry in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot test metrics (solid lines)
    ax.plot(layers, test_acc, 'o-', linewidth=2, markersize=6, label='Test Accuracy', color='#2E86AB')
    ax.plot(layers, test_f1, '^-', linewidth=2, markersize=6, label='Test F1', color='#F18F01')

    # Plot train metrics (dashed lines, lighter)
    ax.plot(layers, train_acc, 's--', linewidth=1.5, markersize=4, label='Train Accuracy', color='#2E86AB', alpha=0.4)
    ax.plot(layers, train_f1, 'v--', linewidth=1.5, markersize=4, label='Train F1', color='#F18F01', alpha=0.4)

    # Formatting
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')

    # Create title based on label type
    if label_type == 'correctness':
        title = f'Test Accuracy & F1: {feature_set} (Predicting Errors)'
    else:
        title = f'Test Accuracy & F1: {feature_set} (Predicting Error Type)'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    all_scores = test_acc + test_f1 + train_acc + train_f1
    y_min = max(0.0, min(all_scores) - 0.05)
    y_max = min(1.0, max(all_scores) + 0.05)
    ax.set_ylim(y_min, y_max)

    # Highlight best test F1
    best_f1_idx = np.argmax(test_f1)
    best_layer = layers[best_f1_idx]
    best_f1 = test_f1[best_f1_idx]
    ax.plot(best_layer, best_f1, 'r*', markersize=20, label=f'Best F1: Layer {best_layer} ({best_f1:.3f})')
    ax.legend(loc='best', fontsize=10)

    # Add text box with statistics
    stats_text = f"Best Test F1: {best_f1:.4f} (Layer {best_layer})\n"
    stats_text += f"Best Test Acc: {max(test_acc):.4f} (Layer {layers[np.argmax(test_acc)]})\n"
    stats_text += f"Mean Test F1: {np.mean(test_f1):.4f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Save figure
    filename = f"acc_f1_{feature_set}_{label_type}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def plot_comparison_across_features(organized_data: Dict, label_type: str, output_dir: Path):
    """Plot comparison of all feature sets for a given label type"""

    # Filter for this label type
    relevant_data = {k: v for k, v in organized_data.items() if k[1] == label_type}

    if len(relevant_data) == 0:
        return

    # Create figure with 3 subplots (AUC, F1, and Accuracy)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Define 9 distinct colors for up to 9 feature sets (expanded palette)
    colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#9B59B6', '#E74C3C', '#3498DB', '#E67E22', '#1ABC9C']
    markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'X', 'P']

    for idx, ((feature_set, _), data) in enumerate(sorted(relevant_data.items())):
        layers = [entry['layer_idx'] for entry in data]
        test_auc = [entry['test_roc_auc'] for entry in data]
        test_f1 = [entry['test_f1'] for entry in data]
        test_acc = [entry['test_accuracy'] for entry in data]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Plot AUC
        axes[0].plot(layers, test_auc, marker=marker, linestyle='-', linewidth=2,
                     markersize=5, label=feature_set, color=color)

        # Plot F1
        axes[1].plot(layers, test_f1, marker=marker, linestyle='-', linewidth=2,
                     markersize=5, label=feature_set, color=color)

        # Plot Accuracy
        axes[2].plot(layers, test_acc, marker=marker, linestyle='-', linewidth=2,
                     markersize=5, label=feature_set, color=color)

    # Format AUC plot
    axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
    axes[0].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test ROC-AUC', fontsize=12, fontweight='bold')
    axes[0].set_title('Test AUC Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Format F1 plot
    axes[1].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Test F1 Comparison', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Format Accuracy plot
    axes[2].set_xlabel('Layer', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    axes[2].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Overall title
    if label_type == 'correctness':
        fig.suptitle('Feature Comparison: Predicting Errors', fontsize=15, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Feature Comparison: Predicting Error Types', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    filename = f"comparison_{label_type}.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def plot_master_comparison_all_settings(organized_data: Dict, output_dir: Path, publication_ready: bool = False):
    """Create master comparison plot showing ALL feature×label combinations in one figure

    Args:
        organized_data: Dict mapping (feature_set, label_type) -> list of results
        output_dir: Output directory for plots
        publication_ready: If True, use publication-ready formatting with filtered feature sets
    """
    if len(organized_data) == 0:
        return

    # Publication-ready mode: filter to specific feature sets only
    if publication_ready:
        # Define the feature sets to include and their display names
        feature_display_names = {
            'hash_last_diffs_pca_joint': 'Late-step trajectory (PCA)',
            'hash_pca': 'Final-state geometry (PCA)',
            'step2_minus_step1': 'Early-step distance (S1→S2)'
        }

        # Define colors and line widths for each feature in publication mode
        feature_styles = {
            'hash_last_diffs_pca_joint': {'color': '#2E86AB', 'linewidth': 3.0},  # Primary - thicker
            'hash_pca': {'color': '#F18F01', 'linewidth': 3.0},  # Primary - thicker
            'step2_minus_step1': {'color': '#A0A0A0', 'linewidth': 2.5}  # Light grey - baseline comparison
        }

        # Filter organized_data to only include these feature sets
        filtered_data = {
            k: v for k, v in organized_data.items()
            if k[0] in feature_display_names
        }

        if len(filtered_data) == 0:
            print("  Warning: None of the publication-ready feature sets found in data")
            return

        organized_data = filtered_data

    # Separate by label type for clarity
    correctness_data = {k: v for k, v in organized_data.items() if k[1] == 'correctness'}
    error_type_data = {k: v for k, v in organized_data.items() if k[1] == 'error_type'}

    # Determine layout based on what data we have
    has_correctness = len(correctness_data) > 0
    has_error_type = len(error_type_data) > 0

    if has_correctness and has_error_type:
        # Two subplots: one for each label type
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        axes_list = [axes[0], axes[1]]
        data_list = [correctness_data, error_type_data]
        titles = ['Predicting Errors (Correctness)', 'Predicting Error Types']
    elif has_correctness:
        # Single plot for correctness
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes_list = [ax]
        data_list = [correctness_data]
        titles = ['Predicting Errors (Correctness)']
    elif has_error_type:
        # Single plot for error_type
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes_list = [ax]
        data_list = [error_type_data]
        titles = ['Predicting Error Types']
    else:
        return

    # Expanded color palette for up to 9 feature sets
    colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#9B59B6', '#E74C3C', '#3498DB', '#E67E22', '#1ABC9C']
    markers = ['o', '^', 's', 'D', 'v', 'p', '*', 'X', 'P']

    # Plot each label type
    for ax, data_dict, title in zip(axes_list, data_list, titles):
        for idx, ((feature_set, _), data) in enumerate(sorted(data_dict.items())):
            if len(data) == 0:
                continue

            layers = [entry['layer_idx'] for entry in data]
            test_auc = [entry['test_roc_auc'] for entry in data]

            # Use publication-ready styles if in publication mode
            if publication_ready and feature_set in feature_styles:
                color = feature_styles[feature_set]['color']
                linewidth = feature_styles[feature_set]['linewidth']
                marker = markers[idx % len(markers)]
                label = feature_display_names[feature_set]
            else:
                color = colors[idx % len(colors)]
                linewidth = 2.5
                marker = markers[idx % len(markers)]
                label = feature_set

            # Plot with feature set as label
            ax.plot(layers, test_auc, marker=marker, linestyle='-',
                   linewidth=linewidth, markersize=6, label=label,
                   color=color, alpha=0.85)

        # Formatting
        if not publication_ready:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Random')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
            ax.set_ylabel('Test ROC-AUC', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2 if len(data_dict) > 6 else 1)
        else:
            ax.set_xlabel('Layer', fontsize=13)
            ax.set_ylabel('Test ROC-AUC', fontsize=13)
            # Legend at bottom left with title in publication mode
            # ax.legend(loc='lower left', fontsize=10, framealpha=0.95, title='Features', title_fontsize=10)
            ax.legend(
                loc='lower left',
                fontsize=13,              # bigger legend text
                title='Features',
                title_fontsize=13,        # bigger legend title
                framealpha=0.95,
                handlelength=2.2,         # longer line samples
                markerscale=1.4,          # bigger markers in legend
                labelspacing=0.6,         # more vertical spacing
                borderpad=0.8             # more padding inside legend box
            )

        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_facecolor('#f9f9f9')
        ax.tick_params(labelsize=11)

        # Set reasonable y-axis limits
        y_min = max(0.3, min([min([e['test_roc_auc'] for e in d]) for d in data_dict.values()]) - 0.05)
        y_max = min(1.0, max([max([e['test_roc_auc'] for e in d]) for d in data_dict.values()]) + 0.05)
        ax.set_ylim(y_min, y_max)

    # Overall title (skip in publication mode)
    if not publication_ready:
        fig.suptitle('Master Comparison: All Feature Sets and Tasks',
                    fontsize=18, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    else:
        plt.tight_layout()

    # Save figure
    filename = "master_comparison_all.png" if not publication_ready else "master_comparison_all_pub.png"
    output_path = output_dir / filename
    dpi = 300 if publication_ready else 150
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def create_detailed_tables(organized_data: Dict, output_dir: Path):
    """Create detailed tables for each feature×label combination"""

    for (feature_set, label_type), data in sorted(organized_data.items()):
        # Create filename
        filename = f"table_{feature_set}_{label_type}.txt"
        output_path = output_dir / filename

        with open(output_path, 'w') as f:
            # Header
            f.write("="*120 + "\n")
            f.write(f"DETAILED RESULTS: {feature_set} × {label_type}\n")
            f.write("="*120 + "\n\n")

            if label_type == 'correctness':
                f.write("Task: Predicting Errors (y=1: incorrect/error, y=0: correct)\n")
            else:
                f.write("Task: Predicting Error Types (y=1: referencing_value_error, y=0: other error types)\n")

            f.write(f"Feature Set: {feature_set}\n")
            f.write(f"Total Layers: {len(data)}\n\n")

            # Table header
            f.write(f"{'Key':<45} {'Layer':<7} {'Feature':<20} {'Label':<15} {'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10}\n")
            f.write("-"*120 + "\n")

            # Sort by layer
            for entry in sorted(data, key=lambda x: x['layer_idx']):
                key = entry['key']
                layer = entry['layer_idx']
                feature = entry['feature_set']
                label = entry['label_type']
                test_acc = entry['test_accuracy']
                test_f1 = entry['test_f1']
                test_auc = entry['test_roc_auc']

                f.write(f"{key:<45} {layer:<7} {feature:<20} {label:<15} "
                       f"{test_acc:<10.4f} {test_f1:<10.4f} {test_auc:<10.4f}\n")

            # Summary statistics
            f.write("\n" + "="*120 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*120 + "\n\n")

            test_accs = [entry['test_accuracy'] for entry in data]
            test_f1s = [entry['test_f1'] for entry in data]
            test_aucs = [entry['test_roc_auc'] for entry in data]

            best_auc_idx = np.argmax(test_aucs)
            best_f1_idx = np.argmax(test_f1s)
            best_acc_idx = np.argmax(test_accs)

            f.write(f"Best Test AUC:  {test_aucs[best_auc_idx]:.4f} at Layer {data[best_auc_idx]['layer_idx']}\n")
            f.write(f"Best Test F1:   {test_f1s[best_f1_idx]:.4f} at Layer {data[best_f1_idx]['layer_idx']}\n")
            f.write(f"Best Test Acc:  {test_accs[best_acc_idx]:.4f} at Layer {data[best_acc_idx]['layer_idx']}\n\n")

            f.write(f"Mean Test AUC:  {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}\n")
            f.write(f"Mean Test F1:   {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}\n")
            f.write(f"Mean Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\n\n")

            # Sample counts
            if len(data) > 0:
                f.write(f"\nSample Information (from best AUC layer {data[best_auc_idx]['layer_idx']}):\n")
                f.write(f"  Train samples: {data[best_auc_idx]['n_train_samples']}\n")
                f.write(f"  Test samples:  {data[best_auc_idx]['n_test_samples']}\n")

                # Class distribution if available
                if 'train_label_dist' in data[best_auc_idx]:
                    train_dist = data[best_auc_idx]['train_label_dist']
                    test_dist = data[best_auc_idx]['test_label_dist']
                    f.write(f"\n  Train class distribution: {train_dist}\n")
                    f.write(f"  Test class distribution:  {test_dist}\n")

            f.write("\n" + "="*120 + "\n")

        print(f"    Saved: {filename}")


def print_comprehensive_summary_table(organized_data: Dict):
    """Print comprehensive summary table to command line showing all test metrics"""

    print("\n" + "="*140)
    print("COMPREHENSIVE SUMMARY: ALL TEST METRICS")
    print("="*140)

    # Group by label type
    for label_type in ['correctness', 'error_type']:
        if label_type == 'correctness':
            print(f"\nTask: Predicting Errors (y=1: incorrect, y=0: correct)")
        else:
            print(f"\nTask: Predicting Error Types (y=1: referencing_value_error, y=0: other)")
        print("-"*140)

        relevant = {k: v for k, v in organized_data.items() if k[1] == label_type}

        if len(relevant) == 0:
            print("  No results found\n")
            continue

        # Print header
        print(f"{'Feature Set':<28} {'Best Layer':<12} {'Test AUC':<12} {'Test F1':<12} {'Test Acc':<12} "
              f"{'Avg AUC':<12} {'Avg F1':<12} {'Avg Acc':<12} {'#Layers':<10}")
        print("-"*140)

        for (feature_set, _), data in sorted(relevant.items()):
            if len(data) == 0:
                continue

            # Find best by AUC
            test_aucs = [entry['test_roc_auc'] for entry in data]
            test_f1s = [entry['test_f1'] for entry in data]
            test_accs = [entry['test_accuracy'] for entry in data]

            best_auc_idx = np.argmax(test_aucs)
            best_entry = data[best_auc_idx]

            # Compute averages
            avg_auc = np.mean(test_aucs)
            avg_f1 = np.mean(test_f1s)
            avg_acc = np.mean(test_accs)

            print(f"{feature_set:<28} "
                  f"{best_entry['layer_idx']:<12} "
                  f"{best_entry['test_roc_auc']:<12.4f} "
                  f"{best_entry['test_f1']:<12.4f} "
                  f"{best_entry['test_accuracy']:<12.4f} "
                  f"{avg_auc:<12.4f} "
                  f"{avg_f1:<12.4f} "
                  f"{avg_acc:<12.4f} "
                  f"{len(data):<10}")

        print()

    print("="*140)


def print_per_layer_summary_table(organized_data: Dict):
    """Print detailed per-layer metrics for all feature sets"""

    print("\n" + "="*140)
    print("PER-LAYER METRICS: ALL FEATURE SETS")
    print("="*140)

    # Group by label type
    for label_type in ['correctness', 'error_type']:
        if label_type == 'correctness':
            print(f"\nTask: Predicting Errors (y=1: incorrect, y=0: correct)")
        else:
            print(f"\nTask: Predicting Error Types (y=1: referencing_value_error, y=0: other)")
        print("-"*140)

        relevant = {k: v for k, v in organized_data.items() if k[1] == label_type}

        if len(relevant) == 0:
            print("  No results found\n")
            continue

        # Get all unique layers
        all_layers = set()
        for data in relevant.values():
            for entry in data:
                all_layers.add(entry['layer_idx'])
        layers = sorted(all_layers)

        # For each metric, create a table
        for metric_name, metric_key in [('Test AUC', 'test_roc_auc'),
                                         ('Test F1', 'test_f1'),
                                         ('Test Accuracy', 'test_accuracy')]:
            print(f"\n{metric_name}:")
            print(f"{'Layer':<8}", end="")

            feature_sets = sorted([fs for fs, lt in relevant.keys()])
            for fs in feature_sets:
                print(f"{fs:<18}", end="")
            print()
            print("-" * (8 + 18 * len(feature_sets)))

            # Print each layer
            for layer_idx in layers:
                print(f"{layer_idx:<8}", end="")

                for feature_set in feature_sets:
                    key = (feature_set, label_type)
                    if key in relevant:
                        # Find entry for this layer
                        layer_entry = None
                        for entry in relevant[key]:
                            if entry['layer_idx'] == layer_idx:
                                layer_entry = entry
                                break

                        if layer_entry:
                            value = layer_entry[metric_key]
                            print(f"{value:<18.4f}", end="")
                        else:
                            print(f"{'N/A':<18}", end="")
                    else:
                        print(f"{'N/A':<18}", end="")
                print()
            print()

    print("="*140)


def create_summary_table(organized_data: Dict, output_dir: Path):
    """Create a text summary table of best results"""

    output_path = output_dir / "summary_best.txt"

    with open(output_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("PREDICTOR TRAINING RESULTS - BEST PERFORMERS\n")
        f.write("="*120 + "\n\n")

        # Group by label type
        for label_type in ['correctness', 'error_type']:
            if label_type == 'correctness':
                f.write("Task: Predicting Errors (y=1: incorrect, y=0: correct)\n")
            else:
                f.write("Task: Predicting Error Types (y=1: referencing_value_error, y=0: other)\n")
            f.write("-"*120 + "\n")

            relevant = {k: v for k, v in organized_data.items() if k[1] == label_type}

            if len(relevant) == 0:
                f.write("  No results found\n\n")
                continue

            f.write(f"{'Feature Set':<25} {'Best Layer':<12} {'Test AUC':<12} {'Test F1':<12} {'Test Acc':<12} {'Samples':<12}\n")
            f.write("-"*120 + "\n")

            for (feature_set, _), data in sorted(relevant.items()):
                # Find best by AUC
                best_auc_idx = np.argmax([entry['test_roc_auc'] for entry in data])
                best_entry = data[best_auc_idx]

                f.write(f"{feature_set:<25} "
                       f"{best_entry['layer_idx']:<12} "
                       f"{best_entry['test_roc_auc']:<12.4f} "
                       f"{best_entry['test_f1']:<12.4f} "
                       f"{best_entry['test_accuracy']:<12.4f} "
                       f"{best_entry['n_train_samples']:<12}\n")

            f.write("\n")

        f.write("="*120 + "\n")

    print(f"  Saved: summary_best.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predictor training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single summary file
  python scripts/predictors/visualize_results.py --input output/predictor_results/summary.json

  # Directory with multiple summary files (auto-detects *summary*.json)
  python scripts/predictors/visualize_results.py --input output/predictor_results/

  # Custom pattern for JSON files in directory
  python scripts/predictors/visualize_results.py --input output/predictors/ --pattern "*predictor*.json"
        """
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/predictor_results/summary.json"),
        help="Path to summary JSON file or directory containing multiple summary files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir/analysis/ for file, input_dir/combined_analysis/ for directory)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*summary*.json",
        help="Glob pattern for finding JSON files in directory (default: *summary*.json)"
    )
    parser.add_argument(
        "--publication-ready",
        action="store_true",
        help="Publication-ready mode: Plot only Test AUC; Remove random baseline; "
             "Remove train curves; No master title; Follow visualize_linear_probe_results.py style"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input path not found: {args.input}")
        return 1

    # Set output directory
    if args.output_dir is None:
        if args.input.is_file():
            args.output_dir = args.input.parent / "analysis"
        else:
            args.output_dir = args.input / "combined_analysis"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*100}")
    print("PREDICTOR RESULTS VISUALIZATION")
    print(f"{'='*100}")
    print(f"Input: {args.input}")
    if args.input.is_dir():
        print(f"Pattern: {args.pattern}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*100}\n")

    # Load data
    data = load_summary(args.input, pattern=args.pattern)

    # Check if we got any data
    if len(data) == 0:
        print("\nERROR: No data loaded. Exiting.")
        return 1

    # Organize by feature set and label type
    organized = organize_by_feature_label(data)

    print(f"\nFound {len(organized)} feature×label combinations:")
    for (feature_set, label_type), entries in sorted(organized.items()):
        print(f"  {feature_set} × {label_type}: {len(entries)} layers")

    # Print comprehensive summary tables to command line
    print_comprehensive_summary_table(organized)
    print_per_layer_summary_table(organized)

    # Generate plots
    print(f"\n{'='*100}")
    if args.publication_ready:
        print("GENERATING PLOTS (PUBLICATION-READY MODE)")
    else:
        print("GENERATING PLOTS")
    print(f"{'='*100}\n")

    if args.publication_ready:
        # Publication mode: only AUC plots
        for (feature_set, label_type), data_subset in sorted(organized.items()):
            print(f"\n[{feature_set} × {label_type}]")
            plot_auc_by_layer(data_subset, feature_set, label_type, args.output_dir, publication_ready=True)

        # Master comparison plot in publication mode
        print(f"\n[Master Comparison - All Settings]")
        plot_master_comparison_all_settings(organized, args.output_dir, publication_ready=True)
    else:
        # Standard mode: all plots
        # Individual plots for each feature×label combination
        for (feature_set, label_type), data_subset in sorted(organized.items()):
            print(f"\n[{feature_set} × {label_type}]")
            plot_auc_by_layer(data_subset, feature_set, label_type, args.output_dir)
            plot_accuracy_f1_by_layer(data_subset, feature_set, label_type, args.output_dir)

        # Comparison plots
        print(f"\n[Comparison Plots]")
        plot_comparison_across_features(organized, 'correctness', args.output_dir)
        plot_comparison_across_features(organized, 'error_type', args.output_dir)

        # Master comparison plot showing all settings in one figure
        print(f"\n[Master Comparison - All Settings]")
        plot_master_comparison_all_settings(organized, args.output_dir, publication_ready=False)

    # Detailed tables for each feature×label combination (skip in publication mode)
    if not args.publication_ready:
        print(f"\n{'='*100}")
        print("GENERATING DETAILED TABLES")
        print(f"{'='*100}\n")
        create_detailed_tables(organized, args.output_dir)

        # Summary table
        print(f"\n[Best Results Summary]")
        create_summary_table(organized, args.output_dir)

    print(f"\n{'='*100}")
    print(f"✓ Visualization complete!")
    print(f"  All outputs saved to: {args.output_dir}")
    print(f"{'='*100}\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
