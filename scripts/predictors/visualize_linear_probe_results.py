#!/usr/bin/env python
"""Compare stepwise binary classifier performance across different model versions

Reads results from multiple model subdirectories (e.g., llamabase, llamainst, llamar1)
and creates comparison plots showing test accuracy across layers for each model.

Usage:
    python scripts/predictors/compare_model_stepwise_classifiers.py \
        --master-dir output/stepwise_classifiers_pca \
        --output output/model_comparison.png

    python scripts/predictors/compare_model_stepwise_classifiers.py \
        --master-dir output/stepwise_classifiers_pca \
        --models llamabase llamainst llamar1 \
        --output output/model_comparison.png

Directory structure expected:
    output/stepwise_classifiers_pca/
    ├── llamabase/
    │   └── results.json
    ├── llamainst/
    │   └── results.json
    └── llamar1/
        └── results.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_results_from_models(master_dir: Path, model_names: List[str] = None) -> Dict:
    """Load results.json from each model subdirectory

    Args:
        master_dir: Master directory containing model subdirectories
        model_names: List of model subdirectory names. If None, auto-detect.

    Returns:
        Dict mapping model_name -> list of result dicts
    """
    print(f"\nLoading results from: {master_dir}")

    if model_names is None:
        # Auto-detect subdirectories
        model_names = [d.name for d in master_dir.iterdir() if d.is_dir()]
        print(f"  Auto-detected models: {model_names}")

    all_results = {}

    for model_name in model_names:
        model_dir = master_dir / model_name
        results_path = model_dir / "results.json"

        if not results_path.exists():
            print(f"  Warning: {results_path} not found, skipping {model_name}")
            continue

        with open(results_path, 'r') as f:
            results = json.load(f)

        all_results[model_name] = results
        print(f"  ✓ Loaded {len(results)} results from {model_name}")

    if not all_results:
        raise ValueError(f"No valid results found in {master_dir}")

    return all_results


def organize_results_by_target(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Organize results by target (step_1, step_2, ..., hash)

    Args:
        results: List of result dictionaries

    Returns:
        Dict mapping target -> list of results for that target
    """
    by_target = defaultdict(list)
    for result in results:
        target = result['target']
        by_target[target].append(result)

    # Sort each target's results by layer_idx
    for target in by_target:
        by_target[target] = sorted(by_target[target], key=lambda r: r['layer_idx'])

    return dict(by_target)


def get_best_layer_and_metric(results: List[Dict], metric: str = 'test_accuracy') -> Tuple[int, float]:
    """Get the layer with best metric value

    Args:
        results: List of result dictionaries for a target
        metric: Metric to use ('test_accuracy', 'test_auc', 'test_f1')

    Returns:
        (best_layer_idx, best_metric_value)
    """
    if not results:
        return -1, 0.0

    best = max(results, key=lambda r: r[metric])
    return best['layer_idx'], best[metric]


def plot_comparison(
    all_model_results: Dict[str, Dict[str, List[Dict]]],
    output_path: Path,
    metric: str = 'test_accuracy',
    targets: List[str] = None,
    publication_ready: bool = False,
    publication_appendix: bool = False
):
    """Create comparison plots for all targets across models

    Args:
        all_model_results: Dict[model_name -> Dict[target -> List[results]]]
        output_path: Path to save plot
        metric: Metric to plot ('test_accuracy', 'test_auc', 'test_f1')
        targets: List of targets to plot. If None, auto-detect.
        publication_ready: If True, use 1×4 layout with Steps 2,3,5 and Final Answer
        publication_appendix: If True, use 2×3 layout with Steps 1-5 and Final Answer
    """
    print("\n" + "="*100)
    print("CREATING COMPARISON PLOTS")
    print("="*100 + "\n")

    # Publication-appendix mode: filter to Steps 1-5 and hash (in that order)
    if publication_appendix:
        if targets is None:
            targets = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']
            print(f"  Publication appendix mode - using targets: {targets}")
    # Publication-ready mode: filter to specific targets only
    elif publication_ready:
        # Filter to step_2, step_3, step_5, and hash only (in that order)
        if targets is None:
            targets = ['step_2', 'step_3', 'step_5', 'hash']
            print(f"  Publication mode - using targets: {targets}")
    else:
        # Auto-detect targets from first model
        if targets is None:
            first_model = list(all_model_results.keys())[0]
            targets = sorted(all_model_results[first_model].keys())
            print(f"  Auto-detected targets: {targets}")

    # Color scheme for models
    model_names = list(all_model_results.keys())
    colors = ['#2E86AB', '#F18F01', '#6A994E', '#A23B72', '#C73E1D', '#5E548E']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}

    # Marker styles for models
    markers = ['o', 's', '^', 'D', 'v', 'p']
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(model_names)}

    # Create clean model name mapping
    def clean_model_name(model_name: str) -> str:
        """Convert model name to clean label"""
        model_lower = model_name.lower()
        if 'inst' in model_lower:
            return 'Instruct'
        elif 'r1' in model_lower:
            return 'R1-Distilled'
        elif 'base' in model_lower:
            return 'Base'
        else:
            return model_name  # Return as-is if no match

    # Create subplots (1×4 for publication-ready, 2×3 for publication-appendix or regular)
    if publication_ready:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes = axes.flatten() if len(targets) > 1 else [axes]
    elif publication_appendix:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

    for idx, target in enumerate(targets):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot each model
        for model_name in model_names:
            if target not in all_model_results[model_name]:
                continue

            results = all_model_results[model_name][target]

            if not results:
                continue

            # Extract data
            layers = [r['layer_idx'] for r in results]
            metric_values = [r[metric] for r in results]

            # Plot
            if publication_ready or publication_appendix:
                # Clean label in publication mode
                label = clean_model_name(model_name)
            else:
                # Full label with best performance in regular mode
                best_layer, best_value = get_best_layer_and_metric(results, metric)
                label = f"{model_name} (Best: {best_value:.3f} @ L{best_layer})"

            ax.plot(layers, metric_values,
                   marker=model_markers[model_name],
                   linestyle='-',
                   linewidth=2.5,
                   markersize=7,
                   label=label,
                   color=model_colors[model_name],
                   alpha=0.9)

            # Mark best layer with a star (skip in publication modes)
            if not publication_ready and not publication_appendix:
                best_layer, best_value = get_best_layer_and_metric(results, metric)
                ax.plot(best_layer, best_value,
                       marker='*',
                       markersize=15,
                       color=model_colors[model_name],
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       zorder=10)

        # Random baseline (skip in publication modes)
        if not publication_ready and not publication_appendix:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
                      alpha=0.5, label='Random', zorder=0)

        # Formatting
        if publication_ready or publication_appendix:
            target_name = "Final Answer Marker" if target == "hash" else f"Step {target.split('_')[1]}"
            ax.set_title(target_name, fontsize=16, fontweight='bold')
        else:
            target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
            ax.set_title(f"{target_name} vs Others", fontsize=16, fontweight='bold')

        ax.set_xlabel("Layer", fontsize=13)

        # Y-axis label based on metric
        metric_labels = {
            'test_accuracy': 'Test Accuracy',
            'test_auc': 'Test AUC',
            'test_f1': 'Test F1'
        }
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=13)

        # Show legend only in the first subplot (leftmost) in publication modes
        if publication_ready or publication_appendix:
            if idx == 0:
                ax.legend(loc='best', fontsize=9, framealpha=0.95)
        else:
            ax.legend(loc='best', fontsize=9, framealpha=0.95)

        ax.grid(True, alpha=0.3, linewidth=0.8)

        # Set y-axis range
        if publication_ready or publication_appendix:
            ax.set_ylim(0.55, 1.05)
        else:
            ax.set_ylim(0.3, 1.05)

        ax.tick_params(labelsize=11)

        # Add subtle background
        ax.set_facecolor('#f9f9f9')

    # Hide unused subplots
    for idx in range(len(targets), len(axes)):
        axes[idx].axis('off')

    # Overall title (skip in publication modes)
    if not publication_ready and not publication_appendix:
        model_list = ", ".join(model_names)
        metric_name = metric_labels.get(metric, metric)
        plt.suptitle(f"Stepwise Binary Classification: {metric_name} Comparison\nModels: {model_list}",
                    fontsize=18, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0.01, 1, 0.995])
    else:
        plt.tight_layout()

    # Save
    print(f"\n  Saving comparison plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison plot saved!")


def print_per_target_detailed_tables(all_model_results: Dict[str, Dict[str, List[Dict]]]):
    """Print detailed tables for each target showing all three metrics across layers

    Args:
        all_model_results: Dict[model_name -> Dict[target -> List[results]]]
    """
    # Get all targets
    first_model = list(all_model_results.keys())[0]
    targets = sorted(all_model_results[first_model].keys())
    model_names = list(all_model_results.keys())

    for target in targets:
        target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"

        print("\n" + "="*120)
        print(f"DETAILED METRICS FOR {target_name.upper()} VS OTHERS")
        print("="*120)

        # Collect all layers
        all_layers = set()
        for model in model_names:
            if target in all_model_results[model]:
                for result in all_model_results[model][target]:
                    all_layers.add(result['layer_idx'])
        layers = sorted(all_layers)

        if not layers:
            print("  No data available")
            continue

        # Print header
        header = f"{'Layer':<8}"
        for model in model_names:
            header += f"{model:<36}"
        print(header)

        subheader = f"{'':8}"
        for _ in model_names:
            subheader += f"{'Acc':>10} {'AUC':>10} {'F1':>10}      "
        print(subheader)
        print("-" * 120)

        # Print each layer
        for layer_idx in layers:
            row = f"{layer_idx:<8}"

            for model in model_names:
                if target in all_model_results[model]:
                    # Find result for this layer
                    layer_result = None
                    for result in all_model_results[model][target]:
                        if result['layer_idx'] == layer_idx:
                            layer_result = result
                            break

                    if layer_result:
                        acc = layer_result['test_accuracy']
                        auc = layer_result['test_auc']
                        f1 = layer_result['test_f1']
                        row += f"{acc:>10.4f} {auc:>10.4f} {f1:>10.4f}      "
                    else:
                        row += f"{'N/A':>10} {'N/A':>10} {'N/A':>10}      "
                else:
                    row += f"{'N/A':>10} {'N/A':>10} {'N/A':>10}      "

            print(row)

        # Print best performers
        print("-" * 120)
        best_row = f"{'BEST':<8}"
        for model in model_names:
            if target in all_model_results[model]:
                results = all_model_results[model][target]
                best_acc_layer, best_acc = get_best_layer_and_metric(results, 'test_accuracy')
                best_auc_layer, best_auc = get_best_layer_and_metric(results, 'test_auc')
                best_f1_layer, best_f1 = get_best_layer_and_metric(results, 'test_f1')
                best_row += f"L{best_acc_layer:2d}:{best_acc:.3f} L{best_auc_layer:2d}:{best_auc:.3f} L{best_f1_layer:2d}:{best_f1:.3f}      "
            else:
                best_row += f"{'N/A':>10} {'N/A':>10} {'N/A':>10}      "

        print(best_row)
        print("="*120)


def print_summary_table(all_model_results: Dict[str, Dict[str, List[Dict]]]):
    """Print a summary table of best performance per model per target

    Args:
        all_model_results: Dict[model_name -> Dict[target -> List[results]]]
    """
    print("\n" + "="*100)
    print("BEST PERFORMANCE SUMMARY (Test Accuracy)")
    print("="*100)

    # Get all targets
    first_model = list(all_model_results.keys())[0]
    targets = sorted(all_model_results[first_model].keys())
    model_names = list(all_model_results.keys())

    # Print header
    header = f"{'Target':<15}"
    for model in model_names:
        header += f"{model:<25}"
    print(header)
    print("-" * 100)

    # Print each target
    for target in targets:
        target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
        row = f"{target_name:<15}"

        for model in model_names:
            if target in all_model_results[model]:
                results = all_model_results[model][target]
                best_layer, best_acc = get_best_layer_and_metric(results, 'test_accuracy')
                row += f"L{best_layer:2d}: {best_acc:.4f}          "
            else:
                row += f"{'N/A':<25}"

        print(row)

    print("="*100 + "\n")


def print_average_performance(all_model_results: Dict[str, Dict[str, List[Dict]]]):
    """Print average performance across all layers for each model

    Args:
        all_model_results: Dict[model_name -> Dict[target -> List[results]]]
    """
    print("\n" + "="*100)
    print("AVERAGE PERFORMANCE ACROSS ALL LAYERS")
    print("="*100)

    model_names = list(all_model_results.keys())

    # Print header
    header = f"{'Model':<15}{'Avg Acc':<12}{'Avg AUC':<12}{'Avg F1':<12}{'# Results':<12}"
    print(header)
    print("-" * 60)

    for model in model_names:
        all_results = []
        for target_results in all_model_results[model].values():
            all_results.extend(target_results)

        if not all_results:
            continue

        avg_acc = np.mean([r['test_accuracy'] for r in all_results])
        avg_auc = np.mean([r['test_auc'] for r in all_results])
        avg_f1 = np.mean([r['test_f1'] for r in all_results])
        n_results = len(all_results)

        print(f"{model:<15}{avg_acc:<12.4f}{avg_auc:<12.4f}{avg_f1:<12.4f}{n_results:<12d}")

    print("="*100 + "\n")


def print_accuracy_tables(all_model_results: Dict[str, Dict[str, List[Dict]]]):
    """Print accuracy tables for each target (Steps 1-5 and Final Answer Marker)

    Creates 6 tables where each table shows:
    - Rows: Layers (0-31)
    - Columns: Models
    - Values: Test accuracy

    Args:
        all_model_results: Dict[model_name -> Dict[target -> List[results]]]
    """
    # Define targets in order: Steps 1-5 and Final Answer Marker
    targets = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']
    model_names = list(all_model_results.keys())

    print("\n" + "="*100)
    print("LINEAR PROBE ACCURACY TABLES")
    print("="*100)

    for target in targets:
        # Create target display name
        if target == 'hash':
            target_name = "Final Answer Marker"
        else:
            step_num = target.split('_')[1]
            target_name = f"Step {step_num}"

        print("\n" + "="*100)
        print(f"{target_name.upper()}")
        print("="*100)

        # Check if target exists in all models
        target_exists = any(target in all_model_results[model] for model in model_names)
        if not target_exists:
            print(f"  No data available for {target_name}")
            continue

        # Collect all layers across all models for this target
        all_layers = set()
        for model in model_names:
            if target in all_model_results[model]:
                for result in all_model_results[model][target]:
                    all_layers.add(result['layer_idx'])

        layers = sorted(all_layers)

        if not layers:
            print(f"  No data available for {target_name}")
            continue

        # Determine column width based on model name lengths
        max_model_len = max(len(model) for model in model_names)
        col_width = max(max_model_len + 2, 12)

        # Print header
        header = f"{'Layer':<8}"
        for model in model_names:
            header += f"{model:<{col_width}}"
        print(header)
        print("-" * (8 + col_width * len(model_names)))

        # Print each layer
        for layer_idx in layers:
            row = f"{layer_idx:<8}"

            for model in model_names:
                if target in all_model_results[model]:
                    # Find result for this layer
                    layer_result = None
                    for result in all_model_results[model][target]:
                        if result['layer_idx'] == layer_idx:
                            layer_result = result
                            break

                    if layer_result:
                        acc = layer_result['test_accuracy']
                        row += f"{acc:<{col_width}.4f}"
                    else:
                        row += f"{'N/A':<{col_width}}"
                else:
                    row += f"{'N/A':<{col_width}}"

            print(row)

        print("="*100)

    print("\n" + "="*100)
    print("TABLE MODE COMPLETE")
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare stepwise binary classifier performance across model versions"
    )
    parser.add_argument("--master-dir", type=Path, required=True,
                        help="Master directory containing model subdirectories")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="List of model subdirectory names (auto-detect if not specified)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for comparison plot")
    parser.add_argument("--metric", type=str, default='test_accuracy',
                        choices=['test_accuracy', 'test_auc', 'test_f1'],
                        help="Metric to plot (default: test_accuracy)")
    parser.add_argument("--targets", type=str, nargs='+', default=None,
                        help="List of targets to plot (auto-detect if not specified)")
    parser.add_argument("--publication-ready", action="store_true",
                        help="Publication-ready mode: 1×4 layout with Steps 2,3,5 and Final Answer Marker; "
                             "Remove master title; Remove random baseline; Set y-axis 0.55-1.05; "
                             "Remove 'vs Others' from titles")
    parser.add_argument("--publication-appendix", action="store_true",
                        help="Publication-appendix mode: 2×3 layout with Steps 1-5 and Final Answer Marker; "
                             "Same formatting as publication-ready (no master title, no baseline, etc.)")
    parser.add_argument("--table", action="store_true",
                        help="Table mode: Print accuracy tables for Steps 1-5 and Final Answer Marker; "
                             "No plots are created in this mode")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("STEPWISE BINARY CLASSIFIER MODEL COMPARISON")
    print("="*100)
    print(f"Master directory: {args.master_dir}")
    print(f"Output: {args.output}")
    print(f"Metric: {args.metric}")
    if args.models:
        print(f"Models: {args.models}")
    else:
        print(f"Models: auto-detect")
    print("="*100)

    # Check master directory exists
    if not args.master_dir.exists():
        print(f"\nError: Master directory not found: {args.master_dir}")
        return 1

    # Load results from all models
    all_model_results_raw = load_results_from_models(args.master_dir, args.models)

    # Organize by target for each model
    all_model_results = {}
    for model_name, results in all_model_results_raw.items():
        all_model_results[model_name] = organize_results_by_target(results)

    # Table mode: print accuracy tables and exit
    if args.table:
        print_accuracy_tables(all_model_results)
        print("\n" + "="*100)
        print("TABLE MODE COMPLETE!")
        print("="*100 + "\n")
        return 0

    # Print detailed per-target tables
    print_per_target_detailed_tables(all_model_results)

    # Print summary tables
    print_summary_table(all_model_results)
    print_average_performance(all_model_results)

    # Create comparison plot
    plot_comparison(all_model_results, args.output, args.metric, args.targets,
                   publication_ready=args.publication_ready,
                   publication_appendix=args.publication_appendix)

    # Final message
    print("\n" + "="*100)
    print("COMPARISON COMPLETE!")
    print("="*100)
    print(f"\nComparison plot saved to: {args.output}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
