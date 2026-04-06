#!/usr/bin/env python
"""Train stepwise binary classifiers with SHUFFLED step labels (null hypothesis baseline)

This script creates a null hypothesis baseline by randomly shuffling step labels
across trajectories. If step separability is meaningful, the original (non-shuffled)
probes should perform significantly better than shuffled probes.

Key difference from train_stepwise_probes.py:
- Step numbers are randomly permuted across all samples
- This destroys the correspondence between activations and their true step numbers
- Expected result: Performance should drop to near-random (0.5 AUC)

Usage:
    python scripts/predictors/train_stepwise_probes_shuffled.py \
        --data output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --output output/stepwise_probes_shuffled \
        --seed 42
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm

# Add parent directory to path to import from train_stepwise_probes
sys.path.insert(0, str(Path(__file__).parent))
from train_stepwise_probes import (
    ClassifierResult,
    load_steering_data,
    train_classifier_for_layer,
    plot_pca_with_exact_decision_boundary
)


def shuffle_step_labels(step_numbers: List[np.ndarray], seed: int = 42) -> List[np.ndarray]:
    """Shuffle step labels across all trajectories

    Args:
        step_numbers: List of [n_step] arrays for each layer
        seed: Random seed for reproducibility

    Returns:
        shuffled_step_numbers: List of [n_step] arrays with shuffled labels
    """
    np.random.seed(seed)

    shuffled = []
    for layer_idx, step_nums in enumerate(step_numbers):
        if len(step_nums) == 0:
            shuffled.append(step_nums)
            continue

        # Shuffle step numbers for this layer
        shuffled_nums = step_nums.copy()
        np.random.shuffle(shuffled_nums)
        shuffled.append(shuffled_nums)

    return shuffled


def main():
    parser = argparse.ArgumentParser(
        description="Train stepwise binary classifiers with SHUFFLED labels (null hypothesis)"
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to steering vectors NPZ file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for classifiers and results")
    parser.add_argument("--use-pca", action="store_true",
                        help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=32,
                        help="Number of PCA components (default: 32)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size (default: 0.2)")
    parser.add_argument("--max-step", type=int, default=5,
                        help="Maximum step number to include (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("STEPWISE BINARY CLASSIFIER TRAINING - SHUFFLED LABELS (NULL HYPOTHESIS)")
    print("="*100)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Use PCA: {args.use_pca}")
    if args.use_pca:
        print(f"  PCA components: {args.pca_components}")
    print(f"Test size: {args.test_size}")
    print(f"Max step: {args.max_step}")
    print(f"Random seed (for shuffling): {args.seed}")
    print("="*100 + "\n")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    step_acts, hash_acts, step_numbers, num_layers, hidden_dim = load_steering_data(args.data)

    # SHUFFLE STEP LABELS
    print("\n" + "="*100)
    print("SHUFFLING STEP LABELS (Creating Null Hypothesis Baseline)")
    print("="*100)
    print("\nOriginal step distribution (layer 0):")
    if len(step_numbers[0]) > 0:
        unique, counts = np.unique(step_numbers[0], return_counts=True)
        print(f"  {dict(zip(unique, counts))}")

    shuffled_step_numbers = shuffle_step_labels(step_numbers, seed=args.seed)

    print(f"\nShuffled step distribution (layer 0):")
    if len(shuffled_step_numbers[0]) > 0:
        unique, counts = np.unique(shuffled_step_numbers[0], return_counts=True)
        print(f"  {dict(zip(unique, counts))}")
    print("\n⚠️  Step labels have been randomly shuffled!")
    print("   This destroys the correspondence between activations and true step numbers.")
    print("   Expected result: Performance should drop to near-random (~0.5 AUC)")
    print("="*100)

    # Define targets
    targets = [f"step_{i}" for i in range(1, args.max_step + 1)] + ["hash"]
    print(f"\nTargets: {targets}")

    # Storage for results
    all_results = []
    all_classifiers = {}  # {target: {layer_idx: (clf, pca)}}

    # Train classifiers with SHUFFLED labels
    print("\n" + "="*100)
    print("TRAINING CLASSIFIERS WITH SHUFFLED LABELS")
    print("="*100 + "\n")

    for target in targets:
        print(f"\n[Target: {target}]")
        all_classifiers[target] = {}

        for layer_idx in tqdm(range(num_layers), desc=f"  Layers", ncols=100):
            result, clf, pca = train_classifier_for_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                shuffled_step_numbers[layer_idx],  # Use SHUFFLED labels
                target,
                layer_idx,
                use_pca=args.use_pca,
                pca_components=args.pca_components,
                test_size=args.test_size,
                random_state=args.seed
            )

            if result is not None:
                all_results.append(result)
                all_classifiers[target][layer_idx] = (clf, pca)

        # Print summary for this target
        target_results = [r for r in all_results if r.target == target]
        if len(target_results) > 0:
            avg_test_acc = np.mean([r.test_accuracy for r in target_results])
            avg_test_f1 = np.mean([r.test_f1 for r in target_results])
            avg_test_auc = np.mean([r.test_auc for r in target_results])
            best_layer = max(target_results, key=lambda r: r.test_auc).layer_idx

            print(f"\n  Summary for {target} (SHUFFLED):")
            print(f"    Layers trained: {len(target_results)}/{num_layers}")
            print(f"    Avg Test Acc: {avg_test_acc:.4f}")
            print(f"    Avg Test F1:  {avg_test_f1:.4f}")
            print(f"    Avg Test AUC: {avg_test_auc:.4f}")
            print(f"    Best layer:   {best_layer}")

    # Save classifiers
    print("\n" + "="*100)
    print("SAVING CLASSIFIERS")
    print("="*100 + "\n")

    classifiers_dir = args.output / "classifiers"
    classifiers_dir.mkdir(exist_ok=True)

    for target in targets:
        for layer_idx, (clf, pca) in all_classifiers[target].items():
            clf_path = classifiers_dir / f"{target}_layer_{layer_idx:02d}.pkl"
            with open(clf_path, 'wb') as f:
                pickle.dump({'clf': clf, 'pca': pca}, f)

    print(f"  Saved {sum(len(v) for v in all_classifiers.values())} classifiers to {classifiers_dir}/")

    # Save results JSON
    results_path = args.output / "results.json"
    results_dict = {
        'shuffled': True,
        'shuffle_seed': args.seed,
        'note': 'Step labels were randomly shuffled across trajectories (null hypothesis baseline)',
        'results': [asdict(r) for r in all_results]
    }
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved results to {results_path}")

    # Create summary plot comparing to random baseline
    print("\n" + "="*100)
    print("CREATING SUMMARY PLOTS")
    print("="*100 + "\n")

    plots_dir = args.output / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]

        target_results = sorted(
            [r for r in all_results if r.target == target],
            key=lambda r: r.layer_idx
        )

        if len(target_results) > 0:
            layers = [r.layer_idx for r in target_results]
            test_auc = [r.test_auc for r in target_results]
            test_acc = [r.test_accuracy for r in target_results]
            test_f1 = [r.test_f1 for r in target_results]

            # Plot metrics
            ax.plot(layers, test_auc, 'o-', linewidth=2, markersize=6,
                   label='Test AUC (Shuffled)', color='#E63946', alpha=0.7)
            ax.plot(layers, test_acc, '^-', linewidth=2, markersize=6,
                   label='Test Acc (Shuffled)', color='#F77F00', alpha=0.7)
            ax.plot(layers, test_f1, 's-', linewidth=2, markersize=6,
                   label='Test F1 (Shuffled)', color='#FCBF49', alpha=0.7)

            # Random baseline
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
                      alpha=0.8, label='Random Baseline')

            # Shaded region around 0.5
            ax.axhspan(0.45, 0.55, alpha=0.1, color='red', label='Near-Random Region')

            target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
            ax.set_title(f"{target_name} vs Others (SHUFFLED LABELS)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)

            # Add stats box
            avg_auc = np.mean(test_auc)
            max_auc = max(test_auc)
            stats_text = f"Avg AUC: {avg_auc:.3f}\nMax AUC: {max_auc:.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Hide unused subplot
    if len(targets) < 6:
        axes[5].axis('off')

    pca_suffix = f" (PCA-{args.pca_components})" if args.use_pca else ""
    plt.suptitle(f"Stepwise Classification with SHUFFLED Labels{pca_suffix}\n(Null Hypothesis Baseline)",
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0.01, 1, 0.995])

    summary_path = plots_dir / "shuffled_performance.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary plot saved to {summary_path}")

    # Final summary
    print("\n" + "="*100)
    print("TRAINING COMPLETE (SHUFFLED LABELS)")
    print("="*100)
    print(f"\nResults saved to: {args.output}")
    print(f"  Classifiers: {classifiers_dir}/")
    print(f"  Results JSON: {results_path}")
    print(f"  Plots: {plots_dir}/")
    print()

    # Print best performers and compare to random
    print("Performance Summary (SHUFFLED LABELS vs Random Baseline):")
    print("-" * 100)
    print(f"{'Target':<12} {'Avg AUC':<12} {'Max AUC':<12} {'Interpretation'}")
    print("-" * 100)

    for target in targets:
        target_results = [r for r in all_results if r.target == target]
        if len(target_results) > 0:
            avg_auc = np.mean([r.test_auc for r in target_results])
            max_auc = max([r.test_auc for r in target_results])
            target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"

            # Interpretation
            if avg_auc > 0.55:
                interp = "⚠️  Above random (unexpected!)"
            elif avg_auc < 0.45:
                interp = "⚠️  Below random (unexpected!)"
            else:
                interp = "✓ Near random (expected)"

            print(f"{target_name:<12} {avg_auc:<12.4f} {max_auc:<12.4f} {interp}")

    print("-" * 100)
    print("\nExpected: With shuffled labels, performance should be near 0.5 (random)")
    print("If performance is significantly > 0.5, it suggests:")
    print("  - The model may be learning spurious correlations")
    print("  - OR there may be data leakage")
    print("\nCompare these results to non-shuffled probes to validate step separability!")
    print("="*100 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
