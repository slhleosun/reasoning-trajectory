#!/usr/bin/env python
"""Train stepwise binary classifiers: Step 1 vs Others, Step 2 vs Others, ..., #### vs Others

For each target (Step 1, Step 2, ..., Step 5, ####):
- Train one binary classifier per layer (32 classifiers total)
- Use raw activations or PCA-reduced activations (32 PCs)
- Evaluate with accuracy, F1, and AUC
- Create exact PCA visualizations with mathematically correct decision boundaries
- Add prediction accuracy to subplot legends

Mathematical Rigor:
For visualization, we use exact PCA projection (not t-SNE):
1. Fit PCA to project activations x_i ∈ R^d to 2D: z_i = U @ x_i
2. Project classifier weights to 2D: w_2D = U @ w
3. Plot exact decision boundary in PCA plane: w_2D^T @ z + b = 0

This yields mathematically exact projection of the probe, suitable for
quantitative separability arguments.

Outputs:
- Trained classifiers (one per layer per target)
- Performance metrics JSON
- Exact PCA visualization grids with decision boundaries

Usage:
    # Train with high-dim features, visualize with PCA-2D
 python scripts/predictors/train_stepwise_probes.py \
      --data output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
      --output output/stepwise_probes

    # Train with PCA-reduced features (e.g., 32 components), visualize with PCA-2D
        --use-pca --pca-components 32
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


@dataclass
class ClassifierResult:
    """Results for a single classifier"""
    target: str  # "step_1", "step_2", ..., "step_5", "hash"
    layer_idx: int
    use_pca: bool
    pca_components: int

    # Performance metrics
    train_accuracy: float
    test_accuracy: float
    train_f1: float
    test_f1: float
    train_auc: float
    test_auc: float
    train_precision: float
    test_precision: float
    train_recall: float
    test_recall: float

    # Data statistics
    n_train_samples: int
    n_test_samples: int
    n_positive_train: int
    n_positive_test: int
    n_negative_train: int
    n_negative_test: int

    # Confusion matrix
    confusion_matrix: List[List[int]]


def load_steering_data(npz_path: Path) -> Tuple:
    """Load steering vector data

    Returns:
        step_acts: List of [n_step, hidden_dim] for each layer
        hash_acts: List of [n_hash, hidden_dim] for each layer
        step_numbers: List of [n_step] for each layer
        num_layers: int
        hidden_dim: int
    """
    print(f"\nLoading data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])
    step_acts = data["step_activations"]
    hash_acts = data["hash_activations"]
    step_numbers = data.get("step_numbers", None)

    # Handle old format without step numbers
    if step_numbers is None:
        print("  Warning: No step_numbers found, creating dummy (all steps = 1)")
        step_numbers = [
            np.ones(len(step_acts[i]), dtype=np.int32) if len(step_acts[i]) > 0
            else np.array([], dtype=np.int32)
            for i in range(num_layers)
        ]

    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Step activations (layer 0): {len(step_acts[0])}")
    print(f"  Hash activations (layer 0): {len(hash_acts[0])}")

    if len(step_numbers[0]) > 0:
        unique, counts = np.unique(step_numbers[0], return_counts=True)
        print(f"  Step distribution (layer 0): {dict(zip(unique, counts))}")

    return step_acts, hash_acts, step_numbers, num_layers, hidden_dim


def prepare_binary_dataset(
    step_acts: np.ndarray,
    hash_acts: np.ndarray,
    step_numbers: np.ndarray,
    target: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare binary classification dataset

    Args:
        step_acts: [n_step, hidden_dim]
        hash_acts: [n_hash, hidden_dim]
        step_numbers: [n_step]
        target: "step_1", "step_2", ..., "step_5", "hash"

    Returns:
        X: [n_samples, hidden_dim]
        y: [n_samples] - 1 for target class, 0 for others
    """
    if target == "hash":
        # Target: #### (hash)
        # Positive: hash activations
        # Negative: all step activations
        X_pos = hash_acts
        X_neg = step_acts

    else:
        # Target: Step N
        # Extract step number
        step_num = int(target.split("_")[1])

        # Positive: this step's activations
        # Negative: all other steps + hash
        mask = step_numbers == step_num
        X_pos = step_acts[mask]
        X_neg_steps = step_acts[~mask]
        X_neg = np.vstack([X_neg_steps, hash_acts]) if len(X_neg_steps) > 0 else hash_acts

    # Combine
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    return X, y


def train_classifier_for_layer(
    step_acts: np.ndarray,
    hash_acts: np.ndarray,
    step_numbers: np.ndarray,
    target: str,
    layer_idx: int,
    use_pca: bool = False,
    pca_components: int = 32,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[ClassifierResult, LogisticRegression, PCA]:
    """Train binary classifier for one layer

    Returns:
        result: ClassifierResult with metrics
        clf: Trained LogisticRegression
        pca: Fitted PCA (or None if not used)
    """
    # Prepare dataset
    X, y = prepare_binary_dataset(step_acts, hash_acts, step_numbers, target)

    # Check class balance
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos < 2 or n_neg < 2:
        # Not enough samples for training
        return None, None, None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply PCA if requested
    pca = None
    if use_pca:
        n_components = min(pca_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Train classifier
    clf = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    # Count positives/negatives
    n_pos_train = np.sum(y_train == 1)
    n_pos_test = np.sum(y_test == 1)
    n_neg_train = np.sum(y_train == 0)
    n_neg_test = np.sum(y_test == 0)

    result = ClassifierResult(
        target=target,
        layer_idx=layer_idx,
        use_pca=use_pca,
        pca_components=pca_components if use_pca else 0,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        train_f1=train_f1,
        test_f1=test_f1,
        train_auc=train_auc,
        test_auc=test_auc,
        train_precision=train_precision,
        test_precision=test_precision,
        train_recall=train_recall,
        test_recall=test_recall,
        n_train_samples=len(y_train),
        n_test_samples=len(y_test),
        n_positive_train=int(n_pos_train),
        n_positive_test=int(n_pos_test),
        n_negative_train=int(n_neg_train),
        n_negative_test=int(n_neg_test),
        confusion_matrix=cm
    )

    return result, clf, pca


def plot_pca_with_exact_decision_boundary(
    step_acts: np.ndarray,
    hash_acts: np.ndarray,
    step_numbers: np.ndarray,
    target: str,
    layer_idx: int,
    clf: LogisticRegression,
    pca_train: Optional[PCA],
    test_acc: float,
    ax=None,
    use_pca_training: bool = False
):
    """Create exact PCA projection with mathematically correct decision boundary

    This is the rigorous approach:
    1. Fit PCA to project activations x_i ∈ R^d to 2D: z_i = U @ x_i
    2. Project classifier weights to 2D: w_2D = U @ w
    3. Plot exact decision boundary: w_2D^T @ z + b = 0

    Args:
        step_acts: [n_step, hidden_dim]
        hash_acts: [n_hash, hidden_dim]
        step_numbers: [n_step]
        target: "step_1", ..., "hash"
        layer_idx: Layer index
        clf: Trained classifier
        pca_train: PCA used during training (if any)
        test_acc: Test accuracy to display
        ax: Matplotlib axis
        use_pca_training: Whether PCA was used during training
    """
    # Prepare data
    X, y = prepare_binary_dataset(step_acts, hash_acts, step_numbers, target)

    if len(X) == 0:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNo data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    # If PCA was used during training, apply it first
    X_for_vis = X.copy()
    if use_pca_training and pca_train is not None:
        X_for_vis = pca_train.transform(X)
        # Classifier operates in PCA-reduced space
        clf_weights = clf.coef_[0]  # [pca_components]
        clf_bias = clf.intercept_[0]
    else:
        # Classifier operates in original space
        clf_weights = clf.coef_[0]  # [hidden_dim]
        clf_bias = clf.intercept_[0]

    # Fit PCA to project to 2D for visualization
    n_samples = len(X_for_vis)
    if n_samples < 2:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNot enough data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    n_components_vis = min(2, n_samples, X_for_vis.shape[1])
    if n_components_vis < 2:
        if ax is not None:
            ax.text(0.5, 0.5, f"Layer {layer_idx}\nNot enough data",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    pca_vis = PCA(n_components=2, random_state=42)
    X_2d = pca_vis.fit_transform(X_for_vis)  # [n_samples, 2]

    # Project classifier weights to 2D: w_2D = U^T @ w
    # U is pca_vis.components_ [2, d] where d = X_for_vis.shape[1]
    w_2d = pca_vis.components_ @ clf_weights  # [2]
    b_2d = clf_bias

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone = True
    else:
        standalone = False

    # Plot points
    mask_pos = y == 1
    mask_neg = y == 0

    ax.scatter(X_2d[mask_pos, 0], X_2d[mask_pos, 1],
              c='red', alpha=0.8, s=50, label='Target',
              edgecolors='darkred', linewidths=0.5)
    ax.scatter(X_2d[mask_neg, 0], X_2d[mask_neg, 1],
              c='blue', alpha=0.6, s=30, label='Others',
              edgecolors='darkblue', linewidths=0.3)

    # Plot exact decision boundary: w_2D^T @ z + b = 0
    # => w_2d[0] * z0 + w_2d[1] * z1 + b_2d = 0
    # => z1 = -(w_2d[0] * z0 + b_2d) / w_2d[1]
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    if np.abs(w_2d[1]) > 1e-6:
        # Boundary is not vertical
        z0_line = np.linspace(x_min, x_max, 200)
        z1_line = -(w_2d[0] * z0_line + b_2d) / w_2d[1]
        ax.plot(z0_line, z1_line, 'k-', linewidth=2, label='Exact Boundary')
    else:
        # Boundary is vertical (w_2d[1] ≈ 0)
        if np.abs(w_2d[0]) > 1e-6:
            z0_boundary = -b_2d / w_2d[0]
            ax.axvline(x=z0_boundary, color='black', linewidth=2, label='Exact Boundary')

    # Fill regions
    # Create mesh grid for background shading
    h = 0.02 * (x_max - x_min)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid using EXACT decision function
    # Decision: w_2d^T @ z + b_2d > 0 => class 1
    decision_values = Z_grid @ w_2d + b_2d
    Z = (decision_values > 0).astype(int).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.15, levels=1, colors=['lightblue', 'lightcoral'])

    # Title with accuracy
    target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
    ax.set_title(f"Layer {layer_idx}: {target_name} vs Others\nAcc={test_acc:.3f}",
                fontsize=10, fontweight='bold')
    ax.set_xlabel("PCA Component 1", fontsize=8)
    ax.set_ylabel("PCA Component 2", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Legend with sample counts
    n_pos = np.sum(mask_pos)
    n_neg = np.sum(mask_neg)
    handles = [
        Patch(facecolor='red', edgecolor='darkred', label=f'Target: {n_pos}'),
        Patch(facecolor='blue', edgecolor='darkblue', label=f'Others: {n_neg}'),
        Patch(facecolor='black', edgecolor='black', label='Exact Boundary')
    ]
    ax.legend(handles=handles, fontsize=7, loc='best', framealpha=0.9)

    if standalone:
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train stepwise binary classifiers for Step 1-5 and ####"
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
                        help="Random seed")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("STEPWISE BINARY CLASSIFIER TRAINING")
    print("="*100)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Use PCA: {args.use_pca}")
    if args.use_pca:
        print(f"  PCA components: {args.pca_components}")
    print(f"Test size: {args.test_size}")
    print(f"Max step: {args.max_step}")
    print(f"Random seed: {args.seed}")
    print("="*100 + "\n")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    step_acts, hash_acts, step_numbers, num_layers, hidden_dim = load_steering_data(args.data)

    # Define targets
    targets = [f"step_{i}" for i in range(1, args.max_step + 1)] + ["hash"]
    print(f"\nTargets: {targets}")

    # Storage for results
    all_results = []
    all_classifiers = {}  # {target: {layer_idx: (clf, pca)}}

    # Train classifiers
    print("\n" + "="*100)
    print("TRAINING CLASSIFIERS")
    print("="*100 + "\n")

    for target in targets:
        print(f"\n[Target: {target}]")
        all_classifiers[target] = {}

        for layer_idx in tqdm(range(num_layers), desc=f"  Layers", ncols=100):
            result, clf, pca = train_classifier_for_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
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

            print(f"\n  Summary for {target}:")
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
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"  Saved results to {results_path}")

    # Create visualizations
    print("\n" + "="*100)
    print("CREATING EXACT PCA VISUALIZATIONS WITH DECISION BOUNDARIES")
    print("="*100 + "\n")

    plots_dir = args.output / "pca_plots"
    plots_dir.mkdir(exist_ok=True)

    for target in targets:
        print(f"\n[Visualizing: {target}]")

        # Create grid plot (8x4 for 32 layers)
        n_cols = 8
        n_rows = (num_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for layer_idx in tqdm(range(num_layers), desc="  Layers", ncols=100):
            if layer_idx in all_classifiers[target]:
                clf, pca = all_classifiers[target][layer_idx]

                # Get test accuracy for this layer
                result = next((r for r in all_results
                              if r.target == target and r.layer_idx == layer_idx), None)
                test_acc = result.test_accuracy if result else 0.0

                plot_pca_with_exact_decision_boundary(
                    step_acts[layer_idx],
                    hash_acts[layer_idx],
                    step_numbers[layer_idx],
                    target,
                    layer_idx,
                    clf,
                    pca,
                    test_acc,
                    ax=axes[layer_idx],
                    use_pca_training=args.use_pca
                )
            else:
                axes[layer_idx].text(0.5, 0.5, f"Layer {layer_idx}\nNo classifier",
                                    ha='center', va='center',
                                    transform=axes[layer_idx].transAxes)
                axes[layer_idx].set_xticks([])
                axes[layer_idx].set_yticks([])

        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

        # Title
        target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
        pca_suffix = f" (PCA-{args.pca_components})" if args.use_pca else ""
        plt.suptitle(f"Binary Classification: {target_name} vs Others{pca_suffix}\n(Exact PCA Projection)",
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0.01, 1, 0.995])

        # Save
        output_path = plots_dir / f"{target}_decision_boundaries.png"
        print(f"  Saving to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {target} visualization saved!")

    # Create summary plots (AUC across layers for each target)
    print("\n[Creating summary plots]")

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
                   label='Test AUC', color='#2E86AB')
            ax.plot(layers, test_acc, '^-', linewidth=2, markersize=6,
                   label='Test Acc', color='#F18F01')
            ax.plot(layers, test_f1, 's-', linewidth=2, markersize=6,
                   label='Test F1', color='#6A994E')

            # Random baseline
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1,
                      alpha=0.5, label='Random')

            target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
            ax.set_title(f"{target_name} vs Others", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)

            # Add stats box
            best_auc = max(test_auc)
            best_layer = layers[test_auc.index(best_auc)]
            stats_text = f"Best: {best_auc:.3f} @ L{best_layer}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplot
    if len(targets) < 6:
        axes[5].axis('off')

    pca_suffix = f" (PCA-{args.pca_components})" if args.use_pca else ""
    plt.suptitle(f"Stepwise Binary Classification Performance{pca_suffix}",
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0.01, 1, 0.995])

    summary_path = plots_dir / "summary_performance.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary plot saved to {summary_path}")

    # Final summary
    print("\n" + "="*100)
    print("TRAINING COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {args.output}")
    print(f"  Classifiers: {classifiers_dir}/")
    print(f"  Results JSON: {results_path}")
    print(f"  Visualizations: {plots_dir}/")
    print()

    # Print best performers
    print("Best Performers (by AUC):")
    print("-" * 80)
    for target in targets:
        target_results = [r for r in all_results if r.target == target]
        if len(target_results) > 0:
            best = max(target_results, key=lambda r: r.test_auc)
            target_name = "####" if target == "hash" else f"Step {target.split('_')[1]}"
            print(f"  {target_name:10s} - Layer {best.layer_idx:2d}: "
                  f"AUC={best.test_auc:.4f}, Acc={best.test_accuracy:.4f}, F1={best.test_f1:.4f}")
    print("="*100 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
