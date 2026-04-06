#!/usr/bin/env python
"""Apply trained linear probes to a new dataset

Loads trained linear probe .pkl files from a directory and applies them to
a specified npz file to compute accuracy for each step and layer.

Usage:
    python scripts/predictors/apply_linear_probes.py \
        --probes-dir output/stepwise_probes/inst_8000/classifiers \
        --data output/steering_vectors/steering_vectors_r1-distill-llama-8b_8000_deepseek.npz \
        --output output/probe_transfer_results.json
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


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


def load_classifier(pkl_path: Path) -> Tuple:
    """Load classifier from .pkl file

    Returns:
        clf: Trained LogisticRegression
        pca: Fitted PCA (or None if not used)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['clf'], data['pca']


def apply_probe_to_layer(
    step_acts: np.ndarray,
    hash_acts: np.ndarray,
    step_numbers: np.ndarray,
    target: str,
    layer_idx: int,
    clf,
    pca
) -> Dict:
    """Apply trained probe to layer data

    Returns:
        Dict with accuracy, f1, auc, and sample counts
    """
    # Prepare dataset
    X, y = prepare_binary_dataset(step_acts, hash_acts, step_numbers, target)

    if len(X) == 0:
        return None

    # Apply PCA if it was used during training
    if pca is not None:
        X = pca.transform(X)

    # Get predictions
    try:
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_proba)

        return {
            'target': target,
            'layer_idx': layer_idx,
            'accuracy': float(accuracy),
            'f1': float(f1),
            'auc': float(auc),
            'n_samples': len(y),
            'n_positive': int(np.sum(y == 1)),
            'n_negative': int(np.sum(y == 0))
        }
    except Exception as e:
        print(f"  Warning: Failed to apply probe for {target} layer {layer_idx}: {e}")
        return None


def print_accuracy_table(results: List[Dict], num_layers: int):
    """Print accuracy table with layers as rows and targets as columns

    Args:
        results: List of result dicts
        num_layers: Number of layers (typically 32)
    """
    # Define targets in order
    targets = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']
    target_names = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Final Answer']

    # Organize results by layer and target
    results_dict = {}
    for result in results:
        if result is not None:
            layer_idx = result['layer_idx']
            target = result['target']
            if layer_idx not in results_dict:
                results_dict[layer_idx] = {}
            results_dict[layer_idx][target] = result

    print("\n" + "="*120)
    print("LINEAR PROBE ACCURACY TABLE")
    print("="*120)

    # Print header
    header = f"{'Layer':<8}"
    for target_name in target_names:
        header += f"{target_name:<15}"
    print(header)
    print("-" * 120)

    # Print each layer
    for layer_idx in range(num_layers):
        row = f"{layer_idx:<8}"

        for target in targets:
            if layer_idx in results_dict and target in results_dict[layer_idx]:
                acc = results_dict[layer_idx][target]['accuracy']
                row += f"{acc:<15.4f}"
            else:
                row += f"{'N/A':<15}"

        print(row)

    # Print average row
    print("-" * 120)
    avg_row = f"{'AVG':<8}"

    for target in targets:
        target_results = [r for r in results if r is not None and r['target'] == target]
        if target_results:
            avg_acc = np.mean([r['accuracy'] for r in target_results])
            avg_row += f"{avg_acc:<15.4f}"
        else:
            avg_row += f"{'N/A':<15}"

    print(avg_row)
    print("="*120)

    # Print detailed statistics
    print("\n" + "="*120)
    print("DETAILED STATISTICS")
    print("="*120)

    for target, target_name in zip(targets, target_names):
        target_results = [r for r in results if r is not None and r['target'] == target]

        if target_results:
            accuracies = [r['accuracy'] for r in target_results]
            f1_scores = [r['f1'] for r in target_results]
            aucs = [r['auc'] for r in target_results]

            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            max_acc = np.max(accuracies)
            min_acc = np.min(accuracies)
            best_layer = target_results[accuracies.index(max_acc)]['layer_idx']

            avg_f1 = np.mean(f1_scores)
            avg_auc = np.mean(aucs)

            print(f"\n{target_name}:")
            print(f"  Accuracy: avg={avg_acc:.4f}, std={std_acc:.4f}, min={min_acc:.4f}, max={max_acc:.4f}")
            print(f"  Best layer: {best_layer} (acc={max_acc:.4f})")
            print(f"  Avg F1: {avg_f1:.4f}")
            print(f"  Avg AUC: {avg_auc:.4f}")
            print(f"  Layers evaluated: {len(target_results)}/{num_layers}")

    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained linear probes to a new dataset"
    )
    parser.add_argument("--probes-dir", type=Path, required=True,
                        help="Directory containing trained probe .pkl files")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to steering vectors NPZ file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for results JSON (optional)")

    args = parser.parse_args()

    print("\n" + "="*120)
    print("APPLY LINEAR PROBES TO NEW DATASET")
    print("="*120)
    print(f"Probes directory: {args.probes_dir}")
    print(f"Data: {args.data}")
    if args.output:
        print(f"Output: {args.output}")
    print("="*120)

    # Check probes directory exists
    if not args.probes_dir.exists():
        print(f"\nError: Probes directory not found: {args.probes_dir}")
        return 1

    # Load data
    step_acts, hash_acts, step_numbers, num_layers, hidden_dim = load_steering_data(args.data)

    # Define targets
    targets = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']

    # Apply probes
    print("\n" + "="*120)
    print("APPLYING PROBES")
    print("="*120 + "\n")

    all_results = []

    for target in targets:
        target_name = "Final Answer" if target == "hash" else f"Step {target.split('_')[1]}"
        print(f"\n[Target: {target_name}]")

        for layer_idx in tqdm(range(num_layers), desc=f"  Layers", ncols=100):
            # Construct probe filename
            pkl_path = args.probes_dir / f"{target}_layer_{layer_idx:02d}.pkl"

            if not pkl_path.exists():
                print(f"  Warning: Probe not found: {pkl_path}")
                continue

            # Load classifier
            clf, pca = load_classifier(pkl_path)

            # Apply probe
            result = apply_probe_to_layer(
                step_acts[layer_idx],
                hash_acts[layer_idx],
                step_numbers[layer_idx],
                target,
                layer_idx,
                clf,
                pca
            )

            if result is not None:
                all_results.append(result)

        # Print summary for this target
        target_results = [r for r in all_results if r['target'] == target]
        if len(target_results) > 0:
            avg_acc = np.mean([r['accuracy'] for r in target_results])
            avg_f1 = np.mean([r['f1'] for r in target_results])
            avg_auc = np.mean([r['auc'] for r in target_results])
            best_result = max(target_results, key=lambda r: r['accuracy'])

            print(f"\n  Summary for {target_name}:")
            print(f"    Layers evaluated: {len(target_results)}/{num_layers}")
            print(f"    Avg Accuracy: {avg_acc:.4f}")
            print(f"    Avg F1:       {avg_f1:.4f}")
            print(f"    Avg AUC:      {avg_auc:.4f}")
            print(f"    Best layer:   {best_result['layer_idx']} (acc={best_result['accuracy']:.4f})")

    # Print accuracy table
    print_accuracy_table(all_results, num_layers)

    # Save results if output path provided
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")

    print("\n" + "="*120)
    print("PROBE APPLICATION COMPLETE!")
    print("="*120 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
