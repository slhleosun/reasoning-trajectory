#!/usr/bin/env python
"""Variance across seeds for probes and correctness predictors.

Reruns the key experiments with multiple random seeds to report mean ± std.
No new forward passes needed — operates on pre-extracted NPZ activations.

Reports:
  1. Linear probe accuracy (per step, per layer) across seeds
  2. Correctness predictor AUC (per feature set, per layer) across seeds

Usage:
  python scripts/predictors/seed_variance.py \
      --npz-path output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
      --output-dir output/seed_variance \
      --seeds 42 123 456
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# Data loading (same as train_predictors.py / train_stepwise_probes.py)
# ======================================================================

def load_npz(npz_path: Path) -> Dict:
    print(f"Loading from: {npz_path}")
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
    print(f"  Layers: {result['num_layers']}, Hidden dim: {result['hidden_dim']}")
    return result


# ======================================================================
# 1. Linear Probe Variance
# ======================================================================

def prepare_probe_data(data, layer_idx, target_step):
    """Prepare binary classification data: target_step vs rest."""
    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    hash_acts = data['hash_activations'][layer_idx]

    if target_step == "hash":
        # hash vs all steps
        n_hash = len(hash_acts)
        n_step = len(step_acts)
        X = np.concatenate([hash_acts, step_acts], axis=0)
        y = np.concatenate([np.ones(n_hash), np.zeros(n_step)])
    else:
        # target_step vs all other steps + hash
        target = int(target_step)
        mask_target = step_nums == target
        mask_other = step_nums != target

        target_acts = step_acts[mask_target]
        other_acts = step_acts[mask_other]

        X = np.concatenate([target_acts, other_acts, hash_acts], axis=0)
        y = np.concatenate([
            np.ones(len(target_acts)),
            np.zeros(len(other_acts) + len(hash_acts))
        ])

    return X, y


def run_probe_variance(data, targets, layers, seeds, test_size=0.2):
    """Run linear probes across seeds and collect variance."""
    print("\n" + "=" * 70)
    print("LINEAR PROBE VARIANCE")
    print("=" * 70)

    results = {}  # target -> layer -> [aucs across seeds]

    for target in targets:
        results[target] = defaultdict(list)

        for layer_idx in layers:
            X, y = prepare_probe_data(data, layer_idx, target)

            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            if n_pos < 5 or n_neg < 5:
                continue

            for seed in seeds:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=seed, stratify=y
                    )
                    clf = LogisticRegression(
                        max_iter=2000, random_state=seed,
                        class_weight='balanced'
                    )
                    clf.fit(X_train, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_test))
                    results[target][layer_idx].append(acc)
                except Exception:
                    continue

    # Print summary table
    print(f"\n{'Target':<12s}", end="")
    print(f"{'Avg Acc':>10s} {'Std':>8s} {'Best Layer':>12s} {'Best Acc':>10s} {'Best Std':>9s}")
    print("-" * 70)

    summary = {}
    for target in targets:
        if not results[target]:
            continue

        # Average across layers, then find best
        layer_means = {}
        layer_stds = {}
        for layer_idx in results[target]:
            vals = results[target][layer_idx]
            if vals:
                layer_means[layer_idx] = np.mean(vals)
                layer_stds[layer_idx] = np.std(vals)

        if not layer_means:
            continue

        avg_across_layers = np.mean(list(layer_means.values()))
        best_layer = max(layer_means, key=layer_means.get)
        best_mean = layer_means[best_layer]
        best_std = layer_stds[best_layer]

        label = f"Step {target}" if target != "hash" else "Ans. Marker"
        print(f"{label:<12s}{avg_across_layers:>10.4f}{'':>8s} L{best_layer:>2d}{'':>8s}"
              f"{best_mean:>10.4f} ±{best_std:>7.4f}")

        summary[str(target)] = {
            "avg_across_layers": float(avg_across_layers),
            "best_layer": int(best_layer),
            "best_mean": float(best_mean),
            "best_std": float(best_std),
        }

    return summary


# ======================================================================
# 2. Correctness Predictor Variance
# ======================================================================

def extract_hash(data, layer_idx):
    return (data['hash_activations'][layer_idx],
            data['question_ids_hash'][layer_idx],
            data['is_correct_hash'][layer_idx])


def extract_last_step(data, layer_idx):
    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    qids = data['question_ids_step'][layer_idx]
    correct = data['is_correct_step'][layer_idx]

    unique_qids = np.unique(qids)
    out = {"acts": [], "qids": [], "correct": []}
    for q in unique_qids:
        mask = qids == q
        max_idx = np.argmax(step_nums[mask])
        out["acts"].append(step_acts[mask][max_idx])
        out["qids"].append(q)
        out["correct"].append(correct[mask][max_idx])
    return np.array(out["acts"]), np.array(out["qids"]), np.array(out["correct"])


def extract_second_last_step(data, layer_idx):
    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    qids = data['question_ids_step'][layer_idx]
    correct = data['is_correct_step'][layer_idx]

    unique_qids = np.unique(qids)
    out = {"acts": [], "qids": [], "correct": []}
    for q in unique_qids:
        mask = qids == q
        q_steps = step_nums[mask]
        if len(q_steps) < 2:
            continue
        sorted_idx = np.argsort(q_steps)
        idx_2nd = sorted_idx[-2]
        out["acts"].append(step_acts[mask][idx_2nd])
        out["qids"].append(q)
        out["correct"].append(correct[mask][idx_2nd])
    return np.array(out["acts"]), np.array(out["qids"]), np.array(out["correct"])


def build_features_for_predictor(data, layer_idx, feature_set):
    """Build features matching train_predictors.py feature sets."""
    h, h_q, h_c = extract_hash(data, layer_idx)
    l, l_q, l_c = extract_last_step(data, layer_idx)

    if feature_set == "hash_only":
        return h, h_q, h_c

    # Need last step aligned with hash
    common = np.intersect1d(h_q, l_q)
    if len(common) == 0:
        return np.array([]), np.array([]), np.array([])

    h_idx = np.array([np.where(h_q == q)[0][0] for q in common])
    l_idx = np.array([np.where(l_q == q)[0][0] for q in common])

    h_a = h[h_idx]
    l_a = l[l_idx]
    c = h_c[h_idx]

    if feature_set == "hash_minus_last":
        return h_a - l_a, common, c

    if feature_set in ("hash_pca", "hash_last_diffs_pca", "hash_last_diffs_pca_joint"):
        s, s_q, s_c = extract_second_last_step(data, layer_idx)
        common3 = np.intersect1d(common, s_q)
        if len(common3) == 0:
            return np.array([]), np.array([]), np.array([])

        h_idx3 = np.array([np.where(h_q == q)[0][0] for q in common3])
        l_idx3 = np.array([np.where(l_q == q)[0][0] for q in common3])
        s_idx3 = np.array([np.where(s_q == q)[0][0] for q in common3])

        h3 = h[h_idx3]
        l3 = l[l_idx3]
        s3 = s[s_idx3]
        c3 = h_c[h_idx3]

        if feature_set == "hash_pca":
            return np.concatenate([h3, h3 - l3], axis=1), common3, c3
        elif feature_set == "hash_last_diffs_pca_joint":
            return np.concatenate([h3, h3 - l3, l3 - s3], axis=1), common3, c3
        elif feature_set == "hash_last_diffs_pca":
            return np.concatenate([h3, h3 - l3, l3 - s3], axis=1), common3, c3

    raise ValueError(f"Unknown feature_set: {feature_set}")


def run_predictor_variance(data, feature_sets, layers, seeds, test_size=0.1):
    """Run correctness predictors across seeds."""
    print("\n" + "=" * 70)
    print("CORRECTNESS PREDICTOR VARIANCE")
    print("=" * 70)

    results = {}

    for fs in feature_sets:
        print(f"\n  Feature set: {fs}")
        results[fs] = defaultdict(list)

        for layer_idx in layers:
            X, qids, y = build_features_for_predictor(data, layer_idx, fs)
            if len(X) == 0 or isinstance(X, np.ndarray) and X.ndim < 2:
                continue

            y = y.astype(int)
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            if n_pos < 10 or n_neg < 10:
                continue

            for seed in seeds:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=seed, stratify=y
                    )

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    n_comp = min(128, X_train_s.shape[0] - 1, X_train_s.shape[1])
                    if n_comp >= 2:
                        pca = PCA(n_components=n_comp, random_state=seed)
                        X_train_s = pca.fit_transform(X_train_s)
                        X_test_s = pca.transform(X_test_s)

                    clf = LogisticRegression(
                        max_iter=2000, random_state=seed, C=1.0,
                        class_weight='balanced'
                    )
                    clf.fit(X_train_s, y_train)

                    y_proba = clf.predict_proba(X_test_s)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    results[fs][layer_idx].append(auc)
                except Exception:
                    continue

        # Summary for this feature set
        layer_means = {}
        layer_stds = {}
        for li in results[fs]:
            vals = results[fs][li]
            if vals:
                layer_means[li] = np.mean(vals)
                layer_stds[li] = np.std(vals)

        if layer_means:
            avg = np.mean(list(layer_means.values()))
            best_layer = max(layer_means, key=layer_means.get)
            print(f"    Avg AUC: {avg:.4f}, Best: L{best_layer} = "
                  f"{layer_means[best_layer]:.4f} ± {layer_stds[best_layer]:.4f}")

    # Final summary table
    print(f"\n{'Feature Set':<30s} {'Avg AUC':>10s} {'Best Layer':>12s} "
          f"{'Best AUC':>10s} {'± Std':>8s}")
    print("-" * 75)

    summary = {}
    for fs in feature_sets:
        layer_means = {}
        layer_stds = {}
        for li in results[fs]:
            vals = results[fs][li]
            if vals:
                layer_means[li] = np.mean(vals)
                layer_stds[li] = np.std(vals)

        if not layer_means:
            continue

        avg = np.mean(list(layer_means.values()))
        best_layer = max(layer_means, key=layer_means.get)
        best_mean = layer_means[best_layer]
        best_std = layer_stds[best_layer]
        print(f"{fs:<30s} {avg:>10.4f} L{best_layer:>2d}{'':>8s}"
              f"{best_mean:>10.4f} ±{best_std:>6.4f}")

        summary[fs] = {
            "avg_auc": float(avg),
            "best_layer": int(best_layer),
            "best_auc_mean": float(best_mean),
            "best_auc_std": float(best_std),
        }

    return summary


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed variance for probes and predictors")
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/seed_variance"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--probe-layers", type=int, nargs="+", default=None,
                        help="Layers for probes (default: all)")
    parser.add_argument("--predictor-layers", type=int, nargs="+", default=None,
                        help="Layers for predictors (default: all)")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-predictors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(args.npz_path)

    num_layers = data['num_layers']
    probe_layers = args.probe_layers or list(range(num_layers))
    pred_layers = args.predictor_layers or list(range(num_layers))

    all_results = {"seeds": args.seeds}

    if not args.skip_probes:
        targets = [1, 2, 3, 4, 5, "hash"]
        all_results["probes"] = run_probe_variance(
            data, targets, probe_layers, args.seeds
        )

    if not args.skip_predictors:
        feature_sets = [
            "hash_only", "hash_minus_last",
            "hash_pca", "hash_last_diffs_pca_joint"
        ]
        all_results["predictors"] = run_predictor_variance(
            data, feature_sets, pred_layers, args.seeds
        )

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    out_path = args.output_dir / "seed_variance_results.json"
    with open(out_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()