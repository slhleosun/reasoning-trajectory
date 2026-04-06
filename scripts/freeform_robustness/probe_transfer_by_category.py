#!/usr/bin/env python
"""Per-category decomposition of probe transfer accuracy.

Breaks down the aggregate freeform probe transfer results by structure
category (step_x, numbered_list, double_newline, single_newline, single_block)
and computes a "non_step_x" aggregate (all categories except step_x).

The logic is identical to apply_trained_to_freeform.py's probe evaluation,
except activations are filtered by structure_types_step/hash before
constructing the positive/negative sets.

Usage:
  python scripts/freeform_robustness/probe_transfer_by_category.py \
      --freeform output/freeform/freeform_structure_activations.npz \
      --probes-dir output/stepwise_probes/inst_8000/classifiers \
      --output-dir output/freeform/per_category_probes
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

STRUCTURE_CODES = {
    "step_x": 0, "numbered_list": 1, "double_newline": 2,
    "single_newline": 3, "single_block": 4,
}
STRUCTURE_NAMES = {v: k for k, v in STRUCTURE_CODES.items()}


# ======================================================================
# Data loading (extended from apply_trained_to_freeform.py)
# ======================================================================

def load_freeform_npz(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    num_layers = int(raw["num_layers"])
    hidden_dim = int(raw["hidden_dim"])

    def _unpack(key, dtype=None):
        arr = raw[key]
        if arr.dtype == object:
            out = [arr[i] for i in range(len(arr))]
            if dtype is not None:
                out = [np.asarray(x, dtype=dtype) for x in out]
            return out
        if arr.ndim >= 2 and len(arr) == num_layers:
            out = [arr[i] for i in range(num_layers)]
            if dtype is not None:
                out = [np.asarray(x, dtype=dtype) for x in out]
            return out
        return [arr]

    d = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "step_activations":    _unpack("step_activations", np.float32),
        "hash_activations":    _unpack("hash_activations", np.float32),
        "step_numbers":        _unpack("step_numbers", np.int32),
        "question_ids_step":   _unpack("question_ids_step", np.int64),
        "question_ids_hash":   _unpack("question_ids_hash", np.int64),
        "is_correct_step":     _unpack("is_correct_step", np.bool_),
        "is_correct_hash":     _unpack("is_correct_hash", np.bool_),
        "structure_types_step": _unpack("structure_types_step", np.int32)
            if "structure_types_step" in raw else None,
        "structure_types_hash": _unpack("structure_types_hash", np.int32)
            if "structure_types_hash" in raw else None,
    }

    n_step = len(d["step_activations"][0])
    n_hash = len(d["hash_activations"][0])
    print(f"Loaded: {num_layers} layers, {n_step} step acts, {n_hash} hash acts")

    if d["structure_types_step"] is not None:
        u, c = np.unique(d["structure_types_step"][0], return_counts=True)
        print(f"Step structure dist: {dict(zip([STRUCTURE_NAMES[int(x)] for x in u], c.tolist()))}")
    if d["structure_types_hash"] is not None:
        u, c = np.unique(d["structure_types_hash"][0], return_counts=True)
        print(f"Hash structure dist: {dict(zip([STRUCTURE_NAMES[int(x)] for x in u], c.tolist()))}")

    return d


def make_offset_data(data: dict) -> dict:
    """Shift step numbers by -1, drop step_number=1 (prompt-end)."""
    d = deepcopy(data)
    for li in range(d["num_layers"]):
        sn = d["step_numbers"][li]
        mask = sn >= 2
        d["step_activations"][li] = d["step_activations"][li][mask]
        d["step_numbers"][li] = sn[mask] - 1
        d["question_ids_step"][li] = d["question_ids_step"][li][mask]
        d["is_correct_step"][li] = d["is_correct_step"][li][mask]
        if d["structure_types_step"] is not None:
            d["structure_types_step"][li] = d["structure_types_step"][li][mask]
    return d


def filter_by_category(data: dict, cat_codes: list) -> dict:
    """Return a copy of data with only activations from the given structure codes.

    cat_codes: list of int structure codes to keep (e.g. [0] for step_x,
    or [1,2,3,4] for non_step_x).
    """
    d = deepcopy(data)
    cat_set = set(cat_codes)

    for li in range(d["num_layers"]):
        # Filter step activations
        if d["structure_types_step"] is not None:
            st = d["structure_types_step"][li]
            mask = np.isin(st, list(cat_set))
            d["step_activations"][li] = d["step_activations"][li][mask]
            d["step_numbers"][li] = d["step_numbers"][li][mask]
            d["question_ids_step"][li] = d["question_ids_step"][li][mask]
            d["is_correct_step"][li] = d["is_correct_step"][li][mask]
            d["structure_types_step"][li] = st[mask]

        # Filter hash activations
        if d["structure_types_hash"] is not None:
            sh = d["structure_types_hash"][li]
            mask = np.isin(sh, list(cat_set))
            d["hash_activations"][li] = d["hash_activations"][li][mask]
            d["question_ids_hash"][li] = d["question_ids_hash"][li][mask]
            d["is_correct_hash"][li] = d["is_correct_hash"][li][mask]
            d["structure_types_hash"][li] = sh[mask]

    n_step = len(d["step_activations"][0])
    n_hash = len(d["hash_activations"][0])
    return d, n_step, n_hash


# ======================================================================
# Probe evaluation (same logic as apply_trained_to_freeform.py)
# ======================================================================

def evaluate_probes(data: dict, probes_dir: Path, min_samples: int = 5) -> list:
    """Apply trained probes to data, return list of result dicts."""
    num_layers = data["num_layers"]
    targets = ["step_1", "step_2", "step_3", "step_4", "step_5", "hash"]
    results = []

    for target in targets:
        for layer_idx in range(num_layers):
            pkl_path = probes_dir / f"{target}_layer_{layer_idx:02d}.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                probe_data = pickle.load(f)
            clf, pca = probe_data["clf"], probe_data["pca"]

            step_acts = data["step_activations"][layer_idx]
            hash_acts = data["hash_activations"][layer_idx]
            step_numbers = data["step_numbers"][layer_idx]

            if target == "hash":
                X_pos = hash_acts
                X_neg = step_acts
            else:
                step_num = int(target.split("_")[1])
                mask = step_numbers == step_num
                X_pos = step_acts[mask]
                X_neg_steps = step_acts[~mask]
                X_neg = (np.vstack([X_neg_steps, hash_acts])
                         if len(X_neg_steps) > 0 else hash_acts)

            if len(X_pos) < min_samples or len(X_neg) < min_samples:
                continue

            X = np.vstack([X_pos, X_neg])
            y = np.array([1] * len(X_pos) + [0] * len(X_neg))

            if pca is not None:
                X = pca.transform(X)

            try:
                y_pred = clf.predict(X)
                y_proba = clf.predict_proba(X)[:, 1]
                results.append({
                    "target": target,
                    "layer_idx": layer_idx,
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "f1": float(f1_score(y, y_pred, zero_division=0)),
                    "auc": float(roc_auc_score(y, y_proba)),
                    "n_pos": int(len(X_pos)),
                    "n_neg": int(len(X_neg)),
                })
            except Exception:
                pass

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeform", type=Path, required=True,
                        help="Freeform activation NPZ")
    parser.add_argument("--probes-dir", type=Path, required=True,
                        help="Directory with trained probe .pkl files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/freeform/per_category_probes"))
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum positive samples to evaluate a cell")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and apply offset alignment
    data = load_freeform_npz(args.freeform)
    data = make_offset_data(data)

    if data["structure_types_step"] is None:
        print("ERROR: NPZ does not contain structure_types_step")
        return 1

    # Define categories to evaluate
    categories = {
        "all":            list(range(5)),           # aggregate (reproduce original)
        "step_x":         [0],
        "numbered_list":  [1],
        "double_newline": [2],
        "single_newline": [3],
        "single_block":   [4],
        "non_step_x":     [1, 2, 3, 4],            # everything except step_x
    }

    all_summaries = {}
    targets = ["step_1", "step_2", "step_3", "step_4", "step_5", "hash"]
    target_names = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Answer"]

    for cat_name, cat_codes in categories.items():
        print(f"\n{'='*80}")
        print(f"Category: {cat_name}  (codes={cat_codes})")
        print(f"{'='*80}")

        filtered, n_step, n_hash = filter_by_category(data, cat_codes)
        print(f"  Filtered: {n_step} step acts, {n_hash} hash acts")

        if n_step < args.min_samples:
            print(f"  SKIP: too few samples ({n_step} < {args.min_samples})")
            continue

        results = evaluate_probes(filtered, args.probes_dir,
                                  min_samples=args.min_samples)

        # Compute avg/best per target
        summary = {}
        for t, tname in zip(targets, target_names):
            t_results = [r for r in results if r["target"] == t]
            if not t_results:
                summary[t] = {"avg_acc": None, "best_acc": None,
                              "best_layer": None, "n_pos": 0}
                continue
            avg_acc = np.mean([r["accuracy"] for r in t_results])
            best = max(t_results, key=lambda r: r["accuracy"])
            summary[t] = {
                "avg_acc": float(avg_acc),
                "best_acc": float(best["accuracy"]),
                "best_layer": int(best["layer_idx"]),
                "n_pos": int(best["n_pos"]),
            }
            print(f"  {tname:>10s}: avg={avg_acc:.3f}, "
                  f"best=L{best['layer_idx']} ({best['accuracy']:.3f}), "
                  f"n_pos={best['n_pos']}")

        all_summaries[cat_name] = {"summary": summary, "full": results}

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n\n{'='*110}")
    print("PER-CATEGORY PROBE TRANSFER ACCURACY (best-layer)")
    print(f"{'='*110}")

    header = f"{'Category':<18s}"
    for tn in target_names:
        header += f"{tn:>14s}"
    print(header)
    print("-" * 110)

    for cat_name in categories:
        if cat_name not in all_summaries:
            continue
        s = all_summaries[cat_name]["summary"]
        row = f"{cat_name:<18s}"
        for t in targets:
            if s[t]["best_acc"] is not None:
                row += f"  {s[t]['best_acc']:.3f} (L{s[t]['best_layer']:>2d})"
            else:
                row += f"{'—':>14s}"
        print(row)

    print("-" * 110)

    # ── Print avg-accuracy table ──────────────────────────────────────
    print(f"\nPER-CATEGORY PROBE TRANSFER ACCURACY (avg across layers)")
    print(f"{'='*110}")
    header = f"{'Category':<18s}"
    for tn in target_names:
        header += f"{tn:>14s}"
    header += f"{'n_questions':>14s}"
    print(header)
    print("-" * 110)

    for cat_name in categories:
        if cat_name not in all_summaries:
            continue
        s = all_summaries[cat_name]["summary"]
        row = f"{cat_name:<18s}"
        for t in targets:
            if s[t]["avg_acc"] is not None:
                row += f"  {s[t]['avg_acc']:.3f}      "
            else:
                row += f"{'—':>14s}"
        # Approximate n_questions from hash n_pos (one per question)
        nh = s["hash"]["n_pos"] if s["hash"]["n_pos"] else 0
        row += f"{nh:>14d}"
        print(row)

    print("=" * 110)

    # ── Save full results ─────────────────────────────────────────────
    # Convert to serializable format
    save_data = {}
    for cat_name, cat_data in all_summaries.items():
        save_data[cat_name] = {
            "summary": cat_data["summary"],
            "full": cat_data["full"],
        }

    out_path = args.output_dir / "per_category_probe_transfer.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n✓ Saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())