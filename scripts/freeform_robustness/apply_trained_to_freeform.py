#!/usr/bin/env python
"""Apply trained predictors and probes to freeform activations.

Two transfer evaluation modes:

1. STEPWISE PROBES (binary: "is this step N?" / "is this the answer?")
   - Loads .pkl files from probes-dir
   - Tests on freeform activations
   - Direct reuse of apply_linear_probes.py logic

2. CORRECTNESS PREDICTORS (binary: correct vs incorrect)
   - Loads .npz files from predictors-dir
   - Reconstructs features (step1_step2, hash_only, etc.) from freeform NPZ
   - Applies saved logistic regression / MLP weights

Both run in TWO alignment modes:
  raw:    freeform step numbers as-is (step 1 = prompt-end)
  offset: shift freeform step numbers by -1 (step 1 = first marker ≈ original Step 1)

Semantic mapping:
  Original:  step_1 = "Step 1:" marker,  step_2 = "Step 2:", ...  hash = ####
  Freeform raw:   step_1 = prompt-end,    step_2 = first marker, ...  hash = reasoning end
  Freeform offset: step_1 = first marker, step_2 = second marker, ... hash = reasoning end
                   (prompt-end activation is dropped)

Usage
-----
# Full transfer evaluation
python scripts/freeform_robustness/apply_trained_to_freeform.py \
    --freeform output/freeform/freeform_structure_activations.npz \
    --probes-dir output/stepwise_probes/inst_8000/classifiers \
    --predictors-dir output/predictors/act_based_linear \
    --output-dir output/freeform/transfer_probes_predictors

python scripts/freeform_robustness/apply_trained_to_freeform.py \
    --freeform output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_math500.npz \
    --predictors-dir output/predictors/act_based_linear \
    --output-dir output/math500_predictor_transfer

# Probes only
python scripts/freeform_robustness/apply_trained_to_freeform.py \\
    --freeform output/freeform/freeform_structure_activations.npz \\
    --probes-dir output/stepwise_probes/inst_8000/classifiers \\
    --output-dir output/freeform/transfer_evaluation

# Predictors only
python scripts/freeform_robustness/apply_trained_to_freeform.py \
    --freeform output/freeform/freeform_structure_activations.npz \
    --predictors-dir output/predictors/act_based_linear \
    --output-dir output/freeform/transfer_predictors
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# ── Add project root ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

STRUCTURE_NAMES = {0: "step_x", 1: "numbered_list", 2: "double_newline",
                   3: "single_newline", 4: "single_block"}


# ======================================================================
# Data loading
# ======================================================================

def load_freeform_npz(path: Path) -> Dict:
    """Load freeform activation NPZ into a dict matching the format
    expected by train_predictors.py's helper functions."""
    print(f"\nLoading freeform NPZ: {path}")
    raw = np.load(path, allow_pickle=True)

    num_layers = int(raw["num_layers"])
    hidden_dim = int(raw["hidden_dim"])

    def _unpack(key, inner_dtype=None):
        arr = raw[key]
        if arr.dtype == object:
            out = [arr[i] for i in range(len(arr))]
            # Object arrays from NPZ lose inner dtype — cast back
            if inner_dtype is not None:
                out = [np.asarray(x, dtype=inner_dtype) for x in out]
            return out
        # Non-object array: if first dim matches num_layers, split by layer
        # This handles NPZs where per-layer arrays are stored as regular
        # ndarrays with shape [num_layers, N_per_layer, ...] instead of
        # object arrays.
        if arr.ndim >= 2 and len(arr) == num_layers:
            out = [arr[i] for i in range(num_layers)]
            if inner_dtype is not None:
                out = [np.asarray(x, dtype=inner_dtype) for x in out]
            return out
        return [arr]

    d = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "step_activations": _unpack("step_activations", np.float32),
        "hash_activations": _unpack("hash_activations", np.float32),
        "step_numbers":     _unpack("step_numbers", np.int32),
        "question_ids_step": _unpack("question_ids_step", np.int64),
        "question_ids_hash": _unpack("question_ids_hash", np.int64),
        "is_correct_step":   _unpack("is_correct_step", np.bool_),
        "is_correct_hash":   _unpack("is_correct_hash", np.bool_),
    }

    if "structure_types_step" in raw:
        d["structure_types_step"] = _unpack("structure_types_step", np.int32)
    else:
        d["structure_types_step"] = None

    n_step = len(d["step_activations"][0]) if len(d["step_activations"][0]) > 0 else 0
    n_hash = len(d["hash_activations"][0]) if len(d["hash_activations"][0]) > 0 else 0
    n_q = len(np.unique(d["question_ids_step"][0])) if n_step > 0 else 0
    print(f"  {num_layers} layers, hidden_dim={hidden_dim}")
    print(f"  {n_step} step acts, {n_hash} hash acts, {n_q} questions")

    if n_step > 0:
        u, c = np.unique(d["step_numbers"][0], return_counts=True)
        print(f"  Step distribution (layer 0): {dict(zip(u.tolist(), c.tolist()))}")

    return d


def make_offset_data(data: Dict) -> Dict:
    """Create offset version: shift step numbers by -1, drop step_number=1 (prompt-end).

    This makes freeform step 1 = first structure marker ≈ original Step 1.
    """
    d = deepcopy(data)
    num_layers = d["num_layers"]

    for li in range(num_layers):
        sn = d["step_numbers"][li]
        # Keep only steps with step_number >= 2, then subtract 1
        mask = sn >= 2
        d["step_activations"][li] = d["step_activations"][li][mask]
        d["step_numbers"][li] = sn[mask] - 1  # 2→1, 3→2, etc.
        d["question_ids_step"][li] = d["question_ids_step"][li][mask]
        d["is_correct_step"][li] = d["is_correct_step"][li][mask]
        if d.get("structure_types_step"):
            d["structure_types_step"][li] = d["structure_types_step"][li][mask]

    n_step = len(d["step_activations"][0])
    print(f"  Offset data: {n_step} step acts (after dropping prompt-end)")
    if n_step > 0:
        u, c = np.unique(d["step_numbers"][0], return_counts=True)
        print(f"  Step distribution (layer 0): {dict(zip(u.tolist(), c.tolist()))}")
    return d


# ======================================================================
# Part 1: Stepwise Probes
# ======================================================================

def apply_stepwise_probes(
    data: Dict,
    probes_dir: Path,
    alignment_label: str,
) -> List[Dict]:
    """Apply trained stepwise probes (step_1..step_5, hash) to the data.

    Exactly mirrors apply_linear_probes.py logic.
    """
    num_layers = data["num_layers"]
    targets = ["step_1", "step_2", "step_3", "step_4", "step_5", "hash"]
    all_results = []

    for target in targets:
        target_name = "Final Answer" if target == "hash" else f"Step {target.split('_')[1]}"
        print(f"\n  [{alignment_label}] Target: {target_name}")

        for layer_idx in tqdm(range(num_layers), desc=f"    Layers", ncols=100, leave=False):
            pkl_path = probes_dir / f"{target}_layer_{layer_idx:02d}.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                probe_data = pickle.load(f)
            clf, pca = probe_data["clf"], probe_data["pca"]

            # Prepare dataset (same as apply_linear_probes.py)
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
                X_neg = np.vstack([X_neg_steps, hash_acts]) if len(X_neg_steps) > 0 else hash_acts

            if len(X_pos) == 0:
                continue

            X = np.vstack([X_pos, X_neg])
            y = np.array([1] * len(X_pos) + [0] * len(X_neg))

            if pca is not None:
                X = pca.transform(X)

            try:
                y_pred = clf.predict(X)
                y_proba = clf.predict_proba(X)[:, 1]
                acc = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, zero_division=0)
                auc = roc_auc_score(y, y_proba)

                all_results.append({
                    "alignment": alignment_label,
                    "target": target,
                    "layer_idx": layer_idx,
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "auc": float(auc),
                    "n_positive": int(len(X_pos)),
                    "n_negative": int(len(X_neg)),
                    "n_total": int(len(X)),
                })
            except Exception as e:
                pass  # silently skip failures

        # Summary for this target
        target_results = [r for r in all_results
                          if r["target"] == target and r["alignment"] == alignment_label]
        if target_results:
            avg_acc = np.mean([r["accuracy"] for r in target_results])
            best = max(target_results, key=lambda r: r["accuracy"])
            print(f"    Avg acc={avg_acc:.4f}, best=L{best['layer_idx']} ({best['accuracy']:.4f})")

    return all_results


def print_probe_table(results: List[Dict], alignment: str, num_layers: int):
    """Print accuracy table for probes."""
    targets = ["step_1", "step_2", "step_3", "step_4", "step_5", "hash"]
    names = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Answer"]

    # Organize
    rd = {}
    for r in results:
        if r["alignment"] != alignment:
            continue
        rd.setdefault(r["layer_idx"], {})[r["target"]] = r

    print(f"\n{'='*110}")
    print(f"STEPWISE PROBE ACCURACY TABLE (alignment={alignment})")
    print(f"{'='*110}")
    header = f"{'Layer':<7}"
    for n in names:
        header += f"{n:<16}"
    print(header)
    print("-" * 110)

    for li in range(num_layers):
        row = f"{li:<7}"
        for t in targets:
            if li in rd and t in rd[li]:
                row += f"{rd[li][t]['accuracy']:<16.4f}"
            else:
                row += f"{'—':<16}"
        print(row)

    # Average row
    print("-" * 110)
    avg_row = f"{'AVG':<7}"
    for t in targets:
        vals = [r["accuracy"] for r in results
                if r["target"] == t and r["alignment"] == alignment]
        if vals:
            avg_row += f"{np.mean(vals):<16.4f}"
        else:
            avg_row += f"{'—':<16}"
    print(avg_row)

    # Best row
    best_row = f"{'BEST':<7}"
    for t in targets:
        vals = [r for r in results
                if r["target"] == t and r["alignment"] == alignment]
        if vals:
            best = max(vals, key=lambda r: r["accuracy"])
            best_row += f"L{best['layer_idx']:>2} {best['accuracy']:.4f}   "
        else:
            best_row += f"{'—':<16}"
    print(best_row)
    print("=" * 110)


# ======================================================================
# Part 2: Correctness Predictors
# ======================================================================

def extract_step_activations(data, layer_idx, step_number):
    """Extract activations for a specific step (matches train_predictors.py)."""
    sa = data["step_activations"][layer_idx]
    sn = data["step_numbers"][layer_idx]
    qi = data["question_ids_step"][layer_idx]
    ic = data["is_correct_step"][layer_idx]
    mask = (sn == step_number)
    return sa[mask], qi[mask], ic[mask]


def extract_hash_activations(data, layer_idx):
    return (data["hash_activations"][layer_idx],
            data["question_ids_hash"][layer_idx],
            data["is_correct_hash"][layer_idx])


def extract_last_step_activations(data, layer_idx):
    sa = data["step_activations"][layer_idx]
    sn = data["step_numbers"][layer_idx]
    qi = data["question_ids_step"][layer_idx]
    ic = data["is_correct_step"][layer_idx]
    unique_qids = np.unique(qi)
    acts, qids, corrs, snums = [], [], [], []
    for qid in unique_qids:
        mask = qi == qid
        idx = np.argmax(sn[mask])
        acts.append(sa[mask][idx])
        qids.append(qid)
        corrs.append(ic[mask][idx])
        snums.append(sn[mask][idx])
    return np.array(acts), np.array(qids), np.array(corrs), np.array(snums)


def extract_second_to_last_step_activations(data, layer_idx):
    sa = data["step_activations"][layer_idx]
    sn = data["step_numbers"][layer_idx]
    qi = data["question_ids_step"][layer_idx]
    ic = data["is_correct_step"][layer_idx]
    unique_qids = np.unique(qi)
    acts, qids, corrs = [], [], []
    for qid in unique_qids:
        mask = qi == qid
        q_sn = sn[mask]
        if len(q_sn) < 2:
            continue
        sorted_idx = np.argsort(q_sn)
        idx = sorted_idx[-2]
        acts.append(sa[mask][idx])
        qids.append(qid)
        corrs.append(ic[mask][idx])
    return np.array(acts), np.array(qids), np.array(corrs)


def build_features(data, layer_idx, feature_set):
    """Build feature set from activations (matches train_predictors.py create_feature_set)."""

    def _align(*groups):
        """Intersect question IDs across groups and align."""
        common = groups[0][1]  # qids of first group
        for _, qids, _ in groups[1:]:
            common = np.intersect1d(common, qids)
        if len(common) == 0:
            return None
        aligned = []
        for acts, qids, corr in groups:
            idx = np.array([np.where(qids == q)[0][0] for q in common])
            aligned.append(acts[idx])
        return aligned, common, groups[0][2][np.array([np.where(groups[0][1] == q)[0][0] for q in common])]

    if feature_set == "step1_step2":
        s1 = extract_step_activations(data, layer_idx, 1)
        s2 = extract_step_activations(data, layer_idx, 2)
        if len(s1[0]) == 0 or len(s2[0]) == 0:
            return None, None, None
        r = _align(s1, s2)
        if r is None: return None, None, None
        aligned, qids, corr = r
        return np.concatenate(aligned, axis=1), qids, corr

    elif feature_set == "step2_minus_step1":
        s1 = extract_step_activations(data, layer_idx, 1)
        s2 = extract_step_activations(data, layer_idx, 2)
        if len(s1[0]) == 0 or len(s2[0]) == 0:
            return None, None, None
        r = _align(s1, s2)
        if r is None: return None, None, None
        aligned, qids, corr = r
        return aligned[1] - aligned[0], qids, corr

    elif feature_set == "step1_step2_step3":
        s1 = extract_step_activations(data, layer_idx, 1)
        s2 = extract_step_activations(data, layer_idx, 2)
        s3 = extract_step_activations(data, layer_idx, 3)
        if len(s1[0]) == 0 or len(s2[0]) == 0 or len(s3[0]) == 0:
            return None, None, None
        r = _align(s1, s2, s3)
        if r is None: return None, None, None
        aligned, qids, corr = r
        return np.concatenate(aligned, axis=1), qids, corr

    elif feature_set == "step_diffs":
        s1 = extract_step_activations(data, layer_idx, 1)
        s2 = extract_step_activations(data, layer_idx, 2)
        s3 = extract_step_activations(data, layer_idx, 3)
        h = extract_hash_activations(data, layer_idx)
        last = extract_last_step_activations(data, layer_idx)
        if any(len(x[0]) == 0 for x in [s1, s2, s3, h, (last[0], last[1], last[2])]):
            return None, None, None
        r = _align(s1, s2, s3, h, (last[0], last[1], last[2]))
        if r is None: return None, None, None
        aligned, qids, corr = r
        d1 = aligned[1] - aligned[0]
        d2 = aligned[2] - aligned[1]
        d3 = aligned[3] - aligned[4]  # hash - last
        return np.concatenate([d1, d2, d3], axis=1), qids, corr

    elif feature_set == "hash_only":
        h = extract_hash_activations(data, layer_idx)
        return h[0], h[1], h[2]

    elif feature_set == "hash_minus_last":
        h = extract_hash_activations(data, layer_idx)
        last = extract_last_step_activations(data, layer_idx)
        if len(last[0]) == 0:
            return None, None, None
        r = _align(h, (last[0], last[1], last[2]))
        if r is None: return None, None, None
        aligned, qids, corr = r
        return aligned[0] - aligned[1], qids, corr

    elif feature_set == "hash_pca":
        # Hash + (hash - last), PCA'd — need PCA from predictor
        return "needs_pca", None, None

    elif feature_set in ("hash_last_diffs_pca", "hash_last_diffs_pca_joint"):
        return "needs_pca", None, None

    else:
        return None, None, None


def predict_probabilities(features, predictor):
    """Matches train_predictors.py predict_probabilities."""
    X_scaled = (features - predictor["scaler_mean"]) / predictor["scaler_std"]

    if "mlp_fc1_weight" in predictor:
        hidden = X_scaled @ predictor["mlp_fc1_weight"].T + predictor["mlp_fc1_bias"]
        hidden = np.maximum(0, hidden)
        logits = hidden @ predictor["mlp_fc2_weight"].T + predictor["mlp_fc2_bias"]
        logits = logits.squeeze(-1)
    else:
        logits = X_scaled @ predictor["coefficients"] + predictor["intercept"]

    return 1 / (1 + np.exp(-logits))


def apply_correctness_predictors(
    data: Dict,
    predictors_dir: Path,
    alignment_label: str,
) -> List[Dict]:
    """Apply trained correctness predictors to freeform data."""

    num_layers = data["num_layers"]

    # Discover available predictor NPZ files
    npz_files = sorted(predictors_dir.glob("*.npz"))
    if not npz_files:
        print(f"  No predictor NPZ files found in {predictors_dir}")
        return []

    # Parse filenames to get (feature_set, label_type, layer_idx, model_type)
    # Two naming conventions exist:
    #   train_predictors.py:  "{feature_set}_{label_type}_layer{NN:02d}.npz"
    #                         e.g. hash_pca_correctness_layer15.npz
    #   older convention:     "{feature_set}_{label_type}_layer_{NN}_{model_type}.npz"
    #                         e.g. hash_only_correctness_layer_15_linear.npz
    import re
    LAYER_RE = re.compile(r"_layer_?(\d+)(?:_(\w+))?$")

    predictor_map = {}  # (feature_set, label_type, layer_idx, model_type) → path
    for p in npz_files:
        name = p.stem
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        model_type = m.group(2) if m.group(2) else "linear"
        prefix = name[:m.start()]  # everything before _layerNN

        # Extract feature_set and label from prefix
        # Known feature sets (order matters for longest-match)
        known_fs = ["step1_step2_step3", "step1_step2", "step2_minus_step1",
                     "step_diffs", "hash_only", "hash_minus_last",
                     "hash_last_diffs_pca_joint",
                     "hash_last_diffs_pca",
                     "hash_pca"]
        feature_set = None
        for fs in known_fs:
            if prefix.startswith(fs + "_"):
                feature_set = fs
                label_type = prefix[len(fs) + 1:]
                break
        if feature_set is None:
            continue

        predictor_map[(feature_set, label_type, layer_idx, model_type)] = p

    print(f"\n  Found {len(predictor_map)} predictor files")
    # Group by feature_set
    fs_counts = defaultdict(int)
    for (fs, lt, li, mt), _ in predictor_map.items():
        fs_counts[f"{fs}_{lt}_{mt}"] += 1
    for k, v in sorted(fs_counts.items()):
        print(f"    {k}: {v} layers")

    # Feature sets that DON'T need PCA from the original training
    DIRECT_FEATURE_SETS = {"step1_step2", "step2_minus_step1", "step1_step2_step3",
                           "step_diffs", "hash_only", "hash_minus_last"}

    all_results = []

    for (fs, lt, li, mt), npz_path in tqdm(sorted(predictor_map.items()),
                                             desc=f"  [{alignment_label}] Predictors",
                                             ncols=120):
        if fs not in DIRECT_FEATURE_SETS:
            # PCA-based features need components from training
            pred_data = np.load(npz_path, allow_pickle=True)
            if "pca_components" not in pred_data:
                continue
            # pca_components shape: [n_pca_objects, n_components, raw_dim]
            # pca_mean shape:       [n_pca_objects, raw_dim]
            pca_components = pred_data["pca_components"]
            pca_mean = pred_data["pca_mean"]

            # Build raw component arrays (matching training's tuple format)
            h = extract_hash_activations(data, li)
            last = extract_last_step_activations(data, li)
            if len(h[0]) == 0 or len(last[0]) == 0:
                continue

            if fs == "hash_pca":
                # 2 components: (hash, hash − last)
                common = np.intersect1d(h[1], last[1])
                if len(common) == 0: continue
                h_idx = np.array([np.where(h[1] == q)[0][0] for q in common])
                l_idx = np.array([np.where(last[1] == q)[0][0] for q in common])
                components = [h[0][h_idx], h[0][h_idx] - last[0][l_idx]]
                labels = np.logical_not(h[2][h_idx])
                qids = common

            elif fs in ("hash_last_diffs_pca", "hash_last_diffs_pca_joint"):
                # Need second-to-last step too
                s2l = extract_second_to_last_step_activations(data, li)
                if len(s2l[0]) == 0: continue
                common = np.intersect1d(np.intersect1d(h[1], last[1]), s2l[1])
                if len(common) == 0: continue
                h_idx = np.array([np.where(h[1] == q)[0][0] for q in common])
                l_idx = np.array([np.where(last[1] == q)[0][0] for q in common])
                s_idx = np.array([np.where(s2l[1] == q)[0][0] for q in common])
                h_a = h[0][h_idx]; l_a = last[0][l_idx]; s_a = s2l[0][s_idx]
                d1 = h_a - l_a; d2 = l_a - s_a

                if fs == "hash_last_diffs_pca":
                    # 3 separate PCAs: (hash, hash−last, last−2ndlast)
                    components = [h_a, d1, d2]
                else:
                    # 1 joint PCA on concatenation
                    components = [np.concatenate([h_a, d1, d2], axis=1)]

                labels = np.logical_not(h[2][h_idx])
                qids = common
            else:
                continue

            # Apply PCA per-component (matching training), then concatenate
            pca_transformed = []
            for i, comp in enumerate(components):
                comp = np.asarray(comp, dtype=np.float32)
                # pca_components[i]: [n_components, raw_dim]
                # pca_mean[i]: [raw_dim]
                transformed = (comp - pca_mean[i]) @ pca_components[i].T
                pca_transformed.append(transformed)
            features = np.concatenate(pca_transformed, axis=1)
        else:
            # Direct feature sets
            features, qids, is_correct = build_features(data, li, fs)
            if features is None or isinstance(features, str):
                continue
            if len(features) == 0:
                continue
            # Labels: incorrect = 1, correct = 0
            labels = np.logical_not(is_correct)

        # Load predictor weights
        pred_data = np.load(npz_path, allow_pickle=True)
        predictor = {
            "scaler_mean": pred_data["scaler_mean"],
            "scaler_std": np.maximum(pred_data["scaler_std"], 1e-10),  # avoid div-by-zero
        }
        if "mlp_fc1_weight" in pred_data:
            predictor["mlp_fc1_weight"] = pred_data["mlp_fc1_weight"]
            predictor["mlp_fc1_bias"] = pred_data["mlp_fc1_bias"]
            predictor["mlp_fc2_weight"] = pred_data["mlp_fc2_weight"]
            predictor["mlp_fc2_bias"] = pred_data["mlp_fc2_bias"]
        else:
            predictor["coefficients"] = pred_data["coefficients"]
            predictor["intercept"] = pred_data["intercept"]

        threshold = float(pred_data.get("best_threshold", 0.5))

        # Check dimension match
        if "coefficients" in predictor:
            expected_dim = len(predictor["coefficients"])
        elif "mlp_fc1_weight" in predictor:
            expected_dim = predictor["mlp_fc1_weight"].shape[1]
        else:
            continue

        # Ensure float32 (object arrays from NPZ can cause silent failures)
        features = np.asarray(features, dtype=np.float32)

        if features.shape[1] != expected_dim:
            print(f"    ⚠ {fs} L{li}: dim mismatch feat={features.shape[1]} vs expected={expected_dim}")
            continue

        # Apply
        try:
            probs = predict_probabilities(features, predictor)
            preds_default = (probs >= 0.5).astype(int)
            preds_tuned = (probs >= threshold).astype(int)

            labels_int = labels.astype(int)

            acc_default = accuracy_score(labels_int, preds_default)
            f1_default = f1_score(labels_int, preds_default, zero_division=0)
            acc_tuned = accuracy_score(labels_int, preds_tuned)
            f1_tuned = f1_score(labels_int, preds_tuned, zero_division=0)
            auc = roc_auc_score(labels_int, probs) if len(np.unique(labels_int)) > 1 else 0.5

            # Original training performance (for comparison)
            orig_test_acc = float(pred_data.get("test_accuracy", 0))
            orig_test_f1 = float(pred_data.get("test_f1", 0))
            orig_test_auc = float(pred_data.get("test_roc_auc", 0))

            all_results.append({
                "alignment": alignment_label,
                "feature_set": fs,
                "label_type": lt,
                "layer_idx": li,
                "model_type": mt,
                "threshold": threshold,
                "n_samples": int(len(labels)),
                "n_incorrect": int(labels_int.sum()),
                "n_correct": int((1 - labels_int).sum()),
                "accuracy_default": float(acc_default),
                "f1_default": float(f1_default),
                "accuracy_tuned": float(acc_tuned),
                "f1_tuned": float(f1_tuned),
                "auc": float(auc),
                "orig_test_accuracy": orig_test_acc,
                "orig_test_f1": orig_test_f1,
                "orig_test_auc": orig_test_auc,
            })
        except Exception as e:
            print(f"    ⚠ {fs} L{li}: {e}")

    return all_results


def print_predictor_table(results: List[Dict], alignment: str):
    """Print summary table for correctness predictors."""

    # Group by feature_set
    by_fs = defaultdict(list)
    for r in results:
        if r["alignment"] != alignment:
            continue
        by_fs[f"{r['feature_set']}_{r['model_type']}"].append(r)

    print(f"\n{'='*130}")
    print(f"CORRECTNESS PREDICTOR TRANSFER TABLE (alignment={alignment})")
    print(f"{'='*130}")
    header = (f"{'Feature Set':<35} | {'Avg AUC':>8} | {'Avg Acc':>8} | "
              f"{'Avg F1':>8} | {'Best AUC':>9} | {'Best Layer':>11} | "
              f"{'Orig AUC':>9} | {'ΔAUC':>8} | {'N':>5}")
    print(header)
    print("-" * 130)

    for fs_key in sorted(by_fs.keys()):
        rs = by_fs[fs_key]
        avg_auc = np.mean([r["auc"] for r in rs])
        avg_acc = np.mean([r["accuracy_tuned"] for r in rs])
        avg_f1 = np.mean([r["f1_tuned"] for r in rs])
        best = max(rs, key=lambda r: r["auc"])
        orig_auc = np.mean([r["orig_test_auc"] for r in rs])
        delta = avg_auc - orig_auc

        print(f"{fs_key:<35} | {avg_auc:>8.4f} | {avg_acc:>8.4f} | "
              f"{avg_f1:>8.4f} | {best['auc']:>9.4f} | "
              f"L{best['layer_idx']:>2} ({best['auc']:.3f}) | "
              f"{orig_auc:>9.4f} | {delta:>+8.4f} | {len(rs):>5}")

    print("=" * 130)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained probes and predictors to freeform activations")
    parser.add_argument("--freeform", type=Path, required=True,
                        help="Freeform activations NPZ")
    parser.add_argument("--probes-dir", type=Path, default=None,
                        help="Directory with trained stepwise probe .pkl files")
    parser.add_argument("--predictors-dir", type=Path, default=None,
                        help="Directory with trained predictor .npz files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/freeform/transfer_evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.probes_dir is None and args.predictors_dir is None:
        parser.error("At least one of --probes-dir or --predictors-dir must be provided")

    print(f"\n{'='*110}")
    print("FREEFORM TRANSFER EVALUATION")
    print(f"{'='*110}")
    print(f"Freeform NPZ:    {args.freeform}")
    if args.probes_dir:
        print(f"Probes dir:      {args.probes_dir}")
    if args.predictors_dir:
        print(f"Predictors dir:  {args.predictors_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"{'='*110}")

    # Load data
    data_raw = load_freeform_npz(args.freeform)
    print("\nCreating offset data (shifting step numbers -1, dropping prompt-end)...")
    data_offset = make_offset_data(data_raw)

    num_layers = data_raw["num_layers"]
    all_results = {"probe_results": [], "predictor_results": []}

    # ── Part 1: Stepwise Probes ───────────────────────────────────────
    if args.probes_dir and args.probes_dir.exists():
        print(f"\n{'='*110}")
        print("PART 1: STEPWISE PROBES")
        print(f"{'='*110}")

        for alignment, data in [("raw", data_raw), ("offset", data_offset)]:
            results = apply_stepwise_probes(data, args.probes_dir, alignment)
            all_results["probe_results"].extend(results)

        for alignment in ["raw", "offset"]:
            print_probe_table(all_results["probe_results"], alignment, num_layers)
    elif args.probes_dir:
        print(f"\n⚠ Probes directory not found: {args.probes_dir}")

    # ── Part 2: Correctness Predictors ────────────────────────────────
    if args.predictors_dir and args.predictors_dir.exists():
        print(f"\n{'='*110}")
        print("PART 2: CORRECTNESS PREDICTORS")
        print(f"{'='*110}")

        for alignment, data in [("raw", data_raw), ("offset", data_offset)]:
            results = apply_correctness_predictors(data, args.predictors_dir, alignment)
            all_results["predictor_results"].extend(results)

        for alignment in ["raw", "offset"]:
            print_predictor_table(all_results["predictor_results"], alignment)
    elif args.predictors_dir:
        print(f"\n⚠ Predictors directory not found: {args.predictors_dir}")

    # ── Save ──────────────────────────────────────────────────────────
    def _sanitize(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_sanitize(x) for x in obj]
        return obj

    json_path = args.output_dir / "transfer_results.json"
    with open(json_path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # ── Summary comparison: raw vs offset ─────────────────────────────
    print(f"\n{'='*110}")
    print("ALIGNMENT COMPARISON SUMMARY")
    print(f"{'='*110}")

    if all_results["probe_results"]:
        print("\nStepwise Probes (avg accuracy across all targets and layers):")
        for al in ["raw", "offset"]:
            vals = [r["accuracy"] for r in all_results["probe_results"]
                    if r["alignment"] == al]
            if vals:
                print(f"  {al:>8}: mean={np.mean(vals):.4f}, "
                      f"median={np.median(vals):.4f}")

    if all_results["predictor_results"]:
        print("\nCorrectness Predictors (avg AUC across all feature sets and layers):")
        for al in ["raw", "offset"]:
            vals = [r["auc"] for r in all_results["predictor_results"]
                    if r["alignment"] == al]
            if vals:
                print(f"  {al:>8}: mean={np.mean(vals):.4f}, "
                      f"median={np.median(vals):.4f}")

        print("\nCorrectness Predictors: Original vs Freeform AUC (offset):")
        by_fs = defaultdict(list)
        for r in all_results["predictor_results"]:
            if r["alignment"] == "offset":
                by_fs[r["feature_set"]].append(r)
        for fs in sorted(by_fs.keys()):
            rs = by_fs[fs]
            orig = np.mean([r["orig_test_auc"] for r in rs])
            free = np.mean([r["auc"] for r in rs])
            print(f"  {fs:>30}: orig={orig:.4f} → free={free:.4f} "
                  f"(Δ={free - orig:+.4f})")

    print(f"\n✓ All outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())