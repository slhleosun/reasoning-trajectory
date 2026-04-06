#!/usr/bin/env python
"""Log-prob / entropy baseline for correctness prediction.

Computes logit-lens features (entropy, rank, probability of #### token)
from pre-extracted NPZ activations WITHOUT re-running the model.

Shortcut: hidden_state @ lm_head.weight.T = logits at that position.
From logits we derive entropy, rank(####), prob(####), etc.

Then trains a correctness predictor on these SCALAR features and compares
AUC against the trajectory-based (geometric) features.

This addresses reviewer jqhQ S3: "Baseline comparisons using token
log-probs and answer marker entropy."

Usage:
  python scripts/predictors/logprob_entropy_baseline.py \
      --npz-path output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
      --model-key llama-3.1-8b-instruct \
      --output-dir output/logprob_baseline \
      --seeds 42 123 456

  # If model loading is slow, pre-extract the lm_head weight:
  python scripts/predictors/logprob_entropy_baseline.py \
      --npz-path ... --extract-unembed-only --unembed-save-path /tmp/lm_head.npy

  # Then reuse it:
  python scripts/predictors/logprob_entropy_baseline.py \
      --npz-path ... --unembed-path /tmp/lm_head.npy
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

HASH_TOKEN_ID = 827  # "####" token in Llama tokenizer


# ======================================================================
# Unembedding matrix loading
# ======================================================================

def load_unembed_from_model(model_path: str) -> np.ndarray:
    """Load lm_head.weight from model, return as numpy [vocab_size, hidden_dim]."""
    import torch
    print(f"  Loading model from: {model_path}")
    print(f"  (Only need lm_head.weight — will free model memory after)")

    # Try safetensors first (fastest, no full model load)
    model_dir = Path(model_path)
    safetensor_files = list(model_dir.glob("*.safetensors"))

    if safetensor_files:
        print(f"  Found {len(safetensor_files)} safetensors files, scanning for lm_head.weight...")
        try:
            from safetensors import safe_open
            for sf in safetensor_files:
                with safe_open(sf, framework="pt", device="cpu") as f:
                    keys = f.keys()
                    if "lm_head.weight" in keys:
                        tensor = f.get_tensor("lm_head.weight")
                        print(f"  ✓ Loaded lm_head.weight from {sf.name}: {tensor.shape}")
                        return tensor.float().numpy()
            print("  lm_head.weight not found in safetensors, falling back to full model load")
        except ImportError:
            print("  safetensors not installed, falling back to full model load")

    # Fallback: load full model
    from transformers import AutoModelForCausalLM
    print("  Loading full model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight.float().detach().cpu().numpy()
    elif hasattr(model, "embed_out"):
        unembed = model.embed_out.weight.float().detach().cpu().numpy()
    else:
        unembed = model.get_input_embeddings().weight.float().detach().cpu().numpy()

    print(f"  ✓ Extracted lm_head.weight: {unembed.shape}")

    # Free memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return unembed


def load_unembed(args) -> np.ndarray:
    """Load unembedding matrix from saved npy or from model."""
    if args.unembed_path and Path(args.unembed_path).exists():
        print(f"  Loading cached unembed from: {args.unembed_path}")
        return np.load(args.unembed_path)

    # Load from model using config
    import yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "paths.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_info = config["models"][args.model_key]
    model_path = model_info["path"]
    unembed = load_unembed_from_model(model_path)

    # Optionally save for reuse
    if args.unembed_save_path:
        np.save(args.unembed_save_path, unembed)
        print(f"  ✓ Saved unembed to: {args.unembed_save_path}")

    return unembed


# ======================================================================
# NPZ loading
# ======================================================================

def load_npz(npz_path: Path) -> Dict:
    print(f"\nLoading NPZ from: {npz_path}")
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
# Logit-lens feature computation
# ======================================================================

def compute_logit_lens_features(
    activations: np.ndarray,
    unembed: np.ndarray,
    target_token_id: int = HASH_TOKEN_ID,
) -> Dict[str, np.ndarray]:
    """Compute logit-lens scalar features for a batch of activations.

    Args:
        activations: [n_samples, hidden_dim]
        unembed: [vocab_size, hidden_dim]
        target_token_id: token to track rank/prob for (default: #### = 827)

    Returns:
        Dict with keys:
          entropy: [n_samples]
          rank_target: [n_samples] (1-based rank of target token)
          prob_target: [n_samples]
          logit_target: [n_samples]
          max_prob: [n_samples]
          log_prob_target: [n_samples]
    """
    n = activations.shape[0]

    # Compute logits: [n_samples, vocab_size]
    # Do in float32 for stability (matching existing codebase)
    acts_f32 = activations.astype(np.float32)
    unembed_f32 = unembed.astype(np.float32)

    # Process in chunks to avoid OOM (vocab_size=128256 × n_samples can be huge)
    chunk_size = 256
    all_entropy = []
    all_rank = []
    all_prob = []
    all_logit = []
    all_max_prob = []
    all_log_prob = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_acts = acts_f32[start:end]  # [chunk, hidden_dim]

        # logits = acts @ W_U^T  →  [chunk, vocab_size]
        logits = chunk_acts @ unembed_f32.T

        # Numerically stable softmax
        logits_max = logits.max(axis=1, keepdims=True)
        logits_shifted = logits - logits_max
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Entropy: H = -sum(p * log(p))
        log_probs = np.log(np.clip(probs, 1e-12, None))
        entropy = -(probs * log_probs).sum(axis=1)

        # Target token features
        target_logits = logits[:, target_token_id]
        target_probs = probs[:, target_token_id]
        target_log_probs = log_probs[:, target_token_id]

        # Rank (1-based): how many tokens have higher logit
        ranks = (logits > target_logits[:, None]).sum(axis=1) + 1

        # Max probability
        max_probs = probs.max(axis=1)

        all_entropy.append(entropy)
        all_rank.append(ranks)
        all_prob.append(target_probs)
        all_logit.append(target_logits)
        all_max_prob.append(max_probs)
        all_log_prob.append(target_log_probs)

    return {
        'entropy': np.concatenate(all_entropy),
        'rank_target': np.concatenate(all_rank),
        'prob_target': np.concatenate(all_prob),
        'logit_target': np.concatenate(all_logit),
        'max_prob': np.concatenate(all_max_prob),
        'log_prob_target': np.concatenate(all_log_prob),
    }


# ======================================================================
# Feature extraction helpers (matching train_predictors.py)
# ======================================================================

def extract_last_step_info(data, layer_idx):
    """Get activations and metadata for last reasoning step per question."""
    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    qids = data['question_ids_step'][layer_idx]
    correct = data['is_correct_step'][layer_idx]

    unique_qids = np.unique(qids)
    out_acts, out_qids, out_correct, out_steps = [], [], [], []

    for q in unique_qids:
        mask = qids == q
        max_idx = np.argmax(step_nums[mask])
        out_acts.append(step_acts[mask][max_idx])
        out_qids.append(q)
        out_correct.append(correct[mask][max_idx])
        out_steps.append(step_nums[mask][max_idx])

    return (np.array(out_acts), np.array(out_qids),
            np.array(out_correct), np.array(out_steps))


def extract_second_last_step_info(data, layer_idx):
    """Get activations for second-to-last step."""
    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    qids = data['question_ids_step'][layer_idx]
    correct = data['is_correct_step'][layer_idx]

    unique_qids = np.unique(qids)
    out_acts, out_qids, out_correct = [], [], []

    for q in unique_qids:
        mask = qids == q
        q_steps = step_nums[mask]
        if len(q_steps) < 2:
            continue
        sorted_idx = np.argsort(q_steps)
        idx_2nd = sorted_idx[-2]
        out_acts.append(step_acts[mask][idx_2nd])
        out_qids.append(q)
        out_correct.append(correct[mask][idx_2nd])

    return np.array(out_acts), np.array(out_qids), np.array(out_correct)


# ======================================================================
# Build feature matrices
# ======================================================================

def build_logprob_features(
    data: Dict, unembed: np.ndarray, layer_idx: int, feature_mode: str = "all"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build scalar logit-lens features for correctness prediction.

    Feature modes:
      "hash_only": features from hash (answer marker) position only
      "last_step": features from last reasoning step only
      "hash_and_last": features from both positions
      "all": hash + last step + second-to-last step + cross-step features

    Returns: (X, question_ids, is_correct)
    """
    # Hash activations
    hash_acts = data['hash_activations'][layer_idx]
    hash_qids = data['question_ids_hash'][layer_idx]
    hash_correct = data['is_correct_hash'][layer_idx]

    # Last step activations
    last_acts, last_qids, last_correct, last_steps = extract_last_step_info(data, layer_idx)

    # Second-to-last step activations
    sl_acts, sl_qids, sl_correct = extract_second_last_step_info(data, layer_idx)

    # Compute logit-lens features for each position
    print(f"    Computing logit-lens at hash positions ({len(hash_acts)} samples)...")
    hash_feats = compute_logit_lens_features(hash_acts, unembed)

    print(f"    Computing logit-lens at last-step positions ({len(last_acts)} samples)...")
    last_feats = compute_logit_lens_features(last_acts, unembed)

    if feature_mode == "hash_only":
        # 6 features from hash position
        X = np.column_stack([
            hash_feats['entropy'],
            hash_feats['rank_target'],
            hash_feats['prob_target'],
            hash_feats['log_prob_target'],
            hash_feats['max_prob'],
            hash_feats['logit_target'],
        ])
        return X, hash_qids, hash_correct

    if feature_mode == "last_step":
        X = np.column_stack([
            last_feats['entropy'],
            last_feats['rank_target'],
            last_feats['prob_target'],
            last_feats['log_prob_target'],
            last_feats['max_prob'],
            last_feats['logit_target'],
        ])
        return X, last_qids, last_correct

    # For combined modes, align by question ID
    if feature_mode in ("hash_and_last", "all"):
        common = np.intersect1d(hash_qids, last_qids)

        h_idx = np.array([np.where(hash_qids == q)[0][0] for q in common])
        l_idx = np.array([np.where(last_qids == q)[0][0] for q in common])

        features = []

        # Hash features (6)
        for key in ['entropy', 'rank_target', 'prob_target', 'log_prob_target', 'max_prob', 'logit_target']:
            features.append(hash_feats[key][h_idx])

        # Last-step features (6)
        for key in ['entropy', 'rank_target', 'prob_target', 'log_prob_target', 'max_prob', 'logit_target']:
            features.append(last_feats[key][l_idx])

        # Cross-position features (6)
        features.append(hash_feats['entropy'][h_idx] - last_feats['entropy'][l_idx])
        features.append(np.log1p(hash_feats['rank_target'][h_idx].astype(float)) -
                        np.log1p(last_feats['rank_target'][l_idx].astype(float)))
        features.append(hash_feats['prob_target'][h_idx] - last_feats['prob_target'][l_idx])
        features.append(hash_feats['logit_target'][h_idx] - last_feats['logit_target'][l_idx])
        features.append(hash_feats['max_prob'][h_idx] - last_feats['max_prob'][l_idx])
        features.append(last_steps[l_idx].astype(float))  # step count

        correct = hash_correct[h_idx]

        if feature_mode == "all" and len(sl_acts) > 0:
            # Add second-to-last features if available
            common3 = np.intersect1d(common, sl_qids)
            if len(common3) > 50:
                print(f"    Computing logit-lens at second-to-last positions ({len(sl_acts)} samples)...")
                sl_feats = compute_logit_lens_features(sl_acts, unembed)

                h3_idx = np.array([np.where(hash_qids == q)[0][0] for q in common3])
                l3_idx = np.array([np.where(last_qids == q)[0][0] for q in common3])
                s3_idx = np.array([np.where(sl_qids == q)[0][0] for q in common3])

                features_3 = []

                # Hash features (6)
                for key in ['entropy', 'rank_target', 'prob_target', 'log_prob_target', 'max_prob', 'logit_target']:
                    features_3.append(hash_feats[key][h3_idx])

                # Last-step features (6)
                for key in ['entropy', 'rank_target', 'prob_target', 'log_prob_target', 'max_prob', 'logit_target']:
                    features_3.append(last_feats[key][l3_idx])

                # Cross-position features (6)
                features_3.append(hash_feats['entropy'][h3_idx] - last_feats['entropy'][l3_idx])
                features_3.append(np.log1p(hash_feats['rank_target'][h3_idx].astype(float)) -
                                  np.log1p(last_feats['rank_target'][l3_idx].astype(float)))
                features_3.append(hash_feats['prob_target'][h3_idx] - last_feats['prob_target'][l3_idx])
                features_3.append(hash_feats['logit_target'][h3_idx] - last_feats['logit_target'][l3_idx])
                features_3.append(hash_feats['max_prob'][h3_idx] - last_feats['max_prob'][l3_idx])
                features_3.append(last_steps[l3_idx].astype(float))

                # Second-to-last features (6)
                for key in ['entropy', 'rank_target', 'prob_target', 'log_prob_target', 'max_prob', 'logit_target']:
                    features_3.append(sl_feats[key][s3_idx])

                # Last minus second-to-last deltas (4)
                features_3.append(last_feats['entropy'][l3_idx] - sl_feats['entropy'][s3_idx])
                features_3.append(last_feats['prob_target'][l3_idx] - sl_feats['prob_target'][s3_idx])
                features_3.append(last_feats['logit_target'][l3_idx] - sl_feats['logit_target'][s3_idx])
                features_3.append(last_feats['max_prob'][l3_idx] - sl_feats['max_prob'][s3_idx])

                X = np.column_stack(features_3)
                return X, common3, hash_correct[h3_idx]

        X = np.column_stack(features)
        return X, common, correct

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


# ======================================================================
# Training and evaluation
# ======================================================================

def train_eval(X_train, y_train, X_test, y_test, seed=42):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000, random_state=seed, C=1.0,
        class_weight='balanced', solver='lbfgs'
    )
    clf.fit(X_tr, y_train)

    y_proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, clf.predict(X_te))
    return auc, acc, clf


def build_trajectory_features(data, layer_idx):
    """Build the paper's trajectory features for comparison."""
    hash_acts = data['hash_activations'][layer_idx]
    hash_qids = data['question_ids_hash'][layer_idx]
    hash_correct = data['is_correct_hash'][layer_idx]

    last_acts, last_qids, _, _ = extract_last_step_info(data, layer_idx)
    sl_acts, sl_qids, _ = extract_second_last_step_info(data, layer_idx)

    common = np.intersect1d(np.intersect1d(hash_qids, last_qids), sl_qids)
    if len(common) == 0:
        return None, None, None

    h_idx = np.array([np.where(hash_qids == q)[0][0] for q in common])
    l_idx = np.array([np.where(last_qids == q)[0][0] for q in common])
    s_idx = np.array([np.where(sl_qids == q)[0][0] for q in common])

    h = hash_acts[h_idx]
    l = last_acts[l_idx]
    s = sl_acts[s_idx]

    X = np.concatenate([h, h - l, l - s], axis=1)
    return X, common, hash_correct[h_idx]


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Log-prob/entropy baseline for correctness prediction")
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--model-key", type=str, default="llama-3.1-8b-instruct",
                        help="Model key in config/paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("output/logprob_baseline"))
    parser.add_argument("--unembed-path", type=str, default=None,
                        help="Path to pre-saved lm_head.weight.npy")
    parser.add_argument("--unembed-save-path", type=str, default=None,
                        help="Save extracted lm_head.weight to this path for reuse")
    parser.add_argument("--extract-unembed-only", action="store_true",
                        help="Only extract and save lm_head.weight, then exit")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to evaluate (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load unembedding matrix
    print("\n" + "=" * 70)
    print("Loading unembedding matrix (lm_head.weight)")
    print("=" * 70)
    unembed = load_unembed(args)
    print(f"  Unembed shape: {unembed.shape}")

    if args.extract_unembed_only:
        print("  --extract-unembed-only: done.")
        return

    # Load NPZ
    data = load_npz(args.npz_path)
    num_layers = data['num_layers']
    layers = args.layers or list(range(num_layers))

    feature_modes = ["hash_only", "last_step", "hash_and_last", "all"]

    all_results = {}

    for mode in feature_modes:
        print(f"\n{'=' * 70}")
        print(f"LOGIT-LENS BASELINE: {mode}")
        print(f"{'=' * 70}")

        layer_results = {}

        for layer_idx in layers:
            X, qids, y = build_logprob_features(data, unembed, layer_idx, mode)
            y = y.astype(int)

            if len(X) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
                continue

            aucs = []
            for seed in args.seeds:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=args.test_size, random_state=seed, stratify=y
                    )
                    auc, acc, _ = train_eval(X_train, y_train, X_test, y_test, seed)
                    aucs.append(auc)
                except Exception as e:
                    continue

            if aucs:
                layer_results[layer_idx] = {
                    "mean_auc": float(np.mean(aucs)),
                    "std_auc": float(np.std(aucs)),
                    "n_features": int(X.shape[1]),
                    "n_samples": int(len(y)),
                }

        if layer_results:
            best_layer = max(layer_results, key=lambda l: layer_results[l]["mean_auc"])
            best = layer_results[best_layer]
            print(f"\n  Best: L{best_layer} → AUC = {best['mean_auc']:.4f} ± {best['std_auc']:.4f} "
                  f"({best['n_features']} features, {best['n_samples']} samples)")
            all_results[mode] = {
                "best_layer": int(best_layer),
                "best_auc_mean": best["mean_auc"],
                "best_auc_std": best["std_auc"],
                "per_layer": {str(k): v for k, v in layer_results.items()},
            }

    # Now compare against trajectory features at best layers
    print(f"\n{'=' * 70}")
    print("COMPARISON: Logit-lens vs Trajectory features")
    print(f"{'=' * 70}")

    # Run trajectory features at a few key layers
    traj_layers = sorted(set([29, 31] + [all_results[m]["best_layer"]
                                          for m in all_results if "best_layer" in all_results[m]]))

    traj_results = {}
    for layer_idx in traj_layers:
        X_traj, qids, y = build_trajectory_features(data, layer_idx)
        if X_traj is None:
            continue
        y = y.astype(int)

        from sklearn.decomposition import PCA

        aucs = []
        for seed in args.seeds:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_traj, y, test_size=args.test_size, random_state=seed, stratify=y
                )
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_te = scaler.transform(X_test)

                n_comp = min(128, X_tr.shape[0] - 1, X_tr.shape[1])
                pca = PCA(n_components=n_comp, random_state=seed)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

                clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0,
                                         class_weight='balanced')
                clf.fit(X_tr, y_train)
                auc = roc_auc_score(y_test, clf.predict_proba(X_te)[:, 1])
                aucs.append(auc)
            except Exception:
                continue

        if aucs:
            traj_results[layer_idx] = {
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
            }

    # Print comparison table
    print(f"\n{'Method':<35s} {'Best Layer':>10s} {'AUC':>10s} {'± Std':>8s} {'# Feat':>8s}")
    print("-" * 75)

    for mode in feature_modes:
        if mode in all_results:
            r = all_results[mode]
            n_feat = r["per_layer"][str(r["best_layer"])]["n_features"]
            print(f"Logit-lens ({mode}){'':<{max(0, 22-len(mode))}} "
                  f"L{r['best_layer']:>2d}{'':>6s}"
                  f"{r['best_auc_mean']:>10.4f} ±{r['best_auc_std']:>6.4f} {n_feat:>8d}")

    for layer_idx in sorted(traj_results):
        r = traj_results[layer_idx]
        print(f"Trajectory (L{layer_idx}){'':<20s}"
              f"L{layer_idx:>2d}{'':>6s}"
              f"{r['mean_auc']:>10.4f} ±{r['std_auc']:>6.4f} {'4096×3':>8s}")

    print(f"\nStep-count-only baseline:{'':>24s}{'0.6491':>10s}")

    # Save results
    final = {
        "logit_lens_baselines": all_results,
        "trajectory_comparison": {str(k): v for k, v in traj_results.items()},
        "seeds": args.seeds,
    }
    out_path = args.output_dir / "logprob_baseline_results.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()