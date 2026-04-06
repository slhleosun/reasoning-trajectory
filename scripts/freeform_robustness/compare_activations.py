#!/usr/bin/env python
"""Compare freeform structure activations against Step-enforced (original) activations.

For each question present in BOTH NPZ files, aligns activations by step index
and computes cosine similarity, L2 distance, and CKA between the two sets.

Comparison is per step index:
  - Original step 1 = token at "Step 1:" marker
  - Freeform step 1 = last prompt token (model state before generation)
  - Freeform step 2+ = structure-specific markers

Since freeform step 1 is the prompt-end activation (semantically different from
the original step 1), we report TWO alignment modes:
  A) "raw"    — compare step i (freeform) vs step i (original) directly
  B) "offset" — compare step i+1 (freeform) vs step i (original), skipping
                 the freeform prompt-end activation

We also compare hash/answer-boundary activations (1 per question in both).

Metrics:
  - Cosine similarity  (direction alignment, [-1, 1])
  - L2 distance        (magnitude of difference)
  - Relative L2        (L2 / mean of norms, scale-invariant)
  - Linear CKA         (representational similarity across layers)

Output: JSON + matplotlib PDF with summary plots.

Usage
-----
python scripts/freeform_robustness/compare_activations.py \\
    --original output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \\
    --freeform output/freeform/freeform_structure_activations.npz \\
    --output-dir output/freeform/comparison
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Structure code mapping (must match extraction script) ─────────────
STRUCTURE_NAMES = {0: "step_x", 1: "numbered_list", 2: "double_newline",
                   3: "single_newline", 4: "single_block"}


# ======================================================================
# NPZ loading helpers
# ======================================================================

def load_npz(path: Path, label: str) -> dict:
    """Load and inspect an NPZ file, returning a uniform dict."""
    print(f"Loading {label}: {path}")
    raw = np.load(path, allow_pickle=True)

    num_layers = int(raw["num_layers"])
    hidden_dim = int(raw["hidden_dim"])
    print(f"  {num_layers} layers, hidden_dim={hidden_dim}")

    def _unpack(key, inner_dtype=None):
        """Unpack object-array-of-arrays → list of numpy arrays.

        Handles two NPZ storage formats:
        1. Object array (freeform): each element is a per-layer array (variable sizes)
        2. Stacked 3D array (original): numpy auto-stacked same-shape arrays → [num_layers, n, dim]
        """
        arr = raw[key]
        if arr.dtype == object:
            out = [arr[i] for i in range(len(arr))]
            if inner_dtype is not None:
                out = [np.asarray(x, dtype=inner_dtype) for x in out]
            return out
        # Non-object: if shape[0] == num_layers and ndim >= 2, split by layer
        if arr.ndim >= 2 and arr.shape[0] == num_layers:
            out = [arr[i] for i in range(num_layers)]
            if inner_dtype is not None:
                out = [np.asarray(x, dtype=inner_dtype) for x in out]
            return out
        return [arr]  # scalar or single array

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

    # Freeform-specific fields
    if "structure_types_step" in raw:
        d["structure_types_step"] = _unpack("structure_types_step", np.int32)
        d["structure_types_hash"] = _unpack("structure_types_hash", np.int32)
    else:
        d["structure_types_step"] = None
        d["structure_types_hash"] = None

    # Stats
    n_step = len(d["step_activations"][0]) if len(d["step_activations"][0]) > 0 else 0
    n_hash = len(d["hash_activations"][0]) if len(d["hash_activations"][0]) > 0 else 0
    n_qids = len(np.unique(d["question_ids_step"][0])) if n_step > 0 else 0
    print(f"  {n_step} step activations, {n_hash} hash activations, "
          f"{n_qids} unique questions")
    return d


def group_by_question(data: dict, layer: int) -> dict:
    """Group step activations by question_id for a given layer.

    Returns: {qid: {"steps": [(step_num, activation)], "hash": activation,
                     "is_correct": bool, "structure": int_or_None}}
    """
    sa   = data["step_activations"][layer]
    sn   = data["step_numbers"][layer]
    qids = data["question_ids_step"][layer]
    corr = data["is_correct_step"][layer]
    st   = data["structure_types_step"][layer] if data["structure_types_step"] else None

    ha    = data["hash_activations"][layer]
    qh    = data["question_ids_hash"][layer]
    ch    = data["is_correct_hash"][layer]
    sh    = data["structure_types_hash"][layer] if data["structure_types_hash"] else None

    groups = {}

    # Group step activations
    for i in range(len(sa)):
        qid = int(qids[i])
        if qid not in groups:
            groups[qid] = {"steps": [], "hash": None, "is_correct": bool(corr[i]),
                           "structure": int(st[i]) if st is not None else None}
        groups[qid]["steps"].append((int(sn[i]), sa[i]))

    # Add hash activations
    for i in range(len(ha)):
        qid = int(qh[i])
        if qid in groups:
            groups[qid]["hash"] = ha[i]
        else:
            groups[qid] = {"steps": [], "hash": ha[i],
                           "is_correct": bool(ch[i]),
                           "structure": int(sh[i]) if sh is not None else None}

    # Sort steps by step_number within each question
    for qid in groups:
        groups[qid]["steps"].sort(key=lambda x: x[0])

    return groups


# ======================================================================
# Metrics
# ======================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance normalized by average norm."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    avg_norm = (na + nb) / 2
    if avg_norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(a - b) / avg_norm)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices [n_samples, dim].

    Measures representational similarity: 1.0 = identical geometry,
    0.0 = orthogonal representations.
    """
    n = X.shape[0]
    if n < 2:
        return float('nan')

    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC estimates
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


# ======================================================================
# Core comparison
# ======================================================================

def compare_layer(
    orig_groups: dict,
    free_groups: dict,
    alignment: str = "raw",
) -> dict:
    """Compare activations for one layer.

    alignment="raw":    compare step i (free) vs step i (orig)
    alignment="offset": compare step i+1 (free) vs step i (orig)
                        (skips freeform prompt-end activation)

    Returns dict with overall, per-step, per-structure, and per-correctness
    metrics.  Per-structure includes its own per-step breakdown, hash
    comparison, CKA, and the list of question IDs.
    """
    shared_qids = sorted(set(orig_groups.keys()) & set(free_groups.keys()))

    # ── Global accumulators ───────────────────────────────────────────
    step_cosines   = defaultdict(list)
    step_l2s       = defaultdict(list)
    step_rel_l2s   = defaultdict(list)

    q_cosines, q_l2s, q_n_steps = [], [], []
    hash_cosines, hash_l2s = [], []
    all_orig_acts, all_free_acts = [], []

    by_correct = {True:  {"cosines": [], "l2s": [], "n": 0},
                  False: {"cosines": [], "l2s": [], "n": 0}}

    # ── Per-structure accumulators ────────────────────────────────────
    cat_data = defaultdict(lambda: {
        "question_ids": [],
        "q_cosines": [],        # one mean-cos per question
        "q_l2s": [],            # one mean-L2 per question
        "q_n_steps": [],        # steps compared per question
        "step_cosines": defaultdict(list),   # step_idx → [cos, ...]
        "step_l2s": defaultdict(list),
        "step_rel_l2s": defaultdict(list),
        "hash_cosines": [],
        "hash_l2s": [],
        "orig_acts": [],        # for CKA
        "free_acts": [],
        "n_correct": 0,
        "n_incorrect": 0,
    })

    # ── Main comparison loop ──────────────────────────────────────────
    for qid in shared_qids:
        og = orig_groups[qid]
        fg = free_groups[qid]

        o_steps = og["steps"]
        f_steps = fg["steps"]

        if alignment == "offset" and len(f_steps) > 1:
            f_steps = f_steps[1:]

        n_compare = min(len(o_steps), len(f_steps))
        if n_compare == 0:
            continue

        # Determine category from freeform structure
        struct = fg.get("structure")
        sname = STRUCTURE_NAMES.get(struct, "unknown") if struct is not None else "unknown"
        cat = cat_data[sname]
        cat["question_ids"].append(int(qid))

        q_cos_list, q_l2_list = [], []

        for idx in range(n_compare):
            o_act = o_steps[idx][1].astype(np.float32)
            f_act = f_steps[idx][1].astype(np.float32)

            cs = cosine_sim(o_act, f_act)
            ld = l2_dist(o_act, f_act)
            rl = relative_l2(o_act, f_act)

            step_cosines[idx + 1].append(cs)
            step_l2s[idx + 1].append(ld)
            step_rel_l2s[idx + 1].append(rl)

            cat["step_cosines"][idx + 1].append(cs)
            cat["step_l2s"][idx + 1].append(ld)
            cat["step_rel_l2s"][idx + 1].append(rl)

            q_cos_list.append(cs)
            q_l2_list.append(ld)

            all_orig_acts.append(o_act)
            all_free_acts.append(f_act)
            cat["orig_acts"].append(o_act)
            cat["free_acts"].append(f_act)

        q_cosines.append(np.mean(q_cos_list))
        q_l2s.append(np.mean(q_l2_list))
        q_n_steps.append(n_compare)

        cat["q_cosines"].append(np.mean(q_cos_list))
        cat["q_l2s"].append(np.mean(q_l2_list))
        cat["q_n_steps"].append(n_compare)

        # Hash
        if og["hash"] is not None and fg["hash"] is not None:
            hc = cosine_sim(og["hash"].astype(np.float32),
                            fg["hash"].astype(np.float32))
            hl = l2_dist(og["hash"].astype(np.float32),
                         fg["hash"].astype(np.float32))
            hash_cosines.append(hc)
            hash_l2s.append(hl)
            cat["hash_cosines"].append(hc)
            cat["hash_l2s"].append(hl)

        # Correctness
        corr = og["is_correct"]
        by_correct[corr]["cosines"].extend(q_cos_list)
        by_correct[corr]["l2s"].extend(q_l2_list)
        by_correct[corr]["n"] += 1
        if corr:
            cat["n_correct"] += 1
        else:
            cat["n_incorrect"] += 1

    # ── Helpers ───────────────────────────────────────────────────────

    def _agg(lst):
        if not lst:
            return {"mean": None, "std": None, "median": None,
                    "p5": None, "p95": None, "n": 0}
        a = np.array(lst)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "median": float(np.median(a)),
                "p5": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)),
                "n": len(a)}

    def _cka(orig_list, free_list, max_n=5000):
        if len(orig_list) < 10:
            return float('nan')
        if len(orig_list) > max_n:
            idx = np.random.choice(len(orig_list), max_n, replace=False)
            X = np.stack([orig_list[i] for i in idx])
            Y = np.stack([free_list[i] for i in idx])
        else:
            X = np.stack(orig_list)
            Y = np.stack(free_list)
        return linear_cka(X, Y)

    # ── Build result ──────────────────────────────────────────────────

    result = {
        "alignment": alignment,
        "n_shared_questions": len(shared_qids),
        "n_compared_questions": len(q_cosines),
        "n_compared_pairs": sum(q_n_steps),

        "cosine_similarity": _agg(q_cosines),
        "l2_distance": _agg(q_l2s),
        "linear_cka": _cka(all_orig_acts, all_free_acts),

        "hash_cosine": _agg(hash_cosines),
        "hash_l2": _agg(hash_l2s),

        "per_step": {},
        "per_structure": {},
        "per_correct": {},
        "steps_per_question": _agg(q_n_steps),
    }

    # Per step (global)
    for si in sorted(step_cosines.keys()):
        result["per_step"][si] = {
            "cosine": _agg(step_cosines[si]),
            "l2": _agg(step_l2s[si]),
            "rel_l2": _agg(step_rel_l2s[si]),
        }

    # Per structure — full detail
    for sname in sorted(cat_data.keys()):
        cat = cat_data[sname]
        cat_result = {
            "n_questions": len(cat["question_ids"]),
            "n_correct": cat["n_correct"],
            "n_incorrect": cat["n_incorrect"],
            "n_compared_pairs": sum(cat["q_n_steps"]),
            "question_ids": cat["question_ids"],

            "cosine": _agg(cat["q_cosines"]),
            "l2": _agg(cat["q_l2s"]),
            "linear_cka": _cka(cat["orig_acts"], cat["free_acts"]),

            "hash_cosine": _agg(cat["hash_cosines"]),
            "hash_l2": _agg(cat["hash_l2s"]),

            "steps_per_question": _agg(cat["q_n_steps"]),

            "per_step": {},
        }
        for si in sorted(cat["step_cosines"].keys()):
            cat_result["per_step"][si] = {
                "cosine": _agg(cat["step_cosines"][si]),
                "l2": _agg(cat["step_l2s"][si]),
                "rel_l2": _agg(cat["step_rel_l2s"][si]),
            }
        result["per_structure"][sname] = cat_result

    # Per correctness
    for corr_val in [True, False]:
        label = "correct" if corr_val else "incorrect"
        vals = by_correct[corr_val]
        result["per_correct"][label] = {
            "cosine": _agg(vals["cosines"]),
            "l2": _agg(vals["l2s"]),
            "n_questions": vals["n"],
        }

    return result


# ======================================================================
# Plotting
# ======================================================================

def make_plots(results: dict, output_dir: Path):
    """Generate summary plots from comparison results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")
        return

    layers = sorted(results.keys())
    if not layers:
        return

    # Pick one alignment mode for main plots
    for alignment in ["offset", "raw"]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Freeform vs Step-Enforced Activation Comparison "
                     f"(alignment={alignment})", fontsize=14, y=0.98)

        # Colors for structure types
        CAT_COLORS = {
            "step_x": "#2196F3", "numbered_list": "#4CAF50",
            "double_newline": "#FF9800", "single_newline": "#9C27B0",
            "single_block": "#F44336",
        }

        # ── Plot 1: Overall cosine sim across layers ──────────────────
        ax = axes[0, 0]
        means = [results[l][alignment]["cosine_similarity"]["mean"]
                 for l in layers
                 if results[l][alignment]["cosine_similarity"]["mean"] is not None]
        stds  = [results[l][alignment]["cosine_similarity"]["std"]
                 for l in layers
                 if results[l][alignment]["cosine_similarity"]["mean"] is not None]
        valid_layers = [l for l in layers
                        if results[l][alignment]["cosine_similarity"]["mean"] is not None]
        if means:
            ax.errorbar(valid_layers, means, yerr=stds, fmt='-o',
                        markersize=3, linewidth=1, capsize=2, alpha=0.8,
                        color='black', label='Overall')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title("Mean Cosine Sim by Layer")
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(min(0, min(means) - 0.1), 1.05)

        # ── Plot 2: Per-category cosine sim across layers ─────────────
        ax = axes[0, 1]
        cat_names_plot = sorted(set().union(
            *(results[l][alignment]["per_structure"].keys() for l in layers)))
        for sname in cat_names_plot:
            cat_vals = []
            cat_layers = []
            for l in layers:
                psl = results[l][alignment]["per_structure"].get(sname, {})
                m = psl.get("cosine", {}).get("mean") if psl else None
                if m is not None:
                    cat_vals.append(m)
                    cat_layers.append(l)
            if cat_vals:
                color = CAT_COLORS.get(sname, '#888888')
                ax.plot(cat_layers, cat_vals, '-o', markersize=3,
                        linewidth=1.2, label=sname, alpha=0.85, color=color)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Per-Category Cosine Sim by Layer")
        ax.legend(fontsize=7, loc='best')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        if ax.get_lines():
            ymin = min(np.min(line.get_ydata()) for line in ax.get_lines()
                       if len(line.get_ydata()) > 0)
            ax.set_ylim(min(0, ymin - 0.05), 1.05)

        # ── Plot 3: Per-step cosine at select layers ──────────────────
        ax = axes[0, 2]
        select_layers = [0, 8, 16, 24, 31] if max(layers) >= 31 else \
                        [0, max(layers) // 4, max(layers) // 2,
                         3 * max(layers) // 4, max(layers)]
        select_layers = [l for l in select_layers if l in results]
        for sl in select_layers:
            ps = results[sl][alignment]["per_step"]
            steps = sorted(ps.keys())[:10]
            cos_vals = [ps[s]["cosine"]["mean"] for s in steps
                        if ps[s]["cosine"]["mean"] is not None]
            valid_steps = [s for s in steps
                           if ps[s]["cosine"]["mean"] is not None]
            if cos_vals:
                ax.plot(valid_steps, cos_vals, '-o', markersize=3,
                        linewidth=1, label=f"L{sl}", alpha=0.8)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cosine Sim by Step Index (Global)")
        ax.legend(fontsize=8, loc='lower left')
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)

        # ── Plot 4: Per-category per-step at middle layer ─────────────
        ax = axes[1, 0]
        mid_layer = max(layers) // 2
        if mid_layer in results:
            ps_all = results[mid_layer][alignment]["per_structure"]
            for sname in sorted(ps_all.keys()):
                cat_steps = ps_all[sname].get("per_step", {})
                steps_sorted = sorted(cat_steps.keys(), key=lambda x: int(x))[:10]
                cos_vals = [cat_steps[s]["cosine"]["mean"] for s in steps_sorted
                            if cat_steps[s]["cosine"]["mean"] is not None]
                valid_s = [int(s) for s in steps_sorted
                           if cat_steps[s]["cosine"]["mean"] is not None]
                if cos_vals:
                    color = CAT_COLORS.get(sname, '#888888')
                    ax.plot(valid_s, cos_vals, '-o', markersize=3,
                            linewidth=1.2, label=sname, alpha=0.85, color=color)
            ax.set_xlabel("Step Index")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Per-Category Per-Step (Layer {mid_layer})")
            ax.legend(fontsize=7, loc='best')
            if ax.get_lines():
                ymin = min(np.min(line.get_ydata()) for line in ax.get_lines()
                           if len(line.get_ydata()) > 0)
                ax.set_ylim(min(0, ymin - 0.05), 1.05)

        # ── Plot 5: Correct vs Incorrect ──────────────────────────────
        ax = axes[1, 1]
        corr_cos_by_layer = {"correct": [], "incorrect": []}
        for l in layers:
            for label in ["correct", "incorrect"]:
                v = results[l][alignment]["per_correct"].get(label, {})
                m = v.get("cosine", {}).get("mean")
                corr_cos_by_layer[label].append(m)
        for label, vals in corr_cos_by_layer.items():
            valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if valid:
                ax.plot([x[0] for x in valid], [x[1] for x in valid],
                        '-o', markersize=3, linewidth=1, label=label, alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Correct vs Incorrect")
        ax.legend(fontsize=9)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)

        # ── Plot 6: Per-category hash cosine + CKA ────────────────────
        ax = axes[1, 2]
        # Bar chart: hash cosine and CKA side by side at middle layer
        if mid_layer in results:
            ps_all = results[mid_layer][alignment]["per_structure"]
            names = sorted(ps_all.keys())
            hash_vals = []
            cka_vals = []
            valid_names = []
            for n in names:
                hv = ps_all[n].get("hash_cosine", {}).get("mean")
                cv = ps_all[n].get("linear_cka", float('nan'))
                if hv is not None:
                    hash_vals.append(hv)
                    cka_vals.append(cv if not np.isnan(cv) else 0)
                    valid_names.append(n)
            if hash_vals:
                x = np.arange(len(valid_names))
                w = 0.35
                ax.bar(x - w/2, hash_vals, w, label='Hash Cosine',
                       color=[CAT_COLORS.get(n, '#888') for n in valid_names],
                       alpha=0.7)
                ax.bar(x + w/2, cka_vals, w, label='CKA',
                       color=[CAT_COLORS.get(n, '#888') for n in valid_names],
                       alpha=0.4, hatch='//')
                ax.set_xticks(x)
                ax.set_xticklabels(valid_names, fontsize=7, rotation=30)
                ax.set_ylabel("Score")
                ax.set_title(f"Hash Cosine & CKA (Layer {mid_layer})")
                ax.legend(fontsize=8)
                ax.set_ylim(0, 1.05)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_path = output_dir / f"comparison_{alignment}.pdf"
        plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {pdf_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare freeform vs Step-enforced activations")
    parser.add_argument("--original", type=Path, required=True,
                        help="Original Step-enforced NPZ")
    parser.add_argument("--freeform", type=Path, required=True,
                        help="Freeform structure-aware NPZ")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/freeform/comparison"))
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to compare: 'all' or comma-separated e.g. '0,8,16,24,31'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load both datasets ────────────────────────────────────────────
    orig = load_npz(args.original, "Original (Step-enforced)")
    free = load_npz(args.freeform, "Freeform (structure-aware)")

    assert orig["num_layers"] == free["num_layers"], \
        f"Layer count mismatch: {orig['num_layers']} vs {free['num_layers']}"
    assert orig["hidden_dim"] == free["hidden_dim"], \
        f"Hidden dim mismatch: {orig['hidden_dim']} vs {free['hidden_dim']}"

    num_layers = orig["num_layers"]

    # ── Select layers ─────────────────────────────────────────────────
    if args.layers == "all":
        layers = list(range(num_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    print(f"\nComparing {len(layers)} layers: "
          f"{layers[:5]}{'...' if len(layers) > 5 else ''}\n")

    # ── Compare each layer ────────────────────────────────────────────
    results = {}  # layer → {alignment → comparison_dict}

    for layer in layers:
        print(f"Layer {layer}/{num_layers - 1}...", end=" ", flush=True)

        orig_groups = group_by_question(orig, layer)
        free_groups = group_by_question(free, layer)

        results[layer] = {}
        for alignment in ["raw", "offset"]:
            results[layer][alignment] = compare_layer(
                orig_groups, free_groups, alignment)

        # Quick summary
        r_raw = results[layer]["raw"]
        r_off = results[layer]["offset"]
        cs_raw = r_raw["cosine_similarity"]["mean"]
        cs_off = r_off["cosine_similarity"]["mean"]
        hc_raw = r_raw["hash_cosine"]["mean"]
        cka_off = r_off["linear_cka"]

        print(f"cos(raw)={cs_raw:.4f}  cos(offset)={cs_off:.4f}  "
              f"hash_cos={hc_raw:.4f}  CKA={cka_off:.4f}  "
              f"(n={r_raw['n_compared_questions']} questions, "
              f"{r_raw['n_compared_pairs']} pairs)"
              if cs_raw is not None else "no data")

    # ── Summary across layers ─────────────────────────────────────────
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY ACROSS LAYERS")
    print(f"{'='*100}\n")

    for alignment in ["raw", "offset"]:
        print(f"  Alignment: {alignment}")
        cs_by_layer = [results[l][alignment]["cosine_similarity"]["mean"]
                       for l in layers
                       if results[l][alignment]["cosine_similarity"]["mean"] is not None]
        if cs_by_layer:
            print(f"    Step cosine sim:  "
                  f"mean={np.mean(cs_by_layer):.4f}, "
                  f"min={np.min(cs_by_layer):.4f} (L{layers[np.argmin(cs_by_layer)]}), "
                  f"max={np.max(cs_by_layer):.4f} (L{layers[np.argmax(cs_by_layer)]})")

        hc_by_layer = [results[l][alignment]["hash_cosine"]["mean"]
                       for l in layers
                       if results[l][alignment]["hash_cosine"]["mean"] is not None]
        if hc_by_layer:
            print(f"    Hash cosine sim:  "
                  f"mean={np.mean(hc_by_layer):.4f}, "
                  f"min={np.min(hc_by_layer):.4f}, "
                  f"max={np.max(hc_by_layer):.4f}")

        cka_by_layer = [results[l][alignment]["linear_cka"]
                        for l in layers
                        if not np.isnan(results[l][alignment]["linear_cka"])]
        if cka_by_layer:
            print(f"    Linear CKA:       "
                  f"mean={np.mean(cka_by_layer):.4f}, "
                  f"min={np.min(cka_by_layer):.4f}, "
                  f"max={np.max(cka_by_layer):.4f}")
        print()

    # ── Per-category detailed report ──────────────────────────────────
    # Use the "offset" alignment (the meaningful one for the rebuttal)
    alignment = "offset"
    mid = layers[len(layers) // 2]

    # Gather category names from middle layer
    cat_names = sorted(results[mid][alignment]["per_structure"].keys())

    if cat_names:
        print(f"\n{'='*100}")
        print(f"PER-CATEGORY ANALYSIS (alignment={alignment})")
        print(f"{'='*100}")

        for sname in cat_names:
            ps_mid = results[mid][alignment]["per_structure"].get(sname, {})
            if not ps_mid or ps_mid.get("n_questions", 0) == 0:
                continue

            print(f"\n{'─'*100}")
            print(f"  Category: {sname}")
            print(f"{'─'*100}")
            print(f"    Questions:  {ps_mid['n_questions']}  "
                  f"(correct={ps_mid['n_correct']}, incorrect={ps_mid['n_incorrect']})")
            print(f"    Pairs:      {ps_mid['n_compared_pairs']}")
            print(f"    Steps/q:    mean={ps_mid['steps_per_question']['mean']:.1f}, "
                  f"median={ps_mid['steps_per_question']['median']:.0f}")

            # Cosine sim across layers
            cat_cos_by_layer = []
            cat_hash_by_layer = []
            cat_cka_by_layer = []
            for l in layers:
                psl = results[l][alignment]["per_structure"].get(sname, {})
                if psl and psl.get("cosine", {}).get("mean") is not None:
                    cat_cos_by_layer.append((l, psl["cosine"]["mean"]))
                if psl and psl.get("hash_cosine", {}).get("mean") is not None:
                    cat_hash_by_layer.append((l, psl["hash_cosine"]["mean"]))
                if psl and not np.isnan(psl.get("linear_cka", float('nan'))):
                    cat_cka_by_layer.append((l, psl["linear_cka"]))

            if cat_cos_by_layer:
                vals = [v for _, v in cat_cos_by_layer]
                best_l = cat_cos_by_layer[np.argmax(vals)][0]
                worst_l = cat_cos_by_layer[np.argmin(vals)][0]
                print(f"\n    Step cosine similarity across layers:")
                print(f"      mean={np.mean(vals):.4f}, "
                      f"min={np.min(vals):.4f} (L{worst_l}), "
                      f"max={np.max(vals):.4f} (L{best_l})")

            if cat_hash_by_layer:
                vals = [v for _, v in cat_hash_by_layer]
                print(f"    Hash cosine similarity across layers:")
                print(f"      mean={np.mean(vals):.4f}, "
                      f"min={np.min(vals):.4f}, "
                      f"max={np.max(vals):.4f}")

            if cat_cka_by_layer:
                vals = [v for _, v in cat_cka_by_layer]
                print(f"    Linear CKA across layers:")
                print(f"      mean={np.mean(vals):.4f}, "
                      f"min={np.min(vals):.4f}, "
                      f"max={np.max(vals):.4f}")

            # Per-step at middle layer
            cat_steps = ps_mid.get("per_step", {})
            if cat_steps:
                print(f"\n    Per-step breakdown (Layer {mid}):")
                print(f"      {'Step':>6} | {'Cos mean':>10} | {'Cos std':>10} | "
                      f"{'RelL2 mean':>11} | {'N':>6}")
                print(f"      {'-'*55}")
                for si in sorted(cat_steps.keys(), key=lambda x: int(x)):
                    sv = cat_steps[si]
                    cm = sv["cosine"]["mean"]
                    cs = sv["cosine"]["std"]
                    rl = sv["rel_l2"]["mean"]
                    n  = sv["cosine"]["n"]
                    if cm is not None:
                        print(f"      {si:>6} | {cm:>10.4f} | {cs:>10.4f} | "
                              f"{rl:>11.4f} | {n:>6}")

            # Sample question IDs
            qids = ps_mid.get("question_ids", [])
            if qids:
                sample = qids[:20]
                more = f" ... (+{len(qids) - 20} more)" if len(qids) > 20 else ""
                print(f"\n    Question IDs (first 20): {sample}{more}")

        # ── Cross-category comparison table at middle layer ───────────
        print(f"\n{'='*100}")
        print(f"CROSS-CATEGORY COMPARISON TABLE (Layer {mid}, alignment={alignment})")
        print(f"{'='*100}\n")

        header = (f"  {'Category':>20} | {'N_q':>5} | {'Cos(step)':>10} | "
                  f"{'Cos(hash)':>10} | {'CKA':>8} | {'RelL2':>8} | "
                  f"{'Steps/q':>8}")
        print(header)
        print(f"  {'-'*85}")

        for sname in cat_names:
            ps = results[mid][alignment]["per_structure"].get(sname, {})
            if not ps or ps.get("n_questions", 0) == 0:
                continue

            nq  = ps["n_questions"]
            cos = ps["cosine"]["mean"]
            hcs = ps["hash_cosine"]["mean"]
            cka = ps.get("linear_cka", float('nan'))
            spq = ps["steps_per_question"]["mean"]

            # Mean relative L2 across steps
            rl2_vals = [sv["rel_l2"]["mean"]
                        for sv in ps.get("per_step", {}).values()
                        if sv["rel_l2"]["mean"] is not None]
            rl2 = np.mean(rl2_vals) if rl2_vals else None

            cos_s = f"{cos:10.4f}" if cos is not None else f"{'n/a':>10}"
            hcs_s = f"{hcs:10.4f}" if hcs is not None else f"{'n/a':>10}"
            cka_s = f"{cka:8.4f}"  if not np.isnan(cka) else f"{'n/a':>8}"
            rl2_s = f"{rl2:8.4f}"  if rl2 is not None else f"{'n/a':>8}"
            spq_s = f"{spq:8.1f}"  if spq is not None else f"{'n/a':>8}"

            print(f"  {sname:>20} | {nq:>5} | {cos_s} | {hcs_s} | "
                  f"{cka_s} | {rl2_s} | {spq_s}")

        # ── Cross-category comparison across ALL layers ───────────────
        print(f"\n{'='*100}")
        print(f"PER-CATEGORY COSINE SIMILARITY BY LAYER (alignment={alignment})")
        print(f"{'='*100}\n")

        # Header
        cat_header = f"  {'Layer':>6}"
        for sname in cat_names:
            cat_header += f" | {sname[:12]:>12}"
        cat_header += f" | {'ALL':>10}"
        print(cat_header)
        print(f"  {'-' * (10 + 15 * len(cat_names) + 13)}")

        for l in layers:
            row = f"  {l:>6}"
            for sname in cat_names:
                psl = results[l][alignment]["per_structure"].get(sname, {})
                m = psl.get("cosine", {}).get("mean") if psl else None
                row += f" | {m:>12.4f}" if m is not None else f" | {'':>12}"
            # Overall
            overall_m = results[l][alignment]["cosine_similarity"]["mean"]
            row += f" | {overall_m:>10.4f}" if overall_m is not None else f" | {'':>10}"
            print(row)

    # ── Save results ──────────────────────────────────────────────────
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_path = args.output_dir / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"\n✓ Results: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("Generating plots...")
    make_plots(results, args.output_dir)

    # ── Compact table for paper/rebuttal ──────────────────────────────
    table_path = args.output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(f"{'Layer':>6} | {'Cos(raw)':>10} | {'Cos(offset)':>12} | "
                f"{'Hash Cos':>10} | {'CKA(off)':>10} | "
                f"{'RelL2(off)':>11} | {'N pairs':>8}\n")
        f.write("-" * 85 + "\n")
        for l in layers:
            cr = results[l]["raw"]["cosine_similarity"]
            co = results[l]["offset"]["cosine_similarity"]
            hc = results[l]["raw"]["hash_cosine"]
            cka = results[l]["offset"]["linear_cka"]
            rl2_vals = []
            for si, sv in results[l]["offset"]["per_step"].items():
                if sv["rel_l2"]["mean"] is not None:
                    rl2_vals.append(sv["rel_l2"]["mean"])
            rl2_mean = np.mean(rl2_vals) if rl2_vals else None
            np_ = results[l]["raw"]["n_compared_pairs"]

            cr_s  = f"{cr['mean']:10.4f}" if cr['mean'] is not None else f"{'n/a':>10}"
            co_s  = f"{co['mean']:12.4f}" if co['mean'] is not None else f"{'n/a':>12}"
            hc_s  = f"{hc['mean']:10.4f}" if hc['mean'] is not None else f"{'n/a':>10}"
            cka_s = f"{cka:10.4f}" if not np.isnan(cka) else f"{'n/a':>10}"
            rl_s  = f"{rl2_mean:11.4f}" if rl2_mean is not None else f"{'n/a':>11}"

            f.write(f"{l:>6} | {cr_s} | {co_s} | {hc_s} | "
                    f"{cka_s} | {rl_s} | {np_:>8}\n")
    print(f"✓ Table: {table_path}")

    print(f"\n✓ All outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())