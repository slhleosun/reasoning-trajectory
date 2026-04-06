#!/usr/bin/env python
"""Extract freeform CoT activations at structure-specific positions.

Single forward pass approach — no re-generation needed.

Why this works:
  In a decoder-only transformer with causal attention, the hidden state at
  position t depends ONLY on tokens 0..t (everything after is masked out).
  This means a single forward pass on tokenize(prompt + reasoning_text)
  produces IDENTICAL hidden states at every position as autoregressive
  generation would.  This is the same property that makes KV caching work:
  earlier positions' activations never change when new tokens are appended.

Pipeline per question:
  1. From saved JSON:  get produced_text → find reasoning boundary (text-level)
  2. tokenize(prompt + reasoning_text) → full_ids, split at prompt_len
  3. Decode gen_ids → reasoning_decoded  (self-consistent with token indices)
  4. Classify structure on reasoning_decoded
  5. Build char→token map, find structure-specific marker positions
  6. model(full_ids, output_hidden_states=True) — ONE forward pass
  7. Index hidden states at marker positions + prompt-end + answer boundary

Extraction conventions:
  Index 1 (ALL types):  last prompt token  (model state before generation)
  step_x:               token BEFORE each "Step N:" marker
  numbered_list:        token BEFORE each "1." / "1)" marker
  double_newline:       token AT the \\n\\n boundary
  single_newline:       token AT each \\n  (excluding \\n\\n pairs)
  single_block:         token AT each sentence-ending punctuation (.!?)
  Hash / answer:        last token of reasoning_text

Output NPZ (compatible with existing probe/predictor scripts):
  step_activations[layer]     : [n_total_markers, hidden_dim]
  hash_activations[layer]     : [n_questions, hidden_dim]
  step_numbers[layer]         : [n_total_markers]  (1=prompt, 2+=markers)
  question_ids_{step,hash}    : IDs
  is_correct_{step,hash}      : bool
  structure_types_{step,hash} : int (0–4, see STRUCTURE_CODES)
  num_layers, hidden_dim, stats, structure_codes

Usage:
  # Single GPU
  python extract_freeform_activations.py \\
      --master-json output/freeform/gsm8k_freeform.json

  # Sharded (called by multi_gpu_extract_freeform.py)
  python extract_freeform_activations.py \\
      --master-json output/freeform/gsm8k_freeform.json \\
      --shard-id 0 --num-shards 8
"""

# ── Force fully-offline HF ────────────────────────────────────────────
import os
os.environ["HF_HUB_OFFLINE"]           = "1"
os.environ["HF_DATASETS_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"]      = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, json, re, argparse
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ======================================================================
# Constants
# ======================================================================

MINIMAL_COT_PROMPT = (
    "Solve the following problem. Think step by step.\n\n"
    "Question: {question}\n\n"
    "Solution:\n"
)

STRUCTURE_CODES = {
    "step_x": 0, "numbered_list": 1, "double_newline": 2,
    "single_newline": 3, "single_block": 4,
}
STRUCTURE_NAMES = {v: k for k, v in STRUCTURE_CODES.items()}

# ======================================================================
# Config helpers
# ======================================================================

def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "paths.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_model_path(config: dict, key: str) -> str:
    models = config.get("models", {})
    if key not in models:
        raise ValueError(f"'{key}' not in config. Available: {list(models.keys())}")
    return models[key]["path"]

# ======================================================================
# Structure classification  (exclusive, first-match wins)
# ======================================================================

_STRUCT_PATTERNS = [
    ("step_x",         re.compile(r"(?m)^Step\s+\d+\s*[:.]\s", re.I)),
    ("numbered_list",  re.compile(r"(?m)^\d+[.)]\s")),
    ("double_newline", re.compile(r"\n\n")),
    ("single_newline", re.compile(r"\n")),
]
_STRUCT_THRESH = {"step_x": 2, "numbered_list": 2,
                  "double_newline": 2, "single_newline": 2}

def classify_structure(text: str) -> str:
    for name, pat in _STRUCT_PATTERNS:
        if len(pat.findall(text)) >= _STRUCT_THRESH[name]:
            return name
    return "single_block"

# ======================================================================
# Text-level reasoning boundary
# ======================================================================

def find_reasoning_boundary(txt: str, gold: str = None) -> Tuple[int, str]:
    """Return (char_position, method) where reasoning ends."""
    cands = []

    def _bnd(pos):
        """Walk back to nearest line or sentence start."""
        ls = txt.rfind("\n", 0, pos)
        if ls != -1:
            return ls + 1
        ss = max(txt.rfind(". ", 0, pos),
                 txt.rfind("? ", 0, pos),
                 txt.rfind("! ", 0, pos))
        return ss + 2 if ss != -1 else pos

    if (m := re.search(r"####", txt)):
        cands.append((_bnd(m.start()), "####"))
    if (m := re.search(r"(?i)the\s+final\s+answer\s+is", txt)):
        cands.append((_bnd(m.start()), "the_final_answer_is"))
    if (m := re.search(r"(?i)\bthe\s+answer\s+is", txt)):
        cands.append((_bnd(m.start()), "the_answer_is"))

    bi = txt.find("boxed{")
    if bi > 0 and txt[bi - 1] == "\\":
        bs = _bnd(bi - 1)
        lt = txt[bs:bi - 1]
        if re.match(r"\s*Step\s+\d+", lt, re.I) and \
           re.search(r"(?i)(therefore|thus|so,|hence)", lt):
            cands.append((bs, "boxed_conclusion"))
        else:
            cands.append((bs, "boxed_standalone"))

    if (m := re.search(r"(?i)\bAnswer\s*[:=]\s*\$?[\d]", txt)):
        cands.append((_bnd(m.start()), "answer_colon"))

    if gold and str(gold).strip():
        try:
            pat = (r"(?i)(?:therefore|thus|so|hence|in total|altogether"
                   r"|total is|total =)\s*[^0-9]*?"
                   + re.escape(str(gold)) + r"(?:\b|[^0-9])")
            if (m := re.search(pat, txt)):
                cands.append((_bnd(m.start()), "conclusion_with_gold"))
        except re.error:
            pass

    if not cands:
        return len(txt), "no_marker"
    cands.sort(key=lambda x: x[0])
    return cands[0]

# ======================================================================
# Char ↔ Token alignment
# ======================================================================

def build_char_offsets(token_ids: List[int], tokenizer) -> List[int]:
    """char_offsets[i] = len(decode(tokens[:i+1])).

    Used so that char_to_token is a simple scan: the first index i where
    char_offsets[i] >= target_char_pos is the token covering that char.
    """
    out = []
    for i in range(len(token_ids)):
        out.append(len(tokenizer.decode(token_ids[:i + 1],
                                        skip_special_tokens=True)))
    return out

def char_to_token(char_pos: int, offsets: List[int]) -> int:
    for i, cum in enumerate(offsets):
        if cum >= char_pos:
            return i
    return len(offsets) - 1

# ======================================================================
# Position finders  (gen-token-index space, run on reasoning_decoded)
# ======================================================================

def _pos_step_x(text, offsets):
    """Token BEFORE each 'Step N:' marker."""
    return [max(char_to_token(m.start(), offsets) - 1, 0)
            for m in re.finditer(r"(?m)^Step\s+\d+\s*[:.]\s", text, re.I)]

def _pos_numbered(text, offsets):
    """Token BEFORE each numbered marker."""
    return [max(char_to_token(m.start(), offsets) - 1, 0)
            for m in re.finditer(r"(?m)^\d+[.)]\s", text)]

def _pos_double_nl(text, offsets):
    r"""Token AT each '\n\n' (at the second \n)."""
    out, s = [], 0
    while (idx := text.find("\n\n", s)) != -1:
        out.append(char_to_token(idx + 1, offsets))
        s = idx + 2
    return out

def _pos_single_nl(text, offsets):
    r"""Token AT each lone '\n' (skip those in \n\n pairs)."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\n":
            if i + 1 < len(text) and text[i + 1] == "\n":
                i += 2; continue
            out.append(char_to_token(i, offsets))
        i += 1
    return out

def _pos_sentences(text, offsets):
    """Token AT each sentence-ending punctuation (.!?)."""
    return [char_to_token(m.start(), offsets)
            for m in re.finditer(r"[.!?](?=\s|$)", text)]

FINDERS = {
    "step_x":         _pos_step_x,
    "numbered_list":  _pos_numbered,
    "double_newline": _pos_double_nl,
    "single_newline": _pos_single_nl,
    "single_block":   _pos_sentences,
}

# ======================================================================
# Hidden-state extraction (single forward pass)
# ======================================================================

@torch.no_grad()
def extract_hidden_states(
    full_ids: List[int],
    positions: List[int],
    model,
    num_layers: int,
) -> Optional[np.ndarray]:
    """Forward pass → read hidden_states at given absolute positions.

    Returns [n_pos, num_layers, hidden_dim] float32, or None.
    """
    device = next(model.parameters()).device
    ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = ids.shape[1]

    out = model(ids, use_cache=False, return_dict=True,
                output_hidden_states=True)
    # out.hidden_states: tuple(num_layers+1) of [1, seq_len, D]
    # index 0 = embedding, 1..num_layers = transformer layer outputs

    results = []
    for p in positions:
        if not (0 <= p < seq_len):
            continue
        layers = torch.stack(
            [out.hidden_states[l + 1][0, p, :] for l in range(num_layers)],
            dim=0,
        )
        results.append(layers.float().cpu().numpy())

    del out, ids
    torch.cuda.empty_cache()
    return np.stack(results) if results else None

# ======================================================================
# Prompt-length finder (handles BPE boundary merging)
# ======================================================================

def find_prompt_length(
    full_ids: List[int],
    prompt_ids: List[int],
    tokenizer,
) -> Tuple[int, bool]:
    """Return (prompt_len_in_full_ids, was_adjusted).

    Normally encode(prompt+text)[:len(encode(prompt))] == encode(prompt).
    When BPE merges tokens across the boundary, we fall back to incremental
    decoding to find the token where the prompt text ends.
    """
    n = len(prompt_ids)
    if full_ids[:n] == prompt_ids:
        return n, False

    # BPE boundary merge — find split by decoded length
    target = len(tokenizer.decode(prompt_ids, skip_special_tokens=True))
    for k in range(len(full_ids)):
        if len(tokenizer.decode(full_ids[:k + 1],
                                skip_special_tokens=True)) >= target:
            return k + 1, True
    return n, False          # fallback: use original length

# ======================================================================
# Main
# ======================================================================

def main():
    try:
        cfg = load_config()
        default_model = get_model_path(cfg, "llama-3.1-8b-instruct")
    except Exception:
        default_model = None

    ap = argparse.ArgumentParser(
        description="Single-pass freeform CoT activation extraction")
    ap.add_argument("--master-json",  type=Path, required=True)
    ap.add_argument("--model",        type=str,  default=default_model)
    ap.add_argument("--output",       type=Path,
                    default=Path("output/freeform/freeform_structure_activations.npz"))
    ap.add_argument("--min-positions",      type=int, default=2)
    ap.add_argument("--max-positions",      type=int, default=20)
    ap.add_argument("--min-reasoning-chars", type=int, default=50)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--shard-id",  type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if (args.shard_id is None) != (args.num_shards is None):
        ap.error("--shard-id and --num-shards must both be set or both omitted")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tag = f" (SHARD {args.shard_id}/{args.num_shards})" \
          if args.shard_id is not None else ""
    print(f"\n{'='*90}")
    print(f"FREEFORM STRUCTURE ACTIVATION EXTRACTION (SINGLE-PASS){tag}")
    print(f"{'='*90}")
    print(f"Master JSON:     {args.master_json}")
    print(f"Model:           {args.model}")
    print(f"Min/Max markers: {args.min_positions} / {args.max_positions}")
    print(f"Min reasoning:   {args.min_reasoning_chars} chars")
    print(f"{'='*90}\n")

    # ── Load data ─────────────────────────────────────────────────────
    with open(args.master_json) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} questions")

    if args.shard_id is not None:
        shard_data = [d for i, d in enumerate(all_data)
                      if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_data)} questions")
    else:
        shard_data = all_data

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model_path = args.model
    if not Path(model_path).exists():
        cfg = load_config()
        model_path = get_model_path(cfg, args.model)
        print(f"  Resolved → {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True, local_files_only=True,
    ).eval()

    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  {num_layers} layers, hidden_dim={hidden_dim}\n")

    # ── Accumulators ──────────────────────────────────────────────────
    step_acts  = [[] for _ in range(num_layers)]
    hash_acts  = [[] for _ in range(num_layers)]
    step_nums  = [[] for _ in range(num_layers)]
    qids_step  = [[] for _ in range(num_layers)]
    qids_hash  = [[] for _ in range(num_layers)]
    corr_step  = [[] for _ in range(num_layers)]
    corr_hash  = [[] for _ in range(num_layers)]
    stype_step = [[] for _ in range(num_layers)]
    stype_hash = [[] for _ in range(num_layers)]
    S = defaultdict(int)                       # stats

    # ── Main loop ─────────────────────────────────────────────────────
    it = tqdm(shard_data, desc="Extracting", ncols=120, disable=args.verbose)
    for d in it:
        qid, question = d["question_id"], d["question"]
        is_correct    = d["is_correct"]
        gold          = d.get("gold_answer")
        S["total"] += 1

        # ── 1. Reasoning boundary (text level) ───────────────────────
        bnd_pos, _ = find_reasoning_boundary(d["produced_text"], gold)
        reasoning_text = d["produced_text"][:bnd_pos].rstrip()
        if len(reasoning_text) < args.min_reasoning_chars:
            S["skip_short"] += 1
            continue

        # ── 2. Tokenize  prompt + reasoning_text ─────────────────────
        prompt     = MINIMAL_COT_PROMPT.format(question=question)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_ids   = tokenizer.encode(prompt + reasoning_text,
                                      add_special_tokens=True)

        prompt_len, adjusted = find_prompt_length(
            full_ids, prompt_ids, tokenizer)
        if adjusted:
            S["prompt_adjusted"] += 1

        gen_ids = full_ids[prompt_len:]
        if len(gen_ids) < 5:
            S["skip_few_tokens"] += 1
            continue

        # ── 3. Decode gen_ids → self-consistent reasoning text ────────
        #    We run ALL downstream logic on this decoded text, so
        #    char positions and token indices are exactly aligned.
        reasoning_decoded = tokenizer.decode(gen_ids,
                                             skip_special_tokens=True)

        # ── 4. Classify structure (once, on the decoded text) ─────────
        structure   = classify_structure(reasoning_decoded)
        struct_code = STRUCTURE_CODES[structure]
        S[f"struct_{structure}"] += 1

        # ── 5. Build char→token map ──────────────────────────────────
        offsets = build_char_offsets(gen_ids, tokenizer)
        # Safety: offsets must be monotonically non-decreasing
        assert all(offsets[i] <= offsets[i + 1]
                   for i in range(len(offsets) - 1)), \
            f"Q{qid}: char_offsets non-monotonic"

        # ── 6. Find structure-specific marker positions ───────────────
        markers = FINDERS[structure](reasoning_decoded, offsets)
        # Safety: all positions in [0, len(gen_ids))
        assert all(0 <= p < len(gen_ids) for p in markers), \
            f"Q{qid}: marker out of bounds (max={len(gen_ids)-1})"

        if len(markers) < args.min_positions:
            S["skip_few_markers"] += 1
            continue
        if len(markers) > args.max_positions:
            markers = markers[:args.max_positions]
            S["truncated"] += 1

        # ── 7. Build absolute extraction positions ────────────────────
        #    • pos 0 in list  = prompt_end  (last prompt token)
        #    • pos 1..N       = structure markers
        #    • final entry    = answer boundary (last reasoning token)
        prompt_end = prompt_len - 1
        step_abs   = [prompt_end]
        for gp in markers:
            ap = prompt_len + gp
            step_abs.append(min(ap, len(full_ids) - 1))

        # Deduplicate preserving order
        seen, uniq = set(), []
        for p in step_abs:
            if p not in seen:
                seen.add(p); uniq.append(p)
        step_abs = uniq

        answer_abs = len(full_ids) - 1           # last reasoning token
        all_pos    = step_abs + [answer_abs]

        # Safety: bounds
        assert all(0 <= p < len(full_ids) for p in all_pos), \
            f"Q{qid}: abs pos out of range"
        # Safety: step positions monotonically non-decreasing
        for k in range(1, len(step_abs)):
            if step_abs[k] < step_abs[k - 1]:
                S["non_monotonic"] += 1
                break

        if args.verbose:
            print(f"  Q{qid} | {structure:15s} | "
                  f"{len(step_abs)} steps + 1 hash | "
                  f"{len(gen_ids)} gen toks | seq={len(full_ids)}")

        # ── 8. SINGLE FORWARD PASS ───────────────────────────────────
        try:
            acts = extract_hidden_states(full_ids, all_pos, model,
                                         num_layers)
        except Exception as e:
            if args.verbose:
                print(f"    ✗ forward failed: {e}")
            S["fail_forward"] += 1
            continue

        if acts is None or acts.shape[0] != len(all_pos):
            S["fail_extract"] += 1
            continue

        # ── 9. Store per-layer ────────────────────────────────────────
        n_s = len(step_abs)
        sa, ha = acts[:n_s], acts[n_s:]          # split step / hash

        for si in range(n_s):
            for li in range(num_layers):
                step_acts[li].append(sa[si, li])  # [hidden_dim]
                step_nums[li].append(si + 1)
                qids_step[li].append(qid)
                corr_step[li].append(is_correct)
                stype_step[li].append(struct_code)

        for li in range(num_layers):
            hash_acts[li].append(ha[0, li])
            qids_hash[li].append(qid)
            corr_hash[li].append(is_correct)
            stype_hash[li].append(struct_code)

        S["ok"] += 1
        S["correct" if is_correct else "incorrect"] += 1

        if not args.verbose:
            tot = S["correct"] + S["incorrect"]
            it.set_postfix(ok=S["ok"],
                           acc=f"{S['correct']/tot:.0%}" if tot else "—",
                           s=structure[:6])

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*90}\nSTATISTICS\n{'='*90}")
    for k in ["total", "ok", "skip_short", "skip_few_tokens",
              "skip_few_markers", "truncated", "prompt_adjusted",
              "non_monotonic", "fail_forward", "fail_extract"]:
        print(f"  {k:30s}: {S.get(k, 0)}")
    tot = S["correct"] + S["incorrect"]
    if tot:
        print(f"  {'accuracy':30s}: {S['correct']}/{tot} "
              f"({S['correct']/tot:.1%})")
    print(f"\n  Structure breakdown:")
    for nm in STRUCTURE_CODES:
        print(f"    {nm:20s}: {S.get(f'struct_{nm}', 0)}")
    print(f"{'='*90}\n")

    # ── Save NPZ ─────────────────────────────────────────────────────
    def _arr(lst, nl, dt=np.float32, d2=None):
        r = []
        for i in range(nl):
            if lst[i]:
                r.append(np.stack(lst[i]).astype(dt))
            else:
                r.append(np.empty((0, d2), dtype=dt) if d2
                         else np.array([], dtype=dt))
        return r

    m_sa = _arr(step_acts,  num_layers, np.float32, hidden_dim)
    m_ha = _arr(hash_acts,  num_layers, np.float32, hidden_dim)
    m_sn = _arr(step_nums,  num_layers, np.int32)
    m_qs = _arr(qids_step,  num_layers, np.int64)
    m_qh = _arr(qids_hash,  num_layers, np.int64)
    m_cs = _arr(corr_step,  num_layers, np.bool_)
    m_ch = _arr(corr_hash,  num_layers, np.bool_)
    m_ss = _arr(stype_step, num_layers, np.int32)
    m_sh = _arr(stype_hash, num_layers, np.int32)

    out_dir = (args.output.parent / f"shard_{args.shard_id}"
               if args.shard_id is not None else args.output.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output.name

    print(f"Saving → {out_path}")
    np.savez(
        out_path,
        steering_vectors      = np.zeros((num_layers, hidden_dim), np.float32),
        step_activations      = np.array(m_sa, dtype=object),
        hash_activations      = np.array(m_ha, dtype=object),
        step_numbers          = np.array(m_sn, dtype=object),
        question_ids_step     = np.array(m_qs, dtype=object),
        question_ids_hash     = np.array(m_qh, dtype=object),
        is_correct_step       = np.array(m_cs, dtype=object),
        is_correct_hash       = np.array(m_ch, dtype=object),
        structure_types_step  = np.array(m_ss, dtype=object),
        structure_types_hash  = np.array(m_sh, dtype=object),
        num_layers  = num_layers,
        hidden_dim  = hidden_dim,
        stats       = json.dumps(dict(S)),
        structure_codes = json.dumps(STRUCTURE_CODES),
    )

    mb = out_path.stat().st_size / 1024**2
    ns = len(m_sa[0]) if len(m_sa[0]) else 0
    nh = len(m_ha[0]) if len(m_ha[0]) else 0
    print(f"  ✓ {mb:.1f} MB | {ns} step acts, {nh} hash acts (layer 0)")
    if ns:
        u, c = np.unique(m_sn[0], return_counts=True)
        print(f"  Step dist: {dict(zip(u.tolist(), c.tolist()))}")
    if nh:
        u, c = np.unique(m_sh[0], return_counts=True)
        print(f"  Struct: {{{', '.join(STRUCTURE_NAMES[int(s)]+':'+str(int(v)) for s,v in zip(u,c))}}}")

    with open(out_dir / f"{args.output.stem}_stats.json", "w") as f:
        json.dump(dict(S), f, indent=2)

    print("\n✓ Done!")

if __name__ == "__main__":
    sys.exit(main())