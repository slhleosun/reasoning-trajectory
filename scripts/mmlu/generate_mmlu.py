#!/usr/bin/env python
"""MMLU Generation with Step-enforced CoT prompting.

Generates responses for MMLU multiple-choice questions using the same
Step-enforced CoT format as GSM8K/MATH-500, so that downstream activation
extraction (collect_steering_vectors.py), probes, and predictors all work
unmodified.

Per-question JSON output format (compatible with collect_steering_vectors.py):
  {
    "question_id": int,
    "question": str,          # original question text
    "subject": str,           # MMLU subject
    "choices": ["A_text", ...],
    "gold_answer": str,       # "A" | "B" | "C" | "D"
    "produced_text": str,     # model's full generated text
    "produced_answer": str,   # extracted answer letter
    "is_correct": bool,
    "full_seq_ids": [int],    # full prompt+generation token IDs
    "dp1_idx": int,           # index where generation begins
    "dp2_idx": int|null,      # index of answer token (after ####)
    "prompt": str,            # the formatted prompt
  }

Usage:
    # Single GPU
    python scripts/mmlu/generate_mmlu.py \\
        --parquet /path/to/validation-00000-of-00001.parquet \\
        --output-dir output/mmlu/generations

    # Sharded (launched by multi_gpu_generate_mmlu.py)
    python scripts/mmlu/generate_mmlu.py \\
        --parquet /path/to/parquet \\
        --output-dir output/mmlu/generations \\
        --shard-id 0 --num-shards 8
"""

# ── Force offline before any HF imports ──────────────────────────────
import os
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import re
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ======================================================================
# Constants
# ======================================================================

CHOICE_LETTERS = ["A", "B", "C", "D"]
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


# ======================================================================
# Prompt Templates
# ======================================================================

def format_mmlu_cot_prompt(question: str, choices: list, subject: str) -> str:
    """Format MMLU question with Step-enforced CoT prompt.

    Uses the same "Step [step_number]:" + "####" convention as GSM8K/MATH-500
    so that collect_steering_vectors.py can extract activations at the same
    structural markers.
    """
    subject_formatted = subject.replace("_", " ")

    choice_lines = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip(CHOICE_LETTERS, choices)
    )

    prompt = (
        f'You are a helpful assistant that solves problems step by step '
        f'with each step signified by "Step [step_number]: ".\n'
        f'Always provide your final answer after #### at the end.\n'
        f'\n'
        f'The following is a multiple choice question about {subject_formatted}.\n'
        f'\n'
        f'Question: {question}\n'
        f'{choice_lines}\n'
        f'\n'
        f'Please reason through this step by step, putting each step after '
        f'"Step [step_number]: ". Then provide ONLY the answer letter '
        f'(A, B, C, or D) after ####.\n'
        f'\n'
        f'Solution:\n\n'
    )
    return prompt


# ======================================================================
# Answer Extraction — the critical piece
# ======================================================================

def extract_mcqa_answer(text: str) -> Optional[str]:
    """Robustly extract A/B/C/D answer from model output.

    Strategy (ordered by specificity):
      1. After #### marker: look for first A/B/C/D
      2. "the answer is (X)" / "the answer is X" patterns
      3. "Answer: X" pattern
      4. Isolated letter at very end of text
      5. Last mentioned A/B/C/D in a choice context

    Returns:
        "A", "B", "C", or "D", or None if extraction fails
    """
    if not text or not text.strip():
        return None

    valid = set("ABCD")

    # ── 1. After #### marker ─────────────────────────────────────────
    hash_idx = text.find("####")
    if hash_idx != -1:
        after_hash = text[hash_idx + 4:].strip()
        # Look for first A/B/C/D (possibly with punctuation like "B." or "(B)")
        m = re.search(r'\b([A-D])\b', after_hash)
        if m and m.group(1) in valid:
            return m.group(1)
        # Try without word boundary (might be "####B")
        m = re.search(r'([A-D])', after_hash[:20])
        if m and m.group(1) in valid:
            return m.group(1)

    # ── 2. "the answer is" patterns ──────────────────────────────────
    patterns_answer_is = [
        r'[Tt]he\s+(?:correct\s+)?answer\s+is\s*[\(\[]?\s*([A-D])\s*[\)\]]?',
        r'[Tt]he\s+(?:correct\s+)?(?:option|choice)\s+is\s*[\(\[]?\s*([A-D])\s*[\)\]]?',
        r'[Aa]nswer\s*:\s*[\(\[]?\s*([A-D])\s*[\)\]]?',
        r'[Cc]orrect\s+answer\s*:\s*[\(\[]?\s*([A-D])\s*[\)\]]?',
    ]
    for pattern in patterns_answer_is:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Take the last match (conclusion usually at end)
            letter = matches[-1].group(1)
            if letter in valid:
                return letter

    # ── 3. "I would choose X" / "I'll go with X" ────────────────────
    patterns_choose = [
        r'(?:choose|select|pick|go\s+with)\s+[\(\[]?\s*([A-D])\s*[\)\]]?',
        r'(?:choose|select|pick|go\s+with)\s+(?:option|choice)\s+[\(\[]?\s*([A-D])\s*[\)\]]?',
    ]
    for pattern in patterns_choose:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            letter = matches[-1].group(1).upper()
            if letter in valid:
                return letter

    # ── 4. Bold/emphasized letter: **B** or *B* ─────────────────────
    m = re.search(r'\*\*([A-D])\*\*', text)
    if m and m.group(1) in valid:
        return m.group(1)

    # ── 5. Isolated letter at end of text ────────────────────────────
    # Look at last 30 chars for a standalone letter
    tail = text.strip()[-30:]
    # Match letter at end: "... B" or "... B." or "...(B)"
    m = re.search(r'[\s\n\(\[]([A-D])[\s\.\)\]]*$', tail)
    if m and m.group(1) in valid:
        return m.group(1)
    # Just the letter alone
    m = re.match(r'^([A-D])[\s\.\)\]]*$', tail.strip())
    if m and m.group(1) in valid:
        return m.group(1)

    # ── 6. Fallback: last A/B/C/D in context "X." at start of line ──
    # Matches lines starting with "A." "B." etc. — find last one mentioned
    m_all = re.findall(r'(?:^|\n)\s*([A-D])[\.\)]', text)
    if m_all:
        letter = m_all[-1]
        if letter in valid:
            return letter

    # ── 7. Ultimate fallback: last standalone letter ─────────────────
    m_all = re.findall(r'\b([A-D])\b', text)
    if m_all:
        return m_all[-1]

    return None


def evaluate_mcqa_answer(predicted: Optional[str], gold: str) -> bool:
    """Compare predicted answer letter to gold answer letter.

    Both should be single letters A/B/C/D. Case-insensitive.
    """
    if predicted is None or gold is None:
        return False
    return predicted.strip().upper() == gold.strip().upper()


# ======================================================================
# Generation
# ======================================================================

def generate_greedy(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
) -> Tuple[str, List[int], int]:
    """Generate greedily. Returns (text, full_seq_ids, prompt_length)."""
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_seq_ids = outputs[0].tolist()
    generated_text = tokenizer.decode(
        full_seq_ids[prompt_len:], skip_special_tokens=True
    )

    del outputs, input_ids
    torch.cuda.empty_cache()

    return generated_text, full_seq_ids, prompt_len


def find_dp2_idx(
    full_seq_ids: List[int],
    tokenizer,
    prompt_len: int,
    produced_text: str,
) -> Optional[int]:
    """Find dp2_idx: position of the answer token after #### in full_seq_ids.

    This allows collect_steering_vectors.py to locate the answer boundary.
    """
    hash_pos = produced_text.find("####")
    if hash_pos == -1:
        return None

    # The text after #### is the answer portion
    text_before_hash = produced_text[:hash_pos]
    # Tokenize just the text before hash to find how many gen tokens
    gen_tokens_before = tokenizer.encode(
        text_before_hash, add_special_tokens=False
    )
    # dp2 = prompt_len + tokens_before_hash + tokens_for_"####"
    # "####" typically tokenizes to 1-3 tokens
    hash_tokens = tokenizer.encode("####", add_special_tokens=False)
    dp2 = prompt_len + len(gen_tokens_before) + len(hash_tokens)

    if dp2 < len(full_seq_ids):
        return dp2
    return None


# ======================================================================
# Dataset Loading
# ======================================================================

def load_mmlu_parquet(parquet_path: Path) -> List[Dict]:
    """Load MMLU from parquet, return list of dicts.

    Each dict: {question_id, question, subject, choices, gold_answer, gold_idx}
    """
    df = pd.read_parquet(parquet_path)

    samples = []
    for idx, row in df.iterrows():
        choices = list(row["choices"])
        gold_idx = int(row["answer"])
        gold_letter = IDX_TO_LETTER[gold_idx]

        samples.append({
            "question_id": idx,
            "question": row["question"],
            "subject": row["subject"],
            "choices": choices,
            "gold_answer": gold_letter,
            "gold_idx": gold_idx,
        })

    return samples


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="MMLU CoT Generation")
    ap.add_argument("--parquet", type=Path, required=True,
                    help="Path to MMLU parquet file")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for per-question JSONs")
    ap.add_argument("--model", type=str, default=None,
                    help="Model path (default: from config)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if (args.shard_id is None) != (args.num_shards is None):
        ap.error("--shard-id and --num-shards must both be set or both omitted")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f" (SHARD {args.shard_id}/{args.num_shards})" \
          if args.shard_id is not None else ""
    print(f"\n{'='*90}")
    print(f"MMLU COT GENERATION{tag}")
    print(f"{'='*90}")
    print(f"Parquet:     {args.parquet}")
    print(f"Output:      {args.output_dir}")
    print(f"Max tokens:  {args.max_new_tokens}")
    print(f"{'='*90}\n")

    # ── Load data ─────────────────────────────────────────────────────
    all_samples = load_mmlu_parquet(args.parquet)
    print(f"Loaded {len(all_samples)} MMLU questions "
          f"({pd.read_parquet(args.parquet)['subject'].nunique()} subjects)")

    if args.shard_id is not None:
        shard_samples = [s for i, s in enumerate(all_samples)
                         if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: "
              f"{len(shard_samples)} questions")
    else:
        shard_samples = all_samples

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.model is None:
        try:
            from src import load_config
            from src.config import get_model_path_from_config
            config = load_config()
            model_path = get_model_path_from_config(config, "llama-3.1-8b-instruct")
        except Exception:
            ap.error("--model is required (could not load from config)")
    else:
        model_path = args.model

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  ✓ Model loaded on {next(model.parameters()).device}")

    # ── Output directory ──────────────────────────────────────────────
    if args.shard_id is not None:
        out_dir = args.output_dir / f"shard_{args.shard_id}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate ──────────────────────────────────────────────────────
    stats = {
        "total": 0, "success": 0, "correct": 0,
        "extract_fail": 0, "has_hash": 0, "has_steps": 0,
    }

    iterator = tqdm(shard_samples, desc="Generating", ncols=120,
                    disable=args.verbose)

    for sample in iterator:
        qid = sample["question_id"]
        stats["total"] += 1

        # Format prompt
        prompt = format_mmlu_cot_prompt(
            sample["question"], sample["choices"], sample["subject"]
        )

        # Generate
        try:
            gen_text, full_seq_ids, prompt_len = generate_greedy(
                prompt, model, tokenizer, args.max_new_tokens
            )
        except Exception as e:
            if args.verbose:
                print(f"  Q{qid}: generation error: {e}")
            continue

        # Extract answer
        predicted = extract_mcqa_answer(gen_text)
        is_correct = evaluate_mcqa_answer(predicted, sample["gold_answer"])

        if predicted is None:
            stats["extract_fail"] += 1
        if is_correct:
            stats["correct"] += 1
        if "####" in gen_text:
            stats["has_hash"] += 1
        if re.search(r'Step\s+\d+', gen_text):
            stats["has_steps"] += 1

        stats["success"] += 1

        # Find dp2_idx
        dp2 = find_dp2_idx(full_seq_ids, tokenizer, prompt_len, gen_text)

        # Save JSON
        q_data = {
            "question_id": qid,
            "question": sample["question"],
            "subject": sample["subject"],
            "choices": sample["choices"],
            "gold_answer": sample["gold_answer"],
            "gold_idx": sample["gold_idx"],
            "produced_text": gen_text,
            "produced_answer": predicted,
            "is_correct": is_correct,
            "full_seq_ids": full_seq_ids,
            "dp1_idx": prompt_len,
            "dp2_idx": dp2,
            "prompt": prompt,
            "num_generated_tokens": len(full_seq_ids) - prompt_len,
        }

        json_path = out_dir / f"mmlu_{qid:04d}.json"
        with open(json_path, "w") as f:
            json.dump(q_data, f)

        if args.verbose:
            print(f"  Q{qid} [{sample['subject']}]: "
                  f"pred={predicted} gold={sample['gold_answer']} "
                  f"{'✓' if is_correct else '✗'}")

        # Update progress
        if not args.verbose:
            acc = stats["correct"] / stats["success"] if stats["success"] else 0
            iterator.set_postfix({
                "acc": f"{acc:.1%}",
                "hash": f"{stats['has_hash']}/{stats['success']}",
                "ext_fail": stats["extract_fail"],
            })

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("GENERATION SUMMARY")
    print(f"{'='*90}")
    for k, v in sorted(stats.items()):
        print(f"  {k:20s}: {v}")
    if stats["success"] > 0:
        acc = stats["correct"] / stats["success"]
        ext_rate = 1.0 - stats["extract_fail"] / stats["success"]
        hash_rate = stats["has_hash"] / stats["success"]
        step_rate = stats["has_steps"] / stats["success"]
        print(f"\n  accuracy:            {acc:.1%}")
        print(f"  extraction_rate:     {ext_rate:.1%}")
        print(f"  hash_marker_rate:    {hash_rate:.1%}")
        print(f"  step_marker_rate:    {step_rate:.1%}")

    # Save stats
    stats_path = out_dir / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")
    print(f"  JSONs saved to: {out_dir}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
