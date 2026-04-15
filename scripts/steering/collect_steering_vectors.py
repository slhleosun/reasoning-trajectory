#!/usr/bin/env python
"""Collect steering vectors by computing Step->Hash activation differences

For each question:
1. Collect resid_post activations before each "Step X:" and before "####"
2. Compute differences: act_#### - act_step_x for each step
3. Aggregate across questions using trimmed mean (drop top/bottom 10% by norm)

Output:
- Steering vectors: One per layer [num_layers, hidden_dim]
- Raw activations: For t-SNE visualization

Usage:
    python scripts/collect_steering_vectors.py --num-questions 100
"""

import sys
import json
import argparse
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from scipy.stats import trim_mean

# Add src to path
# Go up 3 levels: scripts/steering/ → scripts/ → project_root/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import evaluate_answer, evaluate_math_answer


def extract_answer_after_hash(text: str) -> Optional[str]:
    """Extract answer after #### marker - ROBUST VERSION

    Rules (as specified):
    1. Find first #### occurrence in generated response
    2. Look for first number after ####
    3. If number has spaces between digits (like "118 000"), concatenate them
    4. If first number is followed by operators (+-*/), look for first number after "=" instead
    5. If number after = is still followed by operators, recursively look for next =
    6. FALLBACK: If no #### found OR extraction fails, extract last number from entire response
    """
    if not text:
        return None

    # Helper function: Extract last number from text as fallback
    def extract_last_number(txt: str) -> Optional[str]:
        all_numbers = re.findall(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', txt)
        if all_numbers:
            return all_numbers[-1].replace(' ', '').replace(',', '')
        return None

    if "####" not in text:
        return extract_last_number(text)

    parts = text.split("####", 1)
    if len(parts) < 2:
        return extract_last_number(text)

    answer_text = parts[1].strip()
    answer_text = re.sub(r'^[\$£€¥₹]+\s*', '', answer_text)

    first_number_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', answer_text)
    if not first_number_match:
        return extract_last_number(text)

    first_number_end = first_number_match.end()
    after_number = answer_text[first_number_end:].lstrip()

    if after_number and after_number[0] in '+-*/':
        remaining_text = answer_text
        while '=' in remaining_text:
            equals_parts = remaining_text.split('=', 1)
            if len(equals_parts) < 2:
                break

            result_text = equals_parts[1].strip()
            result_text = re.sub(r'^[\$£€¥₹]+\s*', '', result_text)
            result_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', result_text)

            if result_match:
                result_number_end = result_match.end()
                after_result = result_text[result_number_end:].lstrip()

                if not after_result or after_result[0] not in '+-*/':
                    return result_match.group(1).replace(' ', '').replace(',', '')

                remaining_text = equals_parts[1]
            else:
                break

        return extract_last_number(text)

    return first_number_match.group(1).replace(' ', '').replace(',', '')


def extract_answer_after_final(text: str) -> Optional[str]:
    """Extract answer after 'Final Answer:' marker - ROBUST VERSION

    Rules (same logic as extract_answer_after_hash but with 'Final Answer:' marker):
    1. Find first 'Final Answer:' occurrence in generated response
    2. Look for first number after 'Final Answer:'
    3. If number has spaces between digits (like "118 000"), concatenate them
    4. If first number is followed by operators (+-*/), look for first number after "=" instead
    5. If number after = is still followed by operators, recursively look for next =
    6. FALLBACK: If no 'Final Answer:' found OR extraction fails, extract last number from entire response
    """
    if not text:
        return None

    # Helper function: Extract last number from text as fallback
    def extract_last_number(txt: str) -> Optional[str]:
        all_numbers = re.findall(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', txt)
        if all_numbers:
            return all_numbers[-1].replace(' ', '').replace(',', '')
        return None

    if "Final Answer:" not in text:
        return extract_last_number(text)

    parts = text.split("Final Answer:", 1)
    if len(parts) < 2:
        return extract_last_number(text)

    answer_text = parts[1].strip()
    answer_text = re.sub(r'^[\$£€¥₹]+\s*', '', answer_text)

    first_number_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', answer_text)
    if not first_number_match:
        return extract_last_number(text)

    first_number_end = first_number_match.end()
    after_number = answer_text[first_number_end:].lstrip()

    if after_number and after_number[0] in '+-*/':
        remaining_text = answer_text
        while '=' in remaining_text:
            equals_parts = remaining_text.split('=', 1)
            if len(equals_parts) < 2:
                break

            result_text = equals_parts[1].strip()
            result_text = re.sub(r'^[\$£€¥₹]+\s*', '', result_text)
            result_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', result_text)

            if result_match:
                result_number_end = result_match.end()
                after_result = result_text[result_number_end:].lstrip()

                if not after_result or after_result[0] not in '+-*/':
                    return result_match.group(1).replace(' ', '').replace(',', '')

                remaining_text = equals_parts[1]
            else:
                break

        return extract_last_number(text)

    return first_number_match.group(1).replace(' ', '').replace(',', '')


def load_config(config_path: Path = None) -> dict:
    """Load configuration from paths.yaml"""
    if config_path is None:
        # Script is at scripts/steering/collect_steering_vectors.py
        # Go up 2 levels to project root, then into config/
        script_dir = Path(__file__).parent
        config_path = script_dir.parent.parent / "config" / "paths.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_model_path_from_config(config: dict, model_key: str = "llama-3.1-8b-instruct") -> str:
    """Extract model path from config"""
    models = config.get("models", {})
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' not found in config")

    model_config = models[model_key]
    model_path = model_config.get("path")

    if not model_path:
        raise ValueError(f"Model '{model_key}' does not have 'path' field")

    return model_path


def load_merged_json(merged_dir: Path, sample_id) -> Optional[dict]:
    """Load merged JSON file for a sample

    Args:
        sample_id: Integer (GSM8K) or string (MATH-500, etc.) sample identifier
    """
    # If sample_id is a string, try it directly first
    if isinstance(sample_id, str):
        direct_path = merged_dir / f"{sample_id}.json"
        if direct_path.exists():
            with open(direct_path, "r") as f:
                return json.load(f)

    # Legacy patterns for integer IDs
    for pattern in [f"gsm8k_{sample_id}.json", f"sample_{sample_id:04d}.json" if isinstance(sample_id, int) else None]:
        if pattern is None:
            continue
        json_path = merged_dir / pattern
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
    return None


def find_step_and_hash_positions(
    produced_text: str,
    full_seq_ids: List[int],
    tokenizer,
    dp1_idx: int,
    dp2_idx: Optional[int] = None,
    use_r1_extraction: bool = False
) -> Optional[Dict]:
    """Find Step and Hash positions in existing generated text

    Args:
        produced_text: Generated text
        full_seq_ids: Full sequence of token IDs
        tokenizer: Tokenizer
        dp1_idx: Index of first generated token (after prompt)
        dp2_idx: Index of first answer token (if available)
        use_r1_extraction: If True, use R1-specific extraction (answer before ####)

    Returns:
        Dict with step_positions, hash_position (in absolute indexing)

    Note on R1 extraction:
        For R1 models, the answer appears BEFORE the #### marker.
        We use dp2_idx (first answer token) to get the activation 1 timestep BEFORE it.
        This captures the model state right before it generates the answer.
    """
    # Find #### position in text
    hash_idx_text = produced_text.find("####")
    if hash_idx_text == -1:
        return None

    # Find all "Step N:" patterns in text (before ####)
    text_before_hash = produced_text[:hash_idx_text]
    pattern = r'\bStep\s+\d+\s*[:.\s]'
    step_matches = list(re.finditer(pattern, text_before_hash, re.IGNORECASE))

    if len(step_matches) == 0:
        return None

    # Get generated portion only (after prompt)
    generated_ids = full_seq_ids[dp1_idx + 1:]

    # Find Step token positions
    step_token_positions = []
    decoded_so_far = ""
    found_steps = set()

    for token_idx in range(len(generated_ids)):
        decoded_so_far = tokenizer.decode(generated_ids[:token_idx + 1], skip_special_tokens=True)

        for match_idx, match in enumerate(step_matches):
            if match_idx in found_steps:
                continue

            pattern_text = match.group()
            if pattern_text.strip() in decoded_so_far or match.start() <= len(decoded_so_far):
                abs_token_pos = dp1_idx + 1 + token_idx
                step_token_positions.append(abs_token_pos)
                found_steps.add(match_idx)

    # Determine hash position based on extraction mode
    hash_token_position = None

    if use_r1_extraction and dp2_idx is not None:
        # R1 MODE: Extract activation 1 timestep BEFORE the first answer token
        # dp2_idx is the absolute index of the first answer token
        # We pass dp2_idx directly to extract_resid_activations, which will automatically
        # extract the hidden state BEFORE this token (it does pos - 1 internally)

        # Validate that this position is reasonable
        if dp2_idx <= dp1_idx:
            # dp2 is too early, fall back to standard #### detection
            print(f"    WARNING: dp2_idx={dp2_idx} is <= dp1_idx={dp1_idx}, falling back to #### detection")
            use_r1_extraction = False
        else:
            hash_token_position = dp2_idx
            print(f"    R1 MODE: Using dp2_idx={dp2_idx}, will extract activation at timestep {hash_token_position - 1} (before token at {hash_token_position})")

    # STANDARD/FALLBACK MODE: Find #### token position if not already set
    if hash_token_position is None:
        decoded_so_far = ""

        for token_idx in range(len(generated_ids)):
            decoded_so_far = tokenizer.decode(generated_ids[:token_idx + 1], skip_special_tokens=True)

            if "####" in decoded_so_far:
                hash_token_position = dp1_idx + 1 + token_idx
                break

    if hash_token_position is None or len(step_token_positions) == 0:
        return None

    return {
        "step_positions": step_token_positions,
        "hash_position": hash_token_position,
        "num_steps": len(step_token_positions),
        "used_r1_extraction": use_r1_extraction
    }


def find_step_and_final_positions(
    produced_text: str,
    full_seq_ids: List[int],
    tokenizer,
    dp1_idx: int
) -> Optional[Dict]:
    """Find Step and Final Answer positions in existing generated text

    Returns:
        Dict with step_positions, final_position (in absolute indexing)
    """
    # Find "Final Answer:" position in text
    final_idx_text = produced_text.find("Final Answer:")
    if final_idx_text == -1:
        return None

    # Find all "Step N:" patterns in text (before Final Answer)
    text_before_final = produced_text[:final_idx_text]
    pattern = r'\bStep\s+\d+\s*[:.\s]'
    step_matches = list(re.finditer(pattern, text_before_final, re.IGNORECASE))

    if len(step_matches) == 0:
        return None

    # Get generated portion only (after prompt)
    generated_ids = full_seq_ids[dp1_idx + 1:]

    # Find Step token positions
    step_token_positions = []
    decoded_so_far = ""
    found_steps = set()

    for token_idx in range(len(generated_ids)):
        decoded_so_far = tokenizer.decode(generated_ids[:token_idx + 1], skip_special_tokens=True)

        for match_idx, match in enumerate(step_matches):
            if match_idx in found_steps:
                continue

            pattern_text = match.group()
            if pattern_text.strip() in decoded_so_far or match.start() <= len(decoded_so_far):
                abs_token_pos = dp1_idx + 1 + token_idx
                step_token_positions.append(abs_token_pos)
                found_steps.add(match_idx)

    # Find "Final" or " Final" token position
    # Get token IDs for both "Final" and " Final"
    final_token_ids = tokenizer.encode("Final", add_special_tokens=False)
    space_final_token_ids = tokenizer.encode(" Final", add_special_tokens=False)

    # Handle both single and multi-token cases
    final_token_id = final_token_ids[0] if final_token_ids else None
    space_final_token_id = space_final_token_ids[0] if space_final_token_ids else None

    final_token_position = None
    found_final_token = False

    # Find the character position of "Final Answer:" in the produced text
    final_answer_char_pos = produced_text.find("Final Answer:")

    for token_idx in range(len(generated_ids)):
        # Check if current token is "Final" or " Final"
        current_token = generated_ids[token_idx]
        if (final_token_id is not None and current_token == final_token_id) or \
           (space_final_token_id is not None and current_token == space_final_token_id):
            # Decode up to this token to get character position
            decoded_up_to_here = tokenizer.decode(generated_ids[:token_idx + 1], skip_special_tokens=True)
            decoded_char_len = len(decoded_up_to_here)

            # Check if this "Final" token is at roughly the right position for "Final Answer:"
            # Allow some tolerance for tokenization differences
            if final_answer_char_pos >= 0 and abs(decoded_char_len - final_answer_char_pos - len("Final")) < 20:
                found_final_token = True
                final_token_position = dp1_idx + 1 + token_idx
                break

    if final_token_position is None:
        return None

    if len(step_token_positions) == 0:
        return None

    return {
        "step_positions": step_token_positions,
        "final_position": final_token_position,
        "num_steps": len(step_token_positions)
    }


@torch.no_grad()
def extract_resid_activations(
    full_seq_ids: List[int],
    positions: List[int],
    model,
    tokenizer
) -> List[np.ndarray]:
    """Extract residual stream activations (resid_post) at specific positions

    Args:
        full_seq_ids: Complete token sequence
        positions: List of positions to extract (absolute indexing)
        model: Model
        tokenizer: Tokenizer

    Returns:
        List of [num_layers, hidden_dim] arrays, one per position
    """
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    activations_list = []

    for pos in positions:
        # Get tokens up to (but not including) this position
        # We want hidden state at position BEFORE the target token
        target_pos = pos - 1
        if target_pos < 0:
            continue

        input_ids = torch.tensor(full_seq_ids[:target_pos + 1], dtype=torch.long).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True
        )

        # Extract last position from each layer (resid_post)
        layer_activations = []
        for layer_idx in range(num_layers):
            # hidden_states[layer_idx + 1] because [0] is embedding
            hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]
            layer_activations.append(hidden)

        # Stack: [num_layers, hidden_dim]
        layer_activations_stacked = torch.cat(layer_activations, dim=0).float().cpu().numpy()
        activations_list.append(layer_activations_stacked)

    return activations_list


def trimmed_mean_by_norm(vectors: List[np.ndarray], trim_proportion: float = 0.1) -> np.ndarray:
    """Compute trimmed mean of vectors, trimming by L2 norm

    Args:
        vectors: List of vectors [hidden_dim]
        trim_proportion: Proportion to trim from each tail (default 0.1 = 10%)

    Returns:
        Trimmed mean vector [hidden_dim]
    """
    if len(vectors) == 0:
        raise ValueError("Empty vector list")

    if len(vectors) == 1:
        return vectors[0]

    # Stack vectors
    vectors_array = np.stack(vectors, axis=0)  # [n_vectors, hidden_dim]

    # Compute norms
    norms = np.linalg.norm(vectors_array, axis=1)  # [n_vectors]

    # Sort by norm
    sorted_indices = np.argsort(norms)

    # Determine trim count
    n_vectors = len(vectors)
    n_trim = int(n_vectors * trim_proportion)

    # Keep middle portion
    keep_indices = sorted_indices[n_trim:-n_trim] if n_trim > 0 else sorted_indices

    # Compute mean of kept vectors
    trimmed_vectors = vectors_array[keep_indices]
    return np.mean(trimmed_vectors, axis=0)


def main():
    try:
        config = load_config()
        default_model = get_model_path_from_config(config, "llama-3.1-8b-instruct")
    except Exception as e:
        print(f"Warning: Could not load model from config: {e}")
        default_model = "meta-llama/Llama-3.1-8B-Instruct"

    parser = argparse.ArgumentParser(
        description="Collect steering vectors from Step->Hash/Final activation differences"
    )
    parser.add_argument("--merged-dir", type=Path,
                        default=Path("output/complete_artifacts/gsm8k_train/merged"))
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("output/steering_vectors.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", type=str, default="hash", choices=["hash", "final"],
                        help="Template mode: 'hash' for #### (default) or 'final' for Final Answer:")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID for multi-GPU processing")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards for multi-GPU processing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question details")

    args = parser.parse_args()

    # Auto-adjust paths for final template
    if args.template == "final":
        # Add _final suffix to merged-dir if using default and not already present
        default_merged_dir = Path("output/complete_artifacts/gsm8k_train/merged")
        if args.merged_dir == default_merged_dir:
            args.merged_dir = Path("output/complete_artifacts/gsm8k_train_final/merged")

        # Add _final suffix to output path if not already present
        if "_final" not in str(args.output):
            args.output = args.output.parent / (args.output.stem + "_final" + args.output.suffix)

    if args.shard_id is not None and args.num_shards is None:
        parser.error("--shard-id requires --num-shards")
    if args.num_shards is not None and args.shard_id is None:
        parser.error("--num-shards requires --shard-id")

    marker_name = "####" if args.template == "hash" else "Final Answer:"
    marker_desc = "Step/####" if args.template == "hash" else "Step/Final Answer"

    # Detect R1/DeepSeek models from merged directory path
    # If "r1" or "deepseek" appears in the directory path, use R1-specific extraction
    merged_dir_str = str(args.merged_dir).lower()
    use_r1_extraction = ("r1" in merged_dir_str or "deepseek" in merged_dir_str) and args.template == "hash"

    # Detect task type from merged directory path
    # If "math" appears in the path, use MATH-specific evaluation
    if "math-500" in merged_dir_str or "math_500" in merged_dir_str:
        task = "math-500"
    elif "math" in merged_dir_str and "mmlu" not in merged_dir_str:
        task = "math"
    elif "mmlu" in merged_dir_str:
        task = "mmlu"
    else:
        task = "gsm8k"

    if args.shard_id is not None:
        print(f"\n{'='*100}")
        print(f"COLLECT STEERING VECTORS - {marker_name} (SHARD {args.shard_id}/{args.num_shards})")
        print(f"{'='*100}")
    else:
        print(f"\n{'='*100}")
        print(f"COLLECT STEERING VECTORS - {marker_name}")
        print(f"{'='*100}")

    print(f"Template mode: {args.template} ({marker_name})")
    print(f"Merged dir: {args.merged_dir}")
    print(f"Task: {task}")
    print(f"Model: {args.model}")
    print(f"Questions: {args.num_questions}")
    if use_r1_extraction:
        print(f"R1 MODE: Detected R1/DeepSeek model - will extract hash activations 1 timestep before first answer token")
    print(f"{'='*100}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print("Loading model...")
    # Use args.model if it's a model key, otherwise use it as path directly
    model_path = args.model
    if not Path(model_path).exists():
        # Assume it's a model key from config
        try:
            config = load_config()
            model_path = get_model_path_from_config(config, args.model)
            print(f"Resolved model key '{args.model}' to path: {model_path}")
        except Exception as e:
            print(f"Warning: Could not resolve model key, using as-is: {e}")
            model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True  # Prevent online downloads
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True  # Prevent online downloads
    ).eval()

    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"Model loaded: {num_layers} layers, hidden_dim={hidden_dim}")

    # Collect question IDs
    print(f"\nCollecting questions from {args.merged_dir}...")

    # Exclude meta-files that aren't individual question results
    META_FILES = {"aggregated_results.json", "merged_results.json",
                   "checkpoint.json", "summary.json", "generation_stats.json"}

    all_question_ids = []     # Integer IDs for NPZ storage
    question_id_to_stem = {}  # int → str mapping for loading

    # Discover ALL json files in merged dir
    all_json_files = sorted([
        f for f in args.merged_dir.glob("*.json")
        if f.name not in META_FILES
    ])

    if not all_json_files:
        print(f"  WARNING: No JSON files found in {args.merged_dir}")
        print(f"  Directory contents: {list(args.merged_dir.iterdir())[:10]}")
    else:
        print(f"  Discovered {len(all_json_files)} JSON files (first: {all_json_files[0].name})")

    # Check if files follow a known naming pattern (prefix_ID.json)
    has_known_prefix = (
        len(all_json_files) > 0 and
        any(all_json_files[0].stem.startswith(p) for p in ("gsm8k_", "mmlu_"))
    )

    for idx, json_file in enumerate(all_json_files):
        stem = json_file.stem
        if has_known_prefix:
            # Extract integer ID from filename: prefix_NNNN → NNNN
            try:
                qid = int(stem.split("_", 1)[1])
            except (ValueError, IndexError):
                continue  # skip non-standard files
        else:
            # MATH-500 / generic: use sequential integer IDs
            qid = idx
        all_question_ids.append(qid)
        question_id_to_stem[qid] = stem

    print(f"Found {len(all_question_ids)} questions")

    selected_questions = all_question_ids[:args.num_questions]
    print(f"Selected {len(selected_questions)} questions (before sharding)")

    # Apply sharding if specified
    if args.shard_id is not None:
        shard_questions = [qid for i, qid in enumerate(selected_questions) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(shard_questions)} questions")
        selected_questions = shard_questions
    else:
        print(f"Processing {len(selected_questions)} questions")

    # Process questions and collect differences
    print(f"\n{'='*100}")
    print("EXTRACTING ACTIVATIONS AND COMPUTING DIFFERENCES")
    print(f"{'='*100}\n")

    # Store differences for each layer
    layer_differences = [[] for _ in range(num_layers)]  # [num_layers][list of difference vectors]

    # Store raw activations for visualization
    all_step_activations = [[] for _ in range(num_layers)]  # [num_layers][list of activations]
    all_hash_activations = [[] for _ in range(num_layers)]  # [num_layers][list of activations]
    all_step_numbers = [[] for _ in range(num_layers)]  # [num_layers][list of step numbers]
    all_question_ids_step = [[] for _ in range(num_layers)]  # [num_layers][list of question IDs]
    all_question_ids_hash = [[] for _ in range(num_layers)]  # [num_layers][list of question IDs]
    all_is_correct_step = [[] for _ in range(num_layers)]  # [num_layers][list of correctness]
    all_is_correct_hash = [[] for _ in range(num_layers)]  # [num_layers][list of correctness]

    stats = {
        "total_questions_attempted": 0,
        "successful_questions": 0,
        "failed_no_json": 0,
        "failed_missing_fields": 0,
        "failed_no_positions": 0,
        "total_differences": 0
    }

    question_iterator = tqdm(selected_questions, desc="Processing questions", ncols=100, disable=args.verbose)

    for question_id in question_iterator:
        if args.verbose:
            print(f"\n[Question {question_id}] Loading merged JSON...")

        # Use stem mapping if available (MATH-500 etc.), else pass ID directly
        load_id = question_id_to_stem.get(question_id, question_id)
        merged_data = load_merged_json(args.merged_dir, load_id)
        if merged_data is None:
            stats["failed_no_json"] += 1
            if args.verbose:
                print(f"[Question {question_id}] ✗ Failed: JSON file not found")
            continue

        stats["total_questions_attempted"] += 1

        produced_text = merged_data.get("produced_text", "")
        full_seq_ids = merged_data.get("full_seq_ids", [])
        dp1_idx = merged_data.get("dp1_idx")
        dp2_idx = merged_data.get("dp2_idx")  # First answer token position (for R1 extraction)

        if not produced_text or not full_seq_ids or dp1_idx is None:
            stats["failed_missing_fields"] += 1
            if args.verbose:
                print(f"[Question {question_id}] ✗ Failed: Missing required fields")
            continue

        # Find positions based on template
        if args.template == "hash":
            # Pass dp2_idx and use_r1_extraction for R1-specific extraction
            positions = find_step_and_hash_positions(
                produced_text,
                full_seq_ids,
                tokenizer,
                dp1_idx,
                dp2_idx=dp2_idx,
                use_r1_extraction=use_r1_extraction
            )
        else:  # final
            positions = find_step_and_final_positions(produced_text, full_seq_ids, tokenizer, dp1_idx)

        if positions is None:
            stats["failed_no_positions"] += 1
            # Always print debug info for failures to help diagnose issues
            print(f"\n[Question {question_id}] ✗ Failed: Could not find {marker_desc} positions")
            print(f"  Produced text (FULL):")
            print(f"  {produced_text}")
            if args.template == "final":
                print(f"  'Final Answer:' in text: {'Final Answer:' in produced_text}")
                print(f"  Text length: {len(produced_text)}")
            continue

        if args.verbose:
            print(f"[Question {question_id}] Found {positions['num_steps']} Steps, extracting activations...")

        # Extract activations for all Step positions
        step_activations = extract_resid_activations(
            full_seq_ids,
            positions["step_positions"],
            model,
            tokenizer
        )

        # Extract activation for Hash/Final position based on template
        if args.template == "hash":
            answer_activations = extract_resid_activations(
                full_seq_ids,
                [positions["hash_position"]],
                model,
                tokenizer
            )
        else:  # final
            answer_activations = extract_resid_activations(
                full_seq_ids,
                [positions["final_position"]],
                model,
                tokenizer
            )

        if not answer_activations or not step_activations:
            stats["failed_no_positions"] += 1
            if args.verbose:
                print(f"[Question {question_id}] ✗ Failed: Could not extract activations")
            continue

        answer_act = answer_activations[0]  # [num_layers, hidden_dim]

        # Get correctness from merged JSON (already evaluated during baseline generation)
        # Do NOT re-extract answer from produced_text to avoid issues with repeated markers
        produced_answer = merged_data.get("produced_answer")
        gold_answer = merged_data.get("gold_answer")

        # Validate that both fields exist
        if produced_answer is None or gold_answer is None:
            stats["failed_missing_fields"] += 1
            if args.verbose:
                print(f"[Question {question_id}] ✗ Failed: Missing produced_answer or gold_answer field")
            continue

        # Evaluate correctness using task-appropriate evaluation
        if task in ("math", "math-500"):
            is_correct = evaluate_math_answer(produced_answer if produced_answer else "", gold_answer)
        else:
            is_correct = evaluate_answer(produced_answer, gold_answer)

        # Compute differences: answer - step for each step
        for step_idx, step_act in enumerate(step_activations):
            difference = answer_act - step_act  # [num_layers, hidden_dim]

            # Store difference for each layer
            for layer_idx in range(num_layers):
                layer_differences[layer_idx].append(difference[layer_idx])

            stats["total_differences"] += 1

        # Store raw activations for visualization (with step numbers, question_id, and correctness)
        for step_idx, step_act in enumerate(step_activations):
            step_number = step_idx + 1  # Step 1, 2, 3, ...
            for layer_idx in range(num_layers):
                all_step_activations[layer_idx].append(step_act[layer_idx])
                all_step_numbers[layer_idx].append(step_number)
                all_question_ids_step[layer_idx].append(question_id)
                all_is_correct_step[layer_idx].append(is_correct)

        for layer_idx in range(num_layers):
            all_hash_activations[layer_idx].append(answer_act[layer_idx])
            all_question_ids_hash[layer_idx].append(question_id)
            all_is_correct_hash[layer_idx].append(is_correct)

        stats["successful_questions"] += 1

        if args.verbose:
            print(f"[Question {question_id}] ✓ Success! Computed {len(step_activations)} differences")

        # Update progress
        if not args.verbose:
            question_iterator.set_postfix({
                'success': stats['successful_questions'],
                'diffs': stats['total_differences']
            })

    # Compute steering vectors using trimmed mean
    print(f"\n{'='*100}")
    print("COMPUTING STEERING VECTORS (TRIMMED MEAN)")
    print(f"{'='*100}\n")

    steering_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    for layer_idx in tqdm(range(num_layers), desc="Computing steering vectors", ncols=100):
        if len(layer_differences[layer_idx]) > 0:
            steering_vectors[layer_idx] = trimmed_mean_by_norm(
                layer_differences[layer_idx],
                trim_proportion=0.1
            )

            if args.verbose:
                norm = np.linalg.norm(steering_vectors[layer_idx])
                print(f"  Layer {layer_idx}: {len(layer_differences[layer_idx])} differences, norm={norm:.4f}")

    # Count correctness for successfully processed questions
    num_correct = 0
    num_incorrect = 0
    if len(all_is_correct_hash) > 0 and len(all_is_correct_hash[0]) > 0:
        # Use hash correctness from layer 0 (same for all layers)
        # Get unique question IDs to avoid double-counting
        # Convert lists to numpy arrays for indexing
        layer_0_qids = np.array(all_question_ids_hash[0], dtype=np.int64)
        layer_0_correct = np.array(all_is_correct_hash[0], dtype=np.bool_)

        unique_qids = np.unique(layer_0_qids)
        for qid in unique_qids:
            # Get correctness for this question (should be same for all occurrences)
            qid_mask = (layer_0_qids == qid)
            qid_correct = layer_0_correct[qid_mask][0]  # Take first occurrence
            if qid_correct:
                num_correct += 1
            else:
                num_incorrect += 1

    # Print statistics
    print(f"\n{'='*100}")
    print("COLLECTION STATISTICS")
    print(f"{'='*100}")
    print(f"Questions attempted: {stats['total_questions_attempted']}")
    print(f"Successful questions: {stats['successful_questions']} ({stats['successful_questions']/max(stats['total_questions_attempted'], 1)*100:.1f}%)")
    print(f"\nCorrectness breakdown (unique questions):")
    total_unique = num_correct + num_incorrect
    if total_unique > 0:
        print(f"  Correct: {num_correct} ({num_correct/total_unique*100:.1f}%)")
        print(f"  Incorrect: {num_incorrect} ({num_incorrect/total_unique*100:.1f}%)")
        print(f"  Total unique questions: {total_unique}")
    else:
        print(f"  No questions processed")
    print(f"\nFailure breakdown:")
    print(f"  No JSON file: {stats['failed_no_json']}")
    print(f"  Missing fields: {stats['failed_missing_fields']}")
    print(f"  No {marker_desc} positions: {stats['failed_no_positions']}")
    print(f"\nComputed data:")
    print(f"  Total differences: {stats['total_differences']}")
    print(f"  Steering vectors shape: {steering_vectors.shape}")
    print(f"{'='*100}\n")

    # Save data
    if args.shard_id is not None:
        output_path = args.output.parent / f"shard_{args.shard_id}" / args.output.name
    else:
        output_path = args.output

    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert lists to arrays for saving
    step_acts_arrays = [np.stack(all_step_activations[i], axis=0) if all_step_activations[i] else np.array([])
                        for i in range(num_layers)]
    hash_acts_arrays = [np.stack(all_hash_activations[i], axis=0) if all_hash_activations[i] else np.array([])
                        for i in range(num_layers)]
    step_numbers_arrays = [np.array(all_step_numbers[i], dtype=np.int32) if all_step_numbers[i] else np.array([], dtype=np.int32)
                           for i in range(num_layers)]
    question_ids_step_arrays = [np.array(all_question_ids_step[i], dtype=np.int64) if all_question_ids_step[i] else np.array([], dtype=np.int64)
                                for i in range(num_layers)]
    question_ids_hash_arrays = [np.array(all_question_ids_hash[i], dtype=np.int64) if all_question_ids_hash[i] else np.array([], dtype=np.int64)
                                for i in range(num_layers)]
    is_correct_step_arrays = [np.array(all_is_correct_step[i], dtype=np.bool_) if all_is_correct_step[i] else np.array([], dtype=np.bool_)
                              for i in range(num_layers)]
    is_correct_hash_arrays = [np.array(all_is_correct_hash[i], dtype=np.bool_) if all_is_correct_hash[i] else np.array([], dtype=np.bool_)
                              for i in range(num_layers)]

    np.savez(
        output_path,
        steering_vectors=steering_vectors,
        step_activations=step_acts_arrays,
        hash_activations=hash_acts_arrays,
        step_numbers=step_numbers_arrays,
        question_ids_step=question_ids_step_arrays,
        question_ids_hash=question_ids_hash_arrays,
        is_correct_step=is_correct_step_arrays,
        is_correct_hash=is_correct_hash_arrays,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        stats=json.dumps(stats),
        task=task,
        question_id_to_stem=json.dumps({str(k): v for k, v in question_id_to_stem.items()})
    )

    print(f"✓ Data saved successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
