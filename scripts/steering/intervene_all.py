#!/usr/bin/env python
"""Unconditional Steering Intervention (No Error Detection)

Applies steering intervention to ALL questions regardless of baseline correctness
or predicted errors. This script is useful for measuring the raw effect of steering
vectors on both correct and incorrect baseline answers.

Strategy:
1. Generate baseline (no intervention) for every question
2. Apply steering intervention to ALL questions unconditionally
3. Track transitions: correct→wrong (regressions) and wrong→correct (improvements)

Usage:
    python scripts/steering/intervene_all.py \
        --steering output/steering_vectors_llamainst8000.npz \
        --mode PROLONG_LAST_N \
        --alpha 1.0 \
        --num-questions 200
"""

import sys
import json
import argparse
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import evaluate_answer, evaluate_math_answer, extract_answer


@dataclass
class InterventionConfig:
    """Configuration for unconditional intervention"""
    mode: str  # PROLONG_LAST_N, PROLONG_FIRST_N, PROLONG_ALL_LAYERS, PROLONG_MID_N
    alpha: float = 1.0
    n_layers: int = 5  # For LAST_N, FIRST_N, or MID_N modes
    center_layer: int = 15  # Center layer for PROLONG_MID_N mode
    multi_timestep: bool = False  # Apply steering across multiple timesteps
    max_new_tokens: int = 1024  # Max tokens to generate


def extract_answer_after_hash(text: str) -> Optional[str]:
    """Extract answer after #### marker - ROBUST VERSION"""
    if not text:
        return None

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


def count_steps(text: str) -> int:
    """Count number of Step N: patterns in text (only before #### marker)"""
    if "####" in text:
        reasoning_part = text.split("####")[0]
    else:
        reasoning_part = text

    pattern = r'\bStep\s+\d+\s*[:.]'
    matches = re.findall(pattern, reasoning_part, re.IGNORECASE)
    return len(matches)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from paths.yaml"""
    if config_path is None:
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


def load_steering_vectors(npz_path: Path) -> Tuple[np.ndarray, int, int]:
    """Load steering vectors from NPZ file

    Returns:
        steering_vectors: [num_layers, hidden_dim]
        num_layers: Number of layers
        hidden_dim: Hidden dimension
    """
    print(f"\nLoading steering vectors from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    steering_vectors = data["steering_vectors"]
    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])

    print(f"  Shape: {steering_vectors.shape}")
    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    return steering_vectors, num_layers, hidden_dim


class SteeringHook:
    """Hook to apply steering intervention during generation"""

    def __init__(
        self,
        steering_vectors: np.ndarray,
        config: InterventionConfig,
        num_layers: int
    ):
        """
        Args:
            steering_vectors: [num_layers, hidden_dim] steering vectors (step - hash)
            config: InterventionConfig
            num_layers: Total number of layers
        """
        self.steering_vectors = torch.from_numpy(steering_vectors).float()
        self.config = config
        self.num_layers = num_layers
        self.device = None

        # Determine which layers to intervene on
        if config.mode == "PROLONG_LAST_N":
            self.intervene_layers = set(range(num_layers - config.n_layers, num_layers))
        elif config.mode == "PROLONG_FIRST_N":
            self.intervene_layers = set(range(min(config.n_layers, num_layers)))
        elif config.mode == "PROLONG_MID_N":
            n = config.n_layers
            center = config.center_layer
            half = n // 2

            if n % 2 == 0:
                start = center - (half - 1)
                end = center + half + 1
            else:
                start = center - half
                end = center + half + 1

            self.intervene_layers = set(range(max(0, start), min(num_layers, end)))
        else:  # PROLONG_ALL_LAYERS
            self.intervene_layers = set(range(num_layers))

    def __call__(self, module, _input, output):
        """Hook function that modifies hidden states"""
        layer_idx = getattr(module, '_layer_idx', None)
        if layer_idx is None or layer_idx not in self.intervene_layers:
            return output

        # Move steering vector to device if needed
        if self.device is None:
            self.device = output[0].device
            self.steering_vectors = self.steering_vectors.to(self.device)

        # Apply steering: hidden_state + alpha * steering
        hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
        steering = self.steering_vectors[layer_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # Apply to last position only
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + self.config.alpha * steering

        return (hidden_states,) + output[1:]


def register_steering_hooks(model, steering_hook: SteeringHook) -> List:
    """Register hooks on model layers"""
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        layer._layer_idx = layer_idx
        handle = layer.register_forward_hook(steering_hook)
        handles.append(handle)
    return handles


def remove_hooks(handles: List):
    """Remove all registered hooks"""
    for handle in handles:
        handle.remove()


@torch.no_grad()
def generate_with_unconditional_intervention(
    input_ids: List[int],
    question_text: str,
    gold_answer: str,
    model,
    tokenizer,
    steering_vectors: np.ndarray,
    config: InterventionConfig,
    device: torch.device,
    question_id: Optional[int] = None,
    verbose: bool = True,
    dp2_idx: Optional[int] = None,
    use_r1_intervention: bool = False,
    task: str = "gsm8k"
) -> Dict:
    """Generate with unconditional intervention on ALL questions

    Strategy:
    1. Generate baseline (no intervention)
    2. Generate token-by-token WITHOUT steering until intervention checkpoint is reached
    3. For R1 models: Apply steering at timestep BEFORE dp2_idx (first answer token)
    4. For non-R1: Apply steering when "####" is about to be generated
    5. Continue generation with steering active for all remaining timesteps

    Args:
        dp2_idx: Absolute index of first answer token (for R1 intervention timing)
        use_r1_intervention: If True, use dp2_idx instead of "####" detection
        task: Task type ("gsm8k", "math", or "math-500") for appropriate extraction/evaluation

    Returns:
        Dict with results including baseline and intervention outcomes
    """
    num_layers = len(model.model.layers)

    if verbose:
        q_prefix = f"Q{question_id}: " if question_id is not None else ""
        print(f"  {q_prefix}Starting generation...")

    # =====================================================================
    # BASELINE GENERATION (deterministic, no intervention)
    # =====================================================================
    if verbose:
        print(f"  {q_prefix}→ Generating baseline (no intervention)...")

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    prompt_len = len(input_ids)

    baseline_outputs = model.generate(
        input_tensor,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,  # Deterministic
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    baseline_ids = baseline_outputs[0].tolist()
    baseline_text = tokenizer.decode(baseline_ids[prompt_len:], skip_special_tokens=True)

    # Use task-appropriate extraction and evaluation
    if task in ("math", "math-500"):
        baseline_answer = extract_answer(baseline_text, task=task)
        baseline_correct = evaluate_math_answer(baseline_answer if baseline_answer else "", gold_answer)
    else:
        baseline_answer = extract_answer_after_hash(baseline_text)
        baseline_correct = evaluate_answer(baseline_answer if baseline_answer else "N/A", gold_answer)

    baseline_num_steps = count_steps(baseline_text)

    # Find #### position for reasoning length
    hash_token_ids = tokenizer.encode("####", add_special_tokens=False)
    baseline_hash_pos = None
    for idx in range(prompt_len, len(baseline_ids)):
        if baseline_ids[idx] in hash_token_ids:
            baseline_hash_pos = idx
            break
    baseline_reasoning_length = (baseline_hash_pos - prompt_len) if baseline_hash_pos else (len(baseline_ids) - prompt_len)

    baseline_result = {
        "text": baseline_text,
        "answer": baseline_answer if baseline_answer else "N/A",
        "is_correct": bool(baseline_correct),
        "num_steps": int(baseline_num_steps),
        "reasoning_length": int(baseline_reasoning_length),
    }

    if verbose:
        correct_mark = "✓" if baseline_correct else "✗"
        print(f"  {q_prefix}→ Baseline: {correct_mark} answer={baseline_answer}, steps={baseline_num_steps}, len={baseline_reasoning_length}")

    # =====================================================================
    # INTERVENTION GENERATION (apply steering at intervention checkpoint)
    # =====================================================================
    if use_r1_intervention and dp2_idx is not None:
        if verbose:
            print(f"  {q_prefix}→ Generating with intervention (R1 MODE: starting before dp2_idx={dp2_idx})...")
    else:
        if verbose:
            print(f"  {q_prefix}→ Generating with intervention (starting at #### checkpoint)...")

    # Start with prompt
    current_ids = input_ids.copy()
    steering_active = False
    steering_handles = None
    hash_detected_at = None

    # Generate tokens one by one, waiting for intervention checkpoint
    max_total_tokens = prompt_len + config.max_new_tokens
    hash_token_id = hash_token_ids[0] if hash_token_ids else tokenizer.encode("####", add_special_tokens=False)[0]

    # For R1 mode: Calculate target position for intervention
    # dp2_idx is the absolute index of the first answer token (e.g., 186)
    # We want to apply steering at the SAME timestep where we extracted activations
    # For activation collection: we extract at timestep 185 (one before dp2_idx)
    # For intervention: we apply steering when generating the token at position 185
    # This happens when len(current_ids) == 185 (we're about to generate token 185)
    r1_intervention_position = None
    if use_r1_intervention and dp2_idx is not None:
        # Intervention position: one position before first answer token
        # When len(current_ids) == dp2_idx - 1, we apply steering to generate token at dp2_idx - 1
        r1_intervention_position = dp2_idx - 1

        # Validate position
        if r1_intervention_position <= prompt_len:
            if verbose:
                print(f"  {q_prefix}  ⚠ WARNING: R1 intervention position {r1_intervention_position} <= prompt_len {prompt_len}")
                print(f"  {q_prefix}    Falling back to standard #### detection")
            use_r1_intervention = False
            r1_intervention_position = None
        elif verbose:
            print(f"  {q_prefix}  R1 intervention will trigger at position {r1_intervention_position - prompt_len} (absolute: {r1_intervention_position})")

    while len(current_ids) < max_total_tokens:
        # R1 MODE: Check if we've reached the intervention position (before first answer token)
        if use_r1_intervention and r1_intervention_position is not None:
            if len(current_ids) == r1_intervention_position and not steering_active:
                # R1 CHECKPOINT: We are at the position before first answer token
                if verbose:
                    print(f"  {q_prefix}  ⚡ R1 CHECKPOINT: Reached position {len(current_ids) - prompt_len} (1 before dp2_idx)")
                    print(f"  {q_prefix}    🎯 INTERVENING: Registering steering hooks...")

                hash_detected_at = len(current_ids) - prompt_len

                # Create and register steering hooks
                steering_hook = SteeringHook(steering_vectors, config, num_layers)
                steering_handles = register_steering_hooks(model, steering_hook)
                steering_active = True

                # Generate next token WITH steering now active
                input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
                outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                next_token_id_steered = torch.argmax(logits, dim=-1).item()

                current_ids.append(next_token_id_steered)

                if verbose:
                    print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")
                    print(f"  {q_prefix}      → Steering ACTIVE for all remaining timesteps...")

                # Check for EOS
                if next_token_id_steered == tokenizer.eos_token_id:
                    break
                continue

        # Generate next token
        input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
        outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_token_id = torch.argmax(logits, dim=-1).item()

        # NON-R1 MODE: Check if next token is "####" - this is our intervention checkpoint!
        if not use_r1_intervention and next_token_id == hash_token_id and not steering_active:
            # Detected "####" about to be generated - register steering hooks NOW
            if verbose:
                print(f"  {q_prefix}  ⚡ CHECKPOINT: Detected #### at position {len(current_ids) - prompt_len}")
                print(f"  {q_prefix}    🎯 INTERVENING: Registering steering hooks...")

            hash_detected_at = len(current_ids) - prompt_len

            # Create and register steering hooks
            steering_hook = SteeringHook(steering_vectors, config, num_layers)
            steering_handles = register_steering_hooks(model, steering_hook)
            steering_active = True

            # Generate next token WITH steering now active
            input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
            outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            next_token_id_steered = torch.argmax(logits, dim=-1).item()

            current_ids.append(next_token_id_steered)

            if verbose:
                print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")
                print(f"  {q_prefix}      → Steering ACTIVE for all remaining timesteps...")

            # Check for EOS
            if next_token_id_steered == tokenizer.eos_token_id:
                break
            continue
        else:
            # Normal token generation (before checkpoint or after steering is active)
            current_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break

    # Cleanup: Remove steering hooks if still active
    if steering_active and steering_handles is not None:
        remove_hooks(steering_handles)
        if verbose:
            print(f"  {q_prefix}→ Steering deactivated at end of generation")

    # Extract intervened results
    intervened_text = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)

    # Use task-appropriate extraction and evaluation
    if task in ("math", "math-500"):
        intervened_answer = extract_answer(intervened_text, task=task)
        intervened_correct = evaluate_math_answer(intervened_answer if intervened_answer else "", gold_answer)
    else:
        intervened_answer = extract_answer_after_hash(intervened_text)
        intervened_correct = evaluate_answer(intervened_answer if intervened_answer else "N/A", gold_answer)

    intervened_num_steps = count_steps(intervened_text)

    # Find #### position for reasoning length
    intervened_hash_pos = None
    for idx in range(prompt_len, len(current_ids)):
        if current_ids[idx] in hash_token_ids:
            intervened_hash_pos = idx
            break
    intervened_reasoning_length = (intervened_hash_pos - prompt_len) if intervened_hash_pos else (len(current_ids) - prompt_len)

    intervened_result = {
        "text": intervened_text,
        "answer": intervened_answer if intervened_answer else "N/A",
        "is_correct": bool(intervened_correct),
        "num_steps": int(intervened_num_steps),
        "reasoning_length": int(intervened_reasoning_length),
        "hash_detected_at": hash_detected_at,  # Position where #### was detected
    }

    if verbose:
        correct_mark = "✓" if intervened_correct else "✗"
        print(f"  {q_prefix}→ Intervened: {correct_mark} answer={intervened_answer}, steps={intervened_num_steps}")
        if hash_detected_at is not None:
            print(f"  {q_prefix}  (Intervention started at position {hash_detected_at})")

        # Show outcome
        if not baseline_correct and intervened_correct:
            print(f"  {q_prefix}🎉 IMPROVEMENT: Fixed wrong answer!")
        elif baseline_correct and not intervened_correct:
            print(f"  {q_prefix}⚠ REGRESSION: Broke correct answer!")
        elif not baseline_correct and not intervened_correct:
            print(f"  {q_prefix}→ No improvement (still incorrect)")
        else:
            print(f"  {q_prefix}→ Maintained correctness")

    return {
        "question": question_text,
        "gold_answer": gold_answer,
        "baseline": baseline_result,
        "intervened": intervened_result,
        "config": asdict(config),
    }


def main():
    try:
        config = load_config()
        default_model = get_model_path_from_config(config, "llama-3.1-8b-instruct")
    except Exception as e:
        print(f"Warning: Could not load model from config: {e}")
        default_model = "meta-llama/Llama-3.1-8B-Instruct"

    parser = argparse.ArgumentParser(
        description="Unconditional steering intervention (no error detection)"
    )
    parser.add_argument("--steering", type=Path, required=True,
                        help="Path to steering vectors NPZ file")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["PROLONG_LAST_N", "PROLONG_FIRST_N", "PROLONG_ALL_LAYERS", "PROLONG_MID_N"],
                        help="Intervention mode")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering strength (default: 1.0)")
    parser.add_argument("--n-layers", type=int, default=5,
                        help="Number of layers for LAST_N, FIRST_N, or MID_N modes (default: 5)")
    parser.add_argument("--center-layer", type=int, default=15,
                        help="Center layer for PROLONG_MID_N mode (default: 15)")
    parser.add_argument("--multi-timestep", action="store_true",
                        help="Apply steering across multiple timesteps")

    parser.add_argument("--merged-dir", type=Path,
                        default=Path("output/complete_artifacts/gsm8k_train/merged"),
                        help="Directory with merged JSON files")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model name or path")
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Number of questions to process")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/unconditional_interventions"),
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # Sharding arguments (for multi-GPU)
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID (for multi-GPU processing)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (for multi-GPU processing)")

    args = parser.parse_args()

    # Detect R1/DeepSeek models from merged directory path
    # If "r1" or "deepseek" appears in the directory path, use R1-specific intervention
    merged_dir_str = str(args.merged_dir).lower()
    use_r1_intervention = "r1" in merged_dir_str or "deepseek" in merged_dir_str

    # Detect task type from merged directory path
    # If "math" appears in the path, use MATH-specific extraction/evaluation
    if "math-500" in merged_dir_str or "math_500" in merged_dir_str:
        task = "math-500"
    elif "math" in merged_dir_str:
        task = "math"
    else:
        task = "gsm8k"

    print(f"\n{'='*100}")
    print("UNCONDITIONAL STEERING INTERVENTION (NO ERROR DETECTION)")
    print(f"{'='*100}")
    print(f"Steering: {args.steering}")
    print(f"Merged dir: {args.merged_dir}")
    print(f"Task: {task}")
    print(f"Mode: {args.mode}")
    print(f"Alpha: {args.alpha}")
    print(f"N layers: {args.n_layers}")
    print(f"Intervention: Applied to ALL questions unconditionally")
    if use_r1_intervention:
        print(f"R1 MODE: Detected R1/DeepSeek model - will apply intervention 1 timestep before first answer token")
    else:
        print(f"Standard MODE: Will apply intervention when #### is detected")
    print(f"{'='*100}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load steering vectors
    steering_vectors, _, _ = load_steering_vectors(args.steering)

    # Create intervention config
    intervention_config = InterventionConfig(
        mode=args.mode,
        alpha=args.alpha,
        n_layers=args.n_layers,
        center_layer=args.center_layer,
        multi_timestep=args.multi_timestep,
        max_new_tokens=args.max_new_tokens,
    )

    # Load model
    print("\nLoading model...")
    model_path = args.model
    if not Path(model_path).exists():
        try:
            config = load_config()
            model_path = get_model_path_from_config(config, args.model)
            print(f"Resolved model key '{args.model}' to path: {model_path}")
        except Exception as e:
            print(f"Warning: Could not resolve model key: {e}")
            model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        local_files_only=True
    ).eval()

    print(f"Model loaded: {len(model.model.layers)} layers")

    # Collect questions
    print(f"\nCollecting questions from {args.merged_dir}...")

    # Detect file pattern based on task
    if task in ("math", "math-500"):
        # MATH format: test_{subject}_{id}.json (e.g., test_algebra_101.json)
        # Collect all JSON files that are NOT special files
        json_files = sorted([
            f for f in args.merged_dir.glob("*.json")
            if f.name not in ('aggregated_results.json', 'merged_results.json', 'checkpoint.json')
        ])
        all_question_files = json_files
        print(f"  MATH format detected: collecting all problem files")
    else:
        # GSM8K format: gsm8k_{id}.json
        json_files = sorted(args.merged_dir.glob("gsm8k_*.json"))
        all_question_files = json_files
        print(f"  GSM8K format detected: pattern gsm8k_*.json")

    print(f"Found {len(all_question_files)} questions")

    # Apply num_questions limit if specified
    if args.num_questions is not None and args.num_questions < len(all_question_files):
        all_question_files = all_question_files[:args.num_questions]
        print(f"Limited to first {args.num_questions} questions")

    # Apply sharding if specified
    if args.shard_id is not None and args.num_shards is not None:
        print(f"\nShard {args.shard_id}/{args.num_shards}: Applying round-robin sharding...")
        selected_question_files = [qf for idx, qf in enumerate(all_question_files) if idx % args.num_shards == args.shard_id]
        print(f"  → Shard {args.shard_id} assigned {len(selected_question_files)}/{len(all_question_files)} questions")
    else:
        selected_question_files = all_question_files
        print(f"\nProcessing {len(selected_question_files)} questions")

    # Process questions
    print(f"\n{'='*100}")
    print("GENERATING WITH UNCONDITIONAL INTERVENTION")
    print(f"{'='*100}\n")

    results = []
    stats = {
        "total": 0,
        "baseline_correct": 0,
        "intervened_correct": 0,
        "flipped_wrong_to_correct": 0,  # Improvements
        "flipped_correct_to_wrong": 0,  # Regressions
    }

    pbar = tqdm(selected_question_files, desc="Processing", ncols=120)
    for json_path in pbar:
        # Load question
        if not json_path.exists():
            continue

        with open(json_path, 'r') as f:
            question_data = json.load(f)

        input_ids = question_data.get("input_ids", [])
        gold_answer = question_data.get("gold_answer", question_data.get("answer", "N/A"))
        question_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Load dp2_idx for R1 intervention timing
        dp2_idx = question_data.get("dp2_idx")

        # Get question identifier (for logging)
        question_id = question_data.get("id", json_path.stem)

        if not input_ids:
            continue

        # Generate with intervention
        try:
            result = generate_with_unconditional_intervention(
                input_ids, question_text, gold_answer,
                model, tokenizer, steering_vectors,
                intervention_config, device,
                question_id=question_id,
                verbose=True,
                dp2_idx=dp2_idx,
                use_r1_intervention=use_r1_intervention,
                task=task
            )

            result["question_id"] = question_id
            results.append(result)

            # Update stats
            stats["total"] += 1
            baseline_correct = result["baseline"]["is_correct"]
            intervened_correct = result["intervened"]["is_correct"]

            if baseline_correct:
                stats["baseline_correct"] += 1
            if intervened_correct:
                stats["intervened_correct"] += 1

            if not baseline_correct and intervened_correct:
                stats["flipped_wrong_to_correct"] += 1
            if baseline_correct and not intervened_correct:
                stats["flipped_correct_to_wrong"] += 1

            # Update progress bar
            pbar.set_postfix({
                "Base": "✓" if baseline_correct else "✗",
                "After": "✓" if intervened_correct else "✗",
                "BaseAcc": f"{stats['baseline_correct']}/{stats['total']}",
                "IntvAcc": f"{stats['intervened_correct']}/{stats['total']}"
            })

        except Exception as e:
            print(f"\n  Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with shard info if applicable
    if args.shard_id is not None and args.num_shards is not None:
        output_file = args.output_dir / f"{args.mode}_alpha{args.alpha}" / f"shard_{args.shard_id}" / "results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = args.output_dir / f"results_{args.mode}_alpha{args.alpha}.json"

    # Compute summary
    if stats["total"] > 0:
        summary = {
            "total_questions": stats["total"],
            "baseline_accuracy": stats["baseline_correct"] / stats["total"],
            "intervened_accuracy": stats["intervened_correct"] / stats["total"],
            "accuracy_change": (stats["intervened_correct"] - stats["baseline_correct"]) / stats["total"],
            "improvements": stats["flipped_wrong_to_correct"],  # wrong→correct
            "regressions": stats["flipped_correct_to_wrong"],   # correct→wrong
            "net_change": stats["flipped_wrong_to_correct"] - stats["flipped_correct_to_wrong"],
        }
    else:
        summary = {}

    output_data = {
        "config": asdict(intervention_config),
        "steering_path": str(args.steering),
        "stats": stats,
        "summary": summary,
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"Total questions processed: {stats['total']}")
    print(f"\nBaseline Performance:")
    print(f"  Correct: {stats['baseline_correct']}/{stats['total']} ({summary.get('baseline_accuracy', 0):.2%})")
    print(f"\nAfter Intervention:")
    print(f"  Correct: {stats['intervened_correct']}/{stats['total']} ({summary.get('intervened_accuracy', 0):.2%})")
    print(f"  Accuracy change: {summary.get('accuracy_change', 0):+.2%}")
    print(f"\nTransitions:")
    print(f"  🎉 Improvements (wrong→correct): {stats['flipped_wrong_to_correct']}")
    print(f"  ⚠  Regressions (correct→wrong):  {stats['flipped_correct_to_wrong']}")
    print(f"  Net change: {'+' if summary.get('net_change', 0) >= 0 else ''}{summary.get('net_change', 0)}")
    print(f"{'='*100}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
