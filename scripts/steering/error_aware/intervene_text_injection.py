#!/usr/bin/env python
"""Text Injection Intervention with Error-Aware Detection

Instead of applying steering vectors, this script directly injects text tokens
(e.g., "Step k+1:", "Wait,", "Hmm,", "Wait, let me double check. ") when the predictor detects a potential error.

Intervention modes:
- STEP: Inject "Step k+1:" when error predicted
- WAIT: Inject "Wait, " when error predicted
- HMM: Inject "Hmm, " when error predicted
- DOUBLE_CHECK: Inject "Wait, let me double check. " when error predicted
- ALWAYS_STEP: Always inject "Step k+1:" at every #### checkpoint (no predictor)
- ALWAYS_WAIT: Always inject "Wait, " at every #### checkpoint (no predictor)
- ALWAYS_HMM: Always inject "Hmm, " at every #### checkpoint (no predictor)
- ALWAYS_DOUBLE_CHECK: Always inject "Wait, let me double check. " at every #### checkpoint (no predictor)

Usage:
    python scripts/steering/error_aware/intervene_text_injection.py \
        --predictor output/predictors/hash_minus_last_correctness_layer31.npz \
        --mode STEP \
        --num-questions 50
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import extract_answer, evaluate_answer, evaluate_math_answer


@dataclass
class InterventionConfig:
    """Configuration for text injection intervention"""
    mode: str  # STEP, WAIT, HMM, DOUBLE_CHECK, ALWAYS_STEP, ALWAYS_WAIT, ALWAYS_HMM, ALWAYS_DOUBLE_CHECK
    max_interventions: int = 1  # Intervene only once
    max_new_tokens: int = 1024  # Max tokens to generate
    predictor_threshold: float = 0.5  # Not used for ALWAYS_* modes
    task: str = "gsm8k"  # Task type for answer extraction (gsm8k, math, math-500)
    use_r1_fallback: bool = False  # Whether to use R1 fallback extraction


def count_steps(text: str) -> int:
    """Count number of Step N: patterns in text (only before #### marker)

    Counts "Step N:" or "Step N." patterns ONLY in the reasoning portion (before #### marker).
    This ensures we don't count spurious steps that may appear after the final answer.
    """
    # Split by #### and only count steps in the reasoning part (before ####)
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
        config_path = script_dir.parent.parent.parent / "config" / "paths.yaml"

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


def load_predictor(npz_path: Path) -> Dict:
    """Load predictor classifier from NPZ file"""
    print(f"\nLoading predictor from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    predictor = {
        'coefficients': data['coefficients'],
        'intercept': float(data['intercept']),
        'scaler_mean': data['scaler_mean'],
        'scaler_std': data['scaler_std'],
        'feature_set': str(data['feature_set']),
        'label_type': str(data['label_type']),
        'layer_idx': int(data['layer_idx']),
    }

    # Load best_threshold if available (from threshold tuning)
    if 'best_threshold' in data:
        predictor['best_threshold'] = float(data['best_threshold'])
        print(f"  Feature set: {predictor['feature_set']}")
        print(f"  Label type: {predictor['label_type']}")
        print(f"  Layer: {predictor['layer_idx']}")
        print(f"  Best threshold: {predictor['best_threshold']:.3f} (tuned)")
    else:
        predictor['best_threshold'] = 0.5  # Default
        print(f"  Feature set: {predictor['feature_set']}")
        print(f"  Label type: {predictor['label_type']}")
        print(f"  Layer: {predictor['layer_idx']}")
        print(f"  Using default threshold: 0.5 (not tuned)")

    return predictor


def predict_correctness(
    hash_minus_last_features: np.ndarray,
    predictor: Dict,
    threshold: float = 0.5
) -> Tuple[bool, float, float, float]:
    """Predict if answer will be correct"""
    features_scaled = (hash_minus_last_features - predictor['scaler_mean']) / predictor['scaler_std']
    score_incorrect = np.dot(predictor['coefficients'], features_scaled) + predictor['intercept']
    score_correct = -score_incorrect
    prob_incorrect = 1.0 / (1.0 + np.exp(-score_incorrect))
    is_predicted_incorrect = (prob_incorrect > threshold)
    return is_predicted_incorrect, float(score_incorrect), float(score_correct), float(prob_incorrect)


@torch.no_grad()
def extract_activations_at_position(
    input_ids: List[int],
    position: int,
    model,
    device: torch.device
) -> np.ndarray:
    """Extract hidden state activations at a specific position"""
    num_layers = len(model.model.layers)
    target_pos = position - 1
    if target_pos < 0:
        raise ValueError(f"Cannot extract activations at position {position}")

    input_tensor = torch.tensor([input_ids[:target_pos + 1]], dtype=torch.long).to(device)
    outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)

    layer_activations = []
    for layer_idx in range(num_layers):
        hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]
        layer_activations.append(hidden.squeeze(0))

    activations = torch.stack(layer_activations, dim=0).float().cpu().numpy()
    return activations


def find_last_step_activation(
    input_ids: List[int],
    model,
    tokenizer,
    device: torch.device
) -> Optional[np.ndarray]:
    """Find and extract the activation at the last 'Step N:' occurrence"""
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    pattern = r'\bStep\s+\d+\s*[:.]'
    step_matches = list(re.finditer(pattern, generated_text, re.IGNORECASE))

    if len(step_matches) == 0:
        return None

    last_step_char_pos = step_matches[-1].start()
    last_step_token_pos = None

    for token_idx in range(len(input_ids)):
        decoded_so_far = tokenizer.decode(input_ids[:token_idx + 1], skip_special_tokens=True)
        if last_step_char_pos <= len(decoded_so_far):
            last_step_token_pos = token_idx
            break

    if last_step_token_pos is None:
        return None

    try:
        activation = extract_activations_at_position(input_ids, last_step_token_pos, model, device)
        return activation
    except ValueError:
        return None


@torch.no_grad()
def generate_with_text_injection(
    input_ids: List[int],
    question_text: str,
    gold_answer: str,
    model,
    tokenizer,
    predictor: Optional[Dict],
    config: InterventionConfig,
    device: torch.device,
    question_id: Optional[int] = None,
    verbose: bool = True,
    dp2_idx: Optional[int] = None,
    use_r1_intervention: bool = False,
    saved_baseline_text: Optional[str] = None,
    saved_baseline_correct: Optional[bool] = None
) -> Dict:
    """Generate with text injection intervention

    Strategy:
    1. Use saved baseline if available, otherwise generate baseline
    2. For intervention:
       - Generate token-by-token
       - For R1 models: Check at position before dp2_idx
       - For non-R1: Check when about to predict ####
       - If error predicted (or always for ALWAYS_* modes), inject text
       - Continue generation from injected text

    Args:
        dp2_idx: Absolute index of first answer token (for R1 intervention timing)
        use_r1_intervention: If True, use dp2_idx instead of "####" detection
        saved_baseline_text: Optional pre-saved baseline text to use instead of regenerating
        saved_baseline_correct: Optional pre-saved baseline correctness
    """
    predictor_layer = predictor['layer_idx'] if predictor else None
    is_always_mode = config.mode.startswith("ALWAYS_")

    if verbose:
        q_prefix = f"Q{question_id}: " if question_id is not None else ""
        print(f"  {q_prefix}Starting generation...")

    # =====================================================================
    # BASELINE GENERATION (or use saved)
    # =====================================================================
    prompt_len = len(input_ids)
    hash_token_ids = tokenizer.encode("####", add_special_tokens=False)

    # Use saved baseline if available (for datasets with pre-saved results)
    if saved_baseline_text is not None and saved_baseline_correct is not None:
        if verbose:
            print(f"  {q_prefix}→ Using saved baseline (no regeneration)...")

        baseline_text = saved_baseline_text
        baseline_correct = saved_baseline_correct
        baseline_answer = extract_answer(baseline_text, task=config.task, use_r1_fallback=config.use_r1_fallback, gold_answer=gold_answer)
        baseline_num_steps = count_steps(baseline_text)

        # Estimate reasoning length from text
        baseline_reasoning_length = len(baseline_text.split())  # Rough estimate

    else:
        # Generate baseline from scratch
        if verbose:
            print(f"  {q_prefix}→ Generating baseline (no intervention)...")

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        baseline_outputs = model.generate(
            input_tensor,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        baseline_ids = baseline_outputs[0].tolist()
        baseline_text = tokenizer.decode(baseline_ids[prompt_len:], skip_special_tokens=True)
        baseline_answer = extract_answer(baseline_text, task=config.task, use_r1_fallback=config.use_r1_fallback, gold_answer=gold_answer)

        # Use appropriate evaluation based on task
        if config.task in ("math", "math-500"):
            baseline_correct = evaluate_math_answer(baseline_answer if baseline_answer else "N/A", gold_answer)
        else:
            baseline_correct = evaluate_answer(baseline_answer if baseline_answer else "N/A", gold_answer)
        baseline_num_steps = count_steps(baseline_text)

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
    # TEXT INJECTION INTERVENTION
    # =====================================================================
    if verbose:
        mode_desc = "always-inject" if is_always_mode else "error-aware"
        if use_r1_intervention and dp2_idx is not None:
            print(f"  {q_prefix}→ Starting {mode_desc} text injection (R1 MODE: before dp2_idx={dp2_idx})...")
        else:
            print(f"  {q_prefix}→ Starting {mode_desc} text injection...")

    current_ids = input_ids.copy()
    intervention_history = []
    num_interventions = 0
    has_checked_once = False  # Flag to ensure we only check predictor once

    max_total_tokens = prompt_len + config.max_new_tokens
    hash_token_id = hash_token_ids[0] if hash_token_ids else tokenizer.encode("####", add_special_tokens=False)[0]

    # For R1 mode: Calculate target position for intervention
    r1_intervention_position = None
    if use_r1_intervention and dp2_idx is not None:
        # Intervention position: one position before first answer token
        r1_intervention_position = dp2_idx - 1

    while len(current_ids) < max_total_tokens:
        # Generate next token
        input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
        outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()

        # Check if we're at R1 intervention position
        if use_r1_intervention and r1_intervention_position is not None and len(current_ids) == r1_intervention_position:
            # R1 CHECKPOINT: Reached position before first answer token
            if not has_checked_once:
                if verbose:
                    print(f"  {q_prefix}  ⚡ R1 CHECKPOINT: Reached position {len(current_ids) - prompt_len} (before dp2_idx)")

                should_intervene = False
                score_incorrect = None
                score_correct = None
                prob_incorrect = None

                # For error-aware modes, run predictor
                if not is_always_mode and predictor is not None:
                    if verbose:
                        print(f"  {q_prefix}    → Running predictor...")

                    # Extract activations
                    current_position = len(current_ids)
                    hash_activation = extract_activations_at_position(current_ids, current_position, model, device)
                    last_step_activation = find_last_step_activation(current_ids, model, tokenizer, device)

                    if last_step_activation is None:
                        if verbose:
                            print(f"  {q_prefix}    ⚠ No steps found, skipping injection")
                        has_checked_once = True
                        current_ids.append(next_token_id)
                        if next_token_id == tokenizer.eos_token_id:
                            break
                        continue

                    # Predict correctness
                    hash_minus_last = hash_activation[predictor_layer] - last_step_activation[predictor_layer]
                    is_predicted_incorrect, score_incorrect, score_correct, prob_incorrect = predict_correctness(
                        hash_minus_last, predictor, config.predictor_threshold
                    )

                    if verbose:
                        pred_label = "INCORRECT" if is_predicted_incorrect else "CORRECT"
                        print(f"  {q_prefix}    → Predictor: {pred_label} (score_incorrect={score_incorrect:.3f}, score_correct={score_correct:.3f})")

                    should_intervene = is_predicted_incorrect

                # For ALWAYS_* modes, always intervene (up to max)
                elif is_always_mode and num_interventions < config.max_interventions:
                    should_intervene = True

                has_checked_once = True

                # Intervene if needed
                if should_intervene:
                    # Count current steps
                    decoded_so_far = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)
                    k = count_steps(decoded_so_far) + 1

                    # Determine injection text based on mode
                    if config.mode in ["STEP", "ALWAYS_STEP"]:
                        continuation_text = f"\nStep {k}: "
                    elif config.mode in ["WAIT", "ALWAYS_WAIT"]:
                        continuation_text = "\nWait, "
                    elif config.mode in ["HMM", "ALWAYS_HMM"]:
                        continuation_text = "\nHmm, "
                    elif config.mode in ["DOUBLE_CHECK", "ALWAYS_DOUBLE_CHECK"]:
                        continuation_text = "\nWait, let me double check. "
                    else:
                        raise ValueError(f"Unknown mode: {config.mode}")

                    if verbose:
                        print(f"  {q_prefix}    🎯 INJECTING: '{continuation_text.strip()}'")

                    # Encode and inject tokens
                    new_tokens = tokenizer.encode(continuation_text, add_special_tokens=False)
                    current_ids.extend(new_tokens)
                    num_interventions += 1

                    intervention_history.append({
                        "position": int(len(current_ids) - prompt_len - len(new_tokens)),
                        "predicted_incorrect": bool(is_predicted_incorrect) if not is_always_mode else None,
                        "score_incorrect": float(score_incorrect) if score_incorrect is not None else None,
                        "score_correct": float(score_correct) if score_correct is not None else None,
                        "prob_incorrect": float(prob_incorrect) if prob_incorrect is not None else None,
                        "intervened": True,
                        "injection": continuation_text,
                        "r1_checkpoint": True,
                    })

                    # Continue generation
                    continue
                else:
                    # Add token and continue
                    current_ids.append(next_token_id)
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    continue

        # Check if next token is #### or EOS (for non-R1 mode)
        if not use_r1_intervention and (next_token_id == hash_token_id or next_token_id == tokenizer.eos_token_id):
            token_name = "####" if next_token_id == hash_token_id else "EOS"

            # Only check with predictor once at the FIRST #### checkpoint
            # (ALWAYS_* modes check at every #### until max_interventions)
            if not has_checked_once and next_token_id == hash_token_id:
                if verbose:
                    print(f"  {q_prefix}  ⚡ CHECKPOINT: Detected {token_name} at position {len(current_ids) - prompt_len}")

                should_intervene = False
                score_incorrect = None
                score_correct = None
                prob_incorrect = None

                # For error-aware modes, run predictor
                if not is_always_mode and predictor is not None:
                    if verbose:
                        print(f"  {q_prefix}    → Running predictor...")

                    # Extract activations
                    current_position = len(current_ids)
                    hash_activation = extract_activations_at_position(current_ids, current_position, model, device)
                    last_step_activation = find_last_step_activation(current_ids, model, tokenizer, device)

                    if last_step_activation is None:
                        if verbose:
                            print(f"  {q_prefix}    ⚠ No steps found, allowing {token_name}")
                        has_checked_once = True  # Mark as checked
                        current_ids.append(next_token_id)
                        if next_token_id == tokenizer.eos_token_id:
                            break
                        continue

                    # Predict correctness
                    hash_minus_last = hash_activation[predictor_layer] - last_step_activation[predictor_layer]
                    is_predicted_incorrect, score_incorrect, score_correct, prob_incorrect = predict_correctness(
                        hash_minus_last, predictor, config.predictor_threshold
                    )

                    if verbose:
                        pred_label = "INCORRECT" if is_predicted_incorrect else "CORRECT"
                        print(f"  {q_prefix}    → Predictor: {pred_label} (score_incorrect={score_incorrect:.3f}, score_correct={score_correct:.3f})")

                    # ALWAYS intervene if predicted incorrect (this is the ONLY check for error-aware)
                    should_intervene = is_predicted_incorrect

                # For ALWAYS_* modes, always intervene (up to max)
                elif is_always_mode and num_interventions < config.max_interventions:
                    should_intervene = True

                # Mark that we've checked once (for error-aware modes only)
                # ALWAYS_* modes continue checking at every #### until max_interventions
                if not is_always_mode:
                    has_checked_once = True

                # Intervene if needed
                if should_intervene:
                    # Count current steps
                    decoded_so_far = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)
                    k = count_steps(decoded_so_far) + 1

                    # Determine injection text based on mode
                    if config.mode in ["STEP", "ALWAYS_STEP"]:
                        continuation_text = f"\nStep {k}: "
                    elif config.mode in ["WAIT", "ALWAYS_WAIT"]:
                        continuation_text = "\nWait, "
                    elif config.mode in ["HMM", "ALWAYS_HMM"]:
                        continuation_text = "\nHmm, "
                    elif config.mode in ["DOUBLE_CHECK", "ALWAYS_DOUBLE_CHECK"]:
                        continuation_text = "\nWait, let me double check. "
                    else:
                        raise ValueError(f"Unknown mode: {config.mode}")

                    if verbose:
                        print(f"  {q_prefix}    🎯 INJECTING: '{continuation_text.strip()}'")

                    # Encode and inject tokens
                    new_tokens = tokenizer.encode(continuation_text, add_special_tokens=False)
                    current_ids.extend(new_tokens)
                    num_interventions += 1

                    intervention_history.append({
                        "position": int(len(current_ids) - prompt_len - len(new_tokens)),
                        "predicted_incorrect": bool(is_predicted_incorrect) if not is_always_mode else None,
                        "score_incorrect": float(score_incorrect) if score_incorrect is not None else None,
                        "score_correct": float(score_correct) if score_correct is not None else None,
                        "prob_incorrect": float(prob_incorrect) if prob_incorrect is not None else None,
                        "intervened": True,
                        "injection": continuation_text,
                    })

                    # Continue generation
                    continue
                else:
                    # Predicted correct OR max interventions reached - allow ####
                    if verbose:
                        if is_always_mode:
                            print(f"  {q_prefix}    ✓ Max interventions reached, allowing {token_name}")
                        elif num_interventions >= config.max_interventions:
                            print(f"  {q_prefix}    ⚠ Max interventions reached, allowing {token_name}")
                        else:
                            print(f"  {q_prefix}    ✓ Predicted correct, allowing {token_name}")

                    current_ids.append(next_token_id)

                    if not is_always_mode and score_incorrect is not None:
                        intervention_history.append({
                            "position": int(len(current_ids) - prompt_len - 1),
                            "predicted_incorrect": False,
                            "score_incorrect": float(score_incorrect),
                            "score_correct": float(score_correct),
                            "prob_incorrect": float(prob_incorrect),
                            "intervened": False,
                        })

                    if next_token_id == tokenizer.eos_token_id:
                        break
                    continue
            else:
                # Already checked once (for error-aware) OR it's EOS - just allow the token
                if verbose and next_token_id == hash_token_id:
                    if is_always_mode:
                        print(f"  {q_prefix}  → Allowing subsequent {token_name} at position {len(current_ids) - prompt_len} (max interventions reached)")
                    else:
                        print(f"  {q_prefix}  → Allowing subsequent {token_name} at position {len(current_ids) - prompt_len} (already checked once)")
                current_ids.append(next_token_id)
                if next_token_id == tokenizer.eos_token_id:
                    break
                continue
        else:
            # Not ####, just add token
            current_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break

    # Extract intervened results
    intervened_text = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)
    intervened_answer = extract_answer(intervened_text, task=config.task, use_r1_fallback=config.use_r1_fallback, gold_answer=gold_answer)

    # Use appropriate evaluation based on task
    if config.task in ("math", "math-500"):
        intervened_correct = evaluate_math_answer(intervened_answer if intervened_answer else "N/A", gold_answer)
    else:
        intervened_correct = evaluate_answer(intervened_answer if intervened_answer else "N/A", gold_answer)
    intervened_num_steps = count_steps(intervened_text)

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
        "num_interventions": int(num_interventions),
        "intervention_history": intervention_history,
    }

    if verbose:
        correct_mark = "✓" if intervened_correct else "✗"
        print(f"  {q_prefix}→ Intervened: {correct_mark} answer={intervened_answer}, steps={intervened_num_steps}, interventions={num_interventions}")

        # Show outcome
        if not baseline_correct and intervened_correct:
            print(f"  {q_prefix}🎉 SUCCESS: Fixed wrong answer!")
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

    parser = argparse.ArgumentParser(description="Text injection intervention")
    parser.add_argument("--predictor", type=Path, default=None,
                        help="Path to predictor NPZ file (required for error-aware modes)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["STEP", "WAIT", "HMM", "DOUBLE_CHECK",
                                "ALWAYS_STEP", "ALWAYS_WAIT", "ALWAYS_HMM", "ALWAYS_DOUBLE_CHECK"],
                        help="Intervention mode")
    parser.add_argument("--max-interventions", type=int, default=1,
                        help="Maximum intervention attempts per question (default: 1, intervene only once)")
    parser.add_argument("--predictor-threshold", type=float, default=0.5,
                        help="Probability threshold for error prediction (default: 0.5)")

    parser.add_argument("--merged-dir", type=Path,
                        default=Path("output/complete_artifacts/gsm8k_train/merged"),
                        help="Directory with merged JSON files")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model name or path")
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Number of questions to process")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/text_injection_interventions"),
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--use-r1-fallback", action="store_true",
                        help="Use R1-specific answer extraction fallback")

    # Sharding arguments (for multi-GPU)
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID (for multi-GPU processing)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (for multi-GPU processing)")

    args = parser.parse_args()

    # Validate predictor for error-aware modes
    if not args.mode.startswith("ALWAYS_") and args.predictor is None:
        print(f"Error: --predictor is required for mode {args.mode}")
        return 1

    print(f"\n{'='*100}")
    print("TEXT INJECTION INTERVENTION")
    print(f"{'='*100}")
    if args.predictor:
        print(f"Predictor: {args.predictor}")
    print(f"Mode: {args.mode}")
    print(f"Max interventions: {args.max_interventions}")
    print(f"{'='*100}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load predictor if needed
    predictor = None
    threshold = args.predictor_threshold
    if args.predictor:
        predictor = load_predictor(args.predictor)
        # Use tuned threshold from predictor if available
        threshold = predictor.get('best_threshold', args.predictor_threshold)
        if 'best_threshold' in predictor and args.predictor_threshold != 0.5:
            print(f"\n⚠ Warning: Using tuned threshold {threshold:.3f} from predictor (ignoring --predictor-threshold {args.predictor_threshold})")
        elif 'best_threshold' not in predictor:
            print(f"\nUsing threshold {threshold:.3f} from command-line argument (predictor not tuned)")

    # Detect task from directory name
    merged_dir_name = str(args.merged_dir).lower()
    if "math-500" in merged_dir_name or "math_500" in merged_dir_name:
        task = "math-500"
        print(f"\nDetected MATH-500 dataset from directory name")
    elif "math" in merged_dir_name:
        task = "math"
        print(f"\nDetected MATH dataset from directory name")
    else:
        task = "gsm8k"
        print(f"\nUsing GSM8K dataset (default)")

    if not args.mode.startswith("ALWAYS_"):
        print(f"\n{'='*100}")
        print(f"FINAL CONFIGURATION")
        print(f"{'='*100}")
        print(f"Task: {task}")
        print(f"Predictor threshold (τ): {threshold:.3f} {'(tuned from validation set)' if predictor and 'best_threshold' in predictor else '(default)'}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"Use R1 fallback: {args.use_r1_fallback}")
        print(f"{'='*100}\n")

    # Create intervention config
    intervention_config = InterventionConfig(
        mode=args.mode,
        max_interventions=args.max_interventions,
        max_new_tokens=args.max_new_tokens,
        predictor_threshold=threshold,
        task=task,
        use_r1_fallback=args.use_r1_fallback,
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
    all_question_files = []

    # Determine file pattern based on task
    if task in ("math", "math-500"):
        # MATH files: test_algebra_101.json, test_counting_and_probability_20.json, etc.
        for json_file in sorted(args.merged_dir.glob("test_*.json")):
            all_question_files.append(json_file)
    else:
        # GSM8K files: gsm8k_0.json, gsm8k_1.json, etc.
        for json_file in sorted(args.merged_dir.glob("gsm8k_*.json")):
            all_question_files.append(json_file)

    print(f"Found {len(all_question_files)} questions")

    # Apply num_questions limit
    if args.num_questions is not None and args.num_questions < len(all_question_files):
        all_question_files = all_question_files[:args.num_questions]
        print(f"Limited to first {args.num_questions} questions")

    # Apply sharding
    if args.shard_id is not None and args.num_shards is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: Filtering questions...")
        selected_files = [f for idx, f in enumerate(all_question_files) if idx % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id} will process {len(selected_files)}/{len(all_question_files)} questions")
    else:
        selected_files = all_question_files
        print(f"Processing {len(selected_files)} questions")

    # Process questions
    print(f"\n{'='*100}")
    print("GENERATING WITH TEXT INJECTION")
    print(f"{'='*100}\n")

    results = []
    stats = {
        "total": 0,
        "baseline_correct": 0,
        "intervened_correct": 0,
        "flipped_wrong_to_correct": 0,
        "flipped_correct_to_wrong": 0,
        "total_interventions": 0,
    }

    is_always_mode = args.mode.startswith("ALWAYS_")
    pbar = tqdm(selected_files, desc="Processing", ncols=120)
    for json_path in pbar:
        if not json_path.exists():
            continue

        with open(json_path, 'r') as f:
            question_data = json.load(f)

        # Use filename as question ID for tracking
        question_id = json_path.stem

        input_ids = question_data.get("input_ids", [])
        gold_answer = question_data.get("gold_answer", question_data.get("answer", "N/A"))
        question_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Load dp2_idx for R1 intervention timing
        dp2_idx = question_data.get("dp2_idx")

        # Check if we should use R1 intervention mode
        use_r1_intervention = False
        if "r1" in args.model.lower() or "deepseek" in args.model.lower():
            use_r1_intervention = True

        # Check if we have saved baseline (for MATH datasets)
        saved_baseline_text = question_data.get("produced_text")
        saved_baseline_correct = question_data.get("is_correct")

        if not input_ids:
            continue

        try:
            result = generate_with_text_injection(
                input_ids, question_text, gold_answer,
                model, tokenizer, predictor,
                intervention_config, device,
                question_id=question_id,
                verbose=True,
                dp2_idx=dp2_idx,
                use_r1_intervention=use_r1_intervention,
                saved_baseline_text=saved_baseline_text,
                saved_baseline_correct=saved_baseline_correct
            )

            result["question_id"] = question_id
            results.append(result)

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

            stats["total_interventions"] += result["intervened"]["num_interventions"]

            # Determine predictor prediction and intervened correctness
            intervention_occurred = result["intervened"]["num_interventions"] > 0
            predictor_checked = len(result["intervened"]["intervention_history"]) > 0
            predictor_pred = "N/A"
            intervened_correctness = "N/A"

            if not is_always_mode and result["intervened"]["intervention_history"]:
                # Find the checkpoint that triggered intervention (if any)
                for checkpoint in result["intervened"]["intervention_history"]:
                    if checkpoint.get("intervened", False):
                        # This checkpoint triggered intervention
                        predictor_pred = "Wrong"
                        intervened_correctness = "✓" if intervened_correct else "✗"
                        break
                else:
                    # No intervention occurred - predictor predicted correct or wrong but didn't intervene
                    if result["intervened"]["intervention_history"]:
                        first_checkpoint = result["intervened"]["intervention_history"][0]
                        if first_checkpoint.get("predicted_incorrect", False):
                            # Predicted wrong but didn't intervene (max interventions or check-once)
                            predictor_pred = "Wrong"
                            intervened_correctness = "✗" if not intervened_correct else "✓"
                        else:
                            predictor_pred = "Correct"
                    intervened_correctness = "N/A" if predictor_pred == "Correct" else intervened_correctness
            elif is_always_mode and intervention_occurred:
                # ALWAYS mode - always intervene, no predictor
                intervened_correctness = "✓" if intervened_correct else "✗"

            # Update tqdm with current question status and running accuracy
            postfix_dict = {
                "Base": "✓" if baseline_correct else "✗",
                "BaseAcc": f"{stats['baseline_correct']}/{stats['total']}",
                "After_Intv": intervened_correctness,
                "IntvAcc": f"{stats['intervened_correct']}/{stats['total']}"
            }
            if not is_always_mode:
                postfix_dict["Pred"] = predictor_pred
            pbar.set_postfix(postfix_dict)

            # Print detailed information only when intervention occurs
            if intervention_occurred:
                print(f"\n{'='*100}")
                print(f"INTERVENTION DETAILS - Q{question_id}")
                print(f"{'='*100}")
                print(f"Gold answer: {gold_answer}")
                print(f"Mode: {args.mode}")

                print(f"\n{'='*100}")
                print(f"BASELINE RESPONSE:")
                print(f"{'='*100}")
                print(result['baseline']['text'])
                print(f"\n{'─'*100}")
                print(f"Extracted answer: {result['baseline']['answer']} {'✓ CORRECT' if baseline_correct else '✗ WRONG'}")
                print(f"Steps: {result['baseline']['num_steps']}")
                print(f"{'─'*100}")

                if not is_always_mode and result["intervened"]["intervention_history"]:
                    print(f"\n{'='*100}")
                    print(f"PREDICTOR DECISION:")
                    print(f"{'='*100}")
                    for i, h in enumerate(result["intervened"]["intervention_history"]):
                        if h["intervened"]:
                            print(f"  Checkpoint {i+1}: Predicted INCORRECT (p={h.get('prob_incorrect', 0):.3f}) → INJECTED '{h.get('injection', '').strip()}'")
                        else:
                            pred_status = "INCORRECT" if h.get("predicted_incorrect", False) else "CORRECT"
                            print(f"  Checkpoint {i+1}: Predicted {pred_status} (p={h.get('prob_incorrect', 0):.3f}) → No intervention")
                elif is_always_mode:
                    print(f"\n{'='*100}")
                    print(f"INJECTION (ALWAYS MODE):")
                    print(f"{'='*100}")
                    for i, h in enumerate(result["intervened"]["intervention_history"]):
                        if h["intervened"]:
                            print(f"  Checkpoint {i+1}: INJECTED '{h.get('injection', '').strip()}'")

                print(f"\n{'='*100}")
                print(f"INTERVENED RESPONSE:")
                print(f"{'='*100}")
                print(result['intervened']['text'])
                print(f"\n{'─'*100}")
                print(f"Extracted answer: {result['intervened']['answer']} {'✓ CORRECT' if intervened_correct else '✗ WRONG'}")
                print(f"Steps: {result['intervened']['num_steps']}")
                print(f"# Interventions: {result['intervened']['num_interventions']}")
                print(f"{'─'*100}")

                print(f"\n{'='*100}")
                print(f"OUTCOME:")
                print(f"{'='*100}")
                if not baseline_correct and intervened_correct:
                    print(f"  🎉 SUCCESS: Fixed wrong answer!")
                elif baseline_correct and not intervened_correct:
                    print(f"  ⚠ REGRESSION: Broke correct answer!")
                elif not baseline_correct and not intervened_correct:
                    print(f"  → No improvement (still incorrect)")
                else:
                    print(f"  → Maintained correctness")
                print(f"{'='*100}\n")

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

    # Generate output filename
    if args.shard_id is not None and args.num_shards is not None:
        output_file = args.output_dir / f"{args.mode}" / f"shard_{args.shard_id}" / "results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = args.output_dir / f"results_{args.mode}.json"

    # Compute summary
    if stats["total"] > 0:
        summary = {
            "total_questions": stats["total"],
            "baseline_accuracy": stats["baseline_correct"] / stats["total"],
            "intervened_accuracy": stats["intervened_correct"] / stats["total"],
            "accuracy_change": (stats["intervened_correct"] - stats["baseline_correct"]) / stats["total"],
            "flipped_wrong_to_correct": stats["flipped_wrong_to_correct"],
            "flipped_correct_to_wrong": stats["flipped_correct_to_wrong"],
            "avg_interventions_per_question": stats["total_interventions"] / stats["total"],
        }
    else:
        summary = {}

    output_data = {
        "config": asdict(intervention_config),
        "predictor_path": str(args.predictor) if args.predictor else None,
        "stats": stats,
        "summary": summary,
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"Total questions: {stats['total']}")
    print(f"Baseline accuracy: {stats['baseline_correct']}/{stats['total']} ({summary.get('baseline_accuracy', 0):.2%})")
    print(f"Intervened accuracy: {stats['intervened_correct']}/{stats['total']} ({summary.get('intervened_accuracy', 0):.2%})")
    print(f"Accuracy change: {summary.get('accuracy_change', 0):+.2%}")
    print(f"Flipped wrong→correct: {stats['flipped_wrong_to_correct']}")
    print(f"Flipped correct→wrong: {stats['flipped_correct_to_wrong']}")
    print(f"Avg interventions per question: {summary.get('avg_interventions_per_question', 0):.2f}")
    print(f"{'='*100}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
