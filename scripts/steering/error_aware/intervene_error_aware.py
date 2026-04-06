#!/usr/bin/env python
"""Error-Aware Steering Intervention

Uses a trained predictor to detect potential errors before generating the final answer.
If an error is predicted, applies steering vectors to prolong reasoning by delaying
the "####" token and encouraging more reasoning steps.

Strategy:
1. Generate deterministically until one timestep before "####" would be predicted
2. Extract hash_minus_last features from activations
3. Use trained predictor to classify if answer will be correct
4. If predicted incorrect:
   - Apply (step - hash) steering to prolong reasoning
   - Continue generation to get new "####" prediction
   - Repeat until predictor predicts correct OR max iterations reached
5. If predicted correct, allow "####" generation

Three intervention modes:
- PROLONG_LAST_N: Apply steering to last N layers (default: 5)
- PROLONG_FIRST_N: Apply steering to first N layers (default: 5)
- PROLONG_ALL_LAYERS: Apply steering to all layers

Usage:
    python scripts/steering/error_aware/intervene_error_aware.py \
        --predictor output/predictor_8000/llamainst/hash_minus_last_correctness_layer31.npz \
        --steering output/steering_vectors_llamainst8000.npz \
        --mode PROLONG_LAST_N \
        --alpha 1.0 \
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

from src.utils import evaluate_answer


@dataclass
class InterventionConfig:
    """Configuration for error-aware intervention"""
    mode: str  # PROLONG_LAST_N, PROLONG_FIRST_N, PROLONG_ALL_LAYERS, PROLONG_MID_N
    alpha: float = 1.0
    n_layers: int = 5  # For LAST_N, FIRST_N, or MID_N modes
    center_layer: int = 15  # Center layer for PROLONG_MID_N mode
    multi_timestep: bool = False  # Apply steering across multiple timesteps (not just one)
    max_interventions: int = 1  # Max number of intervention attempts (intervene only once)
    max_new_tokens: int = 1024  # Max tokens to generate
    predictor_threshold: float = 0.5  # Probability threshold for "incorrect" prediction


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
    """Load predictor classifier from NPZ file

    Returns:
        Dict with coefficients, intercept, scaler_mean, scaler_std, best_threshold (optional)
    """
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


def predict_correctness(
    hash_minus_last_features: np.ndarray,
    predictor: Dict,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """Predict if answer will be correct using the classifier

    Args:
        hash_minus_last_features: [hidden_dim] feature vector
        predictor: Predictor dict with coefficients, intercept, scaler
        threshold: Probability threshold for "incorrect" class

    Returns:
        is_predicted_incorrect: True if predicted to be incorrect
        probability: Probability of being incorrect (y=1)
    """
    # Standardize features
    features_scaled = (hash_minus_last_features - predictor['scaler_mean']) / predictor['scaler_std']

    # Compute logits: w^T x + b
    logit = np.dot(predictor['coefficients'], features_scaled) + predictor['intercept']

    # Sigmoid to get probability
    probability = 1.0 / (1.0 + np.exp(-logit))

    # y=1 means incorrect, so check if prob > threshold
    is_predicted_incorrect = (probability > threshold)

    return is_predicted_incorrect, float(probability)


@torch.no_grad()
def extract_activations_at_position(
    input_ids: List[int],
    position: int,
    model,
    device: torch.device
) -> np.ndarray:
    """Extract hidden state activations at a specific position

    Args:
        input_ids: Token IDs up to and including target position
        position: Position to extract (will extract hidden state at position-1)
        model: Model
        device: Device

    Returns:
        activations: [num_layers, hidden_dim] activations at position
    """
    num_layers = len(model.model.layers)

    # Extract hidden state at position BEFORE the target token
    target_pos = position - 1
    if target_pos < 0:
        raise ValueError(f"Cannot extract activations at position {position} (target_pos={target_pos} < 0)")

    # Forward pass up to target position
    input_tensor = torch.tensor([input_ids[:target_pos + 1]], dtype=torch.long).to(device)

    outputs = model(
        input_tensor,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True
    )

    # Extract activations from each layer
    layer_activations = []
    for layer_idx in range(num_layers):
        # hidden_states[layer_idx + 1] because [0] is embedding
        hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]  # [1, hidden_dim]
        layer_activations.append(hidden.squeeze(0))

    # Stack: [num_layers, hidden_dim]
    activations = torch.stack(layer_activations, dim=0).float().cpu().numpy()

    return activations


def find_last_step_activation(
    input_ids: List[int],
    model,
    tokenizer,
    device: torch.device
) -> Optional[np.ndarray]:
    """Find and extract the activation at the last "Step N:" occurrence

    Args:
        input_ids: Full sequence of token IDs
        model: Model
        tokenizer: Tokenizer
        device: Device

    Returns:
        last_step_activation: [num_layers, hidden_dim] or None if no steps found
    """
    # Decode tokens to find Step positions
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Find all "Step N:" patterns
    pattern = r'\bStep\s+\d+\s*[:.]'
    step_matches = list(re.finditer(pattern, generated_text, re.IGNORECASE))

    if len(step_matches) == 0:
        return None

    # Find the last step's token position
    last_step_char_pos = step_matches[-1].start()

    # Find corresponding token position
    decoded_so_far = ""
    last_step_token_pos = None

    for token_idx in range(len(input_ids)):
        decoded_so_far = tokenizer.decode(input_ids[:token_idx + 1], skip_special_tokens=True)

        if last_step_char_pos <= len(decoded_so_far):
            last_step_token_pos = token_idx
            break

    if last_step_token_pos is None:
        return None

    # Extract activation at this position
    try:
        activation = extract_activations_at_position(
            input_ids, last_step_token_pos, model, device
        )
        return activation
    except ValueError:
        return None


class SteeringHook:
    """Hook to apply steering intervention during generation"""

    def __init__(
        self,
        steering_vectors: np.ndarray,
        config: InterventionConfig,
        num_layers: int,
        predictor_layer: int
    ):
        """
        Args:
            steering_vectors: [num_layers, hidden_dim] steering vectors (step - hash)
            config: InterventionConfig
            num_layers: Total number of layers
            predictor_layer: Layer index used by predictor
        """
        self.steering_vectors = torch.from_numpy(steering_vectors).float()
        self.config = config
        self.num_layers = num_layers
        self.predictor_layer = predictor_layer
        self.device = None

        # Determine which layers to intervene on
        if config.mode == "PROLONG_LAST_N":
            self.intervene_layers = set(range(num_layers - config.n_layers, num_layers))
        elif config.mode == "PROLONG_FIRST_N":
            self.intervene_layers = set(range(min(config.n_layers, num_layers)))
        elif config.mode == "PROLONG_MID_N":
            # Middle layers centered around center_layer
            # For N=5, center=15: layers 13, 14, 15, 16, 17
            # For even N, add extra layer to late side
            n = config.n_layers
            center = config.center_layer
            half = n // 2

            if n % 2 == 0:
                # Even: extra layer on late side
                # N=6, center=15: [13, 14, 15, 16, 17, 18]
                start = center - (half - 1)
                end = center + half + 1
            else:
                # Odd: symmetric
                # N=5, center=15: [13, 14, 15, 16, 17]
                start = center - half
                end = center + half + 1

            self.intervene_layers = set(range(max(0, start), min(num_layers, end)))
        else:  # PROLONG_ALL_LAYERS
            self.intervene_layers = set(range(num_layers))

        # DO NOT normalize steering vectors - preserve original magnitudes!
        # Alpha scaling should work on the original vector magnitudes

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
        # (step - hash) vector should prolong reasoning, delaying ####
        hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
        steering = self.steering_vectors[layer_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]

        # Apply to last position only (multi-timestep is handled by keeping hooks active)
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
def generate_with_error_aware_intervention(
    input_ids: List[int],
    question_text: str,
    gold_answer: str,
    model,
    tokenizer,
    predictor: Dict,
    steering_vectors: np.ndarray,
    config: InterventionConfig,
    device: torch.device,
    question_id: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """Generate with error-aware intervention

    Strategy:
    1. Generate baseline (no intervention)
    2. For intervention attempt:
       - Generate until about to predict "####"
       - Extract hash_minus_last features
       - Predict correctness
       - If predicted incorrect, apply steering and continue
       - Repeat until predicted correct or max iterations

    Returns:
        Dict with results including baseline and intervention outcomes
    """
    num_layers = len(model.model.layers)
    predictor_layer = predictor['layer_idx']

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
    # ERROR-AWARE INTERVENTION GENERATION
    # =====================================================================
    if verbose:
        print(f"  {q_prefix}→ Starting error-aware intervention generation...")

    # Start with prompt
    current_ids = input_ids.copy()
    intervention_history = []
    num_interventions = 0
    has_checked_once = False  # Flag to ensure we only check predictor once
    steering_active = False  # Flag to track if steering hooks are currently active
    steering_handles = None  # Handles for active steering hooks

    # Generate tokens one by one, checking before each potential "####"
    max_total_tokens = prompt_len + config.max_new_tokens
    hash_token_id = hash_token_ids[0] if hash_token_ids else tokenizer.encode("####", add_special_tokens=False)[0]

    while len(current_ids) < max_total_tokens:
        # Generate next token
        input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
        outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        next_token_id = torch.argmax(logits, dim=-1).item()

        # Check if next token is "####"
        if next_token_id == hash_token_id or next_token_id == tokenizer.eos_token_id:
            # About to generate "####" or EOS
            token_name = "####" if next_token_id == hash_token_id else "EOS"

            # Only check with predictor once at the FIRST #### checkpoint
            if not has_checked_once and next_token_id == hash_token_id:
                if verbose:
                    print(f"  {q_prefix}  ⚡ CHECKPOINT: Detected {token_name} at position {len(current_ids) - prompt_len}")

                # Extract current position activations (hash position)
                if verbose:
                    print(f"  {q_prefix}    → Extracting hash activations...")
                current_position = len(current_ids)
                hash_activation = extract_activations_at_position(
                    current_ids, current_position, model, device
                )

                # Extract last step activation
                if verbose:
                    print(f"  {q_prefix}    → Extracting last step activations...")
                last_step_activation = find_last_step_activation(
                    current_ids, model, tokenizer, device
                )

                if last_step_activation is None:
                    # No steps found, cannot compute hash_minus_last
                    # Allow token generation and mark as checked
                    if verbose:
                        print(f"  {q_prefix}    ⚠ No steps found, allowing {token_name}")
                    has_checked_once = True
                    current_ids.append(next_token_id)
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    continue

                # Compute hash_minus_last features for predictor layer
                hash_minus_last = hash_activation[predictor_layer] - last_step_activation[predictor_layer]

                # Predict correctness
                if verbose:
                    print(f"  {q_prefix}    → Running predictor (layer {predictor_layer})...")
                is_predicted_incorrect, prob_incorrect = predict_correctness(
                    hash_minus_last, predictor, config.predictor_threshold
                )

                intervention_history.append({
                    "position": int(len(current_ids) - prompt_len),
                    "predicted_incorrect": bool(is_predicted_incorrect),
                    "prob_incorrect": float(prob_incorrect),
                    "intervened": False,
                })

                if verbose:
                    pred_label = "INCORRECT" if is_predicted_incorrect else "CORRECT"
                    print(f"  {q_prefix}    → Predictor: {pred_label} (p_incorrect={prob_incorrect:.3f})")

                # Mark that we've checked once
                has_checked_once = True

                if is_predicted_incorrect:
                    # ALWAYS intervene if predicted incorrect (this is the ONLY check)
                    if verbose:
                        mode_desc = "multi-timestep" if config.multi_timestep else "single-timestep"
                        print(f"  {q_prefix}    🎯 INTERVENING: Applying steering ({mode_desc})...")

                    # Create steering hook
                    steering_hook = SteeringHook(
                        steering_vectors, config, num_layers, predictor_layer
                    )
                    steering_handles = register_steering_hooks(model, steering_hook)
                    steering_active = True

                    # Generate next token WITH steering
                    input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
                    outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
                    logits = outputs.logits[:, -1, :]
                    next_token_id_steered = torch.argmax(logits, dim=-1).item()

                    current_ids.append(next_token_id_steered)
                    num_interventions += 1

                    intervention_history[-1]["intervened"] = True
                    intervention_history[-1]["steered_token"] = tokenizer.decode([next_token_id_steered])

                    if verbose:
                        print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")

                    # For single-timestep mode, remove hooks immediately
                    if not config.multi_timestep:
                        remove_hooks(steering_handles)
                        steering_handles = None
                        steering_active = False
                        if verbose:
                            print(f"  {q_prefix}      → Steering deactivated (single-timestep mode)")

                    # Continue generation
                    if next_token_id_steered == tokenizer.eos_token_id:
                        if steering_active:
                            remove_hooks(steering_handles)
                            steering_active = False
                        break
                    continue
                else:
                    # Predicted correct - allow "####"
                    if verbose:
                        print(f"  {q_prefix}    ✓ Predicted correct, allowing {token_name}")

                    current_ids.append(next_token_id)
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    continue
            else:
                # Already checked once OR it's EOS - just allow the token
                if verbose and next_token_id == hash_token_id:
                    print(f"  {q_prefix}  → Allowing subsequent {token_name} at position {len(current_ids) - prompt_len} (already checked once)")
                current_ids.append(next_token_id)
                if next_token_id == tokenizer.eos_token_id:
                    break
                continue
        else:
            # Not "####", just add token
            current_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break

    # Cleanup: Remove steering hooks if still active
    if steering_active and steering_handles is not None:
        remove_hooks(steering_handles)
        steering_active = False
        if verbose:
            print(f"  {q_prefix}→ Steering deactivated at end of generation")

    # Extract intervened results
    intervened_text = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)
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

    parser = argparse.ArgumentParser(
        description="Error-aware steering intervention"
    )
    parser.add_argument("--predictor", type=Path, required=True,
                        help="Path to predictor NPZ file (e.g., hash_minus_last_correctness_layer31.npz)")
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
                        help="Apply steering across multiple timesteps (not just one)")
    parser.add_argument("--max-interventions", type=int, default=1,
                        help="Maximum intervention attempts per question (default: 1, intervene only once)")
    parser.add_argument("--predictor-threshold", type=float, default=0.5,
                        help="Probability threshold for 'incorrect' prediction (default: 0.5)")

    parser.add_argument("--merged-dir", type=Path,
                        default=Path("output/complete_artifacts/gsm8k_train/merged"),
                        help="Directory with merged JSON files")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model name or path")
    parser.add_argument("--num-questions", type=int, default=50,
                        help="Number of questions to process")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/error_aware_interventions"),
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # Sharding arguments (for multi-GPU)
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID (for multi-GPU processing)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (for multi-GPU processing)")

    args = parser.parse_args()

    print(f"\n{'='*100}")
    print("ERROR-AWARE STEERING INTERVENTION")
    print(f"{'='*100}")
    print(f"Predictor: {args.predictor}")
    print(f"Steering: {args.steering}")
    print(f"Mode: {args.mode}")
    print(f"Alpha: {args.alpha}")
    print(f"N layers: {args.n_layers}")
    print(f"Max interventions: {args.max_interventions}")
    print(f"{'='*100}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load predictor
    predictor = load_predictor(args.predictor)

    # Load steering vectors
    steering_vectors, _, _ = load_steering_vectors(args.steering)

    # Use tuned threshold from predictor if available, otherwise use command-line arg
    threshold = predictor.get('best_threshold', args.predictor_threshold)
    if 'best_threshold' in predictor and args.predictor_threshold != 0.5:
        print(f"\n⚠ Warning: Using tuned threshold {threshold:.3f} from predictor (ignoring --predictor-threshold {args.predictor_threshold})")
    elif 'best_threshold' not in predictor:
        print(f"\nUsing threshold {threshold:.3f} from command-line argument (predictor not tuned)")

    print(f"\n{'='*100}")
    print(f"FINAL CONFIGURATION")
    print(f"{'='*100}")
    print(f"Predictor threshold (τ): {threshold:.3f} {'(tuned from validation set)' if 'best_threshold' in predictor else '(default)'}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"{'='*100}\n")

    # Create intervention config
    intervention_config = InterventionConfig(
        mode=args.mode,
        alpha=args.alpha,
        n_layers=args.n_layers,
        center_layer=args.center_layer,
        multi_timestep=args.multi_timestep,
        max_interventions=args.max_interventions,
        max_new_tokens=args.max_new_tokens,
        predictor_threshold=threshold,
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
    all_question_ids = []
    for json_file in sorted(args.merged_dir.glob("gsm8k_*.json")):
        try:
            question_id = int(json_file.stem.split("_")[1])
            all_question_ids.append(question_id)
        except (ValueError, IndexError):
            continue

    print(f"Found {len(all_question_ids)} questions")

    # Apply num_questions limit if specified
    if args.num_questions is not None and args.num_questions < len(all_question_ids):
        all_question_ids = all_question_ids[:args.num_questions]
        print(f"Limited to first {args.num_questions} questions")

    # Apply sharding if specified
    if args.shard_id is not None and args.num_shards is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: Filtering questions...")
        selected_questions = [qid for idx, qid in enumerate(all_question_ids) if idx % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id} will process {len(selected_questions)}/{len(all_question_ids)} questions")
    else:
        selected_questions = all_question_ids
        print(f"Processing {len(selected_questions)} questions")

    # Process questions
    print(f"\n{'='*100}")
    print("GENERATING WITH ERROR-AWARE INTERVENTION")
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

    pbar = tqdm(selected_questions, desc="Processing", ncols=120)
    for question_id in pbar:
        # Load question
        json_path = args.merged_dir / f"gsm8k_{question_id}.json"
        if not json_path.exists():
            continue

        with open(json_path, 'r') as f:
            question_data = json.load(f)

        input_ids = question_data.get("input_ids", [])
        gold_answer = question_data.get("gold_answer", question_data.get("answer", "N/A"))
        question_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        if not input_ids:
            continue

        # Generate with error-aware intervention
        try:
            result = generate_with_error_aware_intervention(
                input_ids, question_text, gold_answer,
                model, tokenizer, predictor, steering_vectors,
                intervention_config, device,
                question_id=question_id,
                verbose=True
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

            stats["total_interventions"] += result["intervened"]["num_interventions"]

            # Determine predictor prediction and intervened correctness
            intervention_occurred = result["intervened"]["num_interventions"] > 0
            predictor_checked = len(result["intervened"]["intervention_history"]) > 0
            predictor_pred = "N/A"
            intervened_correctness = "N/A"

            if result["intervened"]["intervention_history"]:
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

            # Update tqdm with current question status and running accuracy
            pbar.set_postfix({
                "Base": "✓" if baseline_correct else "✗",
                "BaseAcc": f"{stats['baseline_correct']}/{stats['total']}",
                "Pred": predictor_pred,
                "After_Intv": intervened_correctness,
                "IntvAcc": f"{stats['intervened_correct']}/{stats['total']}"
            })

            # Print detailed information only when intervention occurs
            if intervention_occurred:
                print(f"\n{'='*100}")
                print(f"INTERVENTION DETAILS - Q{question_id}")
                print(f"{'='*100}")
                print(f"Gold answer: {gold_answer}")

                print(f"\n{'='*100}")
                print(f"BASELINE RESPONSE:")
                print(f"{'='*100}")
                print(result['baseline']['text'])
                print(f"\n{'─'*100}")
                print(f"Extracted answer: {result['baseline']['answer']} {'✓ CORRECT' if baseline_correct else '✗ WRONG'}")
                print(f"Steps: {result['baseline']['num_steps']}")
                print(f"{'─'*100}")

                print(f"\n{'='*100}")
                print(f"PREDICTOR DECISION:")
                print(f"{'='*100}")
                for i, h in enumerate(result["intervened"]["intervention_history"]):
                    if h["intervened"]:
                        print(f"  Checkpoint {i+1}: Predicted INCORRECT (p={h['prob_incorrect']:.3f}) → INTERVENED")
                    else:
                        pred_status = "INCORRECT" if h.get("predicted_incorrect", False) else "CORRECT"
                        print(f"  Checkpoint {i+1}: Predicted {pred_status} (p={h.get('prob_incorrect', 0):.3f}) → No intervention")

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

    # Generate output filename with shard info if applicable
    if args.shard_id is not None and args.num_shards is not None:
        # Include mode and alpha in shard directory to avoid conflicts
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
            "flipped_wrong_to_correct": stats["flipped_wrong_to_correct"],
            "flipped_correct_to_wrong": stats["flipped_correct_to_wrong"],
            "avg_interventions_per_question": stats["total_interventions"] / stats["total"],
        }
    else:
        summary = {}

    output_data = {
        "config": asdict(intervention_config),
        "predictor_path": str(args.predictor),
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
