#!/usr/bin/env python
import sys
import json
import argparse
import yaml
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import evaluate_answer, evaluate_math_answer, extract_answer


@dataclass
class InterventionConfig:
    mode: str
    alpha: float = 1.0
    n_layers: int = 5
    center_layer: int = 15
    multi_timestep: bool = False
    max_new_tokens: int = 1024


def extract_answer_after_hash(text: str) -> Optional[str]:
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
    if "####" in text:
        reasoning_part = text.split("####")[0]
    else:
        reasoning_part = text
    pattern = r'\bStep\s+\d+\s*[:.]'
    matches = re.findall(pattern, reasoning_part, re.IGNORECASE)
    return len(matches)


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent.parent / "config" / "paths.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_path_from_config(config: dict, model_key: str = "llama-3.1-8b-instruct") -> str:
    models = config.get("models", {})
    if model_key not in models:
        raise ValueError(f"Model '{model_key}' not found in config")
    model_config = models[model_key]
    model_path = model_config.get("path")
    if not model_path:
        raise ValueError(f"Model '{model_key}' does not have 'path' field")
    return model_path


def load_steering_vectors(npz_path: Path) -> Tuple[np.ndarray, int, int]:
    print(f"\nLoading steering vectors from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    steering_vectors = data["steering_vectors"]
    num_layers = int(data["num_layers"])
    hidden_dim = int(data["hidden_dim"])
    print(f"  Shape: {steering_vectors.shape}")
    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    return steering_vectors, num_layers, hidden_dim


def _strip_timing_prefixes(mode: str) -> str:
    m = mode
    if m.startswith("ALL_"):
        m = m[len("ALL_"):]
    if m.startswith("AFTER_STEP1_"):
        m = m[len("AFTER_STEP1_"):]
    return m


def _step2_regex() -> re.Pattern:
    return re.compile(r"\bStep\s*2\s*[:.]", flags=re.IGNORECASE)


def _find_step2_start_token_offset(window_token_ids: List[int], tokenizer) -> Optional[int]:
    if not window_token_ids:
        return None
    pat = _step2_regex()
    decoded_full = tokenizer.decode(window_token_ids, skip_special_tokens=True)
    if not pat.search(decoded_full):
        return None

    for j in range(len(window_token_ids)):
        suffix = tokenizer.decode(window_token_ids[j:], skip_special_tokens=True)
        if pat.match(suffix.lstrip()):
            return j

    for j in range(len(window_token_ids)):
        suffix = tokenizer.decode(window_token_ids[j:], skip_special_tokens=True)
        if pat.search(suffix):
            return j

    return None


class SteeringHook:
    def __init__(self, steering_vectors: np.ndarray, config: InterventionConfig, num_layers: int):
        self.steering_vectors = torch.from_numpy(steering_vectors).float()
        self.config = config
        self.num_layers = num_layers
        self.device = None

        mode_for_layers = _strip_timing_prefixes(config.mode)

        if mode_for_layers == "PROLONG_LAST_N":
            self.intervene_layers = set(range(num_layers - config.n_layers, num_layers))
        elif mode_for_layers == "PROLONG_FIRST_N":
            self.intervene_layers = set(range(min(config.n_layers, num_layers)))
        elif mode_for_layers == "PROLONG_MID_N":
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
        else:
            self.intervene_layers = set(range(num_layers))

    def __call__(self, module, _input, output):
        layer_idx = getattr(module, "_layer_idx", None)
        if layer_idx is None or layer_idx not in self.intervene_layers:
            return output

        if self.device is None:
            self.device = output[0].device
            self.steering_vectors = self.steering_vectors.to(self.device)

        hidden_states = output[0]
        steering = self.steering_vectors[layer_idx].unsqueeze(0).unsqueeze(0)
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + self.config.alpha * steering
        return (hidden_states,) + output[1:]


def register_steering_hooks(model, steering_hook: SteeringHook) -> List:
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        layer._layer_idx = layer_idx
        handles.append(layer.register_forward_hook(steering_hook))
    return handles


def remove_hooks(handles: List):
    for h in handles:
        h.remove()


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
    task: str = "gsm8k",
) -> Dict:
    num_layers = len(model.model.layers)
    q_prefix = f"Q{question_id}: " if question_id is not None else ""

    if verbose:
        print(f"  {q_prefix}Starting generation...")

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    prompt_len = len(input_ids)

    if verbose:
        print(f"  {q_prefix}→ Generating baseline (no intervention)...")

    baseline_outputs = model.generate(
        input_tensor,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    baseline_ids = baseline_outputs[0].tolist()
    baseline_text = tokenizer.decode(baseline_ids[prompt_len:], skip_special_tokens=True)

    if task in ("math", "math-500"):
        baseline_answer = extract_answer(baseline_text, task=task)
        baseline_correct = evaluate_math_answer(baseline_answer if baseline_answer else "", gold_answer)
    else:
        baseline_answer = extract_answer_after_hash(baseline_text)
        baseline_correct = evaluate_answer(baseline_answer if baseline_answer else "N/A", gold_answer)

    baseline_num_steps = count_steps(baseline_text)

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

    current_ids = input_ids.copy()
    steering_active = False
    steering_handles = None
    hash_detected_at = None

    all_mode = config.mode.startswith("ALL_")
    after_step1_mode = config.mode.startswith("AFTER_STEP1_")

    if all_mode and after_step1_mode:
        raise ValueError(f"Invalid mode '{config.mode}': cannot combine ALL_ and AFTER_STEP1_ prefixes")

    if all_mode:
        if verbose:
            print(f"  {q_prefix}→ ALL MODE: Registering steering hooks immediately (from first generated token)...")
        steering_hook = SteeringHook(steering_vectors, config, num_layers)
        steering_handles = register_steering_hooks(model, steering_hook)
        steering_active = True
        hash_detected_at = 0
    else:
        if use_r1_intervention and dp2_idx is not None and (not after_step1_mode):
            if verbose:
                print(f"  {q_prefix}→ Generating with intervention (R1 MODE: starting before dp2_idx={dp2_idx})...")
        elif after_step1_mode:
            if verbose:
                print(f"  {q_prefix}→ Generating with intervention (AFTER_STEP1 MODE: start at Step 2 checkpoint)...")
        else:
            if verbose:
                print(f"  {q_prefix}→ Generating with intervention (starting at #### checkpoint)...")

    max_total_tokens = prompt_len + config.max_new_tokens
    hash_token_id = None
    if not after_step1_mode:
        hash_token_ids2 = tokenizer.encode("####", add_special_tokens=False)
        hash_token_id = hash_token_ids2[0] if hash_token_ids2 else tokenizer.encode("####", add_special_tokens=False)[0]

    r1_intervention_position = None
    if use_r1_intervention and dp2_idx is not None and (not after_step1_mode):
        r1_intervention_position = dp2_idx - 1
        if r1_intervention_position <= prompt_len:
            if verbose:
                print(f"  {q_prefix}  ⚠ WARNING: R1 intervention position {r1_intervention_position} <= prompt_len {prompt_len}")
                print(f"  {q_prefix}    Falling back to standard #### detection")
            use_r1_intervention = False
            r1_intervention_position = None
        elif verbose:
            print(f"  {q_prefix}  R1 intervention will trigger at position {r1_intervention_position - prompt_len} (absolute: {r1_intervention_position})")

    step2_triggered = False
    step2_pat = _step2_regex()
    step2_window_k = 64

    while len(current_ids) < max_total_tokens:
        if (not all_mode) and (not after_step1_mode) and use_r1_intervention and r1_intervention_position is not None:
            if len(current_ids) == r1_intervention_position and not steering_active:
                if verbose:
                    print(f"  {q_prefix}  ⚡ R1 CHECKPOINT: Reached position {len(current_ids) - prompt_len} (1 before dp2_idx)")
                    print(f"  {q_prefix}    🎯 INTERVENING: Registering steering hooks...")

                hash_detected_at = len(current_ids) - prompt_len
                steering_hook = SteeringHook(steering_vectors, config, num_layers)
                steering_handles = register_steering_hooks(model, steering_hook)
                steering_active = True

                input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
                outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                next_token_id_steered = torch.argmax(logits, dim=-1).item()
                current_ids.append(next_token_id_steered)

                if verbose:
                    print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")

                if next_token_id_steered == tokenizer.eos_token_id:
                    break
                continue

        input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
        outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()

        if (not all_mode) and (not after_step1_mode) and (not use_r1_intervention) and (hash_token_id is not None) and next_token_id == hash_token_id and (not steering_active):
            if verbose:
                print(f"  {q_prefix}  ⚡ CHECKPOINT: Detected #### at position {len(current_ids) - prompt_len}")
                print(f"  {q_prefix}    🎯 INTERVENING: Registering steering hooks...")

            hash_detected_at = len(current_ids) - prompt_len
            steering_hook = SteeringHook(steering_vectors, config, num_layers)
            steering_handles = register_steering_hooks(model, steering_hook)
            steering_active = True

            input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
            outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            next_token_id_steered = torch.argmax(logits, dim=-1).item()
            current_ids.append(next_token_id_steered)

            if verbose:
                print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")

            if next_token_id_steered == tokenizer.eos_token_id:
                break
            continue

        current_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break

        if after_step1_mode and (not step2_triggered) and (not steering_active):
            gen_ids = current_ids[prompt_len:]
            if gen_ids:
                window_ids = gen_ids[-step2_window_k:]
                window_txt = tokenizer.decode(window_ids, skip_special_tokens=True)
                if step2_pat.search(window_txt):
                    off = _find_step2_start_token_offset(window_ids, tokenizer)
                    if off is not None:
                        step2_start_abs = (len(current_ids) - len(window_ids)) + off
                        if step2_start_abs > prompt_len:
                            current_ids = current_ids[:step2_start_abs]
                            step2_triggered = True
                            hash_detected_at = step2_start_abs - prompt_len

                            if verbose:
                                print(f"  {q_prefix}  ⚡ STEP2 CHECKPOINT: Detected Step 2; rewinding to position {hash_detected_at}")
                                print(f"  {q_prefix}    🎯 INTERVENING: Registering steering hooks from this timestep...")

                            steering_hook = SteeringHook(steering_vectors, config, num_layers)
                            steering_handles = register_steering_hooks(model, steering_hook)
                            steering_active = True

                            input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
                            outputs = model(input_tensor, use_cache=False, return_dict=True, output_hidden_states=True)
                            logits = outputs.logits[:, -1, :]
                            next_token_id_steered = torch.argmax(logits, dim=-1).item()
                            current_ids.append(next_token_id_steered)

                            if verbose:
                                print(f"  {q_prefix}      → Steered token: '{tokenizer.decode([next_token_id_steered])}'")

                            if next_token_id_steered == tokenizer.eos_token_id:
                                break

    if steering_active and steering_handles is not None:
        remove_hooks(steering_handles)
        if verbose:
            print(f"  {q_prefix}→ Steering deactivated at end of generation")

    intervened_text = tokenizer.decode(current_ids[prompt_len:], skip_special_tokens=True)

    if task in ("math", "math-500"):
        intervened_answer = extract_answer(intervened_text, task=task)
        intervened_correct = evaluate_math_answer(intervened_answer if intervened_answer else "", gold_answer)
    else:
        intervened_answer = extract_answer_after_hash(intervened_text)
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
        "hash_detected_at": hash_detected_at,
    }

    if verbose:
        correct_mark = "✓" if intervened_correct else "✗"
        print(f"  {q_prefix}→ Intervened: {correct_mark} answer={intervened_answer}, steps={intervened_num_steps}")
        if hash_detected_at is not None:
            print(f"  {q_prefix}  (Intervention started at position {hash_detected_at})")

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
        cfg = load_config()
        default_model = get_model_path_from_config(cfg, "llama-3.1-8b-instruct")
    except Exception as e:
        print(f"Warning: Could not load model from config: {e}")
        default_model = "meta-llama/Llama-3.1-8B-Instruct"

    parser = argparse.ArgumentParser(description="Unconditional steering intervention (no error detection)")
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "PROLONG_LAST_N",
            "PROLONG_FIRST_N",
            "PROLONG_ALL_LAYERS",
            "PROLONG_MID_N",
            "ALL_PROLONG_LAST_N",
            "ALL_PROLONG_MID_N",
            "AFTER_STEP1_PROLONG_LAST_N",
            "AFTER_STEP1_PROLONG_FIRST_N",
            "AFTER_STEP1_PROLONG_ALL_LAYERS",
            "AFTER_STEP1_PROLONG_MID_N",
        ],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--center-layer", type=int, default=15)
    parser.add_argument("--multi-timestep", action="store_true")
    parser.add_argument("--merged-dir", type=Path, default=Path("output/complete_artifacts/gsm8k_train/merged"))
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("output/unconditional_interventions"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)

    args = parser.parse_args()

    merged_dir_str = str(args.merged_dir).lower()
    use_r1_intervention = ("r1" in merged_dir_str) or ("deepseek" in merged_dir_str)

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
    if args.mode.startswith("ALL_"):
        print("Timing: Steering ACTIVE from first generated token (all timesteps)")
    elif args.mode.startswith("AFTER_STEP1_"):
        print("Timing: Steering starts at Step 2 checkpoint (rewind to 1 timestep before Step 2 begins)")
    print("Intervention: Applied to ALL questions unconditionally")
    if use_r1_intervention and (not args.mode.startswith("AFTER_STEP1_")) and (not args.mode.startswith("ALL_")):
        print("R1 MODE: Detected R1/DeepSeek model - will apply intervention 1 timestep before first answer token")
    elif (not args.mode.startswith("AFTER_STEP1_")) and (not args.mode.startswith("ALL_")):
        print("Standard MODE: Will apply intervention when #### is detected")
    print(f"{'='*100}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    steering_vectors, _, _ = load_steering_vectors(args.steering)

    intervention_config = InterventionConfig(
        mode=args.mode,
        alpha=args.alpha,
        n_layers=args.n_layers,
        center_layer=args.center_layer,
        multi_timestep=args.multi_timestep,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nLoading model...")
    model_path = args.model
    if not Path(model_path).exists():
        try:
            cfg = load_config()
            model_path = get_model_path_from_config(cfg, args.model)
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
        local_files_only=True,
    ).eval()

    print(f"Model loaded: {len(model.model.layers)} layers")

    print(f"\nCollecting questions from {args.merged_dir}...")

    if task in ("math", "math-500"):
        json_files = sorted([
            f for f in args.merged_dir.glob("*.json")
            if f.name not in ("aggregated_results.json", "merged_results.json", "checkpoint.json")
        ])
        all_question_files = json_files
    else:
        all_question_files = sorted(args.merged_dir.glob("gsm8k_*.json"))

    print(f"Found {len(all_question_files)} questions")

    if args.num_questions is not None and args.num_questions < len(all_question_files):
        all_question_files = all_question_files[:args.num_questions]
        print(f"Limited to first {args.num_questions} questions")

    if args.shard_id is not None and args.num_shards is not None:
        print(f"\nShard {args.shard_id}/{args.num_shards}: Applying round-robin sharding...")
        selected_question_files = [qf for idx, qf in enumerate(all_question_files) if idx % args.num_shards == args.shard_id]
        print(f"  → Shard {args.shard_id} assigned {len(selected_question_files)}/{len(all_question_files)} questions")
    else:
        selected_question_files = all_question_files
        print(f"\nProcessing {len(selected_question_files)} questions")

    results = []
    stats = {
        "total": 0,
        "baseline_correct": 0,
        "intervened_correct": 0,
        "flipped_wrong_to_correct": 0,
        "flipped_correct_to_wrong": 0,
    }

    pbar = tqdm(selected_question_files, desc="Processing", ncols=120)
    for json_path in pbar:
        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            question_data = json.load(f)

        input_ids = question_data.get("input_ids", [])
        gold_answer = question_data.get("gold_answer", question_data.get("answer", "N/A"))
        question_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        dp2_idx = question_data.get("dp2_idx")
        question_id = question_data.get("id", json_path.stem)

        if not input_ids:
            continue

        try:
            result = generate_with_unconditional_intervention(
                input_ids=input_ids,
                question_text=question_text,
                gold_answer=gold_answer,
                model=model,
                tokenizer=tokenizer,
                steering_vectors=steering_vectors,
                config=intervention_config,
                device=device,
                question_id=question_id,
                verbose=True,
                dp2_idx=dp2_idx,
                use_r1_intervention=use_r1_intervention,
                task=task,
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

            if (not baseline_correct) and intervened_correct:
                stats["flipped_wrong_to_correct"] += 1
            if baseline_correct and (not intervened_correct):
                stats["flipped_correct_to_wrong"] += 1

            pbar.set_postfix({
                "Base": "✓" if baseline_correct else "✗",
                "After": "✓" if intervened_correct else "✗",
                "BaseAcc": f"{stats['baseline_correct']}/{stats['total']}",
                "IntvAcc": f"{stats['intervened_correct']}/{stats['total']}",
            })

        except Exception as e:
            print(f"\n  Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.shard_id is not None and args.num_shards is not None:
        output_file = args.output_dir / f"{args.mode}_alpha{args.alpha}" / f"shard_{args.shard_id}" / "results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = args.output_dir / f"results_{args.mode}_alpha{args.alpha}.json"

    if stats["total"] > 0:
        summary = {
            "total_questions": stats["total"],
            "baseline_accuracy": stats["baseline_correct"] / stats["total"],
            "intervened_accuracy": stats["intervened_correct"] / stats["total"],
            "accuracy_change": (stats["intervened_correct"] - stats["baseline_correct"]) / stats["total"],
            "improvements": stats["flipped_wrong_to_correct"],
            "regressions": stats["flipped_correct_to_wrong"],
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

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
