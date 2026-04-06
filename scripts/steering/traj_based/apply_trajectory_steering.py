#!/usr/bin/env python
"""Apply trajectory-based steering during generation

Implements trajectory-based steering with the following mathematical framework:

At each reasoning step j (j=1,...,K):
1. Project activation to PCA space: z_j = U^T(h_j - μ_pca)
2. Compute normalized divergence: δ_j = ||z_j - μ_j^(z)||_2 / (σ_j + ε)
3. Accumulate divergence: D_j = Σ_{t=2}^j δ_t (excluding Step 1 to avoid pollution)
4. Check intervention rule: δ_j > τ_local OR D_j > τ_cum (never at j=1)
5. If triggered, apply low-rank steering: h_j' = h_j + α U_r Δz_j^(r)
   where Δz_j^(r) = μ_j^(z,r) - z_j^(r) (first r PCA components)

Note: Step 1 divergence is measured but not accumulated into D_j since we never
intervene at Step 1, and Step 1 often has abnormally high divergence that would
cause interventions at all subsequent steps.

CRITICAL FIX (2024-12):
Activation extraction now happens at position BEFORE "Step N:" marker (step_marker_position - 1)
to align with static trajectory collection in collect_steering_vectors.py.
- Example: If "Step 2:" starts at token 45, extracts activation at token 44
- This fixes the 5-7 token offset that was causing 2-10x divergence differences

USAGE:
------
# With per-step thresholds (RECOMMENDED - uses step-specific thresholds):
python scripts/steering/traj_based/apply_trajectory_steering.py \
    --ideal-trajectory-model output/ideal_trajectory/ideal_trajectory_model_K4_layer31.npz \
    --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
    --thresholds-json output/ideal_trajectory/thresholds_K4_layer31.json \
    --num-steps 4 \
    --layer 31

# With global thresholds (backward compatibility):
python scripts/steering/traj_based/apply_trajectory_steering.py \
    --ideal-trajectory-model output/ideal_trajectory/ideal_trajectory_model_K4_layer31.npz \
    --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
    --num-steps 4 \
    --layer 31 \
    --tau-local 3.5 \
    --tau-cum 13.5 \
    --steering-rank 32 \
    --alpha 0.5

# For multi-GPU, use multi_gpu_trajectory_steering.py instead
"""

import sys
import json
import argparse
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import evaluation utils
from src.utils import evaluate_answer


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


def load_merged_json(merged_dir: Path, sample_id: int) -> Optional[dict]:
    """Load merged JSON file for a sample"""
    for pattern in [f"gsm8k_{sample_id}.json", f"sample_{sample_id:04d}.json"]:
        json_path = merged_dir / pattern
        if json_path.exists():
            with open(json_path, "r") as f:
                return json.load(f)
    return None


@dataclass
class IdealTrajectoryModel:
    """Loaded ideal trajectory model for steering"""
    num_steps: int
    layer_idx: int
    pca_dim: int
    # PCA model components
    pca_components: np.ndarray  # [pca_dim, hidden_dim]
    pca_mean: np.ndarray  # [hidden_dim]
    # Ideal trajectory in PCA space
    mu: np.ndarray  # [num_steps, pca_dim]
    # Per-step spreads
    sigma: np.ndarray  # [num_steps]
    epsilon: float


@dataclass
class SteeringState:
    """State for tracking steering during generation"""
    current_step: int
    cumulative_divergence: float
    interventions: List[Dict]  # Record of all interventions
    divergences: List[float]  # δ_j for each step


def extract_test_trajectories(
    npz_path: Path,
    layer_idx: int,
    num_steps: int,
    include_hash: bool = True
) -> Tuple[List[np.ndarray], List[int], List[bool]]:
    """Extract K-step trajectories from steering vectors NPZ

    Args:
        npz_path: Path to steering vectors NPZ
        layer_idx: Layer index
        num_steps: Number of reasoning steps K
        include_hash: If True, include hash activation as step K+1

    Returns:
        trajectories: List of [K or K+1, hidden_dim] arrays
        question_ids: List of question IDs
        is_correct: List of correctness labels
    """
    print(f"\nLoading steering vectors from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    step_acts = data['step_activations'][layer_idx]
    step_nums = data['step_numbers'][layer_idx]
    question_ids_step = data['question_ids_step'][layer_idx]
    is_correct_step = data['is_correct_step'][layer_idx]

    # Group by question ID
    question_to_steps = {}
    for i, qid in enumerate(question_ids_step):
        if qid not in question_to_steps:
            question_to_steps[qid] = {
                'step_numbers': [],
                'activations': [],
                'is_correct': is_correct_step[i]
            }
        question_to_steps[qid]['step_numbers'].append(step_nums[i])
        question_to_steps[qid]['activations'].append(step_acts[i])

    # Extract trajectories with exactly K steps
    trajectories = []
    question_ids = []
    is_correct = []

    for qid, qdata in question_to_steps.items():
        if len(qdata['step_numbers']) != num_steps:
            continue

        # Sort by step number
        sorted_indices = np.argsort(qdata['step_numbers'])
        traj = [qdata['activations'][i] for i in sorted_indices]

        # Optionally add hash
        if include_hash:
            hash_acts = data['hash_activations'][layer_idx]
            question_ids_hash = data['question_ids_hash'][layer_idx]
            hash_mask = (question_ids_hash == qid)
            if hash_mask.sum() > 0:
                traj.append(hash_acts[hash_mask][0])
            else:
                continue  # Skip if no hash

        trajectories.append(np.array(traj))
        question_ids.append(qid)
        is_correct.append(qdata['is_correct'])

    print(f"  Extracted {len(trajectories)} trajectories with K={num_steps} steps")
    print(f"  Correct: {sum(is_correct)}, Wrong: {len(is_correct) - sum(is_correct)}")

    return trajectories, question_ids, is_correct


def load_ideal_trajectory_model(model_path: Path) -> IdealTrajectoryModel:
    """Load ideal trajectory model from NPZ file

    Args:
        model_path: Path to ideal_trajectory_model_K{num_steps}_layer{layer}.npz

    Returns:
        IdealTrajectoryModel with all components
    """
    print(f"\nLoading ideal trajectory model from: {model_path}")
    data = np.load(model_path)

    model = IdealTrajectoryModel(
        num_steps=int(data['num_steps']),
        layer_idx=int(data['layer_idx']),
        pca_dim=int(data['pca_dim']),
        pca_components=data['pca_components'],
        pca_mean=data['pca_mean'],
        mu=data['mu'],
        sigma=data['sigma'],
        epsilon=float(data['epsilon'])
    )

    print(f"  Num steps: {model.num_steps}")
    print(f"  Layer: {model.layer_idx}")
    print(f"  PCA dim: {model.pca_dim}")
    print(f"  Hidden dim: {model.pca_mean.shape[0]}")

    return model


def load_per_step_thresholds(
    thresholds_json_path: Path,
    expected_num_steps: int = None,
    expected_layer: int = None
) -> Tuple[Dict, Dict]:
    """Load and validate per-step thresholds from compute_ideal_trajectory_thresholds.py output

    Args:
        thresholds_json_path: Path to thresholds JSON file from compute_ideal_trajectory_thresholds.py
        expected_num_steps: Expected number of reasoning steps (K) for validation
        expected_layer: Expected layer index for validation

    Returns:
        Tuple of (per_step_thresholds_dict, config_dict)
        - per_step_thresholds_dict: Dict mapping step names to threshold values
        - config_dict: Configuration from the thresholds file (num_steps, layer_idx, etc.)
    """
    print(f"\nLoading per-step thresholds from: {thresholds_json_path}")

    if not thresholds_json_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_json_path}")

    with open(thresholds_json_path, 'r') as f:
        data = json.load(f)

    # Validate JSON structure
    if 'per_step_thresholds' not in data:
        raise ValueError(
            f"No 'per_step_thresholds' field found in {thresholds_json_path}.\n"
            f"This file may not be from compute_ideal_trajectory_thresholds.py.\n"
            f"Expected JSON structure with 'per_step_thresholds' field."
        )

    if 'config' not in data:
        raise ValueError(
            f"No 'config' field found in {thresholds_json_path}.\n"
            f"This file may not be from compute_ideal_trajectory_thresholds.py."
        )

    per_step_thresholds = data['per_step_thresholds']
    config = data['config']

    # Extract configuration
    num_steps = config.get('num_steps')
    layer_idx = config.get('layer_idx')
    pca_dim = config.get('pca_dim')

    print(f"\n  Thresholds file configuration:")
    print(f"    Num steps (K): {num_steps}")
    print(f"    Layer: {layer_idx}")
    print(f"    PCA dim: {pca_dim}")

    # Validate against expected values
    if expected_num_steps is not None and num_steps != expected_num_steps:
        raise ValueError(
            f"Mismatch in num_steps!\n"
            f"  Thresholds file: {num_steps}\n"
            f"  Expected: {expected_num_steps}\n"
            f"Make sure you're using the correct thresholds file for K={expected_num_steps}."
        )

    if expected_layer is not None and layer_idx != expected_layer:
        raise ValueError(
            f"Mismatch in layer!\n"
            f"  Thresholds file: {layer_idx}\n"
            f"  Expected: {expected_layer}\n"
            f"Make sure you're using the correct thresholds file for layer {expected_layer}."
        )

    # Display loaded thresholds
    print(f"\n  Loaded per-step thresholds:")
    for step_name in sorted(per_step_thresholds.keys(), key=lambda x: (x != 'hash', int(x.split('_')[1]) if '_' in x else 999)):
        thresholds = per_step_thresholds[step_name]
        print(f"    {step_name:8s}: τ_local={thresholds['tau_local']:7.4f}, τ_cum={thresholds['tau_cum']:7.4f}, "
              f"F1={thresholds.get('f1', 0.0):.4f}")

    # Print summary from grid search if available
    if 'grid_search_results' in data and 'overall_best' in data['grid_search_results']:
        overall_best = data['grid_search_results']['overall_best']
        print(f"\n  Grid search overall best: {overall_best['step_name']} "
              f"(F1={overall_best['f1']:.4f})")

    return per_step_thresholds, config


def compute_stepwise_divergence(
    activation: np.ndarray,
    step_idx: int,
    model: IdealTrajectoryModel
) -> Tuple[float, np.ndarray]:
    """Compute stepwise divergence from ideal trajectory

    Args:
        activation: Hidden activation [hidden_dim]
        step_idx: Current step index (0-indexed)
        model: Ideal trajectory model

    Returns:
        delta_j: Normalized divergence
        z_j: PCA projection [pca_dim]
    """
    # Project to PCA space: z_j = U^T (h_j - μ_pca)
    z_j = model.pca_components @ (activation - model.pca_mean)

    # Compute normalized divergence: δ_j = ||z_j - μ_j|| / (σ_j + ε)
    diff = z_j - model.mu[step_idx]
    distance = np.linalg.norm(diff)
    delta_j = distance / (model.sigma[step_idx] + model.epsilon)

    return delta_j, z_j


def compute_steering_update(
    z_j: np.ndarray,
    step_idx: int,
    model: IdealTrajectoryModel,
    steering_rank: int,
    alpha: float
) -> np.ndarray:
    """Compute low-rank steering update

    Args:
        z_j: Current PCA projection [pca_dim]
        step_idx: Current step index
        model: Ideal trajectory model
        steering_rank: Number of PCA components to use (r)
        alpha: Steering strength

    Returns:
        steering_vector: Update to add to activation [hidden_dim]
    """
    # Truncate to first r components
    z_j_r = z_j[:steering_rank]
    mu_j_r = model.mu[step_idx, :steering_rank]

    # Corrective direction in PCA subspace: Δz_j^(r) = μ_j^(r) - z_j^(r)
    delta_z_r = mu_j_r - z_j_r

    # Apply steering strength: z_j^(r)' = z_j^(r) + α Δz_j^(r)
    # This means the update is: α Δz_j^(r)

    # Map back to residual space: h_j' = h_j + α U_r Δz_j^(r)
    # U_r is the first r columns of U (i.e., first r rows of U^T)
    U_r = model.pca_components[:steering_rank, :]  # [r, hidden_dim]

    # Steering vector: α U_r Δz_j^(r)
    steering_vector = alpha * (delta_z_r @ U_r)  # [hidden_dim]

    return steering_vector


class TrajectorySteeringHook:
    """Hook for applying trajectory-based steering during generation

    This class can be registered as a forward hook on the model layer
    to apply steering at detected reasoning step boundaries.
    """

    def __init__(
        self,
        model: IdealTrajectoryModel,
        tau_local: float = None,
        tau_cum: float = None,
        per_step_thresholds: Dict = None,
        steering_rank: int = 32,
        alpha: float = 0.5,
        enable_steering: bool = True,
        ignore_tau_cum: bool = False
    ):
        """Initialize trajectory steering hook

        Args:
            model: Ideal trajectory model
            tau_local: Global local divergence threshold (used if per_step_thresholds not provided)
            tau_cum: Global cumulative divergence threshold (used if per_step_thresholds not provided)
            per_step_thresholds: Dict mapping step names to {"tau_local": ..., "tau_cum": ...}
                                If provided, overrides tau_local and tau_cum
            steering_rank: Number of PCA components for steering
            alpha: Steering strength
            enable_steering: Whether to enable steering
            ignore_tau_cum: If True, only use tau_local for intervention decisions (ignore tau_cum)
        """
        self.model = model
        self.steering_rank = steering_rank
        self.alpha = alpha
        self.enable_steering = enable_steering
        self.ignore_tau_cum = ignore_tau_cum

        # Store thresholds (either global or per-step)
        self.use_per_step_thresholds = per_step_thresholds is not None
        if self.use_per_step_thresholds:
            self.per_step_thresholds = per_step_thresholds
            print(f"  Using per-step thresholds for {len(per_step_thresholds)} steps")
        else:
            # Fallback to global thresholds
            if tau_local is None or tau_cum is None:
                raise ValueError("Must provide either per_step_thresholds OR (tau_local and tau_cum)")
            self.tau_local = tau_local
            self.tau_cum = tau_cum
            print(f"  Using global thresholds: τ_local={tau_local:.4f}, τ_cum={tau_cum:.4f}")

        if self.ignore_tau_cum:
            print(f"  ⚠ Ignoring τ_cum: will only use τ_local for intervention decisions")

        # Device
        self.device = None

        # Control flags set by generation loop
        self.should_process_current_token = False
        self.current_step_idx = 0

        # State (reset for each generation)
        self.reset_state()

    def reset_state(self):
        """Reset steering state for new generation"""
        self.state = SteeringState(
            current_step=0,
            cumulative_divergence=0.0,
            interventions=[],
            divergences=[]
        )
        self.should_process_current_token = False
        self.current_step_idx = 0

    def set_step_boundary(self, step_idx: int):
        """Signal that we're at a step boundary and should process next token

        Called by the generation loop when it detects a step marker.

        Args:
            step_idx: The step index we're entering (0-indexed)
        """
        self.should_process_current_token = True
        self.current_step_idx = step_idx

    def get_thresholds_for_step(self, step_idx: int) -> Tuple[float, float]:
        """Get thresholds for a specific step

        Args:
            step_idx: Step index (0-indexed)

        Returns:
            (tau_local, tau_cum) for this step
        """
        if self.use_per_step_thresholds:
            # Map step_idx to step_name
            # step_idx ranges from 0 to num_steps (K steps + potentially hash at K+1)
            if step_idx < self.model.num_steps:
                step_name = f"step_{step_idx + 1}"
            else:
                step_name = "hash"

            # Look up thresholds for this step
            if step_name in self.per_step_thresholds:
                thresholds = self.per_step_thresholds[step_name]
                return thresholds['tau_local'], thresholds['tau_cum']
            else:
                # Fallback: use step_1 thresholds if specific step not found
                if 'step_1' in self.per_step_thresholds:
                    fallback = self.per_step_thresholds['step_1']
                    return fallback['tau_local'], fallback['tau_cum']
                else:
                    raise ValueError(f"No thresholds found for {step_name} and no fallback available")
        else:
            # Use global thresholds
            return self.tau_local, self.tau_cum

    def should_intervene(self, delta_j: float, D_j: float, step_idx: int) -> bool:
        """Check if intervention should be triggered

        Args:
            delta_j: Stepwise divergence
            D_j: Cumulative divergence
            step_idx: Current step index (0-indexed)

        Returns:
            True if intervention should be applied
        """
        # Never intervene at step 1 (step_idx == 0)
        if step_idx == 0:
            return False

        # Get step-specific thresholds
        tau_local, tau_cum = self.get_thresholds_for_step(step_idx)

        # Intervene based on flag
        if self.ignore_tau_cum:
            # Only use tau_local
            return delta_j > tau_local
        else:
            # Use both thresholds (original behavior)
            return delta_j > tau_local or D_j > tau_cum

    def __call__(self, module, input, output):
        """Forward hook callback - applies steering at step boundaries

        Only processes when should_process_current_token is True,
        which is set by the generation loop when it detects step markers.

        Args:
            module: The model layer
            input: Input to the layer
            output: Output from the layer (hidden_states, ...)

        Returns:
            Modified output tuple
        """
        if not self.enable_steering or not self.should_process_current_token:
            return output

        # Clear the flag so we only process once per step
        self.should_process_current_token = False

        # Move to device if needed
        if self.device is None:
            self.device = output[0].device

        hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
        step_idx = self.current_step_idx

        # Ensure we don't exceed num_steps
        if step_idx >= self.model.num_steps:
            return output

        # Extract activation for last token position
        # Convert to float32 before numpy (to handle BFloat16)
        h_j = hidden_states[0, -1, :].detach().cpu().float().numpy()  # [hidden_dim]

        # Compute divergence from ideal trajectory
        delta_j, z_j = compute_stepwise_divergence(h_j, step_idx, self.model)

        # Update cumulative divergence
        # IMPORTANT: Only accumulate divergence starting from Step 2 (step_idx > 0)
        # since we never intervene at Step 1 anyway, and Step 1 has abnormally high
        # divergence that would pollute D_j for all later steps
        if step_idx > 0:
            self.state.cumulative_divergence += delta_j

        self.state.divergences.append(delta_j)
        self.state.current_step = step_idx + 1

        D_j = self.state.cumulative_divergence

        # Check intervention rule (never intervene at step 0 = Step 1)
        intervene = self.should_intervene(delta_j, D_j, step_idx)

        if intervene:
            # Compute steering update
            steering_vector = compute_steering_update(
                z_j, step_idx, self.model, self.steering_rank, self.alpha
            )

            # Apply steering to last token position
            steering_tensor = torch.from_numpy(steering_vector).to(
                hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states[0, -1, :] = hidden_states[0, -1, :] + steering_tensor

            # Record intervention
            self.state.interventions.append({
                'step_idx': step_idx,
                'delta_j': float(delta_j),
                'D_j': float(D_j),
                'steering_norm': float(np.linalg.norm(steering_vector))
            })

        return (hidden_states,) + output[1:]

    def get_state_summary(self) -> Dict:
        """Get summary of steering state

        Returns:
            Dict with steering statistics
        """
        return {
            'num_steps': self.state.current_step,
            'num_interventions': len(self.state.interventions),
            'cumulative_divergence': float(self.state.cumulative_divergence),
            'divergences': [float(d) for d in self.state.divergences],
            'interventions': self.state.interventions
        }


def register_trajectory_steering_hooks(model, steering_hook: TrajectorySteeringHook, layer_idx: int) -> List:
    """Register trajectory steering hook on specified layer

    Args:
        model: Transformer model
        steering_hook: TrajectorySteeringHook instance
        layer_idx: Layer index to apply steering

    Returns:
        List of hook handles for cleanup
    """
    handles = []

    # Register hook on the specified layer
    layer = model.model.layers[layer_idx]
    layer._layer_idx = layer_idx

    handle = layer.register_forward_hook(steering_hook)
    handles.append(handle)

    return handles


def remove_hooks(handles: List):
    """Remove all registered hooks"""
    for handle in handles:
        handle.remove()


@torch.no_grad()
def generate_baseline(
    input_ids: List[int],
    model,
    tokenizer,
    max_new_tokens: int = 512
) -> Dict:
    """Generate text without intervention (baseline)

    Uses deterministic greedy decoding.

    Args:
        input_ids: Input token IDs
        model: Model
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with generation info and metrics
    """
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    prompt_len = len(input_ids)

    # Generate (deterministic - greedy decoding)
    outputs = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode generated text only (skip prompt)
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    # Compute metrics
    produced_answer = extract_answer_after_hash(generated_text)
    num_steps = count_steps(generated_text)

    # Find #### marker token position to compute reasoning length
    # reasoning_length should be tokens from end of prompt to #### marker
    full_seq_ids = outputs[0].tolist()
    marker_token_id = tokenizer.encode("####", add_special_tokens=False)[0]
    marker_idx = None
    for idx in range(prompt_len, len(full_seq_ids)):
        if full_seq_ids[idx] == marker_token_id:
            marker_idx = idx
            break

    if marker_idx is not None:
        reasoning_length = marker_idx - (prompt_len - 1)
    else:
        # Fallback: if no marker found, use total generated tokens
        reasoning_length = len(outputs[0]) - prompt_len

    return {
        "produced_text": generated_text,
        "produced_answer": produced_answer if produced_answer else "N/A",
        "num_steps": num_steps,
        "reasoning_length": reasoning_length,
    }


@torch.no_grad()
def generate_with_trajectory_steering(
    input_ids: List[int],
    model,
    tokenizer,
    steering_hook: TrajectorySteeringHook,
    layer_idx: int,
    max_new_tokens: int = 512
) -> Dict:
    """Generate text with trajectory-based steering

    Uses manual token-by-token generation with explicit step boundary detection.
    Extracts activations at position BEFORE "Step N:" marker (step_marker_position - 1)
    to align with static trajectory collection.

    Args:
        input_ids: Input token IDs
        model: Model
        tokenizer: Tokenizer
        steering_hook: TrajectorySteeringHook instance
        layer_idx: Layer index to apply steering
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with generation info and metrics
    """
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    prompt_len = len(input_ids)

    # Reset hook state for new generation
    steering_hook.reset_state()

    # Precompute step marker first token IDs for detection
    # We'll check if next token is "Step" (first token of "Step N:")
    step_token_text = "Step"
    step_token_ids = tokenizer.encode(step_token_text, add_special_tokens=False)
    step_first_token_id = step_token_ids[0] if step_token_ids else None

    # Also get hash token ID for completeness
    hash_token_ids = tokenizer.encode("####", add_special_tokens=False)
    hash_first_token_id = hash_token_ids[0] if hash_token_ids else None

    print(f"\n  Detection tokens:")
    print(f"    'Step' first token ID: {step_first_token_id} ('{tokenizer.decode([step_first_token_id])}')")
    print(f"    '####' first token ID: {hash_first_token_id} ('{tokenizer.decode([hash_first_token_id])}')")

    # Create a simple hook that applies pending steering
    class PendingSteering:
        def __init__(self):
            self.vector = None
            self.device = None

        def set_vector(self, vec):
            self.vector = vec

        def __call__(self, module, input, output):
            if self.vector is None:
                return output

            # output can be a tuple or a dataclass-like object from accelerate
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output[0]

            # Apply steering to last token position
            if self.device is None:
                self.device = hidden_states.device

            steering_tensor = torch.from_numpy(self.vector).to(
                hidden_states.device, dtype=hidden_states.dtype
            )

            # Handle both 3D [batch, seq, dim] and 2D [seq, dim]
            # (accelerate device_map="auto" can produce 2D in layer hooks)
            if hidden_states.dim() == 3:
                hidden_states[0, -1, :] = hidden_states[0, -1, :] + steering_tensor
            else:
                hidden_states[-1, :] = hidden_states[-1, :] + steering_tensor

            # Clear after applying
            self.vector = None

            # Return in the same format as input
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                # Dataclass-like output (e.g., BaseModelOutputWithPast)
                # hidden_states was modified in-place, just return as-is
                return output

    # Register the steering hook
    pending_steering_hook = PendingSteering()
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(pending_steering_hook)

    try:
        # Manual token-by-token generation with step detection
        current_ids = input_tensor.clone()
        generated_token_ids = []
        step_count = 0  # Count detected steps

        for _ in range(max_new_tokens):
            # Forward pass to get next token logits
            # Request hidden states so we can extract activations
            outputs = model(current_ids, output_hidden_states=True)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Get hidden states for the steering layer
            hidden_states_at_layer = outputs.hidden_states[layer_idx + 1]  # [batch_size, seq_len, hidden_dim]

            # Greedy decoding: select token with highest probability
            next_token_id = logits[0, -1, :].argmax(dim=-1, keepdim=True).item()  # scalar

            # CRITICAL: Check if next token is "Step" (start of "Step N:")
            # If so, extract activation at CURRENT position (before "Step")
            if next_token_id == step_first_token_id and step_count < steering_hook.model.num_steps:
                step_idx = step_count  # 0-indexed

                # Extract activation from CURRENT position (before "Step")
                # This is the last position in hidden_states
                h_j = hidden_states_at_layer[0, -1, :].detach().cpu().float().numpy()

                # Compute divergence from ideal trajectory
                delta_j, z_j = compute_stepwise_divergence(h_j, step_idx, steering_hook.model)

                # Get thresholds for this step
                tau_local, tau_cum = steering_hook.get_thresholds_for_step(step_idx)

                # Update cumulative divergence (skip step 1 to avoid pollution)
                if step_idx > 0:
                    steering_hook.state.cumulative_divergence += delta_j

                steering_hook.state.divergences.append(delta_j)
                steering_hook.state.current_step = step_idx + 1

                D_j = steering_hook.state.cumulative_divergence

                # Check intervention rule (never intervene at step 0 = Step 1)
                intervene = steering_hook.should_intervene(delta_j, D_j, step_idx)

                # DEBUG: Print extraction and decision details
                print(f"\n  [Step {step_idx+1}] Detected 'Step' token next, extracted at current position")
                print(f"    δ={delta_j:.3f} (thresh={tau_local:.3f}), D={D_j:.3f} (thresh={tau_cum:.3f})")
                print(f"    Decision: {'INTERVENE' if intervene else 'skip'}", end="")

                if intervene:
                    # Compute steering update
                    steering_vector = compute_steering_update(
                        z_j, step_idx, steering_hook.model,
                        steering_hook.steering_rank, steering_hook.alpha
                    )

                    # Store for application on NEXT forward pass
                    pending_steering_hook.set_vector(steering_vector)

                    # Record intervention
                    steering_hook.state.interventions.append({
                        'step_idx': step_idx,
                        'delta_j': float(delta_j),
                        'D_j': float(D_j),
                        'steering_norm': float(np.linalg.norm(steering_vector))
                    })
                    print(f" ✓")
                else:
                    print(f" ✗")

                step_count += 1

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            # Append to sequence
            current_ids = torch.cat([current_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
            generated_token_ids.append(next_token_id)

        # Decode generated text
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Compute metrics
        produced_answer = extract_answer_after_hash(generated_text)
        num_steps = count_steps(generated_text)

        # Find #### marker position to compute reasoning length
        marker_token_id = tokenizer.encode("####", add_special_tokens=False)[0]
        marker_idx = None
        for idx, token_id in enumerate(generated_token_ids):
            if token_id == marker_token_id:
                marker_idx = idx + 1  # +1 because we want length, not index
                break

        if marker_idx is not None:
            reasoning_length = marker_idx
        else:
            reasoning_length = len(generated_token_ids)

        # Get steering statistics
        steering_summary = steering_hook.get_state_summary()

        return {
            "produced_text": generated_text,
            "produced_answer": produced_answer if produced_answer else "N/A",
            "num_steps": num_steps,
            "reasoning_length": reasoning_length,
            "steering_summary": steering_summary
        }

    finally:
        # Remove the steering hook
        handle.remove()


def evaluate_trajectory_steering(
    model_name: str,
    ideal_trajectory_model_path: Path,
    test_npz: Path,
    merged_dir: Path,
    num_questions: int,
    num_steps: int,
    layer: int,
    tau_local: float = None,
    tau_cum: float = None,
    thresholds_json_path: Path = None,
    steering_rank: int = 32,
    alpha: float = 0.5,
    output_dir: Path = None,
    max_new_tokens: int = 512,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
    ignore_tau_cum: bool = False
):
    """Evaluate trajectory-based steering on test set with actual generation

    Args:
        model_name: HuggingFace model name or config key
        ideal_trajectory_model_path: Path to ideal trajectory model NPZ
        test_npz: Path to test steering vectors NPZ (not used for generation, just for reference)
        merged_dir: Path to merged JSON files with questions
        num_questions: Number of questions to process
        num_steps: Number of reasoning steps
        layer: Layer index to apply steering
        tau_local: Global local divergence threshold (optional if thresholds_json_path provided)
        tau_cum: Global cumulative divergence threshold (optional if thresholds_json_path provided)
        thresholds_json_path: Path to JSON file with per-step thresholds (overrides tau_local/tau_cum)
        steering_rank: Number of PCA components for steering (r)
        alpha: Steering strength
        output_dir: Output directory
        max_new_tokens: Maximum tokens to generate
        shard_id: Shard ID for multi-GPU (None = no sharding)
        num_shards: Total shards for multi-GPU
        ignore_tau_cum: If True, only use tau_local for intervention decisions (ignore tau_cum)
    """
    # Set up output directory (shard-specific if sharding)
    if output_dir is None:
        output_dir = Path("output/trajectory_steering_results")

    if shard_id is not None and num_shards is not None:
        output_dir = output_dir / f"shard_{shard_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TRAJECTORY-BASED STEERING EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Ideal trajectory model: {ideal_trajectory_model_path}")
    print(f"  Merged dir: {merged_dir}")
    print(f"  Num questions: {num_questions if num_questions is not None else 'all available'}")
    print(f"  Num steps: {num_steps}")
    print(f"  Layer: {layer}")

    # Will load per-step thresholds after loading ideal model (for validation)
    per_step_thresholds = None
    if thresholds_json_path is not None:
        print(f"  Will use per-step thresholds from: {thresholds_json_path}")
    else:
        print(f"  τ_local (global): {tau_local}")
        print(f"  τ_cum (global): {tau_cum}")

    print(f"  Steering rank: {steering_rank}")
    print(f"  α: {alpha}")
    print(f"  Max new tokens: {max_new_tokens}")
    if shard_id is not None:
        print(f"  Shard: {shard_id}/{num_shards}")
    print(f"  Output: {output_dir}")

    # Load ideal trajectory model
    print("\nLoading ideal trajectory model...")
    ideal_model = load_ideal_trajectory_model(ideal_trajectory_model_path)

    # Use model's config if not specified
    if num_steps is None:
        num_steps = ideal_model.num_steps
        print(f"  Using model's num_steps: {num_steps}")

    if layer is None:
        layer = ideal_model.layer_idx
        print(f"  Using model's layer: {layer}")

    # Verify num_steps and layer match
    if ideal_model.num_steps != num_steps:
        print(f"\n⚠ Warning: Model has {ideal_model.num_steps} steps, but {num_steps} requested")
        print(f"  Using model's num_steps: {ideal_model.num_steps}")
        num_steps = ideal_model.num_steps

    if ideal_model.layer_idx != layer:
        print(f"\n⚠ Warning: Model trained on layer {ideal_model.layer_idx}, but {layer} requested")
        print(f"  Using model's layer: {ideal_model.layer_idx}")
        layer = ideal_model.layer_idx

    # Load per-step thresholds with validation if provided
    thresholds_config = None
    if thresholds_json_path is not None:
        per_step_thresholds, thresholds_config = load_per_step_thresholds(
            thresholds_json_path,
            expected_num_steps=num_steps,
            expected_layer=layer
        )
        print(f"  ✓ Thresholds validated: K={num_steps}, Layer={layer}")

    # Load model and tokenizer
    print("\nLoading language model and tokenizer...")

    # Resolve model path from config if needed
    model_path = model_name
    if not Path(model_path).exists():
        try:
            config = load_config()
            model_path = get_model_path_from_config(config, model_name)
            print(f"Resolved model key '{model_name}' to path: {model_path}")
        except Exception as e:
            print(f"Warning: Could not resolve model key, using as-is: {e}")
            model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
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
        local_files_only=True
    ).eval()

    print(f"Model loaded: {len(model.model.layers)} layers")

    # Create steering hook
    print("\nInitializing steering hook...")
    steering_hook = TrajectorySteeringHook(
        model=ideal_model,
        tau_local=tau_local,
        tau_cum=tau_cum,
        per_step_thresholds=per_step_thresholds,
        steering_rank=steering_rank,
        alpha=alpha,
        enable_steering=True,
        ignore_tau_cum=ignore_tau_cum
    )

    # Display thresholds being used
    print(f"\nActive Thresholds:")
    if steering_hook.use_per_step_thresholds:
        print(f"  Mode: Per-step thresholds")
        for step_idx in range(num_steps):
            tau_l, tau_c = steering_hook.get_thresholds_for_step(step_idx)
            print(f"    Step {step_idx+1}: τ_local={tau_l:.4f}, τ_cum={tau_c:.4f}")
    else:
        print(f"  Mode: Global thresholds")
        print(f"    All steps: τ_local={tau_local:.4f}, τ_cum={tau_cum:.4f}")

    # Extract question IDs that have exactly num_steps from test_npz
    print(f"\nFiltering for questions with exactly K={num_steps} steps from NPZ...")
    _, valid_question_ids, _ = extract_test_trajectories(
        test_npz,
        layer_idx=layer,
        num_steps=num_steps,
        include_hash=False  # We don't need the actual trajectories, just the IDs
    )
    valid_question_ids_set = set(valid_question_ids)
    print(f"  Found {len(valid_question_ids_set)} questions with K={num_steps} steps in NPZ")

    # Collect question IDs from merged directory
    print(f"\nCollecting questions from {merged_dir}...")
    all_question_ids = []
    for json_file in sorted(merged_dir.glob("gsm8k_*.json")):
        try:
            question_id = int(json_file.stem.split("_")[1])
            # Only include questions that have exactly num_steps in the NPZ
            if question_id in valid_question_ids_set:
                all_question_ids.append(question_id)
        except (ValueError, IndexError):
            continue

    print(f"Found {len(all_question_ids)} questions with K={num_steps} steps (intersection of merged dir and NPZ)")

    selected_questions = all_question_ids if num_questions is None else all_question_ids[:num_questions]
    print(f"Selected {len(selected_questions)} questions (before sharding)")

    # Apply sharding if specified
    if shard_id is not None:
        shard_questions = [qid for i, qid in enumerate(selected_questions) if i % num_shards == shard_id]
        print(f"Shard {shard_id}/{num_shards}: Processing {len(shard_questions)} questions")
        selected_questions = shard_questions

    # Process questions
    print(f"\n{'='*80}")
    print(f"GENERATING WITH TRAJECTORY-BASED STEERING")
    print(f"{'='*80}\n")

    results = []
    stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed_no_json": 0,
        "failed_missing_fields": 0,
    }

    # Track intervention stats during generation
    running_interventions = {
        "total_questions": 0,
        "questions_with_interventions": 0,
        "total_interventions": 0
    }

    for idx, question_id in enumerate(tqdm(selected_questions, desc="Questions", ncols=100)):
        merged_data = load_merged_json(merged_dir, question_id)
        if merged_data is None:
            stats["failed_no_json"] += 1
            continue

        stats["total_attempted"] += 1

        # Get data
        gold_answer = merged_data.get("gold_answer", merged_data.get("answer", "N/A"))
        input_ids = merged_data.get("input_ids", [])

        if not input_ids:
            stats["failed_missing_fields"] += 1
            continue

        # Decode input_ids to get question text
        question = tokenizer.decode(input_ids, skip_special_tokens=True)

        print(f"\n{'='*80}")
        print(f"Question {idx+1}/{len(selected_questions)} (ID={question_id})")
        print(f"{'='*80}")

        try:
            # Generate baseline (no steering)
            print("\n[Baseline Generation] (no steering)")
            baseline_result = generate_baseline(
                input_ids,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens
            )

            # Generate with trajectory steering
            print("\n[Steered Generation] (with trajectory steering)")
            intervened_result = generate_with_trajectory_steering(
                input_ids,
                model,
                tokenizer,
                steering_hook,
                layer_idx=layer,
                max_new_tokens=max_new_tokens
            )

            # Evaluate correctness
            baseline_correct = evaluate_answer(baseline_result["produced_answer"], gold_answer)
            intervened_correct = evaluate_answer(intervened_result["produced_answer"], gold_answer)

            # Extract steering summary
            steering_summary = intervened_result.pop("steering_summary")

            result = {
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "baseline": {
                    **baseline_result,
                    "is_correct": baseline_correct
                },
                "intervened": {
                    **intervened_result,
                    "is_correct": intervened_correct
                },
                "steering": steering_summary,
                "tau_local": tau_local,
                "tau_cum": tau_cum,
                "alpha": alpha,
                "steering_rank": steering_rank
            }

            results.append(result)
            stats["successful"] += 1

            # Update running intervention stats
            running_interventions["total_questions"] += 1
            if steering_summary["num_interventions"] > 0:
                running_interventions["questions_with_interventions"] += 1
            running_interventions["total_interventions"] += steering_summary["num_interventions"]

            # Print running stats
            interv_pct = (running_interventions["questions_with_interventions"] /
                         running_interventions["total_questions"] * 100)
            print(f"\n[Q {idx+1}/{len(selected_questions)}] Intervened: "
                  f"{running_interventions['questions_with_interventions']}/{running_interventions['total_questions']} "
                  f"({interv_pct:.1f}%) | "
                  f"Total interventions: {running_interventions['total_interventions']}")

        except Exception as e:
            print(f"\n  [Question {question_id}] ✗ Failed during generation: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    output_file = output_dir / "results.json"

    # Compute summary statistics for JSON
    summary = {}
    if len(results) > 0:
        baseline_correct = sum(1 for r in results if r["baseline"]["is_correct"])
        intervened_correct = sum(1 for r in results if r["intervened"]["is_correct"])
        n_results = len(results)

        flipped_correct_to_wrong = sum(1 for r in results
                                       if r["baseline"]["is_correct"] and not r["intervened"]["is_correct"])
        flipped_wrong_to_correct = sum(1 for r in results
                                       if not r["baseline"]["is_correct"] and r["intervened"]["is_correct"])
        stayed_correct = sum(1 for r in results
                            if r["baseline"]["is_correct"] and r["intervened"]["is_correct"])
        stayed_wrong = sum(1 for r in results
                          if not r["baseline"]["is_correct"] and not r["intervened"]["is_correct"])

        baseline_lengths = [r["baseline"]["reasoning_length"] for r in results]
        intervened_lengths = [r["intervened"]["reasoning_length"] for r in results]
        baseline_steps = [r["baseline"]["num_steps"] for r in results]
        intervened_steps = [r["intervened"]["num_steps"] for r in results]

        total_interventions = sum(r["steering"]["num_interventions"] for r in results)
        num_questions_intervened = sum(1 for r in results if r["steering"]["num_interventions"] > 0)

        summary = {
            "total_samples": n_results,
            "accuracy": {
                "baseline": float(baseline_correct / n_results * 100),
                "intervened": float(intervened_correct / n_results * 100),
                "change": float((intervened_correct - baseline_correct) / n_results * 100)
            },
            "flips": {
                "correct_to_wrong": flipped_correct_to_wrong,
                "wrong_to_correct": flipped_wrong_to_correct,
                "stayed_correct": stayed_correct,
                "stayed_wrong": stayed_wrong
            },
            "reasoning_length": {
                "baseline_avg": float(np.mean(baseline_lengths)),
                "intervened_avg": float(np.mean(intervened_lengths)),
                "avg_change": float(np.mean(intervened_lengths) - np.mean(baseline_lengths))
            },
            "num_steps": {
                "baseline_avg": float(np.mean(baseline_steps)),
                "intervened_avg": float(np.mean(intervened_steps)),
                "avg_change": float(np.mean(intervened_steps) - np.mean(baseline_steps))
            },
            "interventions": {
                "total": total_interventions,
                "num_questions_intervened": num_questions_intervened,
                "avg_per_example": float(total_interventions / n_results)
            }
        }

    save_data = {
        "config": {
            "model_name": model_name,
            "ideal_trajectory_model": str(ideal_trajectory_model_path),
            "num_steps": num_steps,
            "layer": layer,
            "tau_local": tau_local,
            "tau_cum": tau_cum,
            "steering_rank": steering_rank,
            "alpha": alpha,
            "max_new_tokens": max_new_tokens,
        },
        "stats": stats,
        "summary": summary,
        "results": results
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Questions attempted: {stats['total_attempted']}")
    print(f"Successful: {stats['successful']} ({stats['successful']/max(stats['total_attempted'], 1)*100:.1f}%)")
    print(f"Failed - no JSON: {stats['failed_no_json']}")
    print(f"Failed - missing fields: {stats['failed_missing_fields']}")

    # Print intervention effects
    if len(results) > 0:
        print(f"\n{'='*80}")
        print("INTERVENTION EFFECTS")
        print(f"{'='*80}")

        print(f"\nAccuracy:")
        print(f"  Baseline:   {baseline_correct}/{n_results} ({summary['accuracy']['baseline']:.2f}%)")
        print(f"  Intervened: {intervened_correct}/{n_results} ({summary['accuracy']['intervened']:.2f}%)")
        print(f"  Change:     {summary['accuracy']['change']:+.2f}% ({intervened_correct - baseline_correct:+d} questions)")

        print(f"\nAnswer Flips:")
        print(f"  Correct → Wrong: {flipped_correct_to_wrong} questions")
        print(f"  Wrong → Correct: {flipped_wrong_to_correct} questions")
        print(f"  Stayed Correct:  {stayed_correct} questions")
        print(f"  Stayed Wrong:    {stayed_wrong} questions")

        print(f"\nSteering Interventions:")
        print(f"  Total: {summary['interventions']['total']}")
        print(f"  Questions intervened: {summary['interventions']['num_questions_intervened']}/{n_results} ({summary['interventions']['num_questions_intervened']/n_results*100:.1f}%)")
        print(f"  Avg per example: {summary['interventions']['avg_per_example']:.2f}")

        # Analyze per-step intervention patterns
        print(f"\nPer-Step Intervention Analysis:")
        step_intervention_counts = {}
        step_divergence_stats = {}

        for r in results:
            steering = r["steering"]
            # Count interventions per step
            for intervention in steering["interventions"]:
                step_idx = intervention["step_idx"]
                step_name = f"Step {step_idx + 1}"
                if step_name not in step_intervention_counts:
                    step_intervention_counts[step_name] = 0
                step_intervention_counts[step_name] += 1

            # Collect divergence statistics
            for idx, delta in enumerate(steering["divergences"]):
                step_name = f"Step {idx + 1}"
                if step_name not in step_divergence_stats:
                    step_divergence_stats[step_name] = []
                step_divergence_stats[step_name].append(delta)

        print(f"  {'Step':<10} {'Interventions':<15} {'Intervention%':<15} {'Avg δ':<10}")
        print(f"  {'-'*50}")
        for step_idx in range(num_steps):
            step_name = f"Step {step_idx + 1}"
            count = step_intervention_counts.get(step_name, 0)
            pct = count / n_results * 100 if n_results > 0 else 0
            avg_delta = np.mean(step_divergence_stats[step_name]) if step_name in step_divergence_stats else 0.0
            print(f"  {step_name:<10} {count:<15} {pct:<15.1f} {avg_delta:<10.3f}")

        # Show first few questions with intervention details
        print(f"\nFirst 5 Questions - Detailed Intervention Log:")
        for i, r in enumerate(results[:5]):
            print(f"\n  Question {r['question_id']}:")
            print(f"    Baseline: {'✓' if r['baseline']['is_correct'] else '✗'} | "
                  f"Intervened: {'✓' if r['intervened']['is_correct'] else '✗'}")
            steering = r["steering"]
            print(f"    Divergences (δ_j): {[f'{d:.3f}' for d in steering['divergences']]}")
            print(f"    Interventions: {len(steering['interventions'])} at steps {[interv['step_idx']+1 for interv in steering['interventions']]}")
            if steering['interventions']:
                print(f"    First intervention details:")
                first = steering['interventions'][0]
                print(f"      Step {first['step_idx']+1}: δ_j={first['delta_j']:.3f}, D_j={first['D_j']:.3f}")

        print(f"\nReasoning Length (tokens):")
        print(f"  Baseline:   {summary['reasoning_length']['baseline_avg']:.2f} tokens (avg)")
        print(f"  Intervened: {summary['reasoning_length']['intervened_avg']:.2f} tokens (avg)")
        print(f"  Change:     {summary['reasoning_length']['avg_change']:+.2f} tokens")

        print(f"\nReasoning Steps:")
        print(f"  Baseline:   {summary['num_steps']['baseline_avg']:.2f} steps (avg)")
        print(f"  Intervened: {summary['num_steps']['intervened_avg']:.2f} steps (avg)")
        print(f"  Change:     {summary['num_steps']['avg_change']:+.2f} steps")

        print(f"\n{'='*80}\n")


def main():
    try:
        config = load_config()
        default_model = get_model_path_from_config(config, "llama-3.1-8b-instruct")
    except Exception as e:
        print(f"Warning: Could not load model from config: {e}")
        # Use config key instead of HuggingFace ID for offline mode
        default_model = "llama-3.1-8b-instruct"

    parser = argparse.ArgumentParser(
        description="Apply trajectory-based steering during generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=default_model,
        help="HuggingFace model name or config key"
    )
    parser.add_argument(
        "--ideal-trajectory-model",
        type=Path,
        required=True,
        help="Path to ideal trajectory model NPZ file"
    )
    parser.add_argument(
        "--test-npz",
        type=Path,
        required=True,
        help="Path to test steering vectors NPZ file (for reference)"
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("output/complete_artifacts/gsm8k_test/merged"),
        help="Directory with merged JSON files (default: gsm8k_test/merged)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to process (default: all available)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help="Number of reasoning steps (default: 4)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        help="Layer index to apply steering (default: 31)"
    )
    parser.add_argument(
        "--tau-local",
        type=float,
        default=None,
        help="Global local divergence threshold (optional if --thresholds-json provided)"
    )
    parser.add_argument(
        "--tau-cum",
        type=float,
        default=None,
        help="Global cumulative divergence threshold (optional if --thresholds-json provided)"
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Path to JSON file with per-step thresholds (overrides --tau-local and --tau-cum)"
    )
    parser.add_argument(
        "--steering-rank",
        type=int,
        default=32,
        help="Number of PCA components for steering (default: 32)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Steering strength (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/trajectory_steering_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard ID for multi-GPU execution (default: None = no sharding)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU execution"
    )
    parser.add_argument(
        "--ignore-tau-cum",
        action="store_true",
        help="Ignore tau_cum threshold and only use tau_local for intervention decisions"
    )

    args = parser.parse_args()

    evaluate_trajectory_steering(
        model_name=args.model_name,
        ideal_trajectory_model_path=args.ideal_trajectory_model,
        test_npz=args.test_npz,
        merged_dir=args.merged_dir,
        num_questions=args.num_questions,
        num_steps=args.num_steps,
        layer=args.layer,
        tau_local=args.tau_local,
        tau_cum=args.tau_cum,
        thresholds_json_path=args.thresholds_json,
        steering_rank=args.steering_rank,
        alpha=args.alpha,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        ignore_tau_cum=args.ignore_tau_cum
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())