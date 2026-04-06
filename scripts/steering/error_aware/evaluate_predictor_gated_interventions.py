#!/usr/bin/env python
"""Evaluate predictor-gated interventions

This script evaluates how well a predictor can gate interventions to improve accuracy.

Pipeline:
1. Load predictor model (e.g., hash_pca_correctness_layer30.npz)
2. Load steering vectors (hash activations) for test questions
3. Extract hash_pca features and predict which questions are wrong
4. Load "intervene all" results from error_methods_interventions_complete/
5. For predictor-selected questions: use intervened results
   For other questions: use baseline results
6. Compute hybrid metrics and compare to baseline and intervene-all

Optional: Filtering by reasoning steps
- Use --intervene-num-steps-threshold N to only intervene on questions with baseline num_steps >= N
- This also reports "vanilla" (non-predictor-gated) intervention results for comparison
- Questions with num_steps < N use baseline accuracy (no intervention)

Usage:
python scripts/steering/error_aware/evaluate_predictor_gated_interventions.py \
    --predictor output/predictors/hash_pca_correctness_layer30.npz \
    --steering-vectors output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
    --intervention-dir output/error_methods_interventions_complete \
    --output output/error-gated-interventions/hash_pca_correctness_layer30.json

# With num_steps filtering (only intervene on questions with >=5 steps)
python scripts/steering/error_aware/evaluate_predictor_gated_interventions.py \
    --predictor output/predictors/hash_pca_correctness_layer30.npz \
    --steering-vectors output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
    --intervention-dir output/error_methods_interventions_complete \
    --output output/error-gated-interventions/hash_pca_correctness_layer30.json \
    --intervene-num-steps-threshold 5

python scripts/steering/error_aware/evaluate_predictor_gated_interventions.py \
      --predictor-dir output/predictors \
      --steering-vectors output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
      --intervention-dir output/error_methods_interventions_complete \
      --output-dir output/error-gated-interventions


"""


import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class PredictorModel:
    """Loaded predictor model"""
    feature_set: str
    label_type: str
    layer_idx: int
    model_type: str  # "linear" or "mlp"
    coefficients: Optional[np.ndarray]  # For linear: [feature_dim], for MLP: None
    intercept: Optional[float]  # For linear: float, for MLP: None
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    pca_components: Optional[np.ndarray] = None  # [2, n_components, hidden_dim]
    pca_mean: Optional[np.ndarray] = None  # [2, hidden_dim]
    best_threshold: float = 0.5
    # MLP-specific weights
    mlp_fc1_weight: Optional[np.ndarray] = None  # [128, input_dim]
    mlp_fc1_bias: Optional[np.ndarray] = None    # [128]
    mlp_fc2_weight: Optional[np.ndarray] = None  # [1, 128]
    mlp_fc2_bias: Optional[np.ndarray] = None    # [1]


@dataclass
class HybridResult:
    """Result for a single question under hybrid intervention"""
    question_id: int
    baseline_correct: bool
    hybrid_correct: bool  # After applying predictor gating
    intervene_all_correct: bool  # Without gating
    was_intervened: bool  # Whether predictor selected for intervention
    predictor_prob: float  # P(incorrect) from predictor
    baseline_num_steps: int
    hybrid_num_steps: int
    baseline_reasoning_length: int
    hybrid_reasoning_length: int


def load_predictor(npz_path: Path) -> PredictorModel:
    """Load predictor model from NPZ file

    Returns:
        PredictorModel with all parameters
    """
    print(f"\nLoading predictor from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Determine model type
    model_type = str(data.get('model_type', 'linear'))

    # Initialize model based on type
    if model_type == "mlp":
        # Load MLP weights
        model = PredictorModel(
            feature_set=str(data['feature_set']),
            label_type=str(data['label_type']),
            layer_idx=int(data['layer_idx']),
            model_type=model_type,
            coefficients=None,
            intercept=None,
            scaler_mean=data['scaler_mean'],
            scaler_std=data['scaler_std'],
            best_threshold=float(data.get('best_threshold', 0.5)),
            mlp_fc1_weight=data['mlp_fc1_weight'],
            mlp_fc1_bias=data['mlp_fc1_bias'],
            mlp_fc2_weight=data['mlp_fc2_weight'],
            mlp_fc2_bias=data['mlp_fc2_bias']
        )
    else:
        # Load linear model weights
        model = PredictorModel(
            feature_set=str(data['feature_set']),
            label_type=str(data['label_type']),
            layer_idx=int(data['layer_idx']),
            model_type=model_type,
            coefficients=data['coefficients'],
            intercept=float(data['intercept']),
            scaler_mean=data['scaler_mean'],
            scaler_std=data['scaler_std'],
            best_threshold=float(data.get('best_threshold', 0.5))
        )

    # Load PCA components if present (for hash_pca feature set)
    if 'pca_components' in data:
        model.pca_components = data['pca_components']
        model.pca_mean = data['pca_mean']
        print(f"  Loaded PCA components: {model.pca_components.shape}")

    print(f"  Feature set: {model.feature_set}")
    print(f"  Label type: {model.label_type}")
    print(f"  Layer: {model.layer_idx}")
    print(f"  Model type: {model.model_type.upper()}")
    print(f"  Best threshold: {model.best_threshold:.3f}")
    if model.model_type == "mlp":
        print(f"  MLP architecture: fc1[{model.mlp_fc1_weight.shape}] -> ReLU -> fc2[{model.mlp_fc2_weight.shape}]")
    else:
        print(f"  Coefficients shape: {model.coefficients.shape}")

    return model


def load_steering_vectors(npz_path: Path) -> Dict:
    """Load steering vectors NPZ file

    Returns:
        Dict with keys: step_activations, hash_activations, step_numbers,
                       question_ids_step, question_ids_hash, is_correct_step,
                       is_correct_hash, num_layers, hidden_dim
    """
    print(f"\nLoading steering vectors from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    num_layers = int(data['num_layers'])
    hidden_dim = int(data['hidden_dim'])

    print(f"  Num layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")

    result = {
        'step_activations': data['step_activations'],
        'hash_activations': data['hash_activations'],
        'step_numbers': data['step_numbers'],
        'question_ids_step': data['question_ids_step'],
        'question_ids_hash': data['question_ids_hash'],
        'is_correct_step': data['is_correct_step'],
        'is_correct_hash': data['is_correct_hash'],
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
    }

    # Print info about hash activations
    if len(result['hash_activations']) > 0:
        layer_0_hash = result['hash_activations'][0]
        if len(layer_0_hash) > 0:
            print(f"  Hash activations shape (layer 0): {layer_0_hash.shape}")
            print(f"  Num questions with hash: {len(np.unique(result['question_ids_hash'][0]))}")

    return result


def extract_hash_activations(
    steering_data: Dict,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract hash activations for a specific layer

    Returns:
        activations: [n_samples, hidden_dim]
        question_ids: [n_samples]
        is_correct: [n_samples]
    """
    hash_acts = steering_data['hash_activations'][layer_idx]
    question_ids = steering_data['question_ids_hash'][layer_idx]
    is_correct = steering_data['is_correct_hash'][layer_idx]

    return hash_acts, question_ids, is_correct


def extract_last_step_activations(
    steering_data: Dict,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract last step activations for each question at a specific layer

    For each question, finds the maximum step number and returns that step's activation.

    Returns:
        activations: [n_samples, hidden_dim]
        question_ids: [n_samples]
        is_correct: [n_samples]
        last_step_nums: [n_samples] - the actual step number for each sample
    """
    step_acts = steering_data['step_activations'][layer_idx]
    step_nums = steering_data['step_numbers'][layer_idx]
    question_ids_step = steering_data['question_ids_step'][layer_idx]
    is_correct = steering_data['is_correct_step'][layer_idx]

    # Group by question_id and find max step for each
    unique_qids = np.unique(question_ids_step)

    last_step_acts = []
    last_step_qids = []
    last_step_correct = []
    last_step_nums_list = []

    for qid in unique_qids:
        # Get all steps for this question
        mask = (question_ids_step == qid)
        q_step_nums = step_nums[mask]
        q_step_acts = step_acts[mask]
        q_correct = is_correct[mask]

        # Find the maximum step number
        max_step_idx = np.argmax(q_step_nums)

        last_step_acts.append(q_step_acts[max_step_idx])
        last_step_qids.append(qid)
        last_step_correct.append(q_correct[max_step_idx])
        last_step_nums_list.append(q_step_nums[max_step_idx])

    return (np.array(last_step_acts),
            np.array(last_step_qids),
            np.array(last_step_correct),
            np.array(last_step_nums_list))


def extract_second_to_last_step_activations(
    steering_data: Dict,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract second-to-last step activations for each question at a specific layer

    For each question, finds the second highest step number and returns that step's activation.

    Returns:
        activations: [n_samples, hidden_dim]
        question_ids: [n_samples]
        is_correct: [n_samples]
    """
    step_acts = steering_data['step_activations'][layer_idx]
    step_nums = steering_data['step_numbers'][layer_idx]
    question_ids_step = steering_data['question_ids_step'][layer_idx]
    is_correct = steering_data['is_correct_step'][layer_idx]

    # Group by question_id and find second-to-last step for each
    unique_qids = np.unique(question_ids_step)

    second_last_acts = []
    second_last_qids = []
    second_last_correct = []

    for qid in unique_qids:
        # Get all steps for this question
        mask = (question_ids_step == qid)
        q_step_nums = step_nums[mask]
        q_step_acts = step_acts[mask]
        q_correct = is_correct[mask]

        # Skip if less than 2 steps
        if len(q_step_nums) < 2:
            continue

        # Find second-to-last step (second highest step number)
        sorted_indices = np.argsort(q_step_nums)
        second_last_idx = sorted_indices[-2]  # Second to last

        second_last_acts.append(q_step_acts[second_last_idx])
        second_last_qids.append(qid)
        second_last_correct.append(q_correct[second_last_idx])

    return (np.array(second_last_acts),
            np.array(second_last_qids),
            np.array(second_last_correct))


def extract_step_activations(
    steering_data: Dict,
    layer_idx: int,
    step_number: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract activations for a specific step number at a specific layer

    Returns:
        activations: [n_samples, hidden_dim]
        question_ids: [n_samples]
        is_correct: [n_samples]
    """
    step_acts = steering_data['step_activations'][layer_idx]
    step_nums = steering_data['step_numbers'][layer_idx]
    question_ids = steering_data['question_ids_step'][layer_idx]
    is_correct = steering_data['is_correct_step'][layer_idx]

    # Filter for specific step number
    mask = (step_nums == step_number)

    return step_acts[mask], question_ids[mask], is_correct[mask]


def extract_features_for_predictor(
    steering_data: Dict,
    layer_idx: int,
    predictor: PredictorModel
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features matching the predictor's feature set

    Returns:
        features: [n_samples, feature_dim] scaled features
        question_ids: [n_samples]
    """
    feature_set = predictor.feature_set

    if feature_set == "step1_step2":
        # Concatenate Step-1 and Step-2
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)

        common_qids = np.intersect1d(step1_qids, step2_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]

        features_raw = np.concatenate([step1_aligned, step2_aligned], axis=1)

    elif feature_set == "step2_minus_step1":
        # Step-2 - Step-1 difference
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)

        common_qids = np.intersect1d(step1_qids, step2_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]

        features_raw = step2_aligned - step1_aligned

    elif feature_set == "step1_step2_step3":
        # Concatenate Step-1, Step-2, Step-3
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)
        step3_acts, step3_qids, _ = extract_step_activations(steering_data, layer_idx, 3)

        common_qids = np.intersect1d(np.intersect1d(step1_qids, step2_qids), step3_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])
        step3_indices = np.array([np.where(step3_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        step3_aligned = step3_acts[step3_indices]

        features_raw = np.concatenate([step1_aligned, step2_aligned, step3_aligned], axis=1)

    elif feature_set == "step_diffs":
        # Concatenate (Step-2 - Step-1), (Step-3 - Step-2), (Hash - Last-Step)
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)
        step3_acts, step3_qids, _ = extract_step_activations(steering_data, layer_idx, 3)
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        last_acts, last_qids, _, _ = extract_last_step_activations(steering_data, layer_idx)

        common_qids = np.intersect1d(
            np.intersect1d(np.intersect1d(step1_qids, step2_qids), step3_qids),
            np.intersect1d(hash_qids, last_qids)
        )
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])
        step3_indices = np.array([np.where(step3_qids == qid)[0][0] for qid in common_qids])
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        step3_aligned = step3_acts[step3_indices]
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]

        diff1 = step2_aligned - step1_aligned
        diff2 = step3_aligned - step2_aligned
        diff3 = hash_aligned - last_aligned

        features_raw = np.concatenate([diff1, diff2, diff3], axis=1)

    elif feature_set == "hash_only":
        # Hash activations only
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        features_raw = hash_acts
        common_qids = hash_qids

    elif feature_set == "hash_minus_last":
        # Hash - Last-Step difference
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        last_acts, last_qids, _, _ = extract_last_step_activations(steering_data, layer_idx)

        common_qids = np.intersect1d(hash_qids, last_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]

        features_raw = hash_aligned - last_aligned

    elif feature_set == "hash_pca":
        # Hash and (Hash - Last-Step) concatenated, then PCA
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        last_acts, last_qids, _, _ = extract_last_step_activations(steering_data, layer_idx)

        common_qids = np.intersect1d(hash_qids, last_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        hash_minus_last = hash_aligned - last_aligned

        if predictor.pca_components is None:
            raise ValueError("hash_pca predictor missing PCA components")

        # Apply PCA
        hash_centered = hash_aligned - predictor.pca_mean[0]
        hash_pca = hash_centered @ predictor.pca_components[0].T

        hash_minus_last_centered = hash_minus_last - predictor.pca_mean[1]
        hash_minus_last_pca = hash_minus_last_centered @ predictor.pca_components[1].T

        features_raw = np.concatenate([hash_pca, hash_minus_last_pca], axis=1)

    elif feature_set == "hash_last_diffs_pca":
        # Hash + (Hash-Last) + (Last-SecondLast), each PCA'd to 128 dims → 384 total
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        last_acts, last_qids, _, _ = extract_last_step_activations(steering_data, layer_idx)
        second_last_acts, second_last_qids, _ = extract_second_to_last_step_activations(steering_data, layer_idx)

        common_qids = np.intersect1d(
            np.intersect1d(hash_qids, last_qids),
            second_last_qids
        )
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])
        second_last_indices = np.array([np.where(second_last_qids == qid)[0][0] for qid in common_qids])

        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        second_last_aligned = second_last_acts[second_last_indices]
        hash_minus_last = hash_aligned - last_aligned
        last_minus_second_last = last_aligned - second_last_aligned

        if predictor.pca_components is None:
            raise ValueError("hash_last_diffs_pca predictor missing PCA components")

        # Apply PCA to each of the 3 components
        hash_centered = hash_aligned - predictor.pca_mean[0]
        hash_pca = hash_centered @ predictor.pca_components[0].T

        hash_minus_last_centered = hash_minus_last - predictor.pca_mean[1]
        hash_minus_last_pca = hash_minus_last_centered @ predictor.pca_components[1].T

        last_minus_second_last_centered = last_minus_second_last - predictor.pca_mean[2]
        last_minus_second_last_pca = last_minus_second_last_centered @ predictor.pca_components[2].T

        features_raw = np.concatenate([hash_pca, hash_minus_last_pca, last_minus_second_last_pca], axis=1)

    elif feature_set == "hash_last_diffs_pca_joint":
        # Hash + (Hash-Last) + (Last-SecondLast) concatenated, then jointly PCA'd to 128 dims
        # This is different from hash_last_diffs_pca which PCA's each component separately
        hash_acts, hash_qids, _ = extract_hash_activations(steering_data, layer_idx)
        last_acts, last_qids, _, _ = extract_last_step_activations(steering_data, layer_idx)
        second_last_acts, second_last_qids, _ = extract_second_to_last_step_activations(steering_data, layer_idx)

        common_qids = np.intersect1d(
            np.intersect1d(hash_qids, last_qids),
            second_last_qids
        )
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])
        second_last_indices = np.array([np.where(second_last_qids == qid)[0][0] for qid in common_qids])

        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        second_last_aligned = second_last_acts[second_last_indices]
        hash_minus_last = hash_aligned - last_aligned
        last_minus_second_last = last_aligned - second_last_aligned

        # Concatenate all 3 components before PCA (will be jointly PCA'd to 128 dims)
        concatenated = np.concatenate([hash_aligned, hash_minus_last, last_minus_second_last], axis=1)

        if predictor.pca_components is None:
            raise ValueError("hash_last_diffs_pca_joint predictor missing PCA components")

        # Apply single joint PCA to concatenated features
        # pca_components[0] has shape [128, 3*hidden_dim] for joint PCA
        concatenated_centered = concatenated - predictor.pca_mean[0]
        features_raw = concatenated_centered @ predictor.pca_components[0].T

    elif feature_set == "pca_concat":
        # DEPRECATED: Legacy PCA features - Step-1 PCA(64) + Step-2 PCA(64) concatenated (128 dims)
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)

        common_qids = np.intersect1d(step1_qids, step2_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]

        if predictor.pca_components is None:
            raise ValueError("pca_concat predictor missing PCA components")

        # Apply PCA to Step-1 and Step-2 separately
        step1_centered = step1_aligned - predictor.pca_mean[0]
        step1_pca = step1_centered @ predictor.pca_components[0].T

        step2_centered = step2_aligned - predictor.pca_mean[1]
        step2_pca = step2_centered @ predictor.pca_components[1].T

        # Concatenate PCA'd components
        features_raw = np.concatenate([step1_pca, step2_pca], axis=1)

    elif feature_set == "pca_diff":
        # DEPRECATED: Legacy PCA features - Step-2 PCA(64) - Step-1 PCA(64) (64 dims)
        step1_acts, step1_qids, _ = extract_step_activations(steering_data, layer_idx, 1)
        step2_acts, step2_qids, _ = extract_step_activations(steering_data, layer_idx, 2)

        common_qids = np.intersect1d(step1_qids, step2_qids)
        if len(common_qids) == 0:
            return np.array([]), np.array([])

        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]

        if predictor.pca_components is None:
            raise ValueError("pca_diff predictor missing PCA components")

        # Apply PCA to Step-1 and Step-2 separately
        step1_centered = step1_aligned - predictor.pca_mean[0]
        step1_pca = step1_centered @ predictor.pca_components[0].T

        step2_centered = step2_aligned - predictor.pca_mean[1]
        step2_pca = step2_centered @ predictor.pca_components[1].T

        # Compute difference of PCA'd components
        features_raw = step2_pca - step1_pca

    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    # Apply standardization (z-score normalization)
    features_scaled = (features_raw - predictor.scaler_mean) / predictor.scaler_std

    return features_scaled, common_qids


def predict_probabilities(
    features: np.ndarray,
    predictor: PredictorModel
) -> np.ndarray:
    """Compute P(incorrect) using logistic regression or MLP predictor

    Args:
        features: [n_samples, feature_dim] already scaled features
        predictor: PredictorModel

    Returns:
        probabilities: [n_samples] P(y=1|X) = P(incorrect)
    """
    if predictor.model_type == "mlp":
        # MLP forward pass: fc1 -> relu -> fc2 -> sigmoid
        # fc1: [n_samples, input_dim] @ [input_dim, 128].T + [128] = [n_samples, 128]
        hidden = features @ predictor.mlp_fc1_weight.T + predictor.mlp_fc1_bias
        hidden = np.maximum(0, hidden)  # ReLU
        # fc2: [n_samples, 128] @ [128, 1].T + [1] = [n_samples, 1]
        logits = hidden @ predictor.mlp_fc2_weight.T + predictor.mlp_fc2_bias
        logits = logits.squeeze(-1)  # [n_samples]
    else:
        # Logistic regression: P(y=1) = sigmoid(X @ w + b)
        logits = features @ predictor.coefficients + predictor.intercept

    probabilities = 1 / (1 + np.exp(-logits))

    return probabilities


def get_predicted_wrong_questions(
    steering_data: Dict,
    predictor: PredictorModel,
    threshold: float = None
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Get question IDs predicted to be wrong by the predictor

    Args:
        steering_data: Steering vectors
        predictor: Predictor model
        threshold: Decision threshold (default: use predictor's best_threshold)

    Returns:
        predicted_wrong_qids: Array of question IDs predicted as wrong
        all_predictions: Dict mapping question_id -> P(incorrect)
    """
    if threshold is None:
        threshold = predictor.best_threshold

    print(f"\n{'='*80}")
    print(f"RUNNING PREDICTOR ON TEST SET")
    print(f"{'='*80}")
    print(f"Predictor: {predictor.feature_set} | {predictor.label_type} | Layer {predictor.layer_idx}")
    print(f"Threshold: {threshold:.3f}")

    # Extract features (adaptive based on predictor's feature set)
    features, question_ids = extract_features_for_predictor(
        steering_data,
        predictor.layer_idx,
        predictor
    )

    if len(features) == 0:
        print("⚠ WARNING: No features extracted!")
        return np.array([]), {}

    print(f"\nExtracted features for {len(question_ids)} questions")
    print(f"Feature dimension: {features.shape[1]}")

    # Sanity check: verify feature dimensions match expected
    if predictor.model_type == "mlp":
        expected_dim = predictor.mlp_fc1_weight.shape[1]  # input_dim is second dimension
    else:
        expected_dim = predictor.coefficients.shape[0]
    if features.shape[1] != expected_dim:
        raise ValueError(f"Feature dimension mismatch! Got {features.shape[1]}, expected {expected_dim}")

    print(f"✓ Feature dimension matches predictor ({expected_dim})")
    print(f"  Layer {predictor.layer_idx} | PCA components: {predictor.pca_components.shape if predictor.pca_components is not None else 'None'}")

    # Predict probabilities
    probabilities = predict_probabilities(features, predictor)

    # Apply threshold
    predicted_wrong = probabilities >= threshold
    predicted_wrong_qids = question_ids[predicted_wrong]

    # Create dict for all predictions
    all_predictions = {int(qid): float(prob) for qid, prob in zip(question_ids, probabilities)}

    print(f"\nPrediction Summary:")
    print(f"  Total questions: {len(question_ids)}")
    print(f"  Predicted wrong (P(incorrect) >= {threshold:.3f}): {len(predicted_wrong_qids)}")
    print(f"  Predicted correct (P(incorrect) < {threshold:.3f}): {len(question_ids) - len(predicted_wrong_qids)}")
    print(f"  Intervention rate: {len(predicted_wrong_qids) / len(question_ids) * 100:.1f}%")
    print(f"  Mean P(incorrect): {probabilities.mean():.3f}")
    print(f"  Std P(incorrect): {probabilities.std():.3f}")

    return predicted_wrong_qids, all_predictions


def load_intervention_results(json_path: Path) -> Dict:
    """Load intervention results from JSON file

    Returns:
        Dict with keys: config, stats, summary, results
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_hybrid_metrics(
    intervention_data: Dict,
    predicted_wrong_qids: np.ndarray,
    all_predictions: Dict[int, float],
    intervene_num_steps_threshold: Optional[int] = None
) -> Tuple[List[HybridResult], Dict]:
    """Compute hybrid metrics by gating interventions with predictor

    Args:
        intervention_data: Loaded intervention results
        predicted_wrong_qids: Question IDs to intervene on
        all_predictions: Dict mapping question_id -> P(incorrect)
        intervene_num_steps_threshold: If provided, only intervene on questions with
                                       baseline num_steps >= threshold

    Returns:
        results: List of HybridResult for each question
        metrics: Dict with summary metrics
    """
    predicted_wrong_set = set(predicted_wrong_qids)
    results = []

    # Verify question coverage
    intervention_qids = {item['question_id'] for item in intervention_data['results']}
    predictor_qids = set(all_predictions.keys())

    missing_in_predictor = intervention_qids - predictor_qids
    if missing_in_predictor:
        print(f"    ⚠ WARNING: {len(missing_in_predictor)} questions in intervention but not in predictor predictions")
        print(f"      (Will use P(incorrect)=0.0 for these questions)")

    # Counters for metrics
    total = 0
    baseline_correct = 0
    hybrid_correct = 0
    intervene_all_correct = 0

    # Transition counters
    predictor_intervention_count = 0
    tp = 0  # Baseline wrong, predictor intervened, became correct
    fp = 0  # Baseline correct, predictor intervened, became wrong
    tn = 0  # Baseline correct, predictor didn't intervene, stayed correct
    fn = 0  # Baseline wrong, predictor didn't intervene, stayed wrong

    # Intervention breakdown counters (requested metrics)
    baseline_correct_intervened = 0  # Baseline correct questions that were intervened
    baseline_incorrect_intervened = 0  # Baseline incorrect questions that were intervened
    intervened_correct_to_wrong = 0  # Among intervened: correct -> wrong transitions
    intervened_wrong_to_correct = 0  # Among intervened: wrong -> correct transitions

    # Reasoning metrics
    total_baseline_steps = 0
    total_hybrid_steps = 0
    total_baseline_length = 0
    total_hybrid_length = 0

    for item in intervention_data['results']:
        qid = item['question_id']
        baseline = item['baseline']
        intervened = item['intervened']

        # Determine if predictor selected this question for intervention
        predictor_selected = qid in predicted_wrong_set
        predictor_prob = all_predictions.get(qid, 0.0)

        # Check num_steps filter if provided
        meets_num_steps_threshold = (intervene_num_steps_threshold is None or
                                      baseline['num_steps'] >= intervene_num_steps_threshold)

        # Actually intervene only if BOTH predictor selected AND meets num_steps threshold
        was_intervened = predictor_selected and meets_num_steps_threshold

        # Hybrid result: use intervened if we intervened, else use baseline
        if was_intervened:
            is_hybrid_correct = intervened['is_correct']
            hybrid_steps = intervened['num_steps']
            hybrid_length = intervened['reasoning_length']
            predictor_intervention_count += 1
        else:
            is_hybrid_correct = baseline['is_correct']
            hybrid_steps = baseline['num_steps']
            hybrid_length = baseline['reasoning_length']

        # Record result
        result = HybridResult(
            question_id=qid,
            baseline_correct=baseline['is_correct'],
            hybrid_correct=is_hybrid_correct,
            intervene_all_correct=intervened['is_correct'],
            was_intervened=was_intervened,
            predictor_prob=predictor_prob,
            baseline_num_steps=baseline['num_steps'],
            hybrid_num_steps=hybrid_steps,
            baseline_reasoning_length=baseline['reasoning_length'],
            hybrid_reasoning_length=hybrid_length
        )
        results.append(result)

        # Update counters
        total += 1
        if baseline['is_correct']:
            baseline_correct += 1
        if is_hybrid_correct:
            hybrid_correct += 1
        if intervened['is_correct']:
            intervene_all_correct += 1

        # Update reasoning metrics
        total_baseline_steps += baseline['num_steps']
        total_hybrid_steps += hybrid_steps
        total_baseline_length += baseline['reasoning_length']
        total_hybrid_length += hybrid_length

        # Compute transition type for predictor evaluation
        if was_intervened:
            # Track baseline correctness for intervened questions
            if baseline['is_correct']:
                baseline_correct_intervened += 1
                # Track transition: correct -> wrong
                if not is_hybrid_correct:
                    intervened_correct_to_wrong += 1
            else:
                baseline_incorrect_intervened += 1
                # Track transition: wrong -> correct
                if is_hybrid_correct:
                    intervened_wrong_to_correct += 1

            # Original TP/FP logic
            if not baseline['is_correct'] and is_hybrid_correct:
                tp += 1  # Successful intervention (wrong -> correct)
            elif baseline['is_correct'] and not is_hybrid_correct:
                fp += 1  # Harmful intervention (correct -> wrong)
        else:
            if baseline['is_correct']:
                tn += 1  # Correctly did not intervene on correct
            else:
                fn += 1  # Missed intervention on wrong

    # Compute summary metrics
    baseline_acc = baseline_correct / total if total > 0 else 0
    hybrid_acc = hybrid_correct / total if total > 0 else 0
    intervene_all_acc = intervene_all_correct / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_baseline_steps = total_baseline_steps / total if total > 0 else 0
    avg_hybrid_steps = total_hybrid_steps / total if total > 0 else 0
    avg_baseline_length = total_baseline_length / total if total > 0 else 0
    avg_hybrid_length = total_hybrid_length / total if total > 0 else 0

    metrics = {
        'total': total,
        'baseline_correct': baseline_correct,
        'hybrid_correct': hybrid_correct,
        'intervene_all_correct': intervene_all_correct,
        'baseline_accuracy': baseline_acc,
        'hybrid_accuracy': hybrid_acc,
        'intervene_all_accuracy': intervene_all_acc,
        'accuracy_gain_vs_baseline': hybrid_acc - baseline_acc,
        'accuracy_gain_vs_intervene_all': hybrid_acc - intervene_all_acc,
        'predictor_intervention_count': predictor_intervention_count,
        'predictor_intervention_rate': predictor_intervention_count / total if total > 0 else 0,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_baseline_steps': avg_baseline_steps,
        'avg_hybrid_steps': avg_hybrid_steps,
        'avg_baseline_length': avg_baseline_length,
        'avg_hybrid_length': avg_hybrid_length,
        # Intervention breakdown metrics
        'baseline_correct_intervened': baseline_correct_intervened,
        'baseline_incorrect_intervened': baseline_incorrect_intervened,
        'intervened_correct_to_wrong': intervened_correct_to_wrong,
        'intervened_wrong_to_correct': intervened_wrong_to_correct,
    }

    return results, metrics


def compute_vanilla_intervention_metrics(
    intervention_data: Dict,
    intervene_num_steps_threshold: int
) -> Dict:
    """Compute metrics for vanilla (non-predictor-gated) interventions filtered by num_steps

    Args:
        intervention_data: Loaded intervention results
        intervene_num_steps_threshold: Only intervene on questions with baseline num_steps >= threshold

    Returns:
        metrics: Dict with summary metrics
    """
    total = 0
    baseline_correct = 0
    vanilla_correct = 0
    intervene_all_correct = 0
    intervention_count = 0

    correct_to_wrong = 0
    wrong_to_correct = 0

    for item in intervention_data['results']:
        baseline = item['baseline']
        intervened = item['intervened']

        meets_threshold = baseline['num_steps'] >= intervene_num_steps_threshold

        # Vanilla: use intervened if meets threshold, else baseline
        if meets_threshold:
            is_vanilla_correct = intervened['is_correct']
            intervention_count += 1

            if baseline['is_correct'] and not intervened['is_correct']:
                correct_to_wrong += 1
            elif not baseline['is_correct'] and intervened['is_correct']:
                wrong_to_correct += 1
        else:
            is_vanilla_correct = baseline['is_correct']

        total += 1
        if baseline['is_correct']:
            baseline_correct += 1
        if is_vanilla_correct:
            vanilla_correct += 1
        if intervened['is_correct']:
            intervene_all_correct += 1

    baseline_acc = baseline_correct / total if total > 0 else 0
    vanilla_acc = vanilla_correct / total if total > 0 else 0
    intervene_all_acc = intervene_all_correct / total if total > 0 else 0

    return {
        'total': total,
        'baseline_correct': baseline_correct,
        'vanilla_correct': vanilla_correct,
        'intervene_all_correct': intervene_all_correct,
        'baseline_accuracy': baseline_acc,
        'vanilla_accuracy': vanilla_acc,
        'intervene_all_accuracy': intervene_all_acc,
        'accuracy_gain_vs_baseline': vanilla_acc - baseline_acc,
        'accuracy_gain_vs_intervene_all': vanilla_acc - intervene_all_acc,
        'intervention_count': intervention_count,
        'intervention_rate': intervention_count / total if total > 0 else 0,
        'correct_to_wrong': correct_to_wrong,
        'wrong_to_correct': wrong_to_correct,
    }


def evaluate_all_interventions(
    intervention_dir: Path,
    predicted_wrong_qids: np.ndarray,
    all_predictions: Dict[int, float],
    intervene_num_steps_threshold: Optional[int] = None
) -> Tuple[Dict[str, Dict], Optional[Dict[str, Dict]]]:
    """Evaluate predictor gating on all intervention results

    Args:
        intervention_dir: Directory containing intervention JSON files
        predicted_wrong_qids: Questions to intervene on
        all_predictions: Dict mapping question_id -> P(incorrect)
        intervene_num_steps_threshold: If provided, also compute vanilla interventions
                                       filtered by num_steps

    Returns:
        Tuple of (predictor_gated_metrics, vanilla_metrics)
        vanilla_metrics is None if intervene_num_steps_threshold is None
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PREDICTOR-GATED INTERVENTIONS")
    print(f"{'='*80}")
    print(f"Predictor has predictions for {len(all_predictions)} questions")
    if intervene_num_steps_threshold is not None:
        print(f"Num-steps filter: Only intervene on questions with baseline num_steps >= {intervene_num_steps_threshold}")

    all_metrics = {}
    all_vanilla_metrics = {} if intervene_num_steps_threshold is not None else None

    # Process steering interventions
    steering_dir = intervention_dir / "steering"
    if steering_dir.exists():
        print(f"\n📁 Processing steering interventions...")
        for json_file in sorted(steering_dir.glob("results_*.json")):
            intervention_name = json_file.stem.replace("results_", "steering_")
            print(f"\n  {intervention_name}...")

            intervention_data = load_intervention_results(json_file)
            results, metrics = compute_hybrid_metrics(
                intervention_data,
                predicted_wrong_qids,
                all_predictions,
                intervene_num_steps_threshold
            )

            all_metrics[intervention_name] = metrics

            print(f"    Baseline acc: {metrics['baseline_accuracy']:.4f}")
            print(f"    Hybrid acc:   {metrics['hybrid_accuracy']:.4f} ({metrics['accuracy_gain_vs_baseline']:+.4f})")
            print(f"    Intervene-all: {metrics['intervene_all_accuracy']:.4f}")
            print(f"    F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")
            print(f"    Interventions: {metrics['predictor_intervention_count']} total | "
                  f"{metrics['baseline_correct_intervened']} baseline-correct | "
                  f"{metrics['baseline_incorrect_intervened']} baseline-incorrect")
            print(f"    Transitions: {metrics['intervened_wrong_to_correct']} wrong→correct | "
                  f"{metrics['intervened_correct_to_wrong']} correct→wrong")

            # Also compute vanilla metrics if filtering by num_steps
            if intervene_num_steps_threshold is not None:
                vanilla_metrics = compute_vanilla_intervention_metrics(
                    intervention_data,
                    intervene_num_steps_threshold
                )
                all_vanilla_metrics[intervention_name] = vanilla_metrics
                print(f"    [Vanilla] Acc: {vanilla_metrics['vanilla_accuracy']:.4f} "
                      f"({vanilla_metrics['accuracy_gain_vs_baseline']:+.4f}) | "
                      f"Interventions: {vanilla_metrics['intervention_count']}")

    # Process text injection interventions
    text_injection_dir = intervention_dir / "text_injection"
    if text_injection_dir.exists():
        print(f"\n📁 Processing text injection interventions...")
        for json_file in sorted(text_injection_dir.glob("results_*.json")):
            intervention_name = json_file.stem.replace("results_", "text_")
            print(f"\n  {intervention_name}...")

            intervention_data = load_intervention_results(json_file)
            results, metrics = compute_hybrid_metrics(
                intervention_data,
                predicted_wrong_qids,
                all_predictions,
                intervene_num_steps_threshold
            )

            all_metrics[intervention_name] = metrics

            print(f"    Baseline acc: {metrics['baseline_accuracy']:.4f}")
            print(f"    Hybrid acc:   {metrics['hybrid_accuracy']:.4f} ({metrics['accuracy_gain_vs_baseline']:+.4f})")
            print(f"    Intervene-all: {metrics['intervene_all_accuracy']:.4f}")
            print(f"    F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")
            print(f"    Interventions: {metrics['predictor_intervention_count']} total | "
                  f"{metrics['baseline_correct_intervened']} baseline-correct | "
                  f"{metrics['baseline_incorrect_intervened']} baseline-incorrect")
            print(f"    Transitions: {metrics['intervened_wrong_to_correct']} wrong→correct | "
                  f"{metrics['intervened_correct_to_wrong']} correct→wrong")

            # Also compute vanilla metrics if filtering by num_steps
            if intervene_num_steps_threshold is not None:
                vanilla_metrics = compute_vanilla_intervention_metrics(
                    intervention_data,
                    intervene_num_steps_threshold
                )
                all_vanilla_metrics[intervention_name] = vanilla_metrics
                print(f"    [Vanilla] Acc: {vanilla_metrics['vanilla_accuracy']:.4f} "
                      f"({vanilla_metrics['accuracy_gain_vs_baseline']:+.4f}) | "
                      f"Interventions: {vanilla_metrics['intervention_count']}")

    return all_metrics, all_vanilla_metrics


def format_summary_table(all_metrics: Dict[str, Dict], predictor: PredictorModel, threshold: float) -> str:
    """Format summary table as text

    Args:
        all_metrics: Dict mapping intervention_name -> metrics
        predictor: Predictor model used
        threshold: Decision threshold used

    Returns:
        Formatted table as string
    """
    lines = []
    lines.append("=" * 120)
    lines.append("PREDICTOR-GATED INTERVENTION SUMMARY")
    lines.append("=" * 120)
    lines.append(f"Predictor: {predictor.feature_set} | {predictor.label_type} | Layer {predictor.layer_idx}")
    lines.append(f"Threshold: {threshold:.3f}")
    lines.append("")
    lines.append("=" * 120)

    # Sort by hybrid accuracy gain
    sorted_interventions = sorted(
        all_metrics.items(),
        key=lambda x: x[1]['accuracy_gain_vs_baseline'],
        reverse=True
    )

    # Header
    lines.append(f"{'Intervention':<40} {'Base Acc':<10} {'Hybrid Acc':<11} {'Gain':<8} {'Int-All':<10} {'F1':<7} {'Prec':<7} {'Rec':<7}")
    lines.append("-" * 120)

    # Rows
    for intervention_name, metrics in sorted_interventions:
        name_short = intervention_name[:38]
        line = (
            f"{name_short:<40} "
            f"{metrics['baseline_accuracy']:<10.4f} "
            f"{metrics['hybrid_accuracy']:<11.4f} "
            f"{metrics['accuracy_gain_vs_baseline']:+<8.4f} "
            f"{metrics['intervene_all_accuracy']:<10.4f} "
            f"{metrics['f1']:<7.3f} "
            f"{metrics['precision']:<7.3f} "
            f"{metrics['recall']:<7.3f}"
        )
        lines.append(line)

    lines.append("=" * 120)

    # Best intervention
    if sorted_interventions:
        best_name, best_metrics = sorted_interventions[0]
        lines.append("")
        lines.append(f"🏆 Best Intervention: {best_name}")
        lines.append(f"   Baseline Accuracy: {best_metrics['baseline_accuracy']:.4f}")
        lines.append(f"   Hybrid Accuracy:   {best_metrics['hybrid_accuracy']:.4f}")
        lines.append(f"   Gain vs Baseline:  {best_metrics['accuracy_gain_vs_baseline']:+.4f}")
        lines.append(f"   Gain vs Int-All:   {best_metrics['accuracy_gain_vs_intervene_all']:+.4f}")
        lines.append(f"   Intervention Rate: {best_metrics['predictor_intervention_rate']:.1%}")
        lines.append(f"   F1 Score:          {best_metrics['f1']:.4f}")
        lines.append(f"   Precision:         {best_metrics['precision']:.4f}")
        lines.append(f"   Recall:            {best_metrics['recall']:.4f}")
        lines.append(f"")
        lines.append(f"   Intervention Breakdown:")
        lines.append(f"     Total intervened:           {best_metrics['predictor_intervention_count']}")
        lines.append(f"     Baseline correct intervened: {best_metrics['baseline_correct_intervened']}")
        lines.append(f"     Baseline incorrect intervened: {best_metrics['baseline_incorrect_intervened']}")
        lines.append(f"   Intervention Transitions:")
        lines.append(f"     Wrong → Correct:            {best_metrics['intervened_wrong_to_correct']}")
        lines.append(f"     Correct → Wrong:            {best_metrics['intervened_correct_to_wrong']}")
        lines.append(f"")
        lines.append(f"   Avg Steps (Baseline): {best_metrics['avg_baseline_steps']:.1f}")
        lines.append(f"   Avg Steps (Hybrid):   {best_metrics['avg_hybrid_steps']:.1f}")
        lines.append(f"   Avg Length (Baseline): {best_metrics['avg_baseline_length']:.1f}")
        lines.append(f"   Avg Length (Hybrid):   {best_metrics['avg_hybrid_length']:.1f}")

    return "\n".join(lines)


def print_summary_table(all_metrics: Dict[str, Dict], predictor: PredictorModel, threshold: float):
    """Print comprehensive summary table

    Args:
        all_metrics: Dict mapping intervention_name -> metrics
        predictor: Predictor model used
        threshold: Decision threshold used
    """
    table_text = format_summary_table(all_metrics, predictor, threshold)
    print(f"\n{table_text}")


def save_results(
    all_metrics: Dict[str, Dict],
    predicted_wrong_qids: np.ndarray,
    all_predictions: Dict[int, float],
    predictor: PredictorModel,
    threshold: float,
    output_json_path: Path,
    output_txt_path: Path
):
    """Save evaluation results to JSON and TXT

    Args:
        all_metrics: Dict mapping intervention_name -> metrics
        predicted_wrong_qids: Questions selected for intervention
        all_predictions: All predictor predictions
        predictor: Predictor model
        threshold: Decision threshold
        output_json_path: Where to save JSON results
        output_txt_path: Where to save TXT summary
    """
    # Save JSON
    output_data = {
        'predictor': {
            'feature_set': predictor.feature_set,
            'label_type': predictor.label_type,
            'layer_idx': predictor.layer_idx,
            'threshold': threshold,
            'best_threshold': predictor.best_threshold,
        },
        'predictions': {
            'total_questions': len(all_predictions),
            'predicted_wrong_count': len(predicted_wrong_qids),
            'intervention_rate': len(predicted_wrong_qids) / len(all_predictions) if all_predictions else 0,
            'predicted_wrong_qids': predicted_wrong_qids.tolist(),
            'all_probabilities': all_predictions,
        },
        'interventions': all_metrics,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Save TXT summary
    table_text = format_summary_table(all_metrics, predictor, threshold)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, 'w') as f:
        f.write(table_text)
        f.write("\n")

    print(f"\n✓ Results saved to:")
    print(f"   JSON: {output_json_path}")
    print(f"   TXT:  {output_txt_path}")


@dataclass
class PredictorEvaluation:
    """Results for a single predictor evaluation"""
    predictor_name: str
    predictor: PredictorModel
    threshold: float
    all_metrics: Dict[str, Dict]
    best_intervention_name: str
    best_intervention_metrics: Dict


def evaluate_single_predictor(
    predictor_path: Path,
    steering_data: Dict,
    intervention_dir: Path,
    threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    intervene_num_steps_threshold: Optional[int] = None
) -> PredictorEvaluation:
    """Evaluate a single predictor

    Args:
        predictor_path: Path to predictor NPZ file
        steering_data: Pre-loaded steering vectors
        intervention_dir: Directory with intervention results
        threshold: Optional threshold override
        output_dir: Output directory (default: intervention_dir.parent / "predictor_evaluations")
        intervene_num_steps_threshold: If provided, only intervene on questions with
                                       baseline num_steps >= threshold

    Returns:
        PredictorEvaluation with results
    """
    print(f"\n{'='*120}")
    print(f"EVALUATING PREDICTOR: {predictor_path.name}")
    print(f"{'='*120}")

    # Load predictor
    predictor = load_predictor(predictor_path)

    # Determine threshold
    threshold = threshold if threshold is not None else predictor.best_threshold

    # Get predicted wrong questions
    predicted_wrong_qids, all_predictions = get_predicted_wrong_questions(
        steering_data,
        predictor,
        threshold
    )

    # Evaluate all interventions
    all_metrics, all_vanilla_metrics = evaluate_all_interventions(
        intervention_dir,
        predicted_wrong_qids,
        all_predictions,
        intervene_num_steps_threshold
    )

    # Print summary table
    print_summary_table(all_metrics, predictor, threshold)

    # Print vanilla metrics table if available
    if all_vanilla_metrics is not None:
        print(f"\n{'='*120}")
        print(f"VANILLA (NON-GATED) INTERVENTIONS - Filtered by num_steps >= {intervene_num_steps_threshold}")
        print(f"{'='*120}")
        print(f"{'Intervention':<40} {'Base Acc':<10} {'Vanilla Acc':<12} {'Gain':<8} {'Int-All':<10} {'Interv#':<10}")
        print("-" * 120)
        sorted_vanilla = sorted(
            all_vanilla_metrics.items(),
            key=lambda x: x[1]['accuracy_gain_vs_baseline'],
            reverse=True
        )
        for intervention_name, metrics in sorted_vanilla:
            name_short = intervention_name[:38]
            print(
                f"{name_short:<40} "
                f"{metrics['baseline_accuracy']:<10.4f} "
                f"{metrics['vanilla_accuracy']:<12.4f} "
                f"{metrics['accuracy_gain_vs_baseline']:+<8.4f} "
                f"{metrics['intervene_all_accuracy']:<10.4f} "
                f"{metrics['intervention_count']:<10}"
            )

    # Determine output paths
    if output_dir is None:
        output_dir = intervention_dir.parent / "predictor_evaluations"

    base_name = f"{predictor.feature_set}_{predictor.label_type}_layer{predictor.layer_idx}"
    output_json_path = output_dir / f"{base_name}.json"
    output_txt_path = output_dir / f"{base_name}.txt"

    # Save results
    save_results(
        all_metrics,
        predicted_wrong_qids,
        all_predictions,
        predictor,
        threshold,
        output_json_path,
        output_txt_path
    )

    # Find best intervention for this predictor
    sorted_interventions = sorted(
        all_metrics.items(),
        key=lambda x: x[1]['accuracy_gain_vs_baseline'],
        reverse=True
    )
    best_intervention_name, best_intervention_metrics = sorted_interventions[0] if sorted_interventions else (None, None)

    return PredictorEvaluation(
        predictor_name=base_name,
        predictor=predictor,
        threshold=threshold,
        all_metrics=all_metrics,
        best_intervention_name=best_intervention_name,
        best_intervention_metrics=best_intervention_metrics
    )


def format_top_predictors_summary(evaluations: List[PredictorEvaluation]) -> str:
    """Format summary of all predictor setups with positive gain

    Args:
        evaluations: List of all predictor evaluations

    Returns:
        Formatted summary as string
    """
    # Collect all predictor-intervention combinations
    all_setups = []
    for eval_result in evaluations:
        for intervention_name, metrics in eval_result.all_metrics.items():
            all_setups.append({
                'predictor_name': eval_result.predictor_name,
                'predictor': eval_result.predictor,
                'threshold': eval_result.threshold,
                'intervention_name': intervention_name,
                'metrics': metrics
            })

    # Sort by accuracy gain (descending, highest first)
    sorted_setups = sorted(
        all_setups,
        key=lambda x: x['metrics']['accuracy_gain_vs_baseline'],
        reverse=True
    )

    # Filter to only positive gains
    positive_gain_setups = [s for s in sorted_setups if s['metrics']['accuracy_gain_vs_baseline'] > 0]

    # Build summary text
    lines = []
    lines.append("=" * 120)
    lines.append(f"ALL POSITIVE GAINS: PREDICTOR-INTERVENTION SETUPS (RANKED BY GAIN VS BASELINE)")
    lines.append("=" * 120)
    lines.append(f"Showing {len(positive_gain_setups)}/{len(all_setups)} setups with gain > 0%")
    lines.append("")

    for i, setup in enumerate(positive_gain_setups, 1):
        pred = setup['predictor']
        interv = setup['intervention_name']
        m = setup['metrics']

        lines.append(f"#{i}. {setup['predictor_name']} + {interv}")
        lines.append(f"    Predictor: {pred.feature_set} | {pred.label_type} | Layer {pred.layer_idx} | Threshold {setup['threshold']:.3f}")
        lines.append(f"    Baseline Accuracy:     {m['baseline_accuracy']:.4f}")
        lines.append(f"    Hybrid Accuracy:       {m['hybrid_accuracy']:.4f}")
        lines.append(f"    Gain vs Baseline:      {m['accuracy_gain_vs_baseline']:+.4f}")
        lines.append(f"    Gain vs Intervene-All: {m['accuracy_gain_vs_intervene_all']:+.4f}")
        lines.append(f"    Intervention Rate:     {m['predictor_intervention_rate']:.1%}")
        lines.append(f"    F1: {m['f1']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f}")
        lines.append(f"    Interventions: {m['predictor_intervention_count']} total "
                     f"({m['baseline_correct_intervened']} baseline-correct, "
                     f"{m['baseline_incorrect_intervened']} baseline-incorrect)")
        lines.append(f"    Transitions: {m['intervened_wrong_to_correct']} wrong→correct, "
                     f"{m['intervened_correct_to_wrong']} correct→wrong")
        lines.append("")

    lines.append("=" * 120)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictor-gated interventions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mutually exclusive: single predictor or directory of predictors
    predictor_group = parser.add_mutually_exclusive_group(required=True)
    predictor_group.add_argument(
        "--predictor",
        type=Path,
        help="Path to predictor NPZ file (e.g., hash_pca_correctness_layer30.npz)"
    )
    predictor_group.add_argument(
        "--predictor-dir",
        type=Path,
        help="Directory containing multiple predictor NPZ files"
    )

    parser.add_argument(
        "--steering-vectors",
        type=Path,
        required=True,
        help="Path to steering vectors NPZ file for test set"
    )
    parser.add_argument(
        "--intervention-dir",
        type=Path,
        required=True,
        help="Directory containing intervention results (steering/ and text_injection/)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for predictor (default: use predictor's best_threshold)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: intervention_dir.parent / 'predictor_evaluations')"
    )
    parser.add_argument(
        "--intervene-num-steps-threshold",
        type=int,
        default=None,
        help="Only intervene on questions with baseline num_steps >= threshold. "
             "If provided, also reports vanilla (non-gated) intervention results. "
             "Example: --intervene-num-steps-threshold 5 means only intervene on questions with >=5 steps"
    )

    args = parser.parse_args()

    print(f"\n{'='*120}")
    print(f"PREDICTOR-GATED INTERVENTION EVALUATION")
    print(f"{'='*120}")
    print(f"Steering vectors: {args.steering_vectors}")
    print(f"Intervention dir: {args.intervention_dir}")

    # Load steering vectors once (used for all predictors)
    steering_data = load_steering_vectors(args.steering_vectors)

    # Single predictor mode
    if args.predictor is not None:
        print(f"Mode: Single predictor")
        print(f"Predictor: {args.predictor}")
        if args.intervene_num_steps_threshold is not None:
            print(f"Num-steps filter: Only intervene on questions with baseline num_steps >= {args.intervene_num_steps_threshold}")

        evaluate_single_predictor(
            args.predictor,
            steering_data,
            args.intervention_dir,
            args.threshold,
            args.output_dir,
            args.intervene_num_steps_threshold
        )

    # Predictor directory mode
    else:
        print(f"Mode: Predictor directory")
        print(f"Predictor directory: {args.predictor_dir}")
        if args.intervene_num_steps_threshold is not None:
            print(f"Num-steps filter: Only intervene on questions with baseline num_steps >= {args.intervene_num_steps_threshold}")

        # Find all NPZ files in directory
        predictor_files = sorted(args.predictor_dir.glob("*.npz"))

        if not predictor_files:
            print(f"\n❌ Error: No .npz files found in {args.predictor_dir}")
            return 1

        print(f"\nFound {len(predictor_files)} predictor files")

        # Evaluate each predictor and collect results
        evaluations = []
        for i, predictor_path in enumerate(predictor_files, 1):
            print(f"\n\n{'#'*120}")
            print(f"# PREDICTOR {i}/{len(predictor_files)}")
            print(f"{'#'*120}")

            try:
                eval_result = evaluate_single_predictor(
                    predictor_path,
                    steering_data,
                    args.intervention_dir,
                    args.threshold,
                    args.output_dir,
                    args.intervene_num_steps_threshold
                )
                evaluations.append(eval_result)
            except Exception as e:
                print(f"\n❌ Error evaluating {predictor_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print and save positive gains summary
        if evaluations:
            print(f"\n\n{'#'*120}")
            print(f"# SUMMARY: ALL POSITIVE GAINS ACROSS ALL PREDICTORS")
            print(f"{'#'*120}")

            positive_gains_summary = format_top_predictors_summary(evaluations)
            print(f"\n{positive_gains_summary}")

            # Save positive gains summary to file
            output_dir = args.output_dir if args.output_dir else args.intervention_dir.parent / "predictor_evaluations"
            summary_path = output_dir / "positive_gains_summary.txt"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                f.write(positive_gains_summary)
                f.write("\n")

            print(f"\n✓ Positive gains summary saved to: {summary_path}")

    print(f"\n{'='*120}")
    print(f"✓ All evaluations complete!")
    print(f"{'='*120}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
