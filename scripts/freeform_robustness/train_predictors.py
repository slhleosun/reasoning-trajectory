#!/usr/bin/env python
"""Train logistic regression or MLP predictors on steering vectors

This script trains L2-regularized logistic regression or MLP classifiers to predict:
1. Incorrectness (error) of final answer (y=1: incorrect/error, y=0: correct)
2. Error types (referencing_context_value_error vs. others)

Features (8 sets for 'all' mode):
1. [Step-1, Step-2] concatenated
2. [Step-2 - Step-1] difference
3. [Step-1, Step-2, Step-3] concatenated
4. [Step-2 - Step-1, Step-3 - Step-2, Hash - Last-Step] differences concatenated
5. Hash activations only
6. [Hash - Last-Step] difference
7. [Hash, Hash - Last-Step] concatenated, then PCA to 128 dimensions each (hash_pca → 256 dims)
8. [Hash, Hash-Last, Last-SecondLast] concatenated, then PCA to 128 dimensions each (hash_last_diffs_pca → 384 dims)
9. [Hash, Hash-Last, Last-SecondLast] concatenated, then PCA to 128 dimensions jointly (hash_last_diffs_pca_joint → 128 dims)

PCA features (DEPRECATED, not used in 'all' mode):
- Step-1 PCA(64) + Step-2 PCA(64) concatenated
- Step-1 PCA(64) + Step-2 PCA(64) difference

Trains one classifier per layer with cross-validation for hyperparameter C.
Supports parallel training across multiple GPUs/CPUs.

Usage:
    # Train all classifiers with linear model
    python scripts/predictors/train_predictors.py --mode all

    # Train all classifiers with MLP
    python scripts/predictors/train_predictors.py --mode all --model-type mlp

    # Train all classifiers in quick mode (odd layers only: 1, 3, 5, ...)
    python scripts/predictors/train_predictors.py --mode all --quick

    # Train specific feature set and label
    python scripts/predictors/train_predictors.py --feature-set hash_last_diffs_pca --label correctness
"""

import sys
import json
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def setup_logging(log_dir: Path) -> Path:
    """Setup file and console logging

    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"predictor_training_{timestamp}.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return log_file


class LogisticRegressionModel(nn.Module):
    """PyTorch Logistic Regression with L2 regularization"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    """Small MLP with 1 hidden layer (128 units) for classification"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@dataclass
class TrainingConfig:
    """Configuration for training"""
    feature_set: str  # step1_step2, step2_minus_step1, step1_step2_step3, step_diffs, hash_only, hash_minus_last, hash_pca, hash_last_diffs_pca, hash_last_diffs_pca_joint, pca_concat (deprecated), pca_diff (deprecated)
    label_type: str  # correctness, error_type
    layer_idx: int
    model_type: str = "linear"  # "linear" or "mlp"
    test_size: float = 0.1
    cv_folds: int = 5
    random_state: int = 42
    n_pca_components: int = 64  # 64 for legacy pca_concat/pca_diff, 128 for hash_pca and hash_last_diffs_pca (set dynamically)

    # Training hyperparameters
    max_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.01
    patience: int = 50  # Early stopping patience

    # Hyperparameter search grid for L2 regularization (weight_decay)
    c_values: List[float] = None

    def __post_init__(self):
        if self.c_values is None:
            # Convert sklearn C to PyTorch weight_decay (weight_decay = 1/C)
            self.c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass
class TrainingResult:
    """Results from training a single classifier"""
    config: TrainingConfig
    best_c: float
    train_accuracy: float
    train_f1: float
    train_roc_auc: float
    test_accuracy: float
    test_f1: float
    test_roc_auc: float
    coefficients: Union[np.ndarray, Dict]  # np.ndarray for linear, Dict for MLP
    intercept: Optional[float]  # float for linear, None for MLP
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    cv_results: Dict = None
    classification_report: str = ""
    n_train_samples: int = 0
    n_test_samples: int = 0
    train_label_distribution: Dict = None
    test_label_distribution: Dict = None
    # Validation data for threshold tuning
    test_features: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    # Threshold tuning results
    best_threshold: Optional[float] = None
    threshold_metrics: Optional[Dict] = None


def load_steering_vectors(npz_path: Path) -> Dict:
    """Load steering vectors NPZ file

    Handles two storage formats:
    1. Object arrays (freeform / variable-length): each element is a per-layer array
    2. Auto-stacked 3D arrays (GSM8K / same-length): numpy stacks same-shape arrays

    CRITICAL: Object arrays from NPZ lose inner dtype — booleans become Python
    bool objects where ~True == -2. We must cast to proper numpy dtypes.

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

    def _unpack(key, inner_dtype=None):
        """Unpack array → list of per-layer arrays with correct dtypes."""
        arr = data[key]
        if arr.dtype == object:
            # Object array: split elements and cast
            out = [arr[i] for i in range(len(arr))]
        elif arr.ndim >= 2 and arr.shape[0] == num_layers:
            # Auto-stacked 3D: split along first axis
            out = [arr[i] for i in range(num_layers)]
        else:
            out = [arr]
        if inner_dtype is not None:
            out = [np.asarray(x, dtype=inner_dtype) for x in out]
        return out

    # Load arrays per layer with proper dtypes
    result = {
        'step_activations': _unpack('step_activations', np.float32),
        'hash_activations': _unpack('hash_activations', np.float32),
        'step_numbers':     _unpack('step_numbers', np.int32),
        'question_ids_step': _unpack('question_ids_step', np.int64),
        'question_ids_hash': _unpack('question_ids_hash', np.int64),
        'is_correct_step':   _unpack('is_correct_step', np.bool_),
        'is_correct_hash':   _unpack('is_correct_hash', np.bool_),
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
    }

    # Print sample info
    if len(result['step_activations']) > 0:
        layer_0_step = result['step_activations'][0]
        if len(layer_0_step) > 0:
            print(f"  Layer 0 step activations shape: {layer_0_step.shape}")
            print(f"  Layer 0 step numbers: {np.unique(result['step_numbers'][0])}")
            print(f"  Layer 0 is_correct dtype: {result['is_correct_step'][0].dtype}")

    return result


def load_error_annotations(json_path: Path) -> Dict[int, Dict]:
    """Load error annotations JSON

    Returns:
        Dict mapping question_id -> annotation dict
    """
    print(f"\nLoading error annotations from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = {}
    for qid_str, anno in data['annotations'].items():
        question_id = int(qid_str)
        annotations[question_id] = anno

    print(f"  Loaded {len(annotations)} error annotations")

    # Count error types
    error_types = {}
    for anno in annotations.values():
        error_type = anno['annotation']['pred_wrong_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1

    print(f"  Error type distribution:")
    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"    {error_type}: {count}")

    return annotations


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


def create_feature_set(
    steering_data: Dict,
    layer_idx: int,
    feature_set: str,
    n_pca_components: int = 64
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Create feature set for a specific layer

    Feature sets:
    - step1_step2: Concatenation of Step-1, Step-2 activations
    - step2_minus_step1: Step-2 - Step-1 difference
    - step1_step2_step3: Concatenation of Step-1, Step-2, Step-3 activations
    - step_diffs: Concatenation of (Step-2 - Step-1), (Step-3 - Step-2), (Hash - Last-Step)
    - hash_only: Hash activations only
    - hash_minus_last: Hash - Last-Step difference
    - hash_pca: Concatenation of Hash and (Hash - Last-Step), then PCA to 128 dimensions each (256 total)
    - hash_last_diffs_pca: Hash, (Hash-Last), (Last-SecondLast) each PCA'd to 128 dims (384 total)
    - hash_last_diffs_pca_joint: Hash, (Hash-Last), (Last-SecondLast) concatenated then jointly PCA'd to 128 dims
    - pca_concat: (DEPRECATED) Step-1 PCA + Step-2 PCA concatenated
    - pca_diff: (DEPRECATED) Step-2 PCA - Step-1 PCA

    Returns:
        features: Either [n_samples, feature_dim] array OR tuple of (comp1, comp2, ...) raw arrays for PCA features
        question_ids: [n_samples]
        is_correct: [n_samples]
    """

    if feature_set == "step1_step2":
        # Concatenate Step-1 and Step-2
        step1_acts, step1_qids, step1_correct = extract_step_activations(
            steering_data, layer_idx, step_number=1
        )
        step2_acts, step2_qids, step2_correct = extract_step_activations(
            steering_data, layer_idx, step_number=2
        )

        if len(step1_acts) == 0:
            return np.array([]), np.array([]), np.array([])

        # Find common question IDs
        common_qids = np.intersect1d(step1_qids, step2_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        correct_aligned = step1_correct[step1_indices]

        # Concatenate
        features = np.concatenate([step1_aligned, step2_aligned], axis=1)
        return features, common_qids, correct_aligned

    elif feature_set == "step2_minus_step1":
        # Step-2 - Step-1 difference
        step1_acts, step1_qids, step1_correct = extract_step_activations(
            steering_data, layer_idx, step_number=1
        )
        step2_acts, step2_qids, step2_correct = extract_step_activations(
            steering_data, layer_idx, step_number=2
        )

        if len(step1_acts) == 0:
            return np.array([]), np.array([]), np.array([])

        # Find common question IDs
        common_qids = np.intersect1d(step1_qids, step2_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        correct_aligned = step1_correct[step1_indices]

        # Compute difference
        features = step2_aligned - step1_aligned
        return features, common_qids, correct_aligned

    elif feature_set == "step1_step2_step3":
        # Concatenate Step-1, Step-2, Step-3
        step1_acts, step1_qids, step1_correct = extract_step_activations(
            steering_data, layer_idx, step_number=1
        )
        step2_acts, step2_qids, step2_correct = extract_step_activations(
            steering_data, layer_idx, step_number=2
        )
        step3_acts, step3_qids, step3_correct = extract_step_activations(
            steering_data, layer_idx, step_number=3
        )

        if len(step1_acts) == 0:
            return np.array([]), np.array([]), np.array([])

        # Find common question IDs across all three steps
        common_qids = np.intersect1d(np.intersect1d(step1_qids, step2_qids), step3_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])
        step3_indices = np.array([np.where(step3_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        step3_aligned = step3_acts[step3_indices]
        correct_aligned = step1_correct[step1_indices]

        # Concatenate
        features = np.concatenate([step1_aligned, step2_aligned, step3_aligned], axis=1)
        return features, common_qids, correct_aligned

    elif feature_set == "step_diffs":
        # Concatenate differences: (Step-2 - Step-1), (Step-3 - Step-2), (Hash - Last-Step)
        step1_acts, step1_qids, step1_correct = extract_step_activations(
            steering_data, layer_idx, step_number=1
        )
        step2_acts, step2_qids, step2_correct = extract_step_activations(
            steering_data, layer_idx, step_number=2
        )
        step3_acts, step3_qids, step3_correct = extract_step_activations(
            steering_data, layer_idx, step_number=3
        )
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        last_acts, last_qids, last_correct, _ = extract_last_step_activations(
            steering_data, layer_idx
        )

        if len(step1_acts) == 0:
            return np.array([]), np.array([]), np.array([])

        # Find common question IDs across all components
        common_qids = np.intersect1d(
            np.intersect1d(np.intersect1d(step1_qids, step2_qids), step3_qids),
            np.intersect1d(hash_qids, last_qids)
        )

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])
        step3_indices = np.array([np.where(step3_qids == qid)[0][0] for qid in common_qids])
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        step3_aligned = step3_acts[step3_indices]
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        correct_aligned = hash_correct[hash_indices]  # Use hash correctness

        # Compute differences
        diff1 = step2_aligned - step1_aligned  # Step-2 - Step-1
        diff2 = step3_aligned - step2_aligned  # Step-3 - Step-2
        diff3 = hash_aligned - last_aligned    # Hash - Last-Step

        # Concatenate all differences
        features = np.concatenate([diff1, diff2, diff3], axis=1)
        return features, common_qids, correct_aligned

    elif feature_set == "hash_only":
        # Hash activations only
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        return hash_acts, hash_qids, hash_correct

    elif feature_set == "hash_minus_last":
        # Hash - Last-Step difference
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        last_acts, last_qids, last_correct, _ = extract_last_step_activations(
            steering_data, layer_idx
        )

        # Find common question IDs
        common_qids = np.intersect1d(hash_qids, last_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        correct_aligned = hash_correct[hash_indices]

        # Compute difference
        features = hash_aligned - last_aligned
        return features, common_qids, correct_aligned

    elif feature_set == "hash_pca":
        # Concatenate hash and (hash - last), then apply PCA to 128 dimensions
        # Return raw features as tuple - PCA will be fit after train/test split
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        last_acts, last_qids, last_correct, _ = extract_last_step_activations(
            steering_data, layer_idx
        )

        # Find common question IDs
        common_qids = np.intersect1d(hash_qids, last_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        correct_aligned = hash_correct[hash_indices]

        # Compute hash - last difference
        hash_minus_last = hash_aligned - last_aligned

        # Return RAW features as tuple - PCA will be fit after train/test split
        # This avoids data leakage from fitting PCA on all data
        # Format: (hash, hash_minus_last) - will be concatenated and PCA'd to 128 dims
        return (hash_aligned, hash_minus_last), common_qids, correct_aligned

    elif feature_set == "hash_last_diffs_pca":
        # Hash + (Hash-Last) + (Last-SecondLast), each PCA'd to 128 dims → 384 total
        # Return raw features as tuple - PCA will be fit after train/test split
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        last_acts, last_qids, last_correct, _ = extract_last_step_activations(
            steering_data, layer_idx
        )
        second_last_acts, second_last_qids, second_last_correct = extract_second_to_last_step_activations(
            steering_data, layer_idx
        )

        # Find common question IDs across all three
        common_qids = np.intersect1d(
            np.intersect1d(hash_qids, last_qids),
            second_last_qids
        )

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])
        second_last_indices = np.array([np.where(second_last_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        second_last_aligned = second_last_acts[second_last_indices]
        correct_aligned = hash_correct[hash_indices]

        # Compute differences
        hash_minus_last = hash_aligned - last_aligned
        last_minus_second_last = last_aligned - second_last_aligned

        # Return RAW features as tuple of 3 components - PCA will be fit after train/test split
        # Format: (hash, hash_minus_last, last_minus_second_last) - each will be PCA'd to 128 dims
        return (hash_aligned, hash_minus_last, last_minus_second_last), common_qids, correct_aligned

    elif feature_set == "hash_last_diffs_pca_joint":
        # Hash + (Hash-Last) + (Last-SecondLast) concatenated, then jointly PCA'd to 128 dims
        # This is different from hash_last_diffs_pca which PCA's each component separately
        hash_acts, hash_qids, hash_correct = extract_hash_activations(
            steering_data, layer_idx
        )
        last_acts, last_qids, last_correct, _ = extract_last_step_activations(
            steering_data, layer_idx
        )
        second_last_acts, second_last_qids, second_last_correct = extract_second_to_last_step_activations(
            steering_data, layer_idx
        )

        # Find common question IDs across all three
        common_qids = np.intersect1d(
            np.intersect1d(hash_qids, last_qids),
            second_last_qids
        )

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        hash_indices = np.array([np.where(hash_qids == qid)[0][0] for qid in common_qids])
        last_indices = np.array([np.where(last_qids == qid)[0][0] for qid in common_qids])
        second_last_indices = np.array([np.where(second_last_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        hash_aligned = hash_acts[hash_indices]
        last_aligned = last_acts[last_indices]
        second_last_aligned = second_last_acts[second_last_indices]
        correct_aligned = hash_correct[hash_indices]

        # Compute differences
        hash_minus_last = hash_aligned - last_aligned
        last_minus_second_last = last_aligned - second_last_aligned

        # Concatenate all 3 components before PCA (will be jointly PCA'd to 128 dims)
        # Return as a SINGLE-element tuple to indicate joint PCA should be applied
        # Format: (concatenated_features,) - single component for joint PCA
        concatenated = np.concatenate([hash_aligned, hash_minus_last, last_minus_second_last], axis=1)
        return (concatenated,), common_qids, correct_aligned

    elif feature_set in ["pca_concat", "pca_diff"]:
        # DEPRECATED: Legacy PCA features using Step-1 and Step-2
        step1_acts, step1_qids, step1_correct = extract_step_activations(
            steering_data, layer_idx, step_number=1
        )
        step2_acts, step2_qids, step2_correct = extract_step_activations(
            steering_data, layer_idx, step_number=2
        )

        if len(step1_acts) == 0:
            return np.array([]), np.array([]), np.array([])

        # Find common question IDs
        common_qids = np.intersect1d(step1_qids, step2_qids)

        if len(common_qids) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get indices for common questions
        step1_indices = np.array([np.where(step1_qids == qid)[0][0] for qid in common_qids])
        step2_indices = np.array([np.where(step2_qids == qid)[0][0] for qid in common_qids])

        # Extract aligned activations
        step1_aligned = step1_acts[step1_indices]
        step2_aligned = step2_acts[step2_indices]
        correct_aligned = step1_correct[step1_indices]

        # Return RAW features as tuple - PCA will be fit after train/test split
        # This avoids data leakage from fitting PCA on all data
        return (step1_aligned, step2_aligned), common_qids, correct_aligned

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def create_error_type_labels(
    question_ids: np.ndarray,
    is_correct: np.ndarray,
    error_annotations: Dict[int, Dict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create binary labels for error type classification

    Labels:
        y = 1: referencing_context_value_error (or referencing_previous_step_value_error)
        y = 0: other error types

    Returns:
        filtered_features_indices: Indices of samples to keep (incorrect only)
        labels: Binary labels (1 = referencing_context_value_error, 0 = other error types)
        question_ids_filtered: Filtered question IDs
    """
    # Filter for incorrect questions only
    incorrect_mask = ~is_correct
    incorrect_qids = question_ids[incorrect_mask]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Create labels for error types
    labels = []
    valid_indices = []
    valid_qids = []

    for idx, qid in zip(incorrect_indices, incorrect_qids):
        if qid in error_annotations:
            anno = error_annotations[qid]
            error_type = anno['annotation']['pred_wrong_type']

            # Binary classification for error types:
            # y = 1: referencing_context_value_error (or referencing_previous_step_value_error)
            # y = 0: all other error types
            if 'referencing' in error_type and 'value_error' in error_type:
                labels.append(1)  # referencing_context_value_error
            else:
                labels.append(0)  # other error types

            valid_indices.append(idx)
            valid_qids.append(qid)

    return np.array(valid_indices), np.array(labels), np.array(valid_qids)


def train_pytorch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TrainingConfig,
    weight_decay: float,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    """Train a single PyTorch model with early stopping

    Args:
        pos_weight: Weight for positive class (y=1, minority) = n_neg / n_pos

    Returns:
        (trained_model, best_val_auc)
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    pos_weight = pos_weight.to(device)

    # Optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)

    # Loss function with pos_weight to up-weight minority class (y=1, incorrect)
    # pos_weight = n_neg / n_pos (e.g., 1800/200 = 9.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Early stopping
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    # Create data loader for mini-batch training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.max_epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(-1)  # Only squeeze last dim, keep batch dim
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze(-1)  # Only squeeze last dim, keep batch dim
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_y = y_val.cpu().numpy()

            # Compute ROC-AUC
            if len(np.unique(val_y)) > 1:  # Need both classes
                val_auc = roc_auc_score(val_y, val_probs)
            else:
                val_auc = 0.5

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_auc


def train_single_classifier(
    config: TrainingConfig,
    features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    question_ids: np.ndarray,
    device: torch.device,
) -> TrainingResult:
    """Train a single logistic regression classifier with cross-validation using PyTorch + GPU

    Args:
        config: Training configuration
        features: Either [n_samples, feature_dim] OR tuple of (step1_raw, step2_raw) for PCA features
        labels: [n_samples] binary labels
        question_ids: [n_samples] question IDs
        device: PyTorch device (cuda or cpu)

    Returns:
        TrainingResult
    """
    # Check for empty inputs
    if isinstance(features, tuple):
        if len(features[0]) == 0 or len(labels) == 0:
            raise ValueError("Empty features or labels")
    else:
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("Empty features or labels")

    # Handle PCA features (raw components passed as tuple)
    pca_components_list = None
    if isinstance(features, tuple):
        num_components = len(features)

        if num_components == 1:
            # Single-component joint PCA (hash_last_diffs_pca_joint)
            # Features are already concatenated, just apply single PCA
            comp_raw = features[0]

            # Do train/test split on RAW features
            comp_train, comp_test, y_train, y_test, qids_train, qids_test = train_test_split(
                comp_raw, labels, question_ids,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=labels
            )

            # Use 128 components for joint PCA
            target_pca_components = 128
            n_components = min(target_pca_components, comp_train.shape[0] - 1, comp_train.shape[1])

            if n_components < 2:
                raise ValueError(f"Not enough samples for PCA (need at least 3, got {comp_train.shape[0]})")

            pca = PCA(n_components=n_components, random_state=config.random_state)
            comp_train_pca = pca.fit_transform(comp_train)
            comp_test_pca = pca.transform(comp_test)

            pca_components_list = [pca]

            # Use PCA'd features directly
            X_train = comp_train_pca
            X_test = comp_test_pca

        elif num_components == 2:
            # 2-component PCA (hash_pca or legacy pca features)
            comp1_raw, comp2_raw = features

            # Do train/test split on RAW features
            comp1_train, comp1_test, comp2_train, comp2_test, y_train, y_test, qids_train, qids_test = train_test_split(
                comp1_raw, comp2_raw, labels, question_ids,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=labels
            )

            # Determine number of PCA components based on feature set
            if config.feature_set == "hash_pca":
                target_pca_components = 128
            else:
                target_pca_components = config.n_pca_components

            # Fit PCA ONLY on training data
            n_components = min(target_pca_components, comp1_train.shape[0] - 1, comp1_train.shape[1])

            if n_components < 2:
                raise ValueError(f"Not enough samples for PCA (need at least 3, got {comp1_train.shape[0]})")

            pca1 = PCA(n_components=n_components, random_state=config.random_state)
            pca2 = PCA(n_components=n_components, random_state=config.random_state)

            comp1_train_pca = pca1.fit_transform(comp1_train)
            comp2_train_pca = pca2.fit_transform(comp2_train)
            comp1_test_pca = pca1.transform(comp1_test)
            comp2_test_pca = pca2.transform(comp2_test)

            pca_components_list = [pca1, pca2]

            # Create final features based on config
            if config.feature_set in ["pca_concat", "hash_pca"]:
                X_train = np.concatenate([comp1_train_pca, comp2_train_pca], axis=1)
                X_test = np.concatenate([comp1_test_pca, comp2_test_pca], axis=1)
            elif config.feature_set == "pca_diff":
                X_train = comp2_train_pca - comp1_train_pca
                X_test = comp2_test_pca - comp1_test_pca
            else:
                raise ValueError(f"Unexpected feature_set for 2-component PCA: {config.feature_set}")

        elif num_components == 3:
            # 3-component PCA (hash_last_diffs_pca)
            comp1_raw, comp2_raw, comp3_raw = features

            # Do train/test split on RAW features
            comp1_train, comp1_test, comp2_train, comp2_test, comp3_train, comp3_test, y_train, y_test, qids_train, qids_test = train_test_split(
                comp1_raw, comp2_raw, comp3_raw, labels, question_ids,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=labels
            )

            # Use 128 components for each (or less if insufficient samples/features)
            target_pca_components = 128
            n_components = min(target_pca_components, comp1_train.shape[0] - 1, comp1_train.shape[1])

            if n_components < 2:
                raise ValueError(f"Not enough samples for PCA (need at least 3, got {comp1_train.shape[0]})")

            pca1 = PCA(n_components=n_components, random_state=config.random_state)
            pca2 = PCA(n_components=n_components, random_state=config.random_state)
            pca3 = PCA(n_components=n_components, random_state=config.random_state)

            comp1_train_pca = pca1.fit_transform(comp1_train)
            comp2_train_pca = pca2.fit_transform(comp2_train)
            comp3_train_pca = pca3.fit_transform(comp3_train)
            comp1_test_pca = pca1.transform(comp1_test)
            comp2_test_pca = pca2.transform(comp2_test)
            comp3_test_pca = pca3.transform(comp3_test)

            pca_components_list = [pca1, pca2, pca3]

            # Concatenate all 3 PCA'd components
            X_train = np.concatenate([comp1_train_pca, comp2_train_pca, comp3_train_pca], axis=1)
            X_test = np.concatenate([comp1_test_pca, comp2_test_pca, comp3_test_pca], axis=1)

        else:
            raise ValueError(f"Unsupported number of PCA components: {num_components}")
    else:
        # Regular features (no PCA)
        # 90-10 train-test split (stratified)
        X_train, X_test, y_train, y_test, qids_train, qids_test = train_test_split(
            features, labels, question_ids,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=labels
        )

    # Standardize features (z-score normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights for balanced training
    # y=0 = correct (majority), y=1 = incorrect (minority)
    # We need pos_weight = n_neg / n_pos to up-weight the minority class
    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    n_neg = label_counts[0]  # count of y=0 (correct, majority)
    n_pos = label_counts[1]  # count of y=1 (incorrect, minority)
    pos_weight = torch.FloatTensor([n_neg / n_pos])

    # Convert to PyTorch tensors
    input_dim = X_train_scaled.shape[1]

    # Cross-validation for hyperparameter selection (weight_decay = 1/C)
    kfold = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    best_c = None
    best_avg_val_auc = 0
    cv_results = []

    for c_value in config.c_values:
        weight_decay = 1.0 / c_value  # Convert C to weight_decay
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train)):
            # Split data for this fold
            X_fold_train = torch.FloatTensor(X_train_scaled[train_idx])
            y_fold_train = torch.FloatTensor(y_train[train_idx])
            X_fold_val = torch.FloatTensor(X_train_scaled[val_idx])
            y_fold_val = torch.FloatTensor(y_train[val_idx])

            # Create and train model
            if config.model_type == "mlp":
                fold_model = MLPModel(input_dim)
            else:
                fold_model = LogisticRegressionModel(input_dim)
            _, val_auc = train_pytorch_model(
                fold_model, X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                config, weight_decay, pos_weight, device
            )
            fold_aucs.append(val_auc)

        avg_val_auc = np.mean(fold_aucs)
        cv_results.append({'C': c_value, 'mean_val_auc': avg_val_auc, 'fold_aucs': fold_aucs})

        if avg_val_auc > best_avg_val_auc:
            best_avg_val_auc = avg_val_auc
            best_c = c_value

    # Train final model with best C on full training set
    best_weight_decay = 1.0 / best_c
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)

    # Use a portion of train as validation for early stopping
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_tensor, y_train_tensor,
        test_size=0.1,
        random_state=config.random_state,
        stratify=y_train_tensor.numpy()
    )

    if config.model_type == "mlp":
        final_model = MLPModel(input_dim)
    else:
        final_model = LogisticRegressionModel(input_dim)
    final_model, _ = train_pytorch_model(
        final_model, X_train_final, y_train_final, X_val_final, y_val_final,
        config, best_weight_decay, pos_weight, device
    )

    # Evaluate on full train set
    final_model.eval()
    with torch.no_grad():
        train_logits = final_model(X_train_tensor.to(device)).squeeze(-1).cpu()
        train_probs = torch.sigmoid(train_logits).numpy()
        train_preds = (train_probs > 0.5).astype(int)

        train_accuracy = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average='binary')
        train_roc_auc = roc_auc_score(y_train, train_probs)

    # Evaluate on test set
    with torch.no_grad():
        test_logits = final_model(X_test_tensor.to(device)).squeeze(-1).cpu()
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs > 0.5).astype(int)

        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='binary')
        test_roc_auc = roc_auc_score(y_test, test_probs)

    # Classification report
    report = classification_report(y_test, test_preds, zero_division=0)

    # Label distributions
    train_label_dist = {
        'positive': int(np.sum(y_train == 1)),
        'negative': int(np.sum(y_train == 0))
    }
    test_label_dist = {
        'positive': int(np.sum(y_test == 1)),
        'negative': int(np.sum(y_test == 0))
    }

    # Extract model weights (different for linear vs MLP)
    if config.model_type == "mlp":
        # For MLP, extract all layer weights
        # Store as dict: {fc1_weight, fc1_bias, fc2_weight, fc2_bias}
        # We'll flatten and concatenate for storage compatibility
        fc1_w = final_model.fc1.weight.data.cpu().numpy()  # [128, input_dim]
        fc1_b = final_model.fc1.bias.data.cpu().numpy()     # [128]
        fc2_w = final_model.fc2.weight.data.cpu().numpy()   # [1, 128]
        fc2_b = final_model.fc2.bias.data.cpu().numpy()     # [1]

        # Store weights as structured array - we'll update save_results to handle this
        coefficients = {
            'fc1_weight': fc1_w,
            'fc1_bias': fc1_b,
            'fc2_weight': fc2_w,
            'fc2_bias': fc2_b
        }
        intercept = None  # Not applicable for MLP
    else:
        # For linear model, extract coefficients and intercept as before
        coefficients = final_model.linear.weight.data.cpu().numpy().flatten()
        intercept = final_model.linear.bias.data.cpu().item()

    result = TrainingResult(
        config=config,
        best_c=best_c,
        train_accuracy=train_accuracy,
        train_f1=train_f1,
        train_roc_auc=train_roc_auc,
        test_accuracy=test_accuracy,
        test_f1=test_f1,
        test_roc_auc=test_roc_auc,
        coefficients=coefficients,
        intercept=intercept,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
        cv_results={'cv_scores': cv_results, 'best_c': best_c},
        classification_report=report,
        n_train_samples=len(X_train),
        n_test_samples=len(X_test),
        train_label_distribution=train_label_dist,
        test_label_distribution=test_label_dist,
        # Store test data for threshold tuning
        test_features=X_test,
        test_labels=y_test,
    )

    # Add PCA info if PCA was used
    if pca_components_list is not None:
        result.pca_components = np.stack([pca.components_ for pca in pca_components_list])
        result.pca_mean = np.stack([pca.mean_ for pca in pca_components_list])

    return result


def setup_worker_logging(log_dir: Path):
    """Setup logging for a worker process

    Each worker gets its own log file to avoid interleaving
    """
    process = mp.current_process()
    process_name = process.name.replace('/', '_').replace(' ', '_')

    # Get root logger
    logger = logging.getLogger()

    # Check if this worker has already set up logging
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            # Check if this is already a worker-specific handler
            if process_name in str(handler.baseFilename):
                return  # Already set up

    # Clear all existing handlers in spawned process
    # (spawned processes start fresh but might have default handlers)
    logger.handlers.clear()

    # Set logger level to INFO (critical for spawned processes!)
    logger.setLevel(logging.INFO)

    # Create worker-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"predictor_training_{timestamp}_{process_name}.log"

    # Add file handler for this worker
    file_handler = logging.FileHandler(log_file, mode='a')  # Use 'a' in case multiple tasks
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Also add console handler for errors/warnings in worker
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s [%(processName)s]: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Worker {process_name} started (PID: {process.pid})")
    logger.info("="*80)


def train_classifier_wrapper(args: Tuple) -> Tuple[str, TrainingResult]:
    """Wrapper for parallel training

    Args:
        args: (key, config, steering_data, error_annotations, device, log_dir)

    Returns:
        (key, result) where key is a unique identifier
    """
    key, config, steering_data, error_annotations, device, log_dir = args

    # Setup worker-specific logging if in subprocess
    if mp.current_process().name != 'MainProcess':
        setup_worker_logging(log_dir)

    logger = logging.getLogger()

    try:
        logger.info(f"[START] {key} | Feature: {config.feature_set} | Label: {config.label_type} | Layer: {config.layer_idx} | Model: {config.model_type} | Device: {device}")

        # Create feature set
        features, question_ids, is_correct = create_feature_set(
            steering_data,
            config.layer_idx,
            config.feature_set,
            config.n_pca_components
        )

        # Check for empty features (handle both array and tuple cases)
        if isinstance(features, tuple):
            if len(features[0]) == 0:
                logger.info(f"[SKIP] {key} | Reason: No samples available")
                return key, None
        else:
            if len(features) == 0:
                logger.info(f"[SKIP] {key} | Reason: No samples available")
                return key, None

        # Create labels
        if config.label_type == "correctness":
            # Flip labels: y=1 for incorrect (error), y=0 for correct
            labels = (~is_correct).astype(int)
        elif config.label_type == "error_type":
            # Filter for incorrect samples and create error type labels
            valid_indices, labels, question_ids = create_error_type_labels(
                question_ids, is_correct, error_annotations
            )

            if len(valid_indices) == 0:
                logger.info(f"[SKIP] {key} | Reason: No incorrect samples with error annotations")
                return key, None

            # Filter features based on valid indices
            if isinstance(features, tuple):
                # For PCA features (tuple of step1, step2)
                step1, step2 = features
                features = (step1[valid_indices], step2[valid_indices])
            else:
                # For regular features
                features = features[valid_indices]
        else:
            raise ValueError(f"Unknown label_type: {config.label_type}")

        # Check if we have enough samples (handle both array and tuple cases)
        n_samples = len(features[0]) if isinstance(features, tuple) else len(features)
        if n_samples < 20:
            logger.info(f"[SKIP] {key} | Reason: Too few samples ({n_samples})")
            return key, None

        # Check class balance
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            logger.info(f"[SKIP] {key} | Reason: Only one class present")
            return key, None

        min_class_count = counts.min()
        if min_class_count < 5:
            logger.info(f"[SKIP] {key} | Reason: Minority class has too few samples ({min_class_count})")
            return key, None

        logger.info(f"[TRAIN] {key} | Samples: {n_samples} | Class balance: {dict(zip(unique, counts))}")

        # Train classifier (PCA fitting happens inside train_single_classifier)
        result = train_single_classifier(config, features, labels, question_ids, device)

        # Log results
        logger.info(
            f"[DONE] {key} | "
            f"Best C: {result.best_c:.4f} | "
            f"Train samples: {result.n_train_samples} | "
            f"Test samples: {result.n_test_samples} | "
            f"Test Acc: {result.test_accuracy:.4f} | "
            f"Test F1: {result.test_f1:.4f} | "
            f"Test AUC: {result.test_roc_auc:.4f}"
        )

        return key, result

    except Exception as e:
        logger.error(f"[ERROR] {key} | {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return key, None


def predict_probabilities(features: np.ndarray, predictor: Dict) -> np.ndarray:
    """Compute P(incorrect) using logistic regression or MLP predictor

    Args:
        features: [n_samples, feature_dim] raw features
        predictor: Dict with keys:
            - For linear: coefficients, intercept, scaler_mean, scaler_std
            - For MLP: mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias, scaler_mean, scaler_std

    Returns:
        probabilities: [n_samples] P(y=1|X) = P(incorrect)
    """
    # Standardize features
    X_scaled = (features - predictor['scaler_mean']) / predictor['scaler_std']

    # Check if this is an MLP or linear model
    if 'mlp_fc1_weight' in predictor:
        # MLP forward pass: fc1 -> relu -> fc2 -> sigmoid
        # fc1: [n_samples, input_dim] @ [input_dim, 128].T + [128] = [n_samples, 128]
        hidden = X_scaled @ predictor['mlp_fc1_weight'].T + predictor['mlp_fc1_bias']
        hidden = np.maximum(0, hidden)  # ReLU
        # fc2: [n_samples, 128] @ [128, 1].T + [1] = [n_samples, 1]
        logits = hidden @ predictor['mlp_fc2_weight'].T + predictor['mlp_fc2_bias']
        logits = logits.squeeze(-1)  # [n_samples]
    else:
        # Logistic regression: P(y=1) = sigmoid(X @ w + b)
        logits = X_scaled @ predictor['coefficients'] + predictor['intercept']

    probabilities = 1 / (1 + np.exp(-logits))

    return probabilities


def compute_metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Compute metrics at a specific threshold

    Args:
        labels: [n_samples] binary labels (1 = incorrect, 0 = correct)
        probabilities: [n_samples] P(incorrect)
        threshold: Decision threshold τ

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    predictions = (probabilities >= threshold).astype(int)

    # Compute metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def tune_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    predictor: Dict,
    num_thresholds: int = 201
) -> Tuple[float, Dict, List[Dict]]:
    """Find optimal decision threshold by maximizing F1 score

    Sweeps thresholds τ ∈ [0, 1] and selects the one that maximizes F1 score
    on the "incorrect" class (y=1).

    Args:
        features: [n_samples, feature_dim] raw validation features
        labels: [n_samples] binary labels (1 = incorrect, 0 = correct)
        predictor: Dict with predictor parameters
        num_thresholds: Number of thresholds to sweep

    Returns:
        best_threshold: Optimal threshold τ*
        best_metrics: Metrics at τ*
        all_metrics: List of metrics for each threshold (for analysis)
    """
    # Compute probabilities
    y_prob = predict_probabilities(features, predictor)

    # Sweep thresholds
    thresholds = np.linspace(0, 1, num_thresholds)
    all_metrics = []
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = None

    for tau in thresholds:
        metrics = compute_metrics_at_threshold(labels, y_prob, tau)
        metrics['threshold'] = tau
        all_metrics.append(metrics)

        # Track best F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = tau
            best_metrics = metrics

    return best_threshold, best_metrics, all_metrics


def save_results(results: Dict[str, TrainingResult], output_dir: Path):
    """Save training results with threshold tuning

    Args:
        results: Dict mapping classifier key -> TrainingResult
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Tune decision thresholds on validation (test) sets
    print(f"\n{'='*120}")
    print("TUNING DECISION THRESHOLDS")
    print(f"{'='*120}\n")

    num_tuned = 0
    for key, result in results.items():
        if result is None:
            continue

        # Perform threshold tuning on test set (used as validation set)
        predictor_dict = {
            'coefficients': result.coefficients,
            'intercept': result.intercept,
            'scaler_mean': result.scaler_mean,
            'scaler_std': result.scaler_std,
        }

        try:
            best_threshold, best_metrics, all_metrics = tune_threshold(
                features=result.test_features,
                labels=result.test_labels,
                predictor=predictor_dict,
                num_thresholds=201
            )

            # Store threshold tuning results in result object
            result.best_threshold = best_threshold
            result.threshold_metrics = best_metrics

            num_tuned += 1
            print(f"  {key}: τ* = {best_threshold:.3f}, F1 = {best_metrics['f1']:.4f}, Acc = {best_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"  ⚠ {key}: Threshold tuning failed ({e})")
            result.best_threshold = 0.5  # Fallback to default
            result.threshold_metrics = None

    print(f"\n✓ Tuned thresholds for {num_tuned}/{len(results)} classifiers")

    # STEP 2: Build summary statistics (after threshold tuning)
    summary = []
    for key, result in results.items():
        if result is None:
            continue

        summary_entry = {
            'key': key,
            'feature_set': result.config.feature_set,
            'label_type': result.config.label_type,
            'layer_idx': result.config.layer_idx,
            'model_type': result.config.model_type,
            'best_c': result.best_c,
            'n_train_samples': result.n_train_samples,
            'n_test_samples': result.n_test_samples,
            'train_label_dist': result.train_label_distribution,
            'test_label_dist': result.test_label_distribution,
            'train_accuracy': result.train_accuracy,
            'train_f1': result.train_f1,
            'train_roc_auc': result.train_roc_auc,
            'test_accuracy': result.test_accuracy,
            'test_f1': result.test_f1,
            'test_roc_auc': result.test_roc_auc,
        }

        # Add threshold tuning info
        if result.best_threshold is not None:
            summary_entry['best_threshold'] = float(result.best_threshold)
        if result.threshold_metrics is not None:
            summary_entry['threshold_tuned_accuracy'] = float(result.threshold_metrics['accuracy'])
            summary_entry['threshold_tuned_f1'] = float(result.threshold_metrics['f1'])
            summary_entry['threshold_tuned_precision'] = float(result.threshold_metrics['precision'])
            summary_entry['threshold_tuned_recall'] = float(result.threshold_metrics['recall'])

        summary.append(summary_entry)

    # STEP 3: Save summary JSON (append to existing if it exists)
    summary_path = output_dir / "summary.json"

    # Load existing summary if it exists
    existing_summary = []
    if summary_path.exists():
        print(f"\n📝 Found existing summary.json, loading...")
        try:
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            print(f"   Loaded {len(existing_summary)} existing results")
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ⚠ Warning: Could not load existing summary ({e}), will overwrite")
            existing_summary = []

    # Merge new results with existing (avoiding duplicates by key)
    existing_keys = {entry['key'] for entry in existing_summary}
    new_results = [entry for entry in summary if entry['key'] not in existing_keys]

    # Combine and sort
    merged_summary = existing_summary + new_results
    merged_summary.sort(key=lambda x: (x['layer_idx'], x['feature_set'], x['label_type'], x['model_type']))

    # Save merged summary
    with open(summary_path, 'w') as f:
        json.dump(merged_summary, f, indent=2)

    print(f"\n✓ Saved summary to: {summary_path}")
    print(f"   Added: {len(new_results)} new results")
    print(f"   Skipped: {len(summary) - len(new_results)} duplicate keys")
    print(f"   Total: {len(merged_summary)} results in summary.json")

    # STEP 4: Save detailed NPZ files for each classifier
    for key, result in results.items():
        if result is None:
            continue

        result_path = output_dir / f"{key}.npz"

        save_dict = {
            'feature_set': result.config.feature_set,
            'label_type': result.config.label_type,
            'layer_idx': result.config.layer_idx,
            'model_type': result.config.model_type,
            'best_c': result.best_c,
            'scaler_mean': result.scaler_mean,
            'scaler_std': result.scaler_std,
            'n_train_samples': result.n_train_samples,
            'n_test_samples': result.n_test_samples,
            'train_accuracy': result.train_accuracy,
            'train_f1': result.train_f1,
            'train_roc_auc': result.train_roc_auc,
            'test_accuracy': result.test_accuracy,
            'test_f1': result.test_f1,
            'test_roc_auc': result.test_roc_auc,
            'classification_report': result.classification_report,
            # Add threshold tuning results
            'best_threshold': result.best_threshold if result.best_threshold is not None else 0.5,
        }

        # Save model weights (different format for linear vs MLP)
        if result.config.model_type == "mlp":
            # For MLP, save all layer weights separately
            save_dict['mlp_fc1_weight'] = result.coefficients['fc1_weight']
            save_dict['mlp_fc1_bias'] = result.coefficients['fc1_bias']
            save_dict['mlp_fc2_weight'] = result.coefficients['fc2_weight']
            save_dict['mlp_fc2_bias'] = result.coefficients['fc2_bias']
        else:
            # For linear, save coefficients and intercept
            save_dict['coefficients'] = result.coefficients
            save_dict['intercept'] = result.intercept

        # Add threshold metrics if available
        if result.threshold_metrics is not None:
            save_dict['threshold_tuned_accuracy'] = result.threshold_metrics['accuracy']
            save_dict['threshold_tuned_f1'] = result.threshold_metrics['f1']
            save_dict['threshold_tuned_precision'] = result.threshold_metrics['precision']
            save_dict['threshold_tuned_recall'] = result.threshold_metrics['recall']

        if result.pca_components is not None:
            save_dict['pca_components'] = result.pca_components
            save_dict['pca_mean'] = result.pca_mean

        np.savez(result_path, **save_dict)

    print(f"✓ Saved {len(results)} detailed results to: {output_dir}")

    # Print summary table
    print(f"\n{'='*120}")
    print("TRAINING SUMMARY")
    print(f"{'='*120}")
    print(f"{'Key':<40} {'Layer':<6} {'Feature':<20} {'Label':<15} {'Test Acc':<10} {'Test F1':<10} {'Test AUC':<10}")
    print(f"{'-'*120}")

    for entry in sorted(summary, key=lambda x: (x['layer_idx'], x['feature_set'], x['label_type'])):
        key = entry['key']
        layer = entry['layer_idx']
        feature = entry['feature_set']
        label = entry['label_type']
        test_acc = entry['test_accuracy']
        test_f1 = entry['test_f1']
        test_auc = entry['test_roc_auc']

        print(f"{key:<40} {layer:<6} {feature:<20} {label:<15} {test_acc:<10.4f} {test_f1:<10.4f} {test_auc:<10.4f}")

    print(f"{'='*120}\n")


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any CUDA operations
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass

    parser = argparse.ArgumentParser(
        description="Train logistic regression predictors on steering vectors"
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=Path("output/steering_vectors.npz"),
        help="Path to steering vectors NPZ file"
    )
    parser.add_argument(
        "--error-json",
        type=Path,
        default=Path("output/error_annotations/gsm8k_train_errors.json"),
        help="Path to error annotations JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/predictor_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "single"],
        help="Training mode: 'all' or 'single'"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="step1_step2",
        choices=["step1_step2", "step2_minus_step1", "step1_step2_step3", "step_diffs", "hash_only", "hash_minus_last", "hash_pca", "hash_last_diffs_pca", "hash_last_diffs_pca_joint", "pca_concat", "pca_diff"],
        help="Feature set (for single mode)"
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="correctness",
        choices=["correctness", "error_type"],
        help="Label type (for single mode)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to train (for single mode, default: all layers)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test set size (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed - toy"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Model architecture: 'linear' for logistic regression (default) or 'mlp' for small MLP with 1 hidden layer (128 units)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only train on odd-numbered layers (1, 3, 5, ..., 29, 31)"
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard ID for multi-GPU distribution (0-indexed)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for multi-GPU distribution"
    )

    args = parser.parse_args()

    # Auto-detect single mode if feature-set or label-type is explicitly provided
    # (i.e., different from default values or mode not explicitly set to "all")
    if args.mode == "all":
        # Check if user provided non-default feature-set or label-type
        # If they specified these, they probably want single mode
        parser_defaults = {
            'feature_set': 'step1_step2',
            'label_type': 'correctness',
        }
        if (args.feature_set != parser_defaults['feature_set'] or
            args.label_type != parser_defaults['label_type'] or
            args.layer is not None):
            # User specified custom values, switch to single mode
            print("\n⚠️  Auto-switching to 'single' mode because --feature-set, --label-type, or --layer was specified")
            print("   (Use --mode all to override)")
            args.mode = "single"

    # Validate shard arguments
    if (args.shard_id is not None) != (args.num_shards is not None):
        print("Error: --shard-id and --num-shards must be specified together")
        return 1

    if args.shard_id is not None and args.num_shards is not None:
        if args.shard_id >= args.num_shards or args.shard_id < 0:
            print(f"Error: shard_id ({args.shard_id}) must be in range [0, {args.num_shards-1}]")
            return 1
        # Update output directory to be shard-specific
        args.output_dir = args.output_dir / f"shard_{args.shard_id}"

    # Setup logging
    log_file = setup_logging(Path("logs"))

    print(f"\n{'='*120}")
    print("ERROR PREDICTOR TRAINING")
    print(f"{'='*120}")
    print(f"Mode: {args.mode}")
    print(f"Model type: {args.model_type.upper()} ({'Logistic Regression' if args.model_type == 'linear' else 'MLP (1 hidden layer, 128 units)'})")
    if args.quick:
        print(f"Quick mode: ENABLED (odd layers only: 1, 3, 5, ...)")
    print(f"NPZ path: {args.npz_path}")
    print(f"Error JSON: {args.error_json}")
    print(f"Output dir: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Log file: {log_file}")
    if args.mode == "single":
        print(f"Feature set: {args.feature_set}")
        print(f"Label type: {args.label_type}")
        if args.layer is not None:
            print(f"Layer: {args.layer}")
    print(f"{'='*120}\n")

    # Log configuration
    logger = logging.getLogger()
    logger.info("="*80)
    logger.info("PREDICTOR TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model type: {args.model_type.upper()}")
    if args.quick:
        logger.info("Quick mode: ENABLED (odd layers only)")
    logger.info(f"NPZ path: {args.npz_path}")
    logger.info(f"Error JSON: {args.error_json}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Num workers: {args.num_workers if args.num_workers else 'auto'}")
    if args.num_workers and args.num_workers > 1:
        logger.info("Each worker will create its own log file")
    logger.info("="*80)

    # Load data
    steering_data = load_steering_vectors(args.npz_path)

    num_layers = steering_data['num_layers']

    # Prepare training configurations
    configs = []

    if args.mode == "all":
        # All 9 feature sets (excluding deprecated PCA features)
        feature_sets = ["step1_step2", "step2_minus_step1", "step1_step2_step3", "step_diffs", "hash_only", "hash_minus_last", "hash_pca", "hash_last_diffs_pca", "hash_last_diffs_pca_joint"]
        # Respect --label-type in "all" mode (all feature sets, specified label type)
        label_types = [args.label_type]
        layers = range(num_layers)
    else:  # single
        feature_sets = [args.feature_set]
        label_types = [args.label_type]
        layers = [args.layer] if args.layer is not None else range(num_layers)

    # Load error annotations ONLY if needed for error_type training
    if "error_type" in label_types:
        error_annotations = load_error_annotations(args.error_json)
    else:
        error_annotations = {}  # Empty dict when not needed

    # Apply quick mode: filter to odd-numbered layers only
    if args.quick:
        if isinstance(layers, range):
            # Filter range to odd layers (1, 3, 5, ...)
            layers = [i for i in layers if i % 2 == 1]
        else:
            # Filter list to odd layers
            layers = [i for i in layers if i % 2 == 1]

        if len(layers) == 0:
            print("ERROR: Quick mode resulted in 0 layers (no odd layers in range)")
            return 1

    for feature_set in feature_sets:
        for label_type in label_types:
            for layer_idx in layers:
                config = TrainingConfig(
                    feature_set=feature_set,
                    label_type=label_type,
                    layer_idx=layer_idx,
                    model_type=args.model_type,
                    test_size=args.test_size,
                )

                key = f"{feature_set}_{label_type}_layer{layer_idx:02d}"
                configs.append((key, config))

    print(f"\nPrepared {len(configs)} training configurations")
    print(f"  Feature sets: {feature_sets}")
    print(f"  Label types: {label_types}")
    if args.quick:
        print(f"  Layers (quick mode): {len(layers)} layers -> {layers[:5]}{'...' if len(layers) > 5 else ''}")
    else:
        print(f"  Layers: {len(layers)}")

    # Filter configs by shard (round-robin distribution)
    if args.shard_id is not None and args.num_shards is not None:
        original_count = len(configs)
        configs = [(k, c) for i, (k, c) in enumerate(configs) if i % args.num_shards == args.shard_id]
        print(f"\n📌 Shard {args.shard_id}/{args.num_shards}: Training {len(configs)}/{original_count} configurations")

    # Detect and setup GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        print(f"  Using device: {device}")
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print(f"\n⚠ No GPU detected, using CPU")
        logger.info("No GPU detected, using CPU")

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"\nUsing {num_workers} parallel workers")
    if num_workers > 1:
        print(f"  Each worker will write to its own log file in logs/")

    # Prepare arguments for parallel training
    # Note: For multi-GPU setup, we use a single device for simplicity.
    # Each worker will use the same GPU (PyTorch handles concurrent access).
    # For more advanced multi-GPU distribution, you can implement custom logic.
    log_dir = Path("logs")
    train_args = [
        (key, config, steering_data, error_annotations, device, log_dir)
        for key, config in configs
    ]

    # Train classifiers in parallel
    print(f"\n{'='*120}")
    print("TRAINING CLASSIFIERS")
    print(f"{'='*120}\n")

    logger.info(f"Starting training of {len(configs)} classifiers with {num_workers} workers")

    results = {}
    num_success = 0
    num_skipped = 0
    num_errors = 0

    if num_workers == 1:
        # Sequential training
        pbar = tqdm(train_args, desc="Training", ncols=120)
        for args_tuple in pbar:
            key, result = train_classifier_wrapper(args_tuple)
            results[key] = result

            # Update counts
            if result is not None:
                num_success += 1
            else:
                num_skipped += 1

            # Update progress bar with stats
            pbar.set_postfix({
                'success': num_success,
                'skipped': num_skipped,
                'rate': f'{num_success}/{len(results)}'
            })
    else:
        # Parallel training
        with mp.Pool(num_workers) as pool:
            pbar = tqdm(
                pool.imap_unordered(train_classifier_wrapper, train_args),
                total=len(train_args),
                desc="Training",
                ncols=120
            )
            for key, result in pbar:
                results[key] = result

                # Update counts
                if result is not None:
                    num_success += 1
                else:
                    num_skipped += 1

                # Update progress bar with stats
                pbar.set_postfix({
                    'success': num_success,
                    'skipped': num_skipped,
                    'rate': f'{num_success}/{len(results)}'
                })

    # Filter out None results
    results = {k: v for k, v in results.items() if v is not None}

    print(f"\n✓ Training complete! Successfully trained {len(results)}/{len(configs)} classifiers")
    print(f"  Successful: {num_success}")
    print(f"  Skipped: {num_skipped}")

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total configs: {len(configs)}")
    logger.info(f"Successful: {num_success}")
    logger.info(f"Skipped: {num_skipped}")
    logger.info(f"Success rate: {num_success/len(configs)*100:.1f}%")
    logger.info("="*80)

    # Save results
    save_results(results, args.output_dir)

    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("SESSION COMPLETE")

    print(f"\n{'='*120}")
    print(f"✓ All done! Check detailed logs at: {log_file}")
    print(f"{'='*120}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())