#!/usr/bin/env python
"""Compute ideal trajectory divergence and find intervention thresholds

This script implements the ideal trajectory analysis:
1. Fits PCA on correct K-step trajectories to construct shared reasoning manifold
2. Computes ideal correct trajectory as mean in PCA space
3. Measures divergence (local and cumulative) from ideal path
4. Finds intervention thresholds using percentile-based strategy
5. ALWAYS runs full grid search AND optionally computes thresholds for specified percentiles

Usage:
    # Grid search mode only (finds optimal percentiles):
    python scripts/steering/traj_based/compute_ideal_trajectory_thresholds.py \
        --train-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
        --num-steps 3 \
        --layer 31 \
        --pca-dim 128 \
        --output-dir output/ideal_trajectory

    # Grid search + specified percentiles (saves both in JSON):
    python scripts/steering/traj_based/compute_ideal_trajectory_thresholds.py \
        --train-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz \
        --test-npz output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_test.npz \
        --num-steps 3 \
        --layer 31 \
        --correct-percentile 70.0 \
        --wrong-percentile 20.0

The output JSON contains:
    - grid_search_results.overall_best: Best thresholds from grid search
    - specified_percentiles_thresholds.best_step: Thresholds from your percentiles (if specified)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class TrajectoryDivergence:
    """Divergence metrics for a single trajectory"""
    question_id: int
    is_correct: bool
    num_steps: int  # Trajectory length (K+1 when hash included)
    # PCA-projected activations: [num_steps, pca_dim]
    z: np.ndarray
    # Normalized divergence at each step: [num_steps]
    delta: np.ndarray
    # Cumulative divergence up to each step: [num_steps]
    D: np.ndarray


@dataclass
class IdealTrajectoryModel:
    """Trained ideal trajectory model

    Note: num_steps is K (reasoning steps only, not including hash)
    But mu and sigma have shape [K+1, ...] when hash is included
    """
    num_steps: int  # K (reasoning steps only)
    layer_idx: int
    pca_dim: int
    # PCA model
    pca: PCA
    # Ideal correct trajectory: [K+1, pca_dim] when hash included
    mu: np.ndarray
    # Per-step spread (std dev): [K+1] when hash included
    sigma: np.ndarray
    epsilon: float = 1e-6


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

    return result


def extract_k_step_trajectories(
    steering_data: Dict,
    layer_idx: int,
    num_steps: int,
    include_hash: bool = False
) -> Tuple[List[np.ndarray], List[int], List[bool]]:
    """Extract all trajectories with exactly K steps at a specific layer

    Args:
        steering_data: Loaded steering vectors
        layer_idx: Layer index
        num_steps: Number of reasoning steps (K)
        include_hash: If True, include hash activation as step K+1

    Returns:
        trajectories: List of [K, hidden_dim] or [K+1, hidden_dim] arrays
        question_ids: List of question IDs
        is_correct: List of correctness labels
    """
    step_acts = steering_data['step_activations'][layer_idx]
    step_nums = steering_data['step_numbers'][layer_idx]
    question_ids_step = steering_data['question_ids_step'][layer_idx]
    is_correct_step = steering_data['is_correct_step'][layer_idx]

    # Group by question ID
    question_to_steps = {}
    for i, qid in enumerate(question_ids_step):
        if qid not in question_to_steps:
            question_to_steps[qid] = {
                'steps': [],
                'step_numbers': [],
                'activations': [],
                'is_correct': is_correct_step[i]
            }
        question_to_steps[qid]['steps'].append(step_nums[i])
        question_to_steps[qid]['step_numbers'].append(step_nums[i])
        question_to_steps[qid]['activations'].append(step_acts[i])

    # Filter for trajectories with exactly K steps
    trajectories = []
    question_ids = []
    is_correct = []

    for qid, data in question_to_steps.items():
        if len(data['steps']) != num_steps:
            continue

        # Sort by step number
        sorted_indices = np.argsort(data['step_numbers'])
        traj = [data['activations'][i] for i in sorted_indices]

        # Optionally add hash activation
        if include_hash:
            hash_acts = steering_data['hash_activations'][layer_idx]
            question_ids_hash = steering_data['question_ids_hash'][layer_idx]

            # Find hash for this question
            hash_mask = (question_ids_hash == qid)
            if hash_mask.sum() > 0:
                hash_act = hash_acts[hash_mask][0]
                traj.append(hash_act)
            else:
                # Skip if no hash
                continue

        trajectories.append(np.array(traj))
        question_ids.append(qid)
        is_correct.append(data['is_correct'])

    return trajectories, question_ids, is_correct


def fit_ideal_trajectory_model(
    trajectories: List[np.ndarray],
    is_correct: List[bool],
    num_reasoning_steps: int,
    pca_dim: int = 128,
    epsilon: float = 1e-6
) -> IdealTrajectoryModel:
    """Fit PCA and compute ideal correct trajectory

    Args:
        trajectories: List of [K+1, hidden_dim] trajectory arrays (K reasoning steps + hash)
        is_correct: List of correctness labels
        num_reasoning_steps: Number of reasoning steps K (not including hash)
        pca_dim: Target PCA dimensionality
        epsilon: Small constant for numerical stability

    Returns:
        IdealTrajectoryModel with PCA, ideal path μ, and spreads σ
        Note: model.num_steps will be K (reasoning steps), not K+1
    """
    print("\n" + "="*80)
    print("FITTING IDEAL TRAJECTORY MODEL")
    print("="*80)

    # Filter for correct trajectories only
    correct_trajectories = [traj for traj, correct in zip(trajectories, is_correct) if correct]

    if len(correct_trajectories) == 0:
        raise ValueError("No correct trajectories found!")

    trajectory_length = correct_trajectories[0].shape[0]  # K+1 (includes hash)
    hidden_dim = correct_trajectories[0].shape[1]

    print(f"\nCorrect trajectories: {len(correct_trajectories)}")
    print(f"Num reasoning steps (K): {num_reasoning_steps}")
    print(f"Trajectory length (K+1 with hash): {trajectory_length}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Target PCA dim: {pca_dim}")

    # Collect all step activations from correct trajectories
    # Shape: [N_correct * (K+1), hidden_dim]
    all_correct_activations = []
    for traj in correct_trajectories:
        for step_act in traj:
            all_correct_activations.append(step_act)

    all_correct_activations = np.array(all_correct_activations)
    print(f"\nTotal activations for PCA: {all_correct_activations.shape}")

    # Fit PCA on all correct activations
    print(f"\nFitting PCA...")
    actual_pca_dim = min(pca_dim, all_correct_activations.shape[0] - 1, hidden_dim)
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    pca.fit(all_correct_activations)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA components: {actual_pca_dim}")
    print(f"  Explained variance: {explained_var:.4f}")

    # Project all correct trajectories to PCA space
    print(f"\nProjecting correct trajectories to PCA space...")
    z_correct = []  # List of [K+1, pca_dim] arrays
    for traj in tqdm(correct_trajectories, desc="Projecting"):
        z = pca.transform(traj)  # [K+1, pca_dim]
        z_correct.append(z)

    # Compute ideal correct trajectory (mean at each step)
    # NOTE: We compute for all K+1 steps (including hash)
    print(f"\nComputing ideal trajectory μ_j...")
    mu = []  # [K+1, pca_dim]
    for j in range(trajectory_length):
        # Collect all step-j activations from correct trajectories
        step_j_activations = np.array([z[j] for z in z_correct])  # [N_correct, pca_dim]
        mu_j = step_j_activations.mean(axis=0)  # [pca_dim]
        mu.append(mu_j)

    mu = np.array(mu)  # [K+1, pca_dim]
    print(f"  Ideal trajectory shape: {mu.shape}")

    # Compute per-step spread σ_j
    print(f"\nComputing per-step spreads σ_j...")
    sigma = []  # [K+1]
    for j in range(trajectory_length):
        # Compute divergences from μ_j for all correct trajectories
        divergences = []
        for z in z_correct:
            div = np.linalg.norm(z[j] - mu[j])
            divergences.append(div)

        # Compute std dev of divergences
        sigma_j = np.std(divergences)
        sigma.append(sigma_j)
        step_label = f"Step {j+1}" if j < num_reasoning_steps else "Hash"
        print(f"  {step_label}: σ = {sigma_j:.4f}")

    sigma = np.array(sigma)  # [K+1]

    # Create model with K (reasoning steps), not K+1
    # The mu and sigma arrays contain K+1 elements (including hash)
    model = IdealTrajectoryModel(
        num_steps=num_reasoning_steps,  # K (reasoning steps only)
        layer_idx=-1,  # Will be set by caller
        pca_dim=actual_pca_dim,
        pca=pca,
        mu=mu,  # [K+1, pca_dim]
        sigma=sigma,  # [K+1]
        epsilon=epsilon
    )

    print(f"\n✓ Model trained successfully")

    return model


def compute_trajectory_divergence(
    trajectory: np.ndarray,
    question_id: int,
    is_correct: bool,
    model: IdealTrajectoryModel
) -> TrajectoryDivergence:
    """Compute divergence metrics for a single trajectory

    Args:
        trajectory: [K+1, hidden_dim] activation trajectory (K reasoning + hash)
        question_id: Question ID
        is_correct: Correctness label
        model: Trained ideal trajectory model

    Returns:
        TrajectoryDivergence with δ_j and D_j
        Note: num_steps will be K+1 (trajectory length including hash)
    """
    # Project to PCA space
    z = model.pca.transform(trajectory)  # [K+1, pca_dim]

    # Compute normalized divergence at each step
    # Use trajectory length (K+1) not model.num_steps (K)
    trajectory_length = trajectory.shape[0]
    delta = np.zeros(trajectory_length)
    for j in range(trajectory_length):
        # Euclidean distance from ideal
        dist = np.linalg.norm(z[j] - model.mu[j])
        # Normalize by spread
        delta[j] = dist / (model.sigma[j] + model.epsilon)

    # Compute cumulative divergence
    D = np.cumsum(delta)

    return TrajectoryDivergence(
        question_id=question_id,
        is_correct=is_correct,
        num_steps=trajectory_length,  # K+1 (includes hash)
        z=z,
        delta=delta,
        D=D
    )


def compute_all_divergences(
    trajectories: List[np.ndarray],
    question_ids: List[int],
    is_correct: List[bool],
    model: IdealTrajectoryModel
) -> List[TrajectoryDivergence]:
    """Compute divergences for all trajectories

    Returns:
        List of TrajectoryDivergence objects
    """
    print("\n" + "="*80)
    print("COMPUTING TRAJECTORY DIVERGENCES")
    print("="*80)

    divergences = []
    for traj, qid, correct in tqdm(zip(trajectories, question_ids, is_correct),
                                    total=len(trajectories),
                                    desc="Computing divergences"):
        div = compute_trajectory_divergence(traj, qid, correct, model)
        divergences.append(div)

    print(f"\n✓ Computed divergences for {len(divergences)} trajectories")

    return divergences


def evaluate_question_level(
    divergences: List[TrajectoryDivergence],
    step_idx: int,
    tau_local: float,
    tau_cum: float,
    lambda_penalty: float = 2.0
) -> Dict:
    """Evaluate thresholds as a question-level gate using a single step index.

    Unlike per-step classification, this treats intervention as binary per question:
    "Does this question trigger intervention at this step?"

    Args:
        divergences: List of trajectory divergences
        step_idx: Step index to evaluate at
        tau_local: Local divergence threshold
        tau_cum: Cumulative divergence threshold
        lambda_penalty: Penalty weight for intervening on correct questions (default: 2.0)

    Returns:
        Dict with question-level metrics including net benefit score
    """
    correct_divs = [d for d in divergences if d.is_correct]
    wrong_divs = [d for d in divergences if not d.is_correct]

    # Question-level: intervene if this step triggers
    def flagged(d):
        return (d.delta[step_idx] > tau_local) or (d.D[step_idx] > tau_cum)

    intervene_correct = sum(1 for d in correct_divs if flagged(d))
    intervene_wrong = sum(1 for d in wrong_divs if flagged(d))

    tp = intervene_wrong
    fp = intervene_correct
    fn = len(wrong_divs) - intervene_wrong
    tn = len(correct_divs) - intervene_correct

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    intervention_rate_correct = intervene_correct / len(correct_divs) if len(correct_divs) > 0 else 0.0
    intervention_rate_wrong = intervene_wrong / len(wrong_divs) if len(wrong_divs) > 0 else 0.0

    # NET BENEFIT SCORE: Hit many wrongs, avoid corrects
    # Higher score = better (more wrongs hit, fewer corrects touched)
    score = intervention_rate_wrong - lambda_penalty * intervention_rate_correct

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'intervention_rate_correct': float(intervention_rate_correct),
        'intervention_rate_wrong': float(intervention_rate_wrong),
        'score': float(score),  # NEW: Net benefit score
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'intervene_correct': int(intervene_correct),
        'intervene_wrong': int(intervene_wrong),
    }


def compute_thresholds_for_step(
    divergences: List[TrajectoryDivergence],
    step_idx: int,
    correct_percentile: float,
    wrong_percentile: float,
    lambda_penalty: float = 2.0
) -> Dict:
    """Compute thresholds for a specific step using question-level evaluation

    Args:
        divergences: List of trajectory divergences
        step_idx: Step index (0-indexed, or -1 for last step)
        correct_percentile: High percentile for correct
        wrong_percentile: Mid/high percentile for wrong
        lambda_penalty: Penalty weight for intervening on correct questions

    Returns:
        Dict with thresholds and question-level evaluation metrics
    """
    # Separate by correctness
    correct_divs = [d for d in divergences if d.is_correct]
    wrong_divs = [d for d in divergences if not d.is_correct]

    if len(wrong_divs) == 0:
        return None

    # Extract divergences at specified step
    delta_correct = np.array([d.delta[step_idx] for d in correct_divs])
    delta_wrong = np.array([d.delta[step_idx] for d in wrong_divs])

    D_correct = np.array([d.D[step_idx] for d in correct_divs])
    D_wrong = np.array([d.D[step_idx] for d in wrong_divs])

    # Compute percentile-based thresholds
    t1_local = np.percentile(delta_correct, correct_percentile)
    t2_local = np.percentile(delta_wrong, wrong_percentile)
    tau_local = (t1_local + t2_local) / 2

    t1_cum = np.percentile(D_correct, correct_percentile)
    t2_cum = np.percentile(D_wrong, wrong_percentile)
    tau_cum = (t1_cum + t2_cum) / 2

    # Evaluate using question-level gating
    metrics = evaluate_question_level(divergences, step_idx, tau_local, tau_cum, lambda_penalty)

    return {
        'step_idx': step_idx,
        'tau_local': float(tau_local),
        'tau_cum': float(tau_cum),
        'correct_percentile': correct_percentile,
        'wrong_percentile': wrong_percentile,
        't1_local': float(t1_local),
        't2_local': float(t2_local),
        't1_cum': float(t1_cum),
        't2_cum': float(t2_cum),
        'delta_correct_mean': float(delta_correct.mean()),
        'delta_correct_std': float(delta_correct.std()),
        'delta_wrong_mean': float(delta_wrong.mean()),
        'delta_wrong_std': float(delta_wrong.std()),
        'D_correct_mean': float(D_correct.mean()),
        'D_correct_std': float(D_correct.std()),
        'D_wrong_mean': float(D_wrong.mean()),
        'D_wrong_std': float(D_wrong.std()),
        'n_correct': len(correct_divs),
        'n_wrong': len(wrong_divs),
        **metrics,  # Include all question-level metrics
    }


def print_all_grid_search_results(
    all_results: List[Dict],
    step_name: str
):
    """Print detailed table of all grid search results for a step

    Args:
        all_results: List of result dicts from compute_thresholds_for_step
        step_name: Name of the step for display
    """
    print(f"\n{'='*140}")
    print(f"ALL GRID SEARCH RESULTS FOR {step_name.upper()}")
    print(f"{'='*140}")

    # Guard against empty results
    if not all_results:
        print("\nNo valid results to display (all candidates violated constraints)")
        print(f"{'='*140}")
        return

    # Sort by SCORE descending (not F1!)
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

    # Header
    print(f"\n{'Corr%':<8} {'Wrong%':<8} {'τ_local':<10} {'τ_cum':<10} "
          f"{'Int(C)':<10} {'Int(W)':<10} {'Score':<10} {'F1':<10} {'Prec':<10} {'Rec':<10}")
    print("-" * 140)

    # Print all results
    for result in sorted_results:
        print(f"{result['correct_percentile']:<8.1f} "
              f"{result['wrong_percentile']:<8.1f} "
              f"{result['tau_local']:<10.4f} "
              f"{result['tau_cum']:<10.4f} "
              f"{result['intervention_rate_correct']:<10.2%} "
              f"{result['intervention_rate_wrong']:<10.2%} "
              f"{result['score']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f}")

    print(f"{'='*140}")
    print(f"Best SCORE: {sorted_results[0]['score']:.4f} at "
          f"Corr={sorted_results[0]['correct_percentile']:.1f}%, "
          f"Wrong={sorted_results[0]['wrong_percentile']:.1f}% "
          f"(F1={sorted_results[0]['f1']:.4f})")


def get_k_specific_grid(num_steps: int) -> Tuple[List[float], List[float]]:
    """Get K-specific percentile grids based on observed patterns

    Args:
        num_steps: Number of reasoning steps K

    Returns:
        (correct_percentiles, wrong_percentiles) tuples
    """
    if num_steps in [6, 7]:
        # Hard Ks where steering helps: be more aggressive
        correct_percentiles = [80.0, 85.0, 90.0, 92.5, 95.0]
        wrong_percentiles = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
        print(f"  Using AGGRESSIVE grid for K={num_steps} (steering tends to help)")
    else:
        # Easy Ks where steering often hurts: be very conservative
        correct_percentiles = [92.5, 95.0, 97.5, 99.0]
        wrong_percentiles = [50.0, 55.0, 60.0, 65.0, 70.0]
        print(f"  Using CONSERVATIVE grid for K={num_steps} (high baseline accuracy)")

    return correct_percentiles, wrong_percentiles


def grid_search_thresholds(
    divergences: List[TrajectoryDivergence],
    num_steps: int,
    correct_percentiles: List[float] = None,
    wrong_percentiles: List[float] = None,
    lambda_penalty: float = 2.0,
    max_corr_rate: float = 0.40,
    min_wrong_rate: float = 0.20,
    print_all_results: bool = True
) -> Dict:
    """Grid search over percentile combinations using net benefit score

    Args:
        divergences: List of trajectory divergences
        num_steps: Number of reasoning steps K (for K-specific grids)
        correct_percentiles: List of percentiles for correct (if None, uses K-specific grid)
        wrong_percentiles: List of percentiles for wrong (if None, uses K-specific grid)
        lambda_penalty: Penalty weight for intervening on correct questions (default: 2.0)
        max_corr_rate: Maximum allowed intervention rate on correct questions (hard constraint)
        min_wrong_rate: Minimum required intervention rate on wrong questions (hard constraint)
        print_all_results: If True, print detailed tables for all combinations

    Returns:
        Dict with best thresholds per step and overall best
    """
    print("\n" + "="*80)
    print("GRID SEARCH FOR OPTIMAL THRESHOLDS (SCORE-BASED)")
    print("="*80)

    # Use K-specific grids if not provided
    if correct_percentiles is None or wrong_percentiles is None:
        correct_percentiles, wrong_percentiles = get_k_specific_grid(num_steps)

    print(f"\nGrid search configuration:")
    print(f"  Correct percentiles: {correct_percentiles}")
    print(f"  Wrong percentiles: {wrong_percentiles}")
    print(f"  Total combinations: {len(correct_percentiles) * len(wrong_percentiles)}")
    print(f"  Lambda penalty (λ): {lambda_penalty}")
    print(f"  Hard constraints:")
    print(f"    - Max intervention rate on correct: {max_corr_rate:.1%}")
    print(f"    - Min intervention rate on wrong: {min_wrong_rate:.1%}")

    trajectory_length = divergences[0].num_steps

    # For each step, find best percentile combination
    best_per_step = []
    all_results_per_step = {}  # Store all results for detailed printing

    for step_idx in range(trajectory_length):
        step_name = f"step_{step_idx + 1}" if step_idx < num_steps else "hash"
        print(f"\n{'-'*80}")
        print(f"Grid search for {step_name} (index {step_idx})...")
        print(f"{'-'*80}")

        best_score = -1e9  # Use score, not F1
        best_config = None
        step_results = []
        skipped_by_constraints = 0

        # Try all combinations
        for corr_p in correct_percentiles:
            for wrong_p in wrong_percentiles:
                result = compute_thresholds_for_step(
                    divergences, step_idx, corr_p, wrong_p, lambda_penalty
                )
                if result is None:
                    continue

                # Apply hard constraints
                if result['intervention_rate_correct'] > max_corr_rate:
                    skipped_by_constraints += 1
                    continue  # Too aggressive on correct, skip

                if result['intervention_rate_wrong'] < min_wrong_rate:
                    skipped_by_constraints += 1
                    continue  # Not helpful enough on wrong, skip

                step_results.append(result)

                # Select by SCORE, not F1
                if result['score'] > best_score:
                    best_score = result['score']
                    best_config = result

        if best_config is None:
            print(f"  ⚠ No valid configuration found for {step_name}")
            if skipped_by_constraints > 0:
                print(f"    ({skipped_by_constraints} candidates violated hard constraints)")
                print(f"    Trying with relaxed constraints...")

                # Retry with relaxed constraints
                relaxed_max_corr = 0.60  # Relax from 40% to 60%
                relaxed_min_wrong = 0.10  # Relax from 20% to 10%

                # Re-compute all combinations with relaxed constraints
                for corr_p in correct_percentiles:
                    for wrong_p in wrong_percentiles:
                        result = compute_thresholds_for_step(
                            divergences, step_idx, corr_p, wrong_p, lambda_penalty
                        )
                        if result is None:
                            continue

                        # Apply relaxed constraints
                        if (result['intervention_rate_correct'] <= relaxed_max_corr and
                            result['intervention_rate_wrong'] >= relaxed_min_wrong):
                            if result['score'] > best_score:
                                best_score = result['score']
                                best_config = result

                if best_config is not None:
                    print(f"    ✓ Found valid configuration with relaxed constraints")
                    print(f"      (Max correct: {relaxed_max_corr:.0%}, Min wrong: {relaxed_min_wrong:.0%})")

            if best_config is None:
                print(f"    ✗ Still no valid configuration with relaxed constraints")
                print(f"    Using final fallback: selecting best by score ignoring constraints...")

                # Final fallback: compute all combinations and pick best by score
                # without ANY constraints (ensures every step gets thresholds)
                all_candidates = []
                for corr_p in correct_percentiles:
                    for wrong_p in wrong_percentiles:
                        result = compute_thresholds_for_step(
                            divergences, step_idx, corr_p, wrong_p, lambda_penalty
                        )
                        if result is not None:
                            all_candidates.append(result)

                if all_candidates:
                    # Pick best by score (no constraints)
                    best_config = max(all_candidates, key=lambda x: x['score'])
                    print(f"    ✓ Selected best unconstrained config")
                    print(f"      (Int rates: {best_config['intervention_rate_correct']:.1%} correct, {best_config['intervention_rate_wrong']:.1%} wrong)")
                else:
                    print(f"    ✗ No valid configurations at all, skipping {step_name}")
                    continue

        all_results_per_step[step_name] = step_results

        print(f"\n  Best configuration for {step_name}:")
        print(f"    Correct percentile: {best_config['correct_percentile']}")
        print(f"    Wrong percentile:   {best_config['wrong_percentile']}")
        print(f"    τ_local: {best_config['tau_local']:.4f}")
        print(f"    τ_cum:   {best_config['tau_cum']:.4f}")
        print(f"    SCORE:   {best_config['score']:.4f} (higher is better)")
        print(f"    F1:      {best_config['f1']:.4f} (for reference)")
        print(f"    Intervention rates: {best_config['intervention_rate_correct']:.1%} correct, {best_config['intervention_rate_wrong']:.1%} wrong")
        if skipped_by_constraints > 0:
            print(f"    ({skipped_by_constraints} candidates skipped by constraints)")

        best_per_step.append({
            'step_name': step_name,
            'step_idx': step_idx,
            **best_config
        })

    # Find overall best across all steps (by SCORE)
    if not best_per_step:
        print("\n❌ No valid configurations found!")
        return None

    overall_best = max(best_per_step, key=lambda x: x['score'])

    print(f"\n{'='*80}")
    print(f"OVERALL BEST THRESHOLD (BY SCORE)")
    print(f"{'='*80}")
    print(f"\nBest step: {overall_best['step_name']} (index {overall_best['step_idx']})")
    print(f"  Correct percentile: {overall_best['correct_percentile']}")
    print(f"  Wrong percentile:   {overall_best['wrong_percentile']}")
    print(f"  τ_local: {overall_best['tau_local']:.4f}")
    print(f"  τ_cum:   {overall_best['tau_cum']:.4f}")
    print(f"  SCORE:   {overall_best['score']:.4f}")
    print(f"  F1:      {overall_best['f1']:.4f} (for reference)")
    print(f"  Precision: {overall_best['precision']:.4f}")
    print(f"  Recall:    {overall_best['recall']:.4f}")
    print(f"  Intervention rate (correct): {overall_best['intervention_rate_correct']:.2%}")
    print(f"  Intervention rate (wrong):   {overall_best['intervention_rate_wrong']:.2%}")

    # Print detailed tables for all results
    if print_all_results:
        print("\n" + "="*140)
        print("DETAILED GRID SEARCH RESULTS (ALL VALID COMBINATIONS)")
        print("="*140)

        for step_name, step_results in all_results_per_step.items():
            print_all_grid_search_results(step_results, step_name)

    return {
        'best_per_step': best_per_step,
        'overall_best': overall_best,
        'all_results_per_step': all_results_per_step,
        'grid_search_space': {
            'correct_percentiles': correct_percentiles,
            'wrong_percentiles': wrong_percentiles,
            'lambda_penalty': lambda_penalty,
            'max_corr_rate': max_corr_rate,
            'min_wrong_rate': min_wrong_rate,
        }
    }


def evaluate_thresholds_on_dataset(
    divergences: List[TrajectoryDivergence],
    step_idx: int,
    tau_local: float,
    tau_cum: float,
    dataset_name: str = "Dataset",
    lambda_penalty: float = 2.0
) -> Dict:
    """Evaluate thresholds on a dataset using question-level evaluation

    Args:
        divergences: List of trajectory divergences
        step_idx: Step index to evaluate at
        tau_local: Local divergence threshold
        tau_cum: Cumulative divergence threshold
        dataset_name: Name of dataset for logging
        lambda_penalty: Penalty weight for net benefit score

    Returns:
        Dict with question-level evaluation metrics including score
    """
    print("\n" + "="*80)
    print(f"EVALUATING ON {dataset_name.upper()}")
    print("="*80)

    # Separate by correctness
    correct_divs = [d for d in divergences if d.is_correct]
    wrong_divs = [d for d in divergences if not d.is_correct]

    print(f"\nCorrect trajectories: {len(correct_divs)}")
    print(f"Wrong trajectories: {len(wrong_divs)}")
    print(f"Evaluating at step index: {step_idx}")

    # Use question-level evaluation
    metrics = evaluate_question_level(divergences, step_idx, tau_local, tau_cum, lambda_penalty)

    print(f"\nIntervention rates:")
    print(f"  Correct: {metrics['intervene_correct']}/{len(correct_divs)} = {metrics['intervention_rate_correct']:.2%}")
    print(f"  Wrong:   {metrics['intervene_wrong']}/{len(wrong_divs)} = {metrics['intervention_rate_wrong']:.2%}")

    print(f"\nQuestion-level metrics:")
    print(f"  Score:     {metrics['score']:.4f} (higher is better)")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    return {
        'step_idx': step_idx,
        'n_correct': len(correct_divs),
        'n_wrong': len(wrong_divs),
        **metrics,  # Include all question-level metrics (score, f1, etc.)
    }


def save_results(
    output_dir: Path,
    num_steps: int,
    layer_idx: int,
    model: IdealTrajectoryModel,
    grid_search_results: Dict,
    test_eval: Dict,
    train_divergences: List[TrajectoryDivergence],
    test_divergences: List[TrajectoryDivergence],
    specified_percentiles_thresholds: Dict = None,
    specified_percentiles_test_eval: Dict = None
):
    """Save all results to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-step thresholds from grid search results
    per_step_thresholds = {}
    if grid_search_results and 'best_per_step' in grid_search_results:
        for step_result in grid_search_results['best_per_step']:
            step_name = step_result['step_name']
            per_step_thresholds[step_name] = {
                'tau_local': step_result['tau_local'],
                'tau_cum': step_result['tau_cum'],
                'step_idx': step_result['step_idx'],
                'score': step_result['score'],  # NEW: Net benefit score
                'f1': step_result['f1'],
                'precision': step_result['precision'],
                'recall': step_result['recall'],
                'intervention_rate_correct': step_result['intervention_rate_correct'],
                'intervention_rate_wrong': step_result['intervention_rate_wrong'],
            }

    # Save summary JSON
    summary = {
        'config': {
            'num_steps': num_steps,
            'layer_idx': layer_idx,
            'pca_dim': model.pca_dim,
            'epsilon': model.epsilon,
            'includes_hash': True,  # Always true now
        },
        'per_step_thresholds': per_step_thresholds,  # NEW: Step-specific thresholds
        'grid_search_results': grid_search_results,
        'test_evaluation': test_eval,
    }

    # Add specified percentiles results if provided
    if specified_percentiles_thresholds is not None:
        summary['specified_percentiles_thresholds'] = specified_percentiles_thresholds
        if specified_percentiles_test_eval is not None:
            summary['specified_percentiles_test_evaluation'] = specified_percentiles_test_eval

    summary_path = output_dir / f"thresholds_K{num_steps}_layer{layer_idx}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to: {summary_path}")
    print(f"\nPer-step thresholds saved:")
    for step_name, thresholds in per_step_thresholds.items():
        print(f"  {step_name}: τ_local={thresholds['tau_local']:.4f}, τ_cum={thresholds['tau_cum']:.4f}, F1={thresholds['f1']:.4f}")

    # Save model (PCA + ideal trajectory)
    model_path = output_dir / f"ideal_trajectory_model_K{num_steps}_layer{layer_idx}.npz"
    np.savez(
        model_path,
        num_steps=model.num_steps,
        layer_idx=layer_idx,
        pca_dim=model.pca_dim,
        pca_components=model.pca.components_,
        pca_mean=model.pca.mean_,
        pca_explained_variance_ratio=model.pca.explained_variance_ratio_,
        mu=model.mu,
        sigma=model.sigma,
        epsilon=model.epsilon,
    )

    print(f"✓ Saved model to: {model_path}")

    # Save divergences
    def save_divergences(divs: List[TrajectoryDivergence], split: str):
        div_path = output_dir / f"divergences_{split}_K{num_steps}_layer{layer_idx}.npz"

        # Get actual trajectory length (K+1 when hash is included)
        trajectory_length = divs[0].num_steps if divs else num_steps + 1

        np.savez(
            div_path,
            question_ids=np.array([d.question_id for d in divs]),
            is_correct=np.array([d.is_correct for d in divs]),
            num_reasoning_steps=num_steps,  # K (reasoning steps only, for reference)
            trajectory_length=trajectory_length,  # K+1 (includes hash)
            includes_hash=True,  # Always true in current implementation
            # Stack all z, delta, D arrays
            z=np.array([d.z for d in divs]),  # [N, trajectory_length, pca_dim]
            delta=np.array([d.delta for d in divs]),  # [N, trajectory_length]
            D=np.array([d.D for d in divs]),  # [N, trajectory_length]
        )

        print(f"✓ Saved {split} divergences to: {div_path}")

    save_divergences(train_divergences, 'train')
    save_divergences(test_divergences, 'test')


def main():
    parser = argparse.ArgumentParser(
        description="Compute ideal trajectory divergence and find intervention thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--train-npz",
        type=Path,
        required=True,
        help="Path to training steering vectors NPZ"
    )
    parser.add_argument(
        "--test-npz",
        type=Path,
        required=True,
        help="Path to test steering vectors NPZ"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of reasoning steps K to analyze"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to analyze"
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=128,
        help="Target PCA dimensionality (default: 128)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ideal_trajectory"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--correct-percentile",
        type=float,
        default=None,
        help="Correct percentile for specified threshold (grid search always runs; this is additional)"
    )
    parser.add_argument(
        "--wrong-percentile",
        type=float,
        default=None,
        help="Wrong percentile for specified threshold (grid search always runs; this is additional)"
    )

    args = parser.parse_args()

    print("="*80)
    print("IDEAL TRAJECTORY DIVERGENCE ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Train NPZ: {args.train_npz}")
    print(f"  Test NPZ:  {args.test_npz}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Layer:     {args.layer}")
    print(f"  PCA dim:   {args.pca_dim}")
    print(f"  Include hash: True (always)")

    # Grid search always runs
    print(f"  Grid search: Yes (always runs)")

    # Check if both percentiles are provided
    if args.correct_percentile is not None and args.wrong_percentile is not None:
        print(f"  Additional specified percentiles: Yes")
        print(f"    Correct percentile: {args.correct_percentile}")
        print(f"    Wrong percentile: {args.wrong_percentile}")
    else:
        print(f"  Additional specified percentiles: No")

    print(f"  Output dir: {args.output_dir}")

    # Load data
    train_data = load_steering_vectors(args.train_npz)
    test_data = load_steering_vectors(args.test_npz)

    # Extract K-step trajectories (always include hash)
    print("\n" + "="*80)
    print("EXTRACTING K-STEP TRAJECTORIES (INCLUDING HASH)")
    print("="*80)

    print("\nTraining set:")
    train_trajs, train_qids, train_correct = extract_k_step_trajectories(
        train_data, args.layer, args.num_steps, include_hash=True
    )
    print(f"  Total: {len(train_trajs)}")
    print(f"  Correct: {sum(train_correct)}")
    print(f"  Wrong: {len(train_correct) - sum(train_correct)}")

    print("\nTest set:")
    test_trajs, test_qids, test_correct = extract_k_step_trajectories(
        test_data, args.layer, args.num_steps, include_hash=True
    )
    print(f"  Total: {len(test_trajs)}")
    print(f"  Correct: {sum(test_correct)}")
    print(f"  Wrong: {len(test_correct) - sum(test_correct)}")

    if len(train_trajs) == 0:
        print("\n❌ Error: No training trajectories found!")
        return 1

    if len(test_trajs) == 0:
        print("\n❌ Error: No test trajectories found!")
        return 1

    # Fit ideal trajectory model on training correct trajectories
    model = fit_ideal_trajectory_model(
        train_trajs,
        train_correct,
        num_reasoning_steps=args.num_steps,  # K (reasoning steps, not including hash)
        pca_dim=args.pca_dim
    )
    model.layer_idx = args.layer

    # Compute divergences on training set
    print("\nComputing divergences on TRAINING set...")
    train_divergences = compute_all_divergences(
        train_trajs,
        train_qids,
        train_correct,
        model
    )

    # ALWAYS run full grid search for best thresholds on training set
    print("\nRunning full grid search on training set...")
    grid_search_results = grid_search_thresholds(
        train_divergences,
        num_steps=args.num_steps
    )

    if grid_search_results is None:
        print("\n❌ Error: Grid search failed (no wrong trajectories?)")
        return 1

    best_thresholds = grid_search_results['overall_best']

    # Compute divergences on test set
    print("\nComputing divergences on TEST set...")
    test_divergences = compute_all_divergences(
        test_trajs,
        test_qids,
        test_correct,
        model
    )

    # Evaluate best thresholds on test set
    test_eval = evaluate_thresholds_on_dataset(
        test_divergences,
        best_thresholds['step_idx'],
        best_thresholds['tau_local'],
        best_thresholds['tau_cum'],
        "Test Set (Grid Search Best)"
    )

    # If user specified percentiles, ALSO compute those thresholds
    specified_percentiles_thresholds = None
    specified_percentiles_test_eval = None

    if args.correct_percentile is not None and args.wrong_percentile is not None:
        print("\n" + "="*80)
        print(f"COMPUTING THRESHOLDS FOR SPECIFIED PERCENTILES")
        print(f"  Correct percentile: {args.correct_percentile}")
        print(f"  Wrong percentile: {args.wrong_percentile}")
        print("="*80)

        # Compute thresholds for each step using specified percentiles
        all_step_results = []
        for step_idx in range(train_divergences[0].num_steps):
            step_result = compute_thresholds_for_step(
                train_divergences,
                step_idx,
                args.correct_percentile,
                args.wrong_percentile
            )
            if step_result is not None:
                step_name = f"step_{step_idx + 1}" if step_idx < args.num_steps else "hash"
                step_result['step_name'] = step_name
                all_step_results.append(step_result)

                print(f"\n{step_name} (index {step_idx}):")
                print(f"  τ_local: {step_result['tau_local']:.4f}")
                print(f"  τ_cum:   {step_result['tau_cum']:.4f}")
                print(f"  F1:      {step_result['f1']:.4f}")
                print(f"  Precision: {step_result['precision']:.4f}")
                print(f"  Recall:    {step_result['recall']:.4f}")

        if all_step_results:
            # Find best step based on F1
            best_specified = max(all_step_results, key=lambda x: x['f1'])

            print(f"\n{'='*80}")
            print(f"BEST STEP FOR SPECIFIED PERCENTILES")
            print(f"{'='*80}")
            print(f"\nBest step: {best_specified['step_name']} (index {best_specified['step_idx']})")
            print(f"  τ_local: {best_specified['tau_local']:.4f}")
            print(f"  τ_cum:   {best_specified['tau_cum']:.4f}")
            print(f"  F1:      {best_specified['f1']:.4f}")

            specified_percentiles_thresholds = {
                'correct_percentile': args.correct_percentile,
                'wrong_percentile': args.wrong_percentile,
                'best_step': best_specified,
                'all_steps': all_step_results,
            }

            # Evaluate on test set
            specified_percentiles_test_eval = evaluate_thresholds_on_dataset(
                test_divergences,
                best_specified['step_idx'],
                best_specified['tau_local'],
                best_specified['tau_cum'],
                "Test Set (Specified Percentiles)"
            )

    # Save results
    save_results(
        args.output_dir,
        args.num_steps,
        args.layer,
        model,
        grid_search_results,
        test_eval,
        train_divergences,
        test_divergences,
        specified_percentiles_thresholds,
        specified_percentiles_test_eval
    )

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\n[1] GRID SEARCH BEST THRESHOLDS (SCORE-BASED):")
    print(f"  Step: {best_thresholds['step_name']} (index {best_thresholds['step_idx']})")
    print(f"  Correct percentile: {best_thresholds['correct_percentile']}")
    print(f"  Wrong percentile: {best_thresholds['wrong_percentile']}")
    print(f"  τ_local: {best_thresholds['tau_local']:.4f}")
    print(f"  τ_cum: {best_thresholds['tau_cum']:.4f}")

    print(f"\n{'Metric':<30} {'Training':<15} {'Test':<15}")
    print("-"*60)
    print(f"{'Score (optimization target)':<30} {best_thresholds['score']:<15.4f} {test_eval['score']:<15.4f}")
    print(f"{'Intervention rate (correct)':<30} {best_thresholds['intervention_rate_correct']:<15.2%} {test_eval['intervention_rate_correct']:<15.2%}")
    print(f"{'Intervention rate (wrong)':<30} {best_thresholds['intervention_rate_wrong']:<15.2%} {test_eval['intervention_rate_wrong']:<15.2%}")
    print(f"{'Precision':<30} {best_thresholds['precision']:<15.4f} {test_eval['precision']:<15.4f}")
    print(f"{'Recall':<30} {best_thresholds['recall']:<15.4f} {test_eval['recall']:<15.4f}")
    print(f"{'F1':<30} {best_thresholds['f1']:<15.4f} {test_eval['f1']:<15.4f}")
    print(f"{'Accuracy':<30} {best_thresholds['accuracy']:<15.4f} {test_eval['accuracy']:<15.4f}")

    # Print specified percentiles summary if available
    if specified_percentiles_thresholds is not None:
        best_spec = specified_percentiles_thresholds['best_step']
        spec_test = specified_percentiles_test_eval

        print(f"\n[2] SPECIFIED PERCENTILES THRESHOLDS:")
        print(f"  Correct percentile: {args.correct_percentile}")
        print(f"  Wrong percentile: {args.wrong_percentile}")
        print(f"  Step: {best_spec['step_name']} (index {best_spec['step_idx']})")
        print(f"  τ_local: {best_spec['tau_local']:.4f}")
        print(f"  τ_cum: {best_spec['tau_cum']:.4f}")

        print(f"\n{'Metric':<30} {'Training':<15} {'Test':<15}")
        print("-"*60)
        print(f"{'Score':<30} {best_spec['score']:<15.4f} {spec_test['score']:<15.4f}")
        print(f"{'Intervention rate (correct)':<30} {best_spec['intervention_rate_correct']:<15.2%} {spec_test['intervention_rate_correct']:<15.2%}")
        print(f"{'Intervention rate (wrong)':<30} {best_spec['intervention_rate_wrong']:<15.2%} {spec_test['intervention_rate_wrong']:<15.2%}")
        print(f"{'Precision':<30} {best_spec['precision']:<15.4f} {spec_test['precision']:<15.4f}")
        print(f"{'Recall':<30} {best_spec['recall']:<15.4f} {spec_test['recall']:<15.4f}")
        print(f"{'F1':<30} {best_spec['f1']:<15.4f} {spec_test['f1']:<15.4f}")
        print(f"{'Accuracy':<30} {best_spec['accuracy']:<15.4f} {spec_test['accuracy']:<15.4f}")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
