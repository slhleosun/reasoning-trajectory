"""Step-based artifact capture for trajectory features"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..models.generation_output import CompleteGenerationOutput, TimestepArtifacts


# Token ID for "Step" (with capital S)
STEP_TOKEN_ID = 8468


@dataclass
class WindowConfig:
    """Configuration for step-based artifact capture

    Args:
        step_token_id: Token ID to identify step markers (default: 8468 for "Step")
        rank_high_threshold: Threshold for high confidence rank (default: 10)
                            rank ≤ threshold = High confidence
                            rank > threshold = Low confidence
        percentiles: List of percentile points (0-100) to sample (DEPRECATED - kept for compatibility)
    """

    step_token_id: int = STEP_TOKEN_ID  # Token ID for "Step"
    rank_high_threshold: int = 10  # Threshold for trajectory classification
    percentiles: List[int] = None  # DEPRECATED: kept for backwards compatibility

    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = list(range(0, 101, 10))  # 0%, 10%, ..., 100% (for backwards compat)


def find_step_token_positions(
    timestep_artifacts: List[TimestepArtifacts],
    dp1_idx: int,
    dp2_idx: Optional[int] = None,
    step_token_id: int = STEP_TOKEN_ID,
) -> List[int]:
    """Find positions of "Step" tokens in generated sequence (only before dp2)

    Args:
        timestep_artifacts: List of per-timestep artifacts
        dp1_idx: Start of reasoning (absolute index)
        dp2_idx: Start of final answer (absolute index), or None
        step_token_id: Token ID to search for (default: 8468 for "Step")

    Returns:
        List of absolute indices where step tokens appear (only in reasoning portion before dp2)
    """
    step_positions = []

    for rel_idx, artifact in enumerate(timestep_artifacts):
        absolute_idx = dp1_idx + rel_idx

        # Only look at generated tokens before dp2 (in reasoning portion)
        if dp2_idx is not None and absolute_idx >= dp2_idx:
            break  # Stop searching after dp2

        if artifact.next_token_id == step_token_id:
            step_positions.append(absolute_idx)

    return step_positions


def compute_window_boundaries(
    dp1_idx: int, dp2_idx: int, percentiles: List[int]
) -> List[Tuple[int, int]]:
    """DEPRECATED: Compute window boundaries over reasoning steps using percentiles

    This function is kept for backwards compatibility but is no longer the primary method.
    Use compute_step_boundaries() for step-based artifact capture.

    Args:
        dp1_idx: Start of reasoning
        dp2_idx: Start of final answer
        percentiles: List of percentiles (0-100)

    Returns:
        List of (start_idx, end_idx) tuples for each window

    Note:
        Each window is a SINGLE timestep at the percentile point.
        This gives raw values at that specific moment (not aggregated statistics).
    """
    reasoning_length = dp2_idx - dp1_idx + 1

    if reasoning_length <= 0:
        return []

    windows = []
    for pct in percentiles:
        # Calculate index at this percentile
        step_idx = int(np.round(pct / 100.0 * (reasoning_length - 1)))
        absolute_idx = dp1_idx + step_idx

        # Single timestep window
        windows.append((absolute_idx, absolute_idx + 1))

    return windows


def compute_step_boundaries(
    timestep_artifacts: List[TimestepArtifacts],
    dp1_idx: int,
    dp2_idx: Optional[int],
    step_token_id: int = STEP_TOKEN_ID,
) -> List[Tuple[int, int]]:
    """Compute boundaries for each step (range from one "Step" token to the next)

    Args:
        timestep_artifacts: List of per-timestep artifacts
        dp1_idx: Start of reasoning (absolute index)
        dp2_idx: Start of final answer (absolute index), or None
        step_token_id: Token ID to search for (default: 8468 for "Step")

    Returns:
        List of (start_idx, end_idx) tuples for each step
        Each step is the range from one "Step" token to the next "Step" token (exclusive).
        The last step goes from the last "Step" token to dp2_idx (or end of generation).

        NOTE: Only considers "Step" tokens that appear BEFORE dp2_idx (in reasoning portion).

    Example:
        If "Step" tokens are at positions [10, 25, 40] and dp2_idx=50:
        - step_0: [10, 25)  (from first "Step" to second "Step")
        - step_1: [25, 40)  (from second "Step" to third "Step")
        - step_2: [40, 50)  (from third "Step" to dp2)
    """
    step_positions = find_step_token_positions(timestep_artifacts, dp1_idx, dp2_idx, step_token_id)

    if not step_positions:
        return []

    # Create boundaries between consecutive step positions
    boundaries = []
    for i in range(len(step_positions)):
        start_idx = step_positions[i]

        # End is either the next step position, or dp2/end of generation
        if i < len(step_positions) - 1:
            # Not the last step: end at next step position
            end_idx = step_positions[i + 1]
        else:
            # Last step: end at dp2 if available, otherwise end of generation
            if dp2_idx is not None:
                end_idx = dp2_idx
            else:
                # End at the end of generated sequence
                end_idx = dp1_idx + len(timestep_artifacts)

        boundaries.append((start_idx, end_idx))

    return boundaries


def aggregate_window_features(
    timestep_artifacts: List[TimestepArtifacts],
    window_start: int,
    window_end: int,
    dp1_idx: int,
) -> Dict[str, Any]:
    """Aggregate features over a window/step range

    Args:
        timestep_artifacts: List of per-timestep artifacts
        window_start: Start index (absolute in full sequence)
        window_end: End index (exclusive, absolute in full sequence)
        dp1_idx: Start of reasoning (for relative indexing)

    Returns:
        Dictionary of aggregated features including:
        - gold/prod min_rank and max_p: Aggregated across the entire range (MIN of ranks, MAX of probs)
        - per_layer_features: Extracted from the FIRST timestep in the range (at the step marker)

    Note:
        For step-based aggregation, each "step" is the range from one "Step" token to the next.
        The min_rank/max_p are computed across all timesteps in that range, while per-layer
        features are taken from the first timestep (the "Step" token position).
    """
    # Convert absolute indices to relative (within generated sequence)
    rel_start = max(0, window_start - dp1_idx)
    rel_end = min(len(timestep_artifacts), window_end - dp1_idx)

    if rel_start >= rel_end or rel_end > len(timestep_artifacts):
        return {}

    # Get artifacts in window
    window_artifacts = timestep_artifacts[rel_start:rel_end]

    if not window_artifacts:
        return {}

    # === TRAJECTORY CLASSIFICATION FEATURES ===
    # Aggregate gold and produced ranks/probs at final layer
    gold_ranks = [a.rank_gold for a in window_artifacts if a.rank_gold is not None]
    gold_probs = [a.prob_gold for a in window_artifacts if a.prob_gold is not None]
    prod_ranks = [a.rank_prod for a in window_artifacts if a.rank_prod is not None]
    prod_probs = [a.prob_prod for a in window_artifacts if a.prob_prod is not None]

    aggregated = {
        "gold": {
            "min_rank": min(gold_ranks) if gold_ranks else None,
            "max_p": max(gold_probs) if gold_probs else None,
        },
        "prod": {
            "min_rank": min(prod_ranks) if prod_ranks else None,
            "max_p": max(prod_probs) if prod_probs else None,
        },
    }

    # === RF TRAINING FEATURES (PER-LAYER) ===
    # Extract per-layer features from the FIRST timestep in the range
    # (at the "Step" token position for step-based aggregation)

    # Check if we have per-layer features
    if window_artifacts[0].entropy_per_layer is not None:
        # Get the first artifact in the range (at the "Step" token position)
        artifact = window_artifacts[0]

        # Build per-layer features dictionary with values from first timestep
        per_layer_features = {}

        # Entropy per layer (raw values)
        if artifact.entropy_per_layer is not None:
            per_layer_features["entropy"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.entropy_per_layer)
            }

        # Cross-entropy for next token (raw values)
        if artifact.cross_entropy_next is not None:
            per_layer_features["ce_next"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.cross_entropy_next)
            }

        # Cross-entropy for gold token (raw values, if available)
        if artifact.cross_entropy_gold is not None:
            per_layer_features["ce_gold"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.cross_entropy_gold)
            }

        # Ranks for next token (raw values, if available)
        if artifact.ranks_next is not None:
            per_layer_features["ranks_next"] = {
                f"layer_{i}": int(val)
                for i, val in enumerate(artifact.ranks_next)
            }

        # Ranks for gold token (raw values, if available)
        if artifact.ranks_gold is not None:
            per_layer_features["ranks_gold"] = {
                f"layer_{i}": int(val)
                for i, val in enumerate(artifact.ranks_gold)
            }

        # Ranks for "####" token (raw values, if available)
        if artifact.ranks_final_answer is not None:
            per_layer_features["ranks_final_answer"] = {
                f"layer_{i}": int(val)
                for i, val in enumerate(artifact.ranks_final_answer)
            }

        # Probabilities for "####" token (raw values, if available)
        if artifact.probs_final_answer is not None:
            per_layer_features["probs_final_answer"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.probs_final_answer)
            }

        # Cross-entropy for "####" token (raw values, if available)
        if artifact.cross_entropy_final_answer is not None:
            per_layer_features["ce_final_answer"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.cross_entropy_final_answer)
            }

        # Top-p nucleus presence for gold token (raw values, if available)
        if artifact.top_p_presence_gold is not None:
            for p_value, indicators in artifact.top_p_presence_gold.items():
                # e.g., "top_p_gold_0.5", "top_p_gold_0.9", etc.
                feature_name = f"top_p_gold_{p_value}"
                per_layer_features[feature_name] = {
                    f"layer_{i}": int(val)
                    for i, val in enumerate(indicators)
                }

        # Top-p nucleus presence for "####" token (raw values, if available)
        if artifact.top_p_presence_final_answer is not None:
            for p_value, indicators in artifact.top_p_presence_final_answer.items():
                # e.g., "top_p_final_answer_0.5", "top_p_final_answer_0.9", etc.
                feature_name = f"top_p_final_answer_{p_value}"
                per_layer_features[feature_name] = {
                    f"layer_{i}": int(val)
                    for i, val in enumerate(indicators)
                }

        # Ranks for EOS token (raw values, if available)
        if artifact.ranks_eos is not None:
            per_layer_features["ranks_eos"] = {
                f"layer_{i}": int(val)
                for i, val in enumerate(artifact.ranks_eos)
            }

        # Probabilities for EOS token (raw values, if available)
        if artifact.probs_eos is not None:
            per_layer_features["probs_eos"] = {
                f"layer_{i}": float(val)
                for i, val in enumerate(artifact.probs_eos)
            }

        aggregated["per_layer_features"] = per_layer_features

    return aggregated


def compute_all_windows(
    output: CompleteGenerationOutput, config: WindowConfig
) -> Dict[str, Dict[str, Any]]:
    """Compute step-based artifact features for complete output

    NEW BEHAVIOR: Aggregates artifacts over step ranges (from one "Step" token to the next).

    Args:
        output: CompleteGenerationOutput
        config: WindowConfig (includes step_token_id)

    Returns:
        Dictionary mapping step identifiers to aggregated features over that step range
        Keys format: "step_0", "step_1", "step_2", ...
        Each step is the range from one "Step" token to the next "Step" token.
        Features are aggregated (min_rank, max_p) across the entire step range.
    """
    if not output.timestep_artifacts:
        return {}

    # Get step boundaries (ranges between consecutive "Step" tokens)
    boundaries = compute_step_boundaries(
        output.timestep_artifacts,
        output.dp1_idx,
        output.dp2_idx,
        config.step_token_id
    )

    # Aggregate features across each step range
    steps = {}
    for step_num, (start_idx, end_idx) in enumerate(boundaries):
        features = aggregate_window_features(
            output.timestep_artifacts, start_idx, end_idx, output.dp1_idx
        )
        steps[f"step_{step_num}"] = features

    return steps


def classify_trajectory(
    windows: Dict[str, Dict[str, Any]],
    is_correct: bool,
    rank_high_threshold: int = 10,
) -> Optional[str]:
    """Classify trajectory based on confidence patterns and correctness

    NEW BEHAVIOR: Uses first and last STEP tokens instead of percentile windows.

    Classification based on rank confidence at first (step_0) and last (step_N) steps:
    - High confidence: rank ∈ [1, rank_high_threshold]
    - Low confidence: rank > rank_high_threshold

    For CORRECT answers (2 categories based on produced answer pattern):
    - correct_low_to_high: Prod Low→High (model gains confidence in correct answer)
    - correct_high_to_high: Prod High→High (model stays confident in correct answer)

    For INCORRECT answers (8 bins based on prod and gold patterns):
    - I1: Prod Low→High, Gold Low→High
    - I2: Prod Low→High, Gold High→Low
    - I3: Prod Low→High, Gold Low→Low
    - I4: Prod Low→High, Gold High→High
    - I5: Prod High→High, Gold Low→High
    - I6: Prod High→High, Gold High→Low
    - I7: Prod High→High, Gold Low→Low
    - I8: Prod High→High, Gold High→High

    Args:
        windows: Step features (from compute_all_windows)
        is_correct: Whether the produced answer matches the gold answer
        rank_high_threshold: Threshold for high confidence (default: 10)

    Returns:
        Trajectory classification or None if cannot classify
    """
    if not windows:
        return None

    # Get all step keys and sort them
    step_keys = sorted([k for k in windows.keys() if k.startswith("step_")],
                      key=lambda x: int(x.split("_")[1]))

    # Need at least one step to classify
    if len(step_keys) < 1:
        return None

    # Use first and last steps
    first_step = windows[step_keys[0]]
    last_step = windows[step_keys[-1]]

    # Extract min_rank for gold and prod at first and last steps
    gold_rank_first = first_step.get("gold", {}).get("min_rank")
    gold_rank_last = last_step.get("gold", {}).get("min_rank")
    prod_rank_first = first_step.get("prod", {}).get("min_rank")
    prod_rank_last = last_step.get("prod", {}).get("min_rank")

    # Check if all ranks are available
    if any(r is None for r in [gold_rank_first, gold_rank_last, prod_rank_first, prod_rank_last]):
        return None

    # Determine confidence levels
    # High if rank <= threshold, Low if rank > threshold
    prod_conf_first = "High" if prod_rank_first <= rank_high_threshold else "Low"
    prod_conf_last = "High" if prod_rank_last <= rank_high_threshold else "Low"
    gold_conf_first = "High" if gold_rank_first <= rank_high_threshold else "Low"
    gold_conf_last = "High" if gold_rank_last <= rank_high_threshold else "Low"

    # Determine trajectory patterns
    prod_pattern = f"{prod_conf_first}→{prod_conf_last}"
    gold_pattern = f"{gold_conf_first}→{gold_conf_last}"

    # Classification based on correctness
    if is_correct:
        # For correct answers: only use produced answer pattern
        if prod_pattern == "Low→High":
            return "correct_low_to_high"
        elif prod_pattern == "High→High":
            return "correct_high_to_high"
        else:
            # Handle other patterns (High→Low, Low→Low)
            # Map them to the two categories based on final confidence
            if prod_conf_last == "High":
                return "correct_high_to_high"  # Ends high
            else:
                return "correct_low_to_high"  # Ends low (shouldn't happen for correct, but handle it)
    else:
        # For incorrect answers: use I1-I8 classification
        # Format: (prod_pattern, gold_pattern) → bin
        trajectory_map = {
            ("Low→High", "Low→High"): "I1",
            ("Low→High", "High→Low"): "I2",
            ("Low→High", "Low→Low"): "I3",
            ("Low→High", "High→High"): "I4",
            ("High→High", "Low→High"): "I5",
            ("High→High", "High→Low"): "I6",
            ("High→High", "Low→Low"): "I7",
            ("High→High", "High→High"): "I8",
        }

        return trajectory_map.get((prod_pattern, gold_pattern))
