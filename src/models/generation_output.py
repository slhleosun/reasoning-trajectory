"""Complete generation output with per-timestep artifacts"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class TimestepArtifacts:
    """Per-timestep captured artifacts (GPU-first design)

    Arrays can be stored as either:
    - torch.Tensor (on GPU) - during generation and post-processing
    - np.ndarray (on CPU) - only when saving to disk

    This avoids GPU→CPU→GPU roundtrips for massive speedup.
    """

    # Token information
    next_token_id: int
    next_token_str: str

    # Entropy per layer [n_layers] - SCALARS ONLY
    entropy_per_layer: Optional[List[float]] = None

    # Cross-entropy for different targets [n_layers] - SCALARS ONLY
    cross_entropy_next: Optional[List[float]] = None  # CE for next token
    cross_entropy_gold: Optional[List[float]] = None  # CE for gold answer
    cross_entropy_prod: Optional[List[float]] = None  # CE for produced answer (retroactive)
    cross_entropy_final_answer: Optional[List[float]] = None  # CE for "####" token (827)

    # Per-layer ranks [n_layers] - SCALARS ONLY
    ranks_next: Optional[List[int]] = None  # Rank of next token per layer
    ranks_gold: Optional[List[int]] = None  # Rank of gold answer token per layer
    ranks_final_answer: Optional[List[int]] = None  # Rank of "####" token (827) per layer

    # Per-layer probabilities [n_layers] - SCALARS ONLY
    probs_final_answer: Optional[List[float]] = None  # Probability of "####" token (827) per layer

    # Ranks and probabilities at final layer only - SCALARS ONLY (for backwards compat)
    rank_gold: Optional[int] = None
    rank_prod: Optional[int] = None
    prob_gold: Optional[float] = None
    prob_prod: Optional[float] = None
    rank_final_answer: Optional[int] = None  # Rank of "####" token at final layer
    prob_final_answer: Optional[float] = None  # Probability of "####" token at final layer

    # Top-p nucleus presence indicators per layer - SCALARS ONLY
    # Dict mapping p-value to list of binary indicators (0/1) per layer
    # e.g., {"0.5": [1, 1, 0, ...], "0.9": [1, 1, 1, ...], ...}
    top_p_presence_gold: Optional[Dict[str, List[int]]] = None  # For gold token
    top_p_presence_final_answer: Optional[Dict[str, List[int]]] = None  # For "####" token (827)

    # EOS (end-of-sequence) token features per layer - SCALARS ONLY
    ranks_eos: Optional[List[int]] = None  # Rank of EOS token per layer [n_layers]
    probs_eos: Optional[List[float]] = None  # Probability of EOS token per layer [n_layers]
    # Note: rank_eos at final layer is stored separately for backwards compat
    rank_eos: Optional[int] = None  # Rank of EOS token at final layer only
    prob_eos: Optional[float] = None  # Probability of EOS token at final layer only

    # Optional: Full arrays (can be torch.Tensor or np.ndarray)
    # WARNING: MEMORY INTENSIVE - ~8GB for 512 tokens
    hidden_states: Optional[List[Union["torch.Tensor", np.ndarray]]] = None  # [n_layers, hidden_dim]
    logits_per_layer: Optional[List[Union["torch.Tensor", np.ndarray]]] = None  # [n_layers, vocab_size] - HUGE!

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (including arrays if present)

        Arrays (hidden_states, logits_per_layer) are included if they are not None.
        This allows the caller to control whether arrays are saved by setting them to None.
        """
        result = {
            "next_token_id": self.next_token_id,
            "next_token_str": self.next_token_str,
            "entropy_per_layer": self.entropy_per_layer,
            "cross_entropy_next": self.cross_entropy_next,
            "cross_entropy_gold": self.cross_entropy_gold,
            "cross_entropy_prod": self.cross_entropy_prod,
            "cross_entropy_final_answer": self.cross_entropy_final_answer,
            "ranks_next": self.ranks_next,  # Per-layer ranks for next token
            "ranks_gold": self.ranks_gold,  # Per-layer ranks for gold token
            "ranks_final_answer": self.ranks_final_answer,  # Per-layer ranks for "####" token
            "probs_final_answer": self.probs_final_answer,  # Per-layer probabilities for "####" token
            "rank_gold": self.rank_gold,
            "rank_prod": self.rank_prod,
            "prob_gold": self.prob_gold,
            "prob_prod": self.prob_prod,
            "rank_final_answer": self.rank_final_answer,  # Final layer rank for "####"
            "prob_final_answer": self.prob_final_answer,  # Final layer probability for "####"
            "top_p_presence_gold": self.top_p_presence_gold,
            "top_p_presence_final_answer": self.top_p_presence_final_answer,
            "ranks_eos": self.ranks_eos,  # Per-layer ranks for EOS token
            "probs_eos": self.probs_eos,  # Per-layer probabilities for EOS token
            "rank_eos": self.rank_eos,  # Final layer rank (backwards compat)
            "prob_eos": self.prob_eos,  # Final layer probability (backwards compat)
        }

        # Include arrays if present (not None)
        # These are only included when include_arrays=True in save_generation_output()
        if self.hidden_states is not None:
            # Convert torch tensors or keep numpy arrays
            if isinstance(self.hidden_states, list):
                # List of tensors/arrays - convert each to list
                result["hidden_states"] = [
                    h.tolist() if hasattr(h, 'tolist') else h
                    for h in self.hidden_states
                ]
            else:
                result["hidden_states"] = self.hidden_states

        if self.logits_per_layer is not None:
            # Convert torch tensors or keep numpy arrays
            if isinstance(self.logits_per_layer, list):
                # List of tensors/arrays - convert each to list
                result["logits_per_layer"] = [
                    lg.tolist() if hasattr(lg, 'tolist') else lg
                    for lg in self.logits_per_layer
                ]
            else:
                result["logits_per_layer"] = self.logits_per_layer

        return result


@dataclass
class CompleteGenerationOutput:
    """Complete generation output with all artifacts"""

    # Core sequence data
    input_ids: List[int]
    full_seq_ids: List[int]

    # Decision points
    dp1_idx: int  # Start of reasoning (first generated token)
    dp2_idx: Optional[int] = None  # Start of final answer (after ####)
    reasoning_length: Optional[int] = None  # dp2 - dp1 + 1

    # Text outputs
    produced_text: str = ""
    produced_answer: Optional[str] = None
    gold_answer: Optional[str] = None

    # Per-timestep artifacts
    timestep_artifacts: List[TimestepArtifacts] = field(default_factory=list)

    # Step-based aggregations (filled by post-processing)
    # Keys: "step_0", "step_1", ..., "step_N" for each "Step" token found
    windows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    num_steps: Optional[int] = None  # Total number of "Step" tokens found in generation

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization

        Args:
            minimal: If True, only include core fields needed for steering collection
                    (produced_text, full_seq_ids, dp1_idx, gold_answer)
        """
        if minimal:
            # MINIMAL MODE: Only save what's needed for steering collection and intervention
            return {
                "input_ids": self.input_ids,  # Needed by intervention scripts
                "produced_text": self.produced_text,  # Needed by steering collection
                "full_seq_ids": self.full_seq_ids,  # Needed by steering collection
                "dp1_idx": self.dp1_idx,  # Needed by steering collection
                "gold_answer": self.gold_answer,  # Needed by intervention scripts
                "produced_answer": self.produced_answer,  # Useful for evaluation
                "dp2_idx": self.dp2_idx,  # Useful for analysis
                "reasoning_length": self.reasoning_length,  # Useful for analysis
            }

        # FULL MODE: Include everything
        return {
            "input_ids": self.input_ids,
            "full_seq_ids": self.full_seq_ids,
            "dp1_idx": self.dp1_idx,
            "dp2_idx": self.dp2_idx,
            "reasoning_length": self.reasoning_length,
            "produced_text": self.produced_text,
            "produced_answer": self.produced_answer,
            "gold_answer": self.gold_answer,
            "timesteps": [t.to_dict() for t in self.timestep_artifacts],
            "steps": self.windows,  # Renamed from "windows" to "steps" for clarity
            "num_steps": self.num_steps,
            "metadata": self.metadata,
        }
