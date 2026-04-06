"""Logit-lens utilities: project hidden states through unembedding (numerically stable)"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


@torch.no_grad()
def compute_logit_lens(
    hidden_states: List[torch.Tensor],
    unembed_matrix: torch.Tensor,
    eps: float = 1e-10,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute logit-lens: project each layer's hidden state through W_U.

    Args:
        hidden_states: list of [batch, hidden_dim] tensors, one per layer
        unembed_matrix: [vocab_size, hidden_dim] unembedding/LM-head weight (W_U)
        eps: kept for API compatibility (not needed here)

    Returns:
        logits_per_layer: list of [batch, vocab_size] logits (fp32)
        probs_per_layer:  list of [batch, vocab_size] probabilities (fp32)
    """
    logits_per_layer: List[torch.Tensor] = []
    probs_per_layer: List[torch.Tensor] = []

    # Do the projection in fp32 for stability
    WU32 = unembed_matrix.float()
    for i, h in enumerate(hidden_states):
        h32 = h.float()
        logits = h32 @ WU32.t()                      # [B, V], fp32

        # Optional sanity check: catches upstream NaNs early
        if not torch.isfinite(logits).all():
            raise ValueError(f"NaN/Inf in logits at layer {i}. "
                             "Check hidden_states/weights dtypes and magnitudes.")

        # Softmax in fp32
        probs = F.softmax(logits, dim=-1)

        logits_per_layer.append(logits)
        probs_per_layer.append(probs)

    return logits_per_layer, probs_per_layer


@torch.no_grad()
def compute_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy H(p) in nats from probabilities (numerically safe).

    Args:
        probs: [vocab_size] (or [B, V]) probabilities
        eps: clamp floor to avoid log(0)

    Returns:
        float entropy if input is 1D, else raises to keep API semantics clear.
    """
    if probs.dim() != 1:
        raise ValueError("compute_entropy expects a 1D probs tensor [vocab_size]. "
                         "Use compute_entropy_from_logits for batched inputs.")
    p = probs.float().clamp_min(eps)
    log_p = p.log()
    H = -(p * log_p).sum()
    # Harden against any lingering numerical issues
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return float(H.item())


@torch.no_grad()
def compute_entropy_from_logits(logits: torch.Tensor) -> float:
    """Stable entropy computed from logits via log_softmax (preferred).

    Args:
        logits: [vocab_size] (or [B, V]) logits

    Returns:
        float entropy if input is 1D; for batched inputs, raises (keep API tight).
    """
    if logits.dim() != 1:
        raise ValueError("compute_entropy_from_logits expects 1D logits [vocab_size].")
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H = -(p * log_p).sum()
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return float(H.item())


@torch.no_grad()
def compute_cross_entropy(probs: torch.Tensor, target_id: int, eps: float = 1e-12) -> float:
    """Cross-entropy for a specific target token from probabilities.

    Args:
        probs: [vocab_size] probabilities (GPU tensor)
        target_id: int
        eps: clamp floor for stability

    Returns:
        float cross-entropy = -log p[target]
    """
    if probs.dim() != 1:
        raise ValueError("compute_cross_entropy expects 1D probs [vocab_size].")

    # Keep all operations on GPU until final conversion
    p_t = probs[target_id].float().clamp_min(eps)  # Stay on GPU
    ce = -torch.log(p_t)  # Compute log on GPU

    # Single CPU conversion at the end
    return float(ce.item())


@torch.no_grad()
def compute_rank(logits: torch.Tensor, target_id: int) -> int:
    """1-based rank of target token in logits.

    Args:
        logits: [vocab_size] logits
        target_id: int

    Returns:
        int rank (1 = highest logit)
    """
    if logits.dim() != 1:
        raise ValueError("compute_rank expects 1D logits [vocab_size].")
    target_logit = logits[target_id]
    higher = (logits > target_logit).sum().item()
    return int(higher + 1)


@torch.no_grad()
def get_prob(probs: torch.Tensor, target_id: int) -> float:
    """Get probability of target token.

    Args:
        probs: [vocab_size] probabilities
        target_id: int

    Returns:
        float probability
    """
    if probs.dim() != 1:
        raise ValueError("get_prob expects 1D probs [vocab_size].")
    return float(probs[target_id].item())


@torch.no_grad()
def compute_top_p_presence(probs: torch.Tensor, target_id: int, p: float) -> int:
    """Compute whether target token is in the top-p nucleus.

    This implements the top-p (nucleus) sampling criterion: determine whether
    the target token lies within the smallest set of tokens whose cumulative
    probability mass is >= p.

    Algorithm:
    1. Sort probabilities in descending order
    2. Compute cumulative sum until sum >= p
    3. Check if target_id is in the resulting nucleus set

    Args:
        probs: [vocab_size] probability distribution (must be normalized)
        target_id: int, the token ID to check
        p: float in (0, 1], the nucleus threshold (e.g., 0.9, 0.95, 0.99)

    Returns:
        int: 1 if target_id is in top-p nucleus, 0 otherwise
    """
    if probs.dim() != 1:
        raise ValueError("compute_top_p_presence expects 1D probs [vocab_size].")

    if not 0 < p <= 1:
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=0)

    # Find nucleus: smallest k such that cumsum[k] >= p
    # Use searchsorted for efficiency (finds first position where cumsum >= p)
    nucleus_size = torch.searchsorted(cumsum, p, right=False).item() + 1

    # Get the indices in the nucleus
    nucleus_indices = sorted_indices[:nucleus_size]

    # Check if target_id is in the nucleus
    is_in_nucleus = (nucleus_indices == target_id).any()

    return int(is_in_nucleus.item())
