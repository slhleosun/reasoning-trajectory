"""Two-pass single-sample greedy generation with artifact capture

This module implements a two-pass approach for efficient artifact capture:
- Pass 1: Fast generation using model.generate() (no hidden states)
- Pass 2: Single forward pass to capture all artifacts at once

This is 5-10x faster than autoregressive artifact capture.
"""

import torch
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .generation_output import CompleteGenerationOutput, TimestepArtifacts
from ..features.logit_lens import (
    compute_logit_lens,
    compute_entropy,
    compute_cross_entropy,
    compute_rank,
    get_prob,
    compute_top_p_presence,
)


def greedy_generate_with_artifacts_twopass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 512,
    gold_answer_token_id: Optional[int] = None,
    capture_hidden_states: bool = True,
    pad_token_id: Optional[int] = None,
) -> CompleteGenerationOutput:
    """Two-pass single-sample greedy generation with complete artifact capture

    Pass 1: Fast generation to produce output sequence
    Pass 2: Single forward pass to capture all artifacts

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        input_ids: [1, seq_len] input token IDs
        attention_mask: [1, seq_len] attention mask
        max_new_tokens: Maximum tokens to generate
        gold_answer_token_id: Optional gold answer first token ID for tracking
        capture_hidden_states: Whether to capture hidden states
        pad_token_id: Pad token ID for masking

    Returns:
        CompleteGenerationOutput with all artifacts
    """
    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Store initial input
    initial_input_ids = input_ids[0].cpu().tolist()
    prompt_length = attention_mask[0].sum().item()
    dp1_idx = prompt_length

    # ==========================================================================
    # PASS 1: Fast Generation (no hidden states)
    # ==========================================================================

    with torch.no_grad():
        max_length = input_ids.shape[1] + max_new_tokens

        # CRITICAL: Use these exact parameters for determinism
        generated_sequence = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=False,  # Greedy = fully deterministic
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Use KV cache for speed
            # DO NOT set temperature/top_p - they are ignored when do_sample=False
            # but may cause issues in some transformers versions
        )

    # Extract generated tokens (excluding prompt)
    full_seq = generated_sequence[0]
    generated_tokens = full_seq[len(input_ids[0]):].cpu().tolist()

    # Decode produced text
    produced_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # ==========================================================================
    # PASS 2: Capture Artifacts (single forward pass)
    # ==========================================================================

    if not capture_hidden_states:
        # Skip Pass 2 if not capturing artifacts
        timestep_artifacts = [
            TimestepArtifacts(
                next_token_id=tok_id,
                next_token_str=tokenizer.decode([tok_id]),
            )
            for tok_id in generated_tokens
        ]

        output = CompleteGenerationOutput(
            input_ids=initial_input_ids,
            full_seq_ids=initial_input_ids + generated_tokens,
            dp1_idx=dp1_idx,
            dp2_idx=None,
            reasoning_length=None,
            produced_text=produced_text,
            produced_answer=None,
            gold_answer=None,
            timestep_artifacts=timestep_artifacts,
            metadata={
                "prompt_length": prompt_length,
                "generated_length": len(generated_tokens),
            },
        )

        return output

    # Get unembedding matrix for logit lens
    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight
    elif hasattr(model, "embed_out"):
        unembed = model.embed_out.weight
    else:
        unembed = model.get_input_embeddings().weight

    unembed_cpu = unembed.cpu()

    # Single forward pass with full sequence
    full_seq_ids = initial_input_ids + generated_tokens
    full_seq_tensor = torch.tensor([full_seq_ids], dtype=torch.long, device=device)
    full_attention = torch.ones_like(full_seq_tensor)

    with torch.no_grad():
        forward_outputs = model(
            input_ids=full_seq_tensor,
            attention_mask=full_attention,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract artifacts for each generated token
    timestep_artifacts = []
    num_generated = len(generated_tokens)

    for step in range(num_generated):
        # Position in full sequence
        pos = prompt_length + step

        # Token that was generated at this step
        next_token_id = generated_tokens[step]
        next_token_str = tokenizer.decode([next_token_id])

        # Get hidden states at position pos-1 (which predicted this token)
        hidden_pos = pos - 1

        # Extract hidden states from all layers at this position
        sample_hidden_states = [
            h[0, hidden_pos].cpu() for h in forward_outputs.hidden_states
        ]

        # Compute logit-lens for all layers
        logits_per_layer, probs_per_layer = compute_logit_lens(
            [h.unsqueeze(0) for h in sample_hidden_states],
            unembed_cpu,
        )

        # Compute per-layer metrics
        entropy_per_layer = [
            compute_entropy(probs[0]) for probs in probs_per_layer
        ]

        # Cross-entropy for next token
        ce_next = [
            compute_cross_entropy(probs[0], next_token_id)
            for probs in probs_per_layer
        ]

        # Cross-entropy for gold token (if provided)
        ce_gold = None
        if gold_answer_token_id is not None:
            ce_gold = [
                compute_cross_entropy(probs[0], gold_answer_token_id)
                for probs in probs_per_layer
            ]

        # Top-p nucleus presence for next token (multiple p values)
        p_values = [0.5, 0.9, 0.95, 0.99]
        top_p_presence_next = {
            str(p): [
                compute_top_p_presence(probs[0], next_token_id, p)
                for probs in probs_per_layer
            ]
            for p in p_values
        }

        # Top-p nucleus presence for gold token (if provided)
        top_p_presence_gold = None
        if gold_answer_token_id is not None:
            top_p_presence_gold = {
                str(p): [
                    compute_top_p_presence(probs[0], gold_answer_token_id, p)
                    for probs in probs_per_layer
                ]
                for p in p_values
            }

        # PER-LAYER RANKS and PROBS for EOS token
        eos_token_id = tokenizer.eos_token_id
        ranks_eos = [
            compute_rank(logits[0], eos_token_id)
            for logits in logits_per_layer
        ]
        probs_eos = [
            get_prob(probs[0], eos_token_id)
            for probs in probs_per_layer
        ]

        # Ranks and probs at final layer only
        final_probs = probs_per_layer[-1][0]
        final_logits = logits_per_layer[-1][0]

        rank_gold = None
        prob_gold = None
        if gold_answer_token_id is not None:
            rank_gold = compute_rank(final_logits, gold_answer_token_id)
            prob_gold = get_prob(final_probs, gold_answer_token_id)

        # EOS at final layer (for backwards compatibility)
        rank_eos = compute_rank(final_logits, eos_token_id)
        prob_eos = get_prob(final_probs, eos_token_id)

        # Store timestep artifacts
        artifact = TimestepArtifacts(
            next_token_id=next_token_id,
            next_token_str=next_token_str,
            hidden_states=[h.detach().numpy() for h in sample_hidden_states],
            logits_per_layer=[lg[0].detach().cpu().numpy() for lg in logits_per_layer],
            entropy_per_layer=entropy_per_layer,
            cross_entropy_next=ce_next,
            cross_entropy_gold=ce_gold,
            cross_entropy_prod=None,
            rank_gold=rank_gold,
            rank_prod=None,
            prob_gold=prob_gold,
            prob_prod=None,
            top_p_presence_next=top_p_presence_next,
            top_p_presence_gold=top_p_presence_gold,
            ranks_eos=ranks_eos,  # Per-layer ranks for EOS token
            probs_eos=probs_eos,  # Per-layer probabilities for EOS token
            rank_eos=rank_eos,    # Final layer rank for EOS (backwards compat)
            prob_eos=prob_eos,    # Final layer prob for EOS (backwards compat)
        )
        timestep_artifacts.append(artifact)

    # Create output
    output = CompleteGenerationOutput(
        input_ids=initial_input_ids,
        full_seq_ids=full_seq_ids,
        dp1_idx=dp1_idx,
        dp2_idx=None,
        reasoning_length=None,
        produced_text=produced_text,
        produced_answer=None,
        gold_answer=None,
        timestep_artifacts=timestep_artifacts,
        metadata={
            "prompt_length": prompt_length,
            "generated_length": num_generated,
        },
    )

    # Cleanup
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return output
