"""Two-pass batched greedy generation with artifact capture

This module implements a two-pass approach for efficient artifact capture:
- Pass 1: Fast generation using model.generate() (no hidden states)
- Pass 2: Single forward pass to capture all artifacts at once

This is 5-10x faster than autoregressive artifact capture because:
1. Pass 1 uses optimized generate() with KV caching
2. Pass 2 processes entire sequence in parallel (no loop)
3. No GPU->CPU transfers in generation loop
"""

import torch
from typing import List, Optional
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


def batch_greedy_generate_with_artifacts_twopass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 512,
    gold_answer_token_ids: Optional[List[Optional[int]]] = None,
    capture_hidden_states: bool = True,
    pad_token_id: Optional[int] = None,
) -> List[CompleteGenerationOutput]:
    """Two-pass batched greedy generation with complete artifact capture

    Pass 1: Fast generation to produce output sequence
    Pass 2: Single forward pass to capture all artifacts

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        input_ids: [batch_size, seq_len] input token IDs (with left padding)
        attention_mask: [batch_size, seq_len] attention mask
        max_new_tokens: Maximum tokens to generate
        gold_answer_token_ids: Optional list of gold answer first token IDs
        capture_hidden_states: Whether to capture hidden states
        pad_token_id: Pad token ID for masking

    Returns:
        List of CompleteGenerationOutput (one per sample in batch)
    """
    device = model.device
    batch_size = input_ids.shape[0]

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Track original prompt lengths (before padding removed)
    prompt_lengths = attention_mask.sum(dim=1).cpu().tolist()

    # ==========================================================================
    # PASS 1: Fast Generation (no hidden states)
    # ==========================================================================

    if torch.cuda.is_available() and model.device.type == "cpu":
        raise RuntimeError(
            f"MODEL IS ON CPU BUT CUDA IS AVAILABLE!\n"
            f"  Model device: {model.device}\n"
            f"  CUDA available: {torch.cuda.is_available()}\n"
            f"  CUDA devices: {torch.cuda.device_count()}\n"
            f"This is a model loading issue in huggingface.py adapter.load()"
        )

    # Ensure inputs are on same device as model
    if input_ids.device != model.device:
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        generated_sequences = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
            temperature=None,
            top_p=None,
            early_stopping=True,
        )

    # ==========================================================================
    # Extract information from Pass 1
    # ==========================================================================

    initial_input_ids_list = []
    generated_token_lists = []
    produced_texts = []

    for i in range(batch_size):
        full_seq = generated_sequences[i]

        # Find where real tokens start (after left padding)
        real_start = (full_seq != pad_token_id).nonzero()[0].item()
        real_sequence = full_seq[real_start:]

        # Split into prompt and generated
        prompt_len = prompt_lengths[i]
        input_ids_clean = real_sequence[:prompt_len].cpu().tolist()
        generated_tokens = real_sequence[prompt_len:].cpu().tolist()

        # Decode produced text
        produced_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        initial_input_ids_list.append(input_ids_clean)
        generated_token_lists.append(generated_tokens)
        produced_texts.append(produced_text)

    # ==========================================================================
    # PASS 2: Capture Artifacts (single forward pass per sample)
    # ==========================================================================

    if not capture_hidden_states:
        outputs = []
        for i in range(batch_size):
            timestep_artifacts = [
                TimestepArtifacts(
                    next_token_id=tok_id,
                    next_token_str=tokenizer.decode([tok_id]),
                )
                for tok_id in generated_token_lists[i]
            ]

            output = CompleteGenerationOutput(
                input_ids=initial_input_ids_list[i],
                full_seq_ids=initial_input_ids_list[i] + generated_token_lists[i],
                dp1_idx=len(initial_input_ids_list[i]),
                dp2_idx=None,
                reasoning_length=None,
                produced_text=produced_texts[i],
                produced_answer=None,
                gold_answer=None,
                timestep_artifacts=timestep_artifacts,
                metadata={
                    "prompt_length": len(initial_input_ids_list[i]),
                    "generated_length": len(generated_token_lists[i]),
                },
            )
            outputs.append(output)

        return outputs

    # Get unembedding matrix for logit lens (keep on GPU for speed)
    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight
    elif hasattr(model, "embed_out"):
        unembed = model.embed_out.weight
    else:
        unembed = model.get_input_embeddings().weight

    # Process each sample individually in Pass 2
    outputs = []

    for i in range(batch_size):
        # Get full sequence (prompt + generated)
        full_seq_ids = initial_input_ids_list[i] + generated_token_lists[i]
        full_seq_tensor = torch.tensor([full_seq_ids], dtype=torch.long, device=device)
        full_attention = torch.ones_like(full_seq_tensor)

        prompt_len = len(initial_input_ids_list[i])
        num_generated = len(generated_token_lists[i])

        # Single forward pass with hidden states
        with torch.no_grad():
            forward_outputs = model(
                input_ids=full_seq_tensor,
                attention_mask=full_attention,
                output_hidden_states=True,
                use_cache=False,
            )

        gold_token_id = gold_answer_token_ids[i] if gold_answer_token_ids else None

        # Extract per-timestep artifacts (keep tensors on GPU)
        timestep_data = []

        for step in range(num_generated):
            pos = prompt_len + step
            next_token_id = generated_token_lists[i][step]
            next_token_str = tokenizer.decode([next_token_id])

            # Hidden state at pos-1 predicted the token at pos
            hidden_pos = pos - 1

            sample_hidden_states = [
                h[0, hidden_pos] for h in forward_outputs.hidden_states
            ]

            logits_per_layer, probs_per_layer = compute_logit_lens(
                [h.unsqueeze(0) for h in sample_hidden_states],
                unembed,
            )

            entropy_per_layer = [
                compute_entropy(probs[0]) for probs in probs_per_layer
            ]

            ce_next = [
                compute_cross_entropy(probs[0], next_token_id)
                for probs in probs_per_layer
            ]

            ce_gold = None
            if gold_token_id is not None:
                ce_gold = [
                    compute_cross_entropy(probs[0], gold_token_id)
                    for probs in probs_per_layer
                ]

            ranks_next = [
                compute_rank(logits[0], next_token_id)
                for logits in logits_per_layer
            ]

            ranks_gold = None
            if gold_token_id is not None:
                ranks_gold = [
                    compute_rank(logits[0], gold_token_id)
                    for logits in logits_per_layer
                ]

            p_values = [0.5, 0.9, 0.95, 0.99]
            top_p_presence_gold = None
            if gold_token_id is not None:
                top_p_presence_gold = {
                    str(p): [
                        compute_top_p_presence(probs[0], gold_token_id, p)
                        for probs in probs_per_layer
                    ]
                    for p in p_values
                }

            # "####" token features (token ID 827)
            final_answer_token_id = 827

            ce_final_answer = [
                compute_cross_entropy(probs[0], final_answer_token_id)
                for probs in probs_per_layer
            ]
            ranks_final_answer = [
                compute_rank(logits[0], final_answer_token_id)
                for logits in logits_per_layer
            ]
            probs_final_answer = [
                get_prob(probs[0], final_answer_token_id)
                for probs in probs_per_layer
            ]
            top_p_presence_final_answer = {
                str(p): [
                    compute_top_p_presence(probs[0], final_answer_token_id, p)
                    for probs in probs_per_layer
                ]
                for p in p_values
            }

            # EOS token features
            eos_token_id = tokenizer.eos_token_id
            ranks_eos = [
                compute_rank(logits[0], eos_token_id)
                for logits in logits_per_layer
            ]
            probs_eos = [
                get_prob(probs[0], eos_token_id)
                for probs in probs_per_layer
            ]

            # Final layer features (backwards compatibility)
            final_probs = probs_per_layer[-1][0]
            final_logits = logits_per_layer[-1][0]

            rank_gold = None
            prob_gold = None
            if gold_token_id is not None:
                rank_gold = compute_rank(final_logits, gold_token_id)
                prob_gold = get_prob(final_probs, gold_token_id)

            rank_final_answer = compute_rank(final_logits, final_answer_token_id)
            prob_final_answer = get_prob(final_probs, final_answer_token_id)
            rank_eos = compute_rank(final_logits, eos_token_id)
            prob_eos = get_prob(final_probs, eos_token_id)

            timestep_data.append({
                'next_token_id': next_token_id,
                'next_token_str': next_token_str,
                'hidden_states': sample_hidden_states,
                'logits_per_layer': logits_per_layer,
                'entropy_per_layer': entropy_per_layer,
                'ce_next': ce_next,
                'ce_gold': ce_gold,
                'ce_final_answer': ce_final_answer,
                'ranks_next': ranks_next,
                'ranks_gold': ranks_gold,
                'ranks_final_answer': ranks_final_answer,
                'probs_final_answer': probs_final_answer,
                'rank_gold': rank_gold,
                'prob_gold': prob_gold,
                'rank_final_answer': rank_final_answer,
                'prob_final_answer': prob_final_answer,
                'top_p_presence_gold': top_p_presence_gold,
                'top_p_presence_final_answer': top_p_presence_final_answer,
                'ranks_eos': ranks_eos,
                'probs_eos': probs_eos,
                'rank_eos': rank_eos,
                'prob_eos': prob_eos,
            })

        # Convert to TimestepArtifacts
        timestep_artifacts = []
        for ts_data in timestep_data:
            artifact = TimestepArtifacts(
                next_token_id=ts_data['next_token_id'],
                next_token_str=ts_data['next_token_str'],
                hidden_states=ts_data['hidden_states'],
                logits_per_layer=[lg[0] for lg in ts_data['logits_per_layer']],
                entropy_per_layer=ts_data['entropy_per_layer'],
                cross_entropy_next=ts_data['ce_next'],
                cross_entropy_gold=ts_data['ce_gold'],
                cross_entropy_prod=None,
                cross_entropy_final_answer=ts_data['ce_final_answer'],
                ranks_next=ts_data['ranks_next'],
                ranks_gold=ts_data['ranks_gold'],
                ranks_final_answer=ts_data['ranks_final_answer'],
                probs_final_answer=ts_data['probs_final_answer'],
                rank_gold=ts_data['rank_gold'],
                rank_prod=None,
                prob_gold=ts_data['prob_gold'],
                prob_prod=None,
                rank_final_answer=ts_data['rank_final_answer'],
                prob_final_answer=ts_data['prob_final_answer'],
                top_p_presence_gold=ts_data['top_p_presence_gold'],
                top_p_presence_final_answer=ts_data['top_p_presence_final_answer'],
                ranks_eos=ts_data['ranks_eos'],
                probs_eos=ts_data['probs_eos'],
                rank_eos=ts_data['rank_eos'],
                prob_eos=ts_data['prob_eos'],
            )
            timestep_artifacts.append(artifact)

        del timestep_data

        output = CompleteGenerationOutput(
            input_ids=initial_input_ids_list[i],
            full_seq_ids=full_seq_ids,
            dp1_idx=prompt_len,
            dp2_idx=None,
            reasoning_length=None,
            produced_text=produced_texts[i],
            produced_answer=None,
            gold_answer=None,
            timestep_artifacts=timestep_artifacts,
            metadata={
                "prompt_length": prompt_len,
                "generated_length": num_generated,
            },
        )

        outputs.append(output)

        # Free GPU memory after processing this sample
        del forward_outputs
        del full_seq_tensor
        del full_attention
        del timestep_artifacts

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Cleanup
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return outputs
