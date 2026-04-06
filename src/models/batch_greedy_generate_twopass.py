"""Two-pass batched greedy generation with artifact capture

This module implements a two-pass approach for efficient artifact capture:
- Pass 1: Fast generation using model.generate() (no hidden states)
- Pass 2: Single forward pass to capture all artifacts at once

This is 5-10x faster than autoregressive artifact capture because:
1. Pass 1 uses optimized generate() with KV caching
2. Pass 2 processes entire sequence in parallel (no loop)
3. No GPU→CPU transfers in generation loop
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

    # Debug: Ensure model is on correct device
    import sys
    import time
    import os

    print(f"[DEBUG PID={os.getpid()}] About to call model.generate() on device: {device}", flush=True)
    print(f"[DEBUG PID={os.getpid()}] Model device: {model.device}", flush=True)
    print(f"[DEBUG PID={os.getpid()}] Input shape: {input_ids.shape}, Device: {input_ids.device}", flush=True)
    print(f"[DEBUG PID={os.getpid()}] Batch size: {batch_size}", flush=True)
    sys.stdout.flush()

    # Check CUDA availability
    print(f"[DEBUG PID={os.getpid()}] CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"[DEBUG PID={os.getpid()}] CUDA device count: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG PID={os.getpid()}] Current CUDA device: {torch.cuda.current_device()}", flush=True)
        print(f"[DEBUG PID={os.getpid()}] CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)
    sys.stdout.flush()

    # CRITICAL: If model is on CPU but CUDA is available, this is a BUG
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
        print(f"[WARNING PID={os.getpid()}] Inputs on {input_ids.device} but model on {model.device}, moving inputs...", flush=True)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        print(f"[DEBUG PID={os.getpid()}] Inputs moved to {model.device}", flush=True)
        sys.stdout.flush()

    try:
        with torch.no_grad():
            # Calculate max_length for generation
            max_input_length = input_ids.shape[1]
            max_length = max_input_length + max_new_tokens

            print(f"[DEBUG PID={os.getpid()}] Calling model.generate() with max_length={max_length}...", flush=True)
            print(f"[DEBUG PID={os.getpid()}] Time: {time.time()}", flush=True)
            sys.stdout.flush()

            gen_start = time.time()

            eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            # CRITICAL: Use these exact parameters for determinism
            # Must explicitly unset temperature/top_p to avoid issues
            generated_sequences = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy = fully deterministic
                pad_token_id=pad_token_id,
                eos_token_id=eos_ids, 
                use_cache=True,  # Use KV cache for speed
                temperature=None,
                top_p=None,
                early_stopping=True,
                # NOTE: temperature/top_p/top_k are ignored when do_sample=False
                # but explicitly setting to None prevents warnings in some transformers versions
            )

            gen_time = time.time() - gen_start
            print(f"[DEBUG PID={os.getpid()}] model.generate() returned after {gen_time:.1f}s! Output shape: {generated_sequences.shape}", flush=True)
            sys.stdout.flush()
    except RuntimeError as e:
        error_msg = str(e).lower()
        print(f"\n{'='*80}", flush=True)
        print(f"❌❌❌ RUNTIME ERROR in Pass 1 (Generation) ❌❌❌", flush=True)
        print(f"PID={os.getpid()}, Device={device}, Batch size={batch_size}", flush=True)
        print(f"{'='*80}", flush=True)
        if "out of memory" in error_msg or "oom" in error_msg:
            print(f"🔥 OUT OF MEMORY ERROR", flush=True)
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)
            print(f"💡 Try: --batch-size {batch_size // 2}", flush=True)
        elif "cuda" in error_msg:
            print(f"🔥 CUDA ERROR", flush=True)
            print(f"💡 Check: nvidia-smi for GPU health", flush=True)
        print(f"Full error: {e}", flush=True)
        print(f"{'='*80}\n", flush=True)
        raise
    except Exception as e:
        print(f"\n{'='*80}", flush=True)
        print(f"❌❌❌ UNEXPECTED ERROR in Pass 1 ❌❌❌", flush=True)
        print(f"PID={os.getpid()}, Type: {type(e).__name__}", flush=True)
        print(f"Error: {e}", flush=True)
        print(f"{'='*80}\n", flush=True)
        raise

    # ==========================================================================
    # Extract information from Pass 1
    # ==========================================================================

    print(f"[DEBUG PID={os.getpid()}] Extracting token information from Pass 1...", flush=True)
    sys.stdout.flush()

    # Store initial inputs and generated sequences for each sample
    initial_input_ids_list = []
    generated_token_lists = []
    produced_texts = []

    for i in range(batch_size):
        if i % 8 == 0:  # Print every 8 samples
            print(f"[DEBUG PID={os.getpid()}] Processing sample {i+1}/{batch_size}...", flush=True)
            sys.stdout.flush()

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

    print(f"[DEBUG PID={os.getpid()}] Pass 1 extraction complete. Processing {batch_size} samples.", flush=True)
    sys.stdout.flush()

    # ==========================================================================
    # PASS 2: Capture Artifacts (single forward pass per sample)
    # ==========================================================================

    print(f"[DEBUG PID={os.getpid()}] Starting Pass 2: Artifact capture for {batch_size} samples...", flush=True)
    sys.stdout.flush()

    if not capture_hidden_states:
        print(f"[DEBUG PID={os.getpid()}] Skipping Pass 2 (capture_hidden_states=False)", flush=True)
        sys.stdout.flush()
        # Skip Pass 2 if not capturing artifacts
        outputs = []
        for i in range(batch_size):
            # Create minimal artifacts
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

    # Get unembedding matrix for logit lens (KEEP ON GPU for speed)
    print(f"[DEBUG PID={os.getpid()}] Getting unembed matrix (keeping on GPU)...", flush=True)
    sys.stdout.flush()

    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight
    elif hasattr(model, "embed_out"):
        unembed = model.embed_out.weight
    else:
        unembed = model.get_input_embeddings().weight

    print(f"[DEBUG PID={os.getpid()}] Unembed matrix ready on {unembed.device}, size: {unembed.shape}", flush=True)
    sys.stdout.flush()

    # Process each sample individually in Pass 2
    outputs = []

    print(f"[DEBUG PID={os.getpid()}] Processing {batch_size} samples in Pass 2...", flush=True)
    sys.stdout.flush()

    for i in range(batch_size):
        # Always print sample start for visibility
        print(f"[DEBUG PID={os.getpid()}] Pass 2: Sample {i+1}/{batch_size}...", flush=True)
        sys.stdout.flush()

        # Get full sequence (prompt + generated)
        full_seq_ids = initial_input_ids_list[i] + generated_token_lists[i]
        full_seq_tensor = torch.tensor([full_seq_ids], dtype=torch.long, device=device)
        full_attention = torch.ones_like(full_seq_tensor)

        prompt_len = len(initial_input_ids_list[i])
        num_generated = len(generated_token_lists[i])

        # Single forward pass with hidden states
        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Running forward pass (GPU)...", flush=True)
        sys.stdout.flush()

        try:
            with torch.no_grad():
                forward_outputs = model(
                    input_ids=full_seq_tensor,
                    attention_mask=full_attention,
                    output_hidden_states=True,
                    use_cache=False,
                )
        except RuntimeError as e:
            error_msg = str(e).lower()
            print(f"\n{'='*80}", flush=True)
            print(f"❌❌❌ RUNTIME ERROR in Pass 2 (Artifact Extraction) ❌❌❌", flush=True)
            print(f"PID={os.getpid()}, Sample {i+1}/{batch_size}", flush=True)
            print(f"Seq length: {len(full_seq_ids)} (prompt: {prompt_len}, generated: {num_generated})", flush=True)
            print(f"{'='*80}", flush=True)
            if "out of memory" in error_msg or "oom" in error_msg:
                print(f"🔥 OUT OF MEMORY ERROR in Pass 2", flush=True)
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)
                print(f"💡 Try: --batch-size {batch_size // 2} or reduce max_new_tokens", flush=True)
            elif "cuda" in error_msg:
                print(f"🔥 CUDA ERROR in Pass 2", flush=True)
                print(f"💡 Check: nvidia-smi for GPU health", flush=True)
            print(f"Full error: {e}", flush=True)
            print(f"{'='*80}\n", flush=True)
            raise
        except Exception as e:
            print(f"\n{'='*80}", flush=True)
            print(f"❌❌❌ UNEXPECTED ERROR in Pass 2 Sample {i+1} ❌❌❌", flush=True)
            print(f"PID={os.getpid()}, Type: {type(e).__name__}", flush=True)
            print(f"Error: {e}", flush=True)
            print(f"{'='*80}\n", flush=True)
            raise

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Forward pass complete (GPU), extracting artifacts (CPU)...", flush=True)
        sys.stdout.flush()

        # Extract hidden states and logits
        # hidden_states: tuple of [1, seq_len, hidden_dim] for each layer
        # logits: [1, seq_len, vocab_size]

        # We only need artifacts for generated tokens
        # logits[i] predicts token[i+1]
        # So for generated tokens at positions [prompt_len, prompt_len+1, ..., seq_len-1]:
        # We need logits[prompt_len-1, prompt_len, ..., seq_len-2]

        gold_token_id = gold_answer_token_ids[i] if gold_answer_token_ids else None

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Extracting {num_generated} timesteps (FAST: keeping on GPU)...", flush=True)
        sys.stdout.flush()

        # KEEP ALL TENSORS ON GPU during processing for speed
        # Only convert to CPU/numpy at the very end
        timestep_data = []  # Store as GPU tensors first

        for step in range(num_generated):
            # Log progress every 100 steps for all samples
            if step % 100 == 0:
                print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Timestep {step+1}/{num_generated}... (GPU processing)", flush=True)
                sys.stdout.flush()
            # Position in full sequence
            pos = prompt_len + step

            # Token that was generated at this step
            next_token_id = generated_token_lists[i][step]
            next_token_str = tokenizer.decode([next_token_id])

            # Get hidden states at position pos-1 (which predicted this token)
            # This is the hidden state BEFORE the token was generated
            hidden_pos = pos - 1

            # Extract hidden states from all layers at this position (KEEP ON GPU)
            sample_hidden_states = [
                h[0, hidden_pos] for h in forward_outputs.hidden_states
            ]  # List of [hidden_dim] tensors on GPU

            # Compute logit-lens for all layers (ON GPU - FAST!)
            logits_per_layer, probs_per_layer = compute_logit_lens(
                [h.unsqueeze(0) for h in sample_hidden_states],
                unembed,  # Keep on GPU
            )

            # Compute per-layer metrics (ON GPU)
            entropy_per_layer = [
                compute_entropy(probs[0]) for probs in probs_per_layer
            ]

            # Cross-entropy for next token (the one that was actually generated)
            ce_next = [
                compute_cross_entropy(probs[0], next_token_id)
                for probs in probs_per_layer
            ]

            # Cross-entropy for gold token (if provided)
            ce_gold = None
            if gold_token_id is not None:
                ce_gold = [
                    compute_cross_entropy(probs[0], gold_token_id)
                    for probs in probs_per_layer
                ]

            # PER-LAYER RANKS for next token (the one that was actually generated)
            ranks_next = [
                compute_rank(logits[0], next_token_id)
                for logits in logits_per_layer
            ]

            # PER-LAYER RANKS for gold token (if provided)
            ranks_gold = None
            if gold_token_id is not None:
                ranks_gold = [
                    compute_rank(logits[0], gold_token_id)
                    for logits in logits_per_layer
                ]

            # Top-p nucleus presence for gold token (if provided)
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

            # PER-LAYER FEATURES for "####" token (final answer marker, token ID 827)
            final_answer_token_id = 827  # "####" token

            # Cross-entropy for "####" token
            ce_final_answer = [
                compute_cross_entropy(probs[0], final_answer_token_id)
                for probs in probs_per_layer
            ]

            # Ranks for "####" token per layer
            ranks_final_answer = [
                compute_rank(logits[0], final_answer_token_id)
                for logits in logits_per_layer
            ]

            # Probabilities for "####" token per layer
            probs_final_answer = [
                get_prob(probs[0], final_answer_token_id)
                for probs in probs_per_layer
            ]

            # Top-p nucleus presence for "####" token
            top_p_presence_final_answer = {
                str(p): [
                    compute_top_p_presence(probs[0], final_answer_token_id, p)
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

            # Final layer probs and ranks (for backwards compatibility)
            final_probs = probs_per_layer[-1][0]
            final_logits = logits_per_layer[-1][0]

            rank_gold = None
            prob_gold = None
            if gold_token_id is not None:
                rank_gold = compute_rank(final_logits, gold_token_id)
                prob_gold = get_prob(final_probs, gold_token_id)

            # "####" token at final layer (for backwards compatibility)
            rank_final_answer = compute_rank(final_logits, final_answer_token_id)
            prob_final_answer = get_prob(final_probs, final_answer_token_id)

            # EOS at final layer (for backwards compatibility)
            rank_eos = compute_rank(final_logits, eos_token_id)
            prob_eos = get_prob(final_probs, eos_token_id)

            # Store as GPU tensors (no conversion yet!)
            timestep_data.append({
                'next_token_id': next_token_id,
                'next_token_str': next_token_str,
                'hidden_states': sample_hidden_states,  # Keep as GPU tensors
                'logits_per_layer': logits_per_layer,  # Keep as GPU tensors
                'entropy_per_layer': entropy_per_layer,
                'ce_next': ce_next,
                'ce_gold': ce_gold,
                'ce_final_answer': ce_final_answer,  # CE for "####" token
                'ranks_next': ranks_next,  # Per-layer ranks for next token
                'ranks_gold': ranks_gold,  # Per-layer ranks for gold token
                'ranks_final_answer': ranks_final_answer,  # Per-layer ranks for "####" token
                'probs_final_answer': probs_final_answer,  # Per-layer probabilities for "####" token
                'rank_gold': rank_gold,  # Final layer rank (backwards compat)
                'prob_gold': prob_gold,
                'rank_final_answer': rank_final_answer,  # Final layer rank for "####"
                'prob_final_answer': prob_final_answer,  # Final layer probability for "####"
                'top_p_presence_gold': top_p_presence_gold,  # Top-p nucleus presence for gold token
                'top_p_presence_final_answer': top_p_presence_final_answer,  # Top-p for "####" token
                'ranks_eos': ranks_eos,  # Per-layer ranks for EOS token
                'probs_eos': probs_eos,  # Per-layer probabilities for EOS token
                'rank_eos': rank_eos,    # Final layer rank for EOS (backwards compat)
                'prob_eos': prob_eos,    # Final layer prob for EOS (backwards compat)
            })

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: All {num_generated} timesteps extracted (still on GPU)", flush=True)
        sys.stdout.flush()

        # KEEP TENSORS ON GPU! Only convert to numpy when saving to disk
        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Creating artifacts (keeping tensors on GPU)...", flush=True)
        sys.stdout.flush()

        timestep_artifacts = []
        for ts_data in timestep_data:
            artifact = TimestepArtifacts(
                next_token_id=ts_data['next_token_id'],
                next_token_str=ts_data['next_token_str'],
                hidden_states=ts_data['hidden_states'],  # Keep as GPU tensors!
                logits_per_layer=[lg[0] for lg in ts_data['logits_per_layer']],  # Keep as GPU tensors!
                entropy_per_layer=ts_data['entropy_per_layer'],
                cross_entropy_next=ts_data['ce_next'],
                cross_entropy_gold=ts_data['ce_gold'],
                cross_entropy_prod=None,
                cross_entropy_final_answer=ts_data['ce_final_answer'],  # CE for "####" token
                ranks_next=ts_data['ranks_next'],  # Per-layer ranks for next token
                ranks_gold=ts_data['ranks_gold'],  # Per-layer ranks for gold token
                ranks_final_answer=ts_data['ranks_final_answer'],  # Per-layer ranks for "####" token
                probs_final_answer=ts_data['probs_final_answer'],  # Per-layer probabilities for "####" token
                rank_gold=ts_data['rank_gold'],  # Final layer rank (backwards compat)
                rank_prod=None,
                prob_gold=ts_data['prob_gold'],
                prob_prod=None,
                rank_final_answer=ts_data['rank_final_answer'],  # Final layer rank for "####"
                prob_final_answer=ts_data['prob_final_answer'],  # Final layer probability for "####"
                top_p_presence_gold=ts_data['top_p_presence_gold'],  # Top-p nucleus presence for gold token
                top_p_presence_final_answer=ts_data['top_p_presence_final_answer'],  # Top-p for "####" token
                ranks_eos=ts_data['ranks_eos'],  # Per-layer ranks for EOS token
                probs_eos=ts_data['probs_eos'],  # Per-layer probabilities for EOS token
                rank_eos=ts_data['rank_eos'],    # Final layer rank for EOS (backwards compat)
                prob_eos=ts_data['prob_eos'],    # Final layer prob for EOS (backwards compat)
            )
            timestep_artifacts.append(artifact)

        # Free timestep_data but keep tensors in artifacts (on GPU)
        del timestep_data

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Artifacts created (tensors still on GPU)", flush=True)
        sys.stdout.flush()

        # Calculate data size
        num_layers = len(timestep_artifacts[0].hidden_states) if timestep_artifacts else 0
        hidden_size_mb = 0
        logits_size_mb = 0
        if timestep_artifacts:
            if timestep_artifacts[0].hidden_states:
                hidden_bytes = sum(h.nbytes for h in timestep_artifacts[0].hidden_states)
                hidden_size_mb = (hidden_bytes * len(timestep_artifacts)) / (1024**2)
            if timestep_artifacts[0].logits_per_layer:
                logits_bytes = sum(lg.nbytes for lg in timestep_artifacts[0].logits_per_layer)
                logits_size_mb = (logits_bytes * len(timestep_artifacts)) / (1024**2)

        total_size_mb = hidden_size_mb + logits_size_mb
        cumulative_ram_gb = (total_size_mb * (i + 1)) / 1024

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Data in CPU RAM: ~{total_size_mb:.0f} MB (Total: ~{cumulative_ram_gb:.1f} GB)", flush=True)
        sys.stdout.flush()

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: Creating output object...", flush=True)
        sys.stdout.flush()

        # Create output for this sample
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

        print(f"[DEBUG PID={os.getpid()}] Pass 2 Sample {i+1}: ✓ Complete (stored in RAM)", flush=True)
        sys.stdout.flush()

        outputs.append(output)

        # Free GPU memory after processing this sample
        del forward_outputs
        del full_seq_tensor
        del full_attention
        del timestep_artifacts  # Now in output object

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if i < batch_size - 1:  # Not last sample
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                print(f"[DEBUG PID={os.getpid()}] GPU memory after cleanup: {allocated:.2f} GB allocated", flush=True)
                sys.stdout.flush()

    print(f"[DEBUG PID={os.getpid()}] Pass 2 complete! All {batch_size} samples processed.", flush=True)
    sys.stdout.flush()

    # Cleanup
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"[DEBUG PID={os.getpid()}] Two-pass generation complete, returning {len(outputs)} outputs", flush=True)
    sys.stdout.flush()

    return outputs
