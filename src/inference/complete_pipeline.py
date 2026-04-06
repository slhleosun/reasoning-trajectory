"""Complete pipeline for artifact capture with post-processing"""

from typing import Optional, Dict, Any
import json
from pathlib import Path

from ..models.generation_output import CompleteGenerationOutput
from ..features.span_detection import (
    extract_answer_after_hash,
    detect_dp2_index,
    get_gold_answer_first_token,
)
from ..features.windows import WindowConfig, compute_all_windows, classify_trajectory
from ..features.logit_lens import compute_rank, get_prob
from ..utils import answers_match


def fill_retroactive_features(
    output: CompleteGenerationOutput,
    produced_answer_token_id: Optional[int],
) -> None:
    """Fill retroactive features for produced answer (pure GPU, no transfers!)

    Args:
        output: CompleteGenerationOutput (modified in-place)
        produced_answer_token_id: First token ID of produced answer

    Note: Tensors are already on GPU from generation, so no conversion needed!
    """
    if produced_answer_token_id is None:
        return

    import torch
    import torch.nn.functional as F
    import numpy as np
    import sys

    total_artifacts = len(output.timestep_artifacts)
    print(f"[POST-PROCESS] Starting retroactive features for {total_artifacts} timesteps (pure GPU)", flush=True)
    sys.stdout.flush()

    # Process each timestep (tensors already on GPU - no conversion!)
    for t_idx, artifact in enumerate(output.timestep_artifacts):
        try:
            if artifact.logits_per_layer is None or artifact.hidden_states is None:
                continue

            # Progress logging every 100 timesteps
            if t_idx % 100 == 0:
                print(f"[POST-PROCESS] Progress: {t_idx}/{total_artifacts} timesteps", flush=True)
                sys.stdout.flush()

            # Get final layer logits (already on GPU!)
            final_logits = artifact.logits_per_layer[-1]  # [vocab_size] GPU tensor

            # Ensure it's a tensor (in case it's numpy)
            if isinstance(final_logits, np.ndarray):
                # Convert numpy to tensor
                device = next(iter(output.timestep_artifacts[0].hidden_states)).device if output.timestep_artifacts[0].hidden_states else 'cpu'
                final_logits = torch.from_numpy(final_logits).to(device)
            elif not isinstance(final_logits, torch.Tensor):
                # Unknown type, skip this timestep
                print(f"[POST-PROCESS WARNING] final_logits has unexpected type: {type(final_logits)}", flush=True)
                artifact.rank_prod = None
                artifact.prob_prod = None
                artifact.cross_entropy_prod = []
                continue

            # Softmax on GPU (no transfer!)
            final_probs = F.softmax(final_logits.float(), dim=-1)

            # Compute rank and prob for produced answer (all on GPU)
            artifact.rank_prod = compute_rank(final_logits, produced_answer_token_id)
            artifact.prob_prod = get_prob(final_probs, produced_answer_token_id)

            # BATCH OPTIMIZATION: Stack all layers for this timestep
            # Tensors are already on GPU, so torch.stack is fast!
            # Ensure all are tensors (in case some are numpy)
            logits_list = []
            for lg in artifact.logits_per_layer:
                if isinstance(lg, torch.Tensor):
                    logits_list.append(lg)
                elif isinstance(lg, np.ndarray):
                    # Convert numpy to tensor
                    device = final_logits.device if isinstance(final_logits, torch.Tensor) else 'cpu'
                    logits_list.append(torch.from_numpy(lg).to(device))
                else:
                    # Skip any other type (shouldn't happen)
                    print(f"[POST-PROCESS WARNING] Unexpected logit type: {type(lg)}", flush=True)
                    continue

            if len(logits_list) == 0:
                # No valid logits, skip
                artifact.cross_entropy_prod = []
                continue

            all_layer_logits = torch.stack(logits_list)  # [L, V] on GPU

            # Ensure it's a tensor before calling .float()
            if not isinstance(all_layer_logits, torch.Tensor):
                print(f"[POST-PROCESS ERROR] all_layer_logits is not a tensor: {type(all_layer_logits)}", flush=True)
                artifact.cross_entropy_prod = []
                continue

            # Batch softmax across all layers (all on GPU)
            all_layer_probs = F.softmax(all_layer_logits.float(), dim=-1)  # [L, V]

            # VECTORIZED cross-entropy for all layers at once (FAST!)
            # Extract probabilities for target token across all layers: [L]
            target_probs = all_layer_probs[:, produced_answer_token_id].clamp_min(1e-12)
            # Compute CE for all layers: CE = -log(p)
            all_ces = -torch.log(target_probs)  # [L] on GPU
            # Convert to list of floats in one shot
            artifact.cross_entropy_prod = all_ces.tolist()  # Single CPU transfer for all layers!

        except Exception as e:
            print(f"[POST-PROCESS] ERROR at timestep {t_idx}/{total_artifacts}: {type(e).__name__}: {e}", flush=True)
            sys.stdout.flush()
            # Skip this timestep but continue processing
            artifact.rank_prod = None
            artifact.prob_prod = None
            artifact.cross_entropy_prod = []
            continue

    print(f"[POST-PROCESS] Completed retroactive features (all on GPU!)", flush=True)
    sys.stdout.flush()


def process_complete_generation(
    output: CompleteGenerationOutput,
    gold_answer: str,
    tokenizer,
    window_config: Optional[WindowConfig] = None,
    dataset_format: str = "gsm8k",
) -> CompleteGenerationOutput:
    """Complete post-processing pipeline

    Args:
        output: Raw generation output from greedy_generate_with_artifacts
        gold_answer: Ground truth answer string
        tokenizer: Tokenizer for answer tokenization
        window_config: Optional window configuration
        dataset_format: Dataset format for answer extraction ('gsm8k', 'math-500', etc.)

    Returns:
        Processed CompleteGenerationOutput with all fields filled
    """
    # Set gold answer
    output.gold_answer = gold_answer

    # Extract produced answer using format-specific extraction
    if dataset_format == "math-500" or dataset_format == "math":
        # MATH format: try \boxed{} first, then #### fallback
        from ..utils import extract_answer
        produced_answer = extract_answer(output.produced_text, task=dataset_format)
    else:
        # GSM8K format: use #### marker extraction
        produced_answer = extract_answer_after_hash(output.produced_text)

        # Fallback: if extraction fails, try to find any number in the produced text
        if produced_answer is None and output.produced_text:
            import re
            # Try to find last number in text as fallback
            numbers = re.findall(r'-?\d+\.?\d*', output.produced_text)
            if numbers:
                produced_answer = numbers[-1]

    output.produced_answer = produced_answer

    # Detect dp2 (start of extracted answer in token sequence)
    # Only attempt if we have an answer
    if produced_answer:
        output.dp2_idx = detect_dp2_index(
            output.full_seq_ids,
            tokenizer,
            output.dp1_idx,
            produced_text=output.produced_text,
            produced_answer=produced_answer,  # Pass produced answer for accurate detection
        )

        # DEBUG: Log dp2 detection details
        if output.dp2_idx is not None and output.timestep_artifacts:
            import sys
            print(f"\n[DP2 DEBUG] dp1_idx={output.dp1_idx}, dp2_idx={output.dp2_idx}", flush=True)
            print(f"[DP2 DEBUG] reasoning_length={output.dp2_idx - output.dp1_idx}", flush=True)

            # Get the actual token at dp2
            rel_dp2_idx = output.dp2_idx - output.dp1_idx
            if 0 <= rel_dp2_idx < len(output.timestep_artifacts):
                artifact_at_dp2 = output.timestep_artifacts[rel_dp2_idx]
                token_at_dp2 = artifact_at_dp2.next_token_id
                token_str = artifact_at_dp2.next_token_str
                print(f"[DP2 DEBUG] Token at dp2: id={token_at_dp2}, str='{token_str}'", flush=True)
                print(f"[DP2 DEBUG] Expected: produced_answer='{produced_answer}' should start at this token", flush=True)

                # Check gold/prod ranks at dp2 (after retroactive features filled)
                if hasattr(artifact_at_dp2, 'rank_gold') and hasattr(artifact_at_dp2, 'rank_prod'):
                    print(f"[DP2 DEBUG] Ranks at dp2 will be computed during retroactive feature fill", flush=True)

            sys.stdout.flush()
    else:
        # No answer extracted - dp2 remains None (default)
        output.dp2_idx = None

    # Calculate reasoning length
    if output.dp2_idx is not None:
        output.reasoning_length = output.dp2_idx - output.dp1_idx
    else:
        # No dp2 found - use full generation length as reasoning length
        output.reasoning_length = len(output.timestep_artifacts) if output.timestep_artifacts else None

    # Fill retroactive features for produced answer (only if we have an answer)
    if produced_answer:
        # Get context-aware produced answer token ID
        prompt_ids = output.full_seq_ids[:output.dp1_idx + 1]
        produced_token_id = get_gold_answer_first_token(
            produced_answer,
            tokenizer,
            prompt_ids=prompt_ids
        )

        # DEBUG: Log produced token ID extraction
        import sys
        print(f"\n[PROD TOKEN DEBUG] produced_answer='{produced_answer}'", flush=True)
        print(f"[PROD TOKEN DEBUG] produced_token_id={produced_token_id} (context-aware)", flush=True)

        # Compare with actual token at dp2
        if produced_token_id is not None and output.dp2_idx is not None and output.timestep_artifacts:
            rel_dp2_idx = output.dp2_idx - output.dp1_idx
            if 0 <= rel_dp2_idx < len(output.timestep_artifacts):
                actual_token_at_dp2 = output.timestep_artifacts[rel_dp2_idx].next_token_id
                if produced_token_id != actual_token_at_dp2:
                    print(f"[PROD TOKEN WARNING] Mismatch! produced_token_id={produced_token_id} != actual_token_at_dp2={actual_token_at_dp2}", flush=True)
                else:
                    print(f"[PROD TOKEN DEBUG] Match! produced_token_id == actual_token_at_dp2 = {actual_token_at_dp2}", flush=True)

        sys.stdout.flush()

        # Only fill if we successfully got a token ID
        if produced_token_id is not None:
            fill_retroactive_features(output, produced_token_id)

    # Compute step-based aggregations (at each "Step" token)
    if window_config is None:
        window_config = WindowConfig()  # Default: step_token_id=8468, threshold=10

    # Always compute steps (not dependent on dp2_idx)
    output.windows = compute_all_windows(output, window_config)
    output.num_steps = len(output.windows)  # Set the number of steps found

    # DEBUG: Log step details
    import sys

    print(f"\n[STEP DEBUG] Found {output.num_steps} step ranges in generation", flush=True)
    print(f"[STEP DEBUG] (Each step = range from one 'Step' token to the next)", flush=True)

    if output.num_steps > 0:
        # Get sorted step keys
        step_keys = sorted([k for k in output.windows.keys() if k.startswith("step_")],
                          key=lambda x: int(x.split("_")[1]))

        # Log first step (if exists)
        if len(step_keys) > 0:
            first_step = output.windows[step_keys[0]]
            print(f"\n[STEP DEBUG] {step_keys[0]} (FIRST) aggregated features:", flush=True)
            print(f"[STEP DEBUG]   gold min_rank across step: {first_step.get('gold', {}).get('min_rank')}", flush=True)
            print(f"[STEP DEBUG]   prod min_rank across step: {first_step.get('prod', {}).get('min_rank')}", flush=True)
            print(f"[STEP DEBUG]   gold max_p across step: {first_step.get('gold', {}).get('max_p')}", flush=True)
            print(f"[STEP DEBUG]   prod max_p across step: {first_step.get('prod', {}).get('max_p')}", flush=True)

        # Log last step (if exists and different from first)
        if len(step_keys) > 1:
            last_step = output.windows[step_keys[-1]]
            print(f"\n[STEP DEBUG] {step_keys[-1]} (LAST) aggregated features:", flush=True)
            print(f"[STEP DEBUG]   gold min_rank across step: {last_step.get('gold', {}).get('min_rank')}", flush=True)
            print(f"[STEP DEBUG]   prod min_rank across step: {last_step.get('prod', {}).get('min_rank')}", flush=True)
            print(f"[STEP DEBUG]   gold max_p across step: {last_step.get('gold', {}).get('max_p')}", flush=True)
            print(f"[STEP DEBUG]   prod max_p across step: {last_step.get('prod', {}).get('max_p')}", flush=True)

        # Show trajectory computation (if we have at least one step)
        if len(step_keys) >= 1:
            first = output.windows[step_keys[0]]
            last = output.windows[step_keys[-1]]

            prod_r_first = first.get("prod", {}).get("min_rank")
            prod_r_last = last.get("prod", {}).get("min_rank")
            gold_r_first = first.get("gold", {}).get("min_rank")
            gold_r_last = last.get("gold", {}).get("min_rank")

            if all(r is not None for r in [prod_r_first, prod_r_last, gold_r_first, gold_r_last]):
                threshold = window_config.rank_high_threshold
                prod_start = "High" if prod_r_first <= threshold else "Low"
                prod_end = "High" if prod_r_last <= threshold else "Low"
                gold_start = "High" if gold_r_first <= threshold else "Low"
                gold_end = "High" if gold_r_last <= threshold else "Low"

                print(f"\n[TRAJECTORY COMPUTATION] (threshold={threshold})", flush=True)
                print(f"[TRAJECTORY COMPUTATION] Produced: rank {prod_r_first} ({step_keys[0]}) → rank {prod_r_last} ({step_keys[-1]}) = {prod_start}→{prod_end}", flush=True)
                print(f"[TRAJECTORY COMPUTATION] Gold:     rank {gold_r_first} ({step_keys[0]}) → rank {gold_r_last} ({step_keys[-1]}) = {gold_start}→{gold_end}", flush=True)

    sys.stdout.flush()

    # Determine correctness using robust answer matching
    is_correct = False
    if output.produced_answer is not None and output.gold_answer is not None:
        is_correct = answers_match(str(output.produced_answer), str(output.gold_answer))
        print(f"[CORRECTNESS DEBUG] produced='{output.produced_answer}', gold='{output.gold_answer}', is_correct={is_correct}", flush=True)

    # Classify trajectory based on steps and correctness
    if output.num_steps > 0:
        trajectory_type = classify_trajectory(
            output.windows,
            is_correct=is_correct,
            rank_high_threshold=window_config.rank_high_threshold
        )
        if trajectory_type is not None:
            output.metadata["trajectory_type"] = trajectory_type
            print(f"[TRAJECTORY DEBUG] Classified as: {trajectory_type}", flush=True)

    sys.stdout.flush()

    return output


def save_generation_output(
    output: CompleteGenerationOutput,
    output_path: Path,
    include_arrays: bool = False,
    save_hidden_states_at_windows_only: bool = True,
) -> None:
    """Save generation output to JSON

    Args:
        output: CompleteGenerationOutput
        output_path: Path to save JSON
        include_arrays: Whether to include large arrays (hidden_states, logits)
        save_hidden_states_at_windows_only: If True, save hidden_states only at step timesteps
                                             (default: True for memory efficiency)
                                             Step timesteps are positions where "Step" token (ID 8468) appears.

    Note: This is where GPU tensors are converted to numpy for JSON serialization.
    """
    import torch
    import sys
    import gc
    import numpy as np

    total_timesteps = len(output.timestep_artifacts)
    print(f"[SAVE] Converting GPU tensors to numpy for {total_timesteps} timesteps...", flush=True)
    sys.stdout.flush()

    # Check CUDA availability and memory before starting
    if torch.cuda.is_available():
        try:
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[SAVE] GPU memory before conversion: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[SAVE] Warning: Could not check GPU memory: {e}", flush=True)
            sys.stdout.flush()

    # OPTIMIZATION: Batch all GPU→CPU transfers instead of doing them individually
    # This is MUCH faster: 2 large transfers instead of 32,768 small transfers!
    try:
        # Check if we have any GPU tensors to convert
        has_tensors = any(
            artifact.hidden_states is not None and
            any(isinstance(h, torch.Tensor) for h in artifact.hidden_states)
            for artifact in output.timestep_artifacts
        )

        if has_tensors and torch.cuda.is_available():
            print(f"[SAVE] Using BATCHED GPU→CPU transfer (fast!)...", flush=True)
            sys.stdout.flush()

            # Collect all hidden state tensors across all timesteps
            all_hidden_tensors = []
            hidden_shapes = []  # Track which timestep/layer each tensor belongs to

            for t_idx, artifact in enumerate(output.timestep_artifacts):
                if artifact.hidden_states is not None:
                    for h_idx, h in enumerate(artifact.hidden_states):
                        if isinstance(h, torch.Tensor):
                            all_hidden_tensors.append(h.detach())
                            hidden_shapes.append((t_idx, h_idx, h.shape))

            # Batch transfer all hidden states at once
            if all_hidden_tensors:
                print(f"[SAVE] Transferring {len(all_hidden_tensors)} hidden state tensors in one batch...", flush=True)
                sys.stdout.flush()

                # Stack, transfer, convert in one operation
                stacked_hidden = torch.stack([h.flatten() for h in all_hidden_tensors])
                # Convert BFloat16 to float32 before numpy (BFloat16 not supported by numpy)
                numpy_hidden = stacked_hidden.cpu().float().numpy()  # SINGLE GPU→CPU transfer!

                # Redistribute back to artifacts
                for i, (t_idx, h_idx, shape) in enumerate(hidden_shapes):
                    output.timestep_artifacts[t_idx].hidden_states[h_idx] = numpy_hidden[i].reshape(shape)

                del stacked_hidden, numpy_hidden, all_hidden_tensors
                gc.collect()

            # SKIP LOGITS TRANSFER: Logits are too large for JSON storage
            # Clear them immediately to save memory and processing time
            print(f"[SAVE] Skipping logits transfer (will be cleared for JSON)", flush=True)
            for artifact in output.timestep_artifacts:
                artifact.logits_per_layer = None
            sys.stdout.flush()

            print(f"[SAVE] Batched transfer complete!", flush=True)
            sys.stdout.flush()

        else:
            # Fallback: no GPU tensors or CUDA not available, process normally
            print(f"[SAVE] Using sequential conversion (no GPU tensors detected)...", flush=True)
            sys.stdout.flush()

            for t_idx, artifact in enumerate(output.timestep_artifacts):
                if artifact.hidden_states is not None:
                    artifact.hidden_states = [
                        h.detach().cpu().float().numpy() if isinstance(h, torch.Tensor) else h
                        for h in artifact.hidden_states
                    ]
                if artifact.logits_per_layer is not None:
                    artifact.logits_per_layer = [
                        lg.detach().cpu().float().numpy() if isinstance(lg, torch.Tensor) else lg
                        for lg in artifact.logits_per_layer
                    ]

    except Exception as e:
        print(f"[SAVE ERROR] Failed during batched GPU→CPU transfer", flush=True)
        print(f"[SAVE ERROR] Error type: {type(e).__name__}", flush=True)
        print(f"[SAVE ERROR] Error message: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    print(f"[SAVE] Conversion complete ({total_timesteps} timesteps), creating JSON...", flush=True)
    sys.stdout.flush()

    # Clear arrays if not including them (do this BEFORE to_dict() to avoid serialization overhead)
    if not include_arrays:
        print(f"[SAVE] Excluding arrays from JSON (include_arrays=False)", flush=True)
        for artifact in output.timestep_artifacts:
            artifact.hidden_states = None
            artifact.logits_per_layer = None
        sys.stdout.flush()

    # Note: logits_per_layer are already cleared during GPU→CPU transfer (too large for JSON)

    # Determine step timesteps if saving hidden_states selectively at steps
    step_timestep_indices = None
    if include_arrays and save_hidden_states_at_windows_only:
        from ..features.windows import compute_step_boundaries, WindowConfig

        # Use default step config (step_token_id=8468)
        config = WindowConfig()
        boundaries = compute_step_boundaries(
            output.timestep_artifacts,
            output.dp1_idx,
            output.dp2_idx,
            config.step_token_id
        )

        print(f"[SAVE DEBUG] dp1_idx={output.dp1_idx}, dp2_idx={output.dp2_idx}", flush=True)
        print(f"[SAVE DEBUG] Step boundaries (absolute): {boundaries}", flush=True)
        print(f"[SAVE DEBUG] Total timestep_artifacts: {len(output.timestep_artifacts)}", flush=True)

        # Save hidden states at the START of each step (the "Step" token position)
        # Convert absolute indices to relative indices (within generated sequence)
        step_timestep_indices = set()
        for start_idx, end_idx in boundaries:
            rel_start = start_idx - output.dp1_idx
            print(f"[SAVE DEBUG] Step range ({start_idx}, {end_idx}) -> saving at rel_start={rel_start}", flush=True)
            if 0 <= rel_start < len(output.timestep_artifacts):
                step_timestep_indices.add(rel_start)
            else:
                print(f"[SAVE DEBUG] SKIPPED: rel_start={rel_start} out of range [0, {len(output.timestep_artifacts)})", flush=True)

        print(f"[SAVE] Will save hidden_states at {len(step_timestep_indices)} step positions: {sorted(step_timestep_indices)}", flush=True)
        sys.stdout.flush()

        # Count how many timesteps currently have hidden_states
        n_before = sum(1 for a in output.timestep_artifacts if a.hidden_states is not None)
        print(f"[SAVE DEBUG] Timesteps with hidden_states BEFORE clearing: {n_before}/{len(output.timestep_artifacts)}", flush=True)

        # Clear hidden_states for non-step timesteps BEFORE converting to dict
        for t_idx, artifact in enumerate(output.timestep_artifacts):
            if t_idx not in step_timestep_indices:
                artifact.hidden_states = None

        # Count how many timesteps still have hidden_states
        n_after = sum(1 for a in output.timestep_artifacts if a.hidden_states is not None)
        print(f"[SAVE DEBUG] Timesteps with hidden_states AFTER clearing: {n_after}/{len(output.timestep_artifacts)}", flush=True)

    try:
        # Use minimal mode to save only essential fields (steering collection needs)
        # This dramatically reduces file size and processing time
        output_dict = output.to_dict(minimal=True)
    except Exception as e:
        print(f"[SAVE ERROR] Failed to create dict: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    # Log array inclusion statistics
    if not include_arrays:
        print(f"[SAVE] Arrays excluded from JSON (cleared before to_dict())", flush=True)
    else:
        # Count how many timesteps have hidden states saved
        n_with_hidden = sum(
            1 for ts in output_dict.get("timesteps", [])
            if ts.get("hidden_states") is not None
        )
        print(f"[SAVE] Including hidden_states in JSON: {n_with_hidden}/{len(output_dict.get('timesteps', []))} timesteps", flush=True)
        print(f"[SAVE] Logits excluded (too large - use .npz cache for logits if needed)", flush=True)

        # Log hidden states shape for first timestep with data
        if n_with_hidden > 0:
            for ts in output_dict.get("timesteps", []):
                if ts.get("hidden_states") is not None:
                    hs_shape = np.array(ts["hidden_states"]).shape
                    print(f"[SAVE] Hidden states shape: {hs_shape} (at {n_with_hidden} step positions)", flush=True)
                    break
    sys.stdout.flush()

    # Save to JSON
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"[SAVE] Saved to {output_path}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[SAVE ERROR] Failed to write JSON to {output_path}", flush=True)
        print(f"[SAVE ERROR] Error type: {type(e).__name__}", flush=True)
        print(f"[SAVE ERROR] Error message: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    # CRITICAL: Free memory immediately after saving
    # The numpy arrays are still in the output object, even though excluded from JSON
    # Clear them to prevent OOM when processing batches
    if not include_arrays:
        print(f"[SAVE] Freeing memory (clearing {len(output.timestep_artifacts)} timestep arrays)...", flush=True)
        for artifact in output.timestep_artifacts:
            artifact.hidden_states = None
            artifact.logits_per_layer = None
        sys.stdout.flush()


def run_complete_pipeline(
    model_adapter,
    prompt: str,
    gold_answer: str,
    output_path: Path,
    max_new_tokens: int = 512,
    include_arrays: bool = False,
) -> CompleteGenerationOutput:
    """End-to-end pipeline for complete artifact capture

    Args:
        model_adapter: Loaded model adapter (HuggingFaceAdapter)
        prompt: Input prompt string
        gold_answer: Ground truth answer
        output_path: Path to save output JSON
        max_new_tokens: Maximum tokens to generate
        include_arrays: Whether to save large arrays to JSON

    Returns:
        Processed CompleteGenerationOutput
    """
    # Get gold answer first token for tracking during generation
    gold_token_id = get_gold_answer_first_token(
        gold_answer,
        model_adapter._tokenizer
    )

    # DEBUG: Log gold token extraction
    import sys
    print(f"\n[GOLD TOKEN DEBUG] gold_answer='{gold_answer}'", flush=True)
    print(f"[GOLD TOKEN DEBUG] gold_token_id={gold_token_id}", flush=True)
    sys.stdout.flush()

    # Generate with complete artifacts
    output = model_adapter.generate_with_complete_artifacts(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        gold_answer_token_id=gold_token_id,
    )

    # Post-process
    output = process_complete_generation(
        output,
        gold_answer=gold_answer,
        tokenizer=model_adapter._tokenizer,
    )

    # Save to JSON
    save_generation_output(output, output_path, include_arrays=include_arrays)

    return output
