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

    for t_idx, artifact in enumerate(output.timestep_artifacts):
        try:
            if artifact.logits_per_layer is None or artifact.hidden_states is None:
                continue

            # Get final layer logits (already on GPU!)
            final_logits = artifact.logits_per_layer[-1]  # [vocab_size] GPU tensor

            # Ensure it's a tensor (in case it's numpy)
            if isinstance(final_logits, np.ndarray):
                device = next(iter(output.timestep_artifacts[0].hidden_states)).device if output.timestep_artifacts[0].hidden_states else 'cpu'
                final_logits = torch.from_numpy(final_logits).to(device)
            elif not isinstance(final_logits, torch.Tensor):
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
            logits_list = []
            for lg in artifact.logits_per_layer:
                if isinstance(lg, torch.Tensor):
                    logits_list.append(lg)
                elif isinstance(lg, np.ndarray):
                    device = final_logits.device if isinstance(final_logits, torch.Tensor) else 'cpu'
                    logits_list.append(torch.from_numpy(lg).to(device))

            if len(logits_list) == 0:
                artifact.cross_entropy_prod = []
                continue

            all_layer_logits = torch.stack(logits_list)  # [L, V] on GPU

            if not isinstance(all_layer_logits, torch.Tensor):
                artifact.cross_entropy_prod = []
                continue

            # Batch softmax across all layers (all on GPU)
            all_layer_probs = F.softmax(all_layer_logits.float(), dim=-1)  # [L, V]

            # VECTORIZED cross-entropy for all layers at once
            target_probs = all_layer_probs[:, produced_answer_token_id].clamp_min(1e-12)
            all_ces = -torch.log(target_probs)  # [L] on GPU
            artifact.cross_entropy_prod = all_ces.tolist()

        except Exception:
            artifact.rank_prod = None
            artifact.prob_prod = None
            artifact.cross_entropy_prod = []
            continue


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
            numbers = re.findall(r'-?\d+\.?\d*', output.produced_text)
            if numbers:
                produced_answer = numbers[-1]

    output.produced_answer = produced_answer

    # Detect dp2 (start of extracted answer in token sequence)
    if produced_answer:
        output.dp2_idx = detect_dp2_index(
            output.full_seq_ids,
            tokenizer,
            output.dp1_idx,
            produced_text=output.produced_text,
            produced_answer=produced_answer,
        )
    else:
        output.dp2_idx = None

    # Calculate reasoning length
    if output.dp2_idx is not None:
        output.reasoning_length = output.dp2_idx - output.dp1_idx
    else:
        output.reasoning_length = len(output.timestep_artifacts) if output.timestep_artifacts else None

    # Fill retroactive features for produced answer
    if produced_answer:
        prompt_ids = output.full_seq_ids[:output.dp1_idx + 1]
        produced_token_id = get_gold_answer_first_token(
            produced_answer,
            tokenizer,
            prompt_ids=prompt_ids
        )

        if produced_token_id is not None:
            fill_retroactive_features(output, produced_token_id)

    # Compute step-based aggregations (at each "Step" token)
    if window_config is None:
        window_config = WindowConfig()

    output.windows = compute_all_windows(output, window_config)
    output.num_steps = len(output.windows)

    # Determine correctness using robust answer matching
    is_correct = False
    if output.produced_answer is not None and output.gold_answer is not None:
        is_correct = answers_match(str(output.produced_answer), str(output.gold_answer))

    # Classify trajectory based on steps and correctness
    if output.num_steps > 0:
        trajectory_type = classify_trajectory(
            output.windows,
            is_correct=is_correct,
            rank_high_threshold=window_config.rank_high_threshold
        )
        if trajectory_type is not None:
            output.metadata["trajectory_type"] = trajectory_type

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
    import gc
    import numpy as np

    # OPTIMIZATION: Batch all GPU->CPU transfers instead of doing them individually
    try:
        has_tensors = any(
            artifact.hidden_states is not None and
            any(isinstance(h, torch.Tensor) for h in artifact.hidden_states)
            for artifact in output.timestep_artifacts
        )

        if has_tensors and torch.cuda.is_available():
            # Collect all hidden state tensors across all timesteps
            all_hidden_tensors = []
            hidden_shapes = []

            for t_idx, artifact in enumerate(output.timestep_artifacts):
                if artifact.hidden_states is not None:
                    for h_idx, h in enumerate(artifact.hidden_states):
                        if isinstance(h, torch.Tensor):
                            all_hidden_tensors.append(h.detach())
                            hidden_shapes.append((t_idx, h_idx, h.shape))

            # Batch transfer all hidden states at once
            if all_hidden_tensors:
                stacked_hidden = torch.stack([h.flatten() for h in all_hidden_tensors])
                numpy_hidden = stacked_hidden.cpu().float().numpy()

                for i, (t_idx, h_idx, shape) in enumerate(hidden_shapes):
                    output.timestep_artifacts[t_idx].hidden_states[h_idx] = numpy_hidden[i].reshape(shape)

                del stacked_hidden, numpy_hidden, all_hidden_tensors
                gc.collect()

            # Clear logits (too large for JSON storage)
            for artifact in output.timestep_artifacts:
                artifact.logits_per_layer = None

        else:
            # Fallback: no GPU tensors or CUDA not available
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

    except Exception:
        raise

    # Clear arrays if not including them (do this BEFORE to_dict() to avoid serialization overhead)
    if not include_arrays:
        for artifact in output.timestep_artifacts:
            artifact.hidden_states = None
            artifact.logits_per_layer = None

    # Determine step timesteps if saving hidden_states selectively at steps
    if include_arrays and save_hidden_states_at_windows_only:
        from ..features.windows import compute_step_boundaries, WindowConfig as WC

        config = WC()
        boundaries = compute_step_boundaries(
            output.timestep_artifacts,
            output.dp1_idx,
            output.dp2_idx,
            config.step_token_id
        )

        step_timestep_indices = set()
        for start_idx, end_idx in boundaries:
            rel_start = start_idx - output.dp1_idx
            if 0 <= rel_start < len(output.timestep_artifacts):
                step_timestep_indices.add(rel_start)

        # Clear hidden_states for non-step timesteps
        for t_idx, artifact in enumerate(output.timestep_artifacts):
            if t_idx not in step_timestep_indices:
                artifact.hidden_states = None

    output_dict = output.to_dict(minimal=True)

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    # Free memory after saving
    if not include_arrays:
        for artifact in output.timestep_artifacts:
            artifact.hidden_states = None
            artifact.logits_per_layer = None


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
