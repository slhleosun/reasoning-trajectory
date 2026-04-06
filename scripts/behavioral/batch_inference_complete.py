"""Batch inference with complete artifact capture

This script processes datasets in TRUE BATCHES with full per-timestep artifact capture.
Memory is managed by saving results immediately after each batch and freeing GPU memory.
"""

import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import load_config, prepare_dataset, get_model_adapter
from src.utils import format_prompt, extract_answer, evaluate_answer, evaluate_math_answer, calculate_metrics, print_metrics
from src.inference.complete_pipeline import process_complete_generation, save_generation_output
from src.features.span_detection import get_gold_answer_first_token
from src.features.windows import WindowConfig


def process_batch(
    samples,
    adapter,
    model_name,
    max_new_tokens=1024,
    prompt_template="cot",
    window_config=None,
    log_fn=None,
):
    """Process a batch of samples with complete artifact capture

    Args:
        samples: List of DataSample objects
        adapter: Model adapter
        model_name: Name of the model
        max_new_tokens: Maximum tokens to generate
        prompt_template: Prompt template to use
        window_config: Optional window configuration
        log_fn: Optional logging function for progress

    Returns:
        List of (output, result_dict) tuples
    """
    import os
    import time

    def log(msg):
        if log_fn:
            log_fn(msg)
        print(msg, flush=True)
        sys.stdout.flush()

    log(f"[STAGE 1/4 PID={os.getpid()}] Formatting prompts for {len(samples)} questions...")

    # Format prompts and get gold answers
    prompts = []
    gold_answers = []
    gold_token_ids = []

    try:
        # Detect dataset format from sample metadata
        dataset_format = "gsm8k"  # Default
        if samples and hasattr(samples[0], 'metadata'):
            # Check if it's MATH-500 format
            if 'subject' in samples[0].metadata or 'level' in samples[0].metadata:
                dataset_format = "math-500"

        for i, sample in enumerate(samples):
            prompt = format_prompt(sample.question, template=prompt_template)
            gold_answer = extract_answer(sample.answer, task=dataset_format)
            gold_token_id = get_gold_answer_first_token(gold_answer, adapter._tokenizer) if gold_answer is not None else None

            prompts.append(prompt)
            gold_answers.append(gold_answer)
            gold_token_ids.append(gold_token_id)

            # Log question preview
            if i < 3:  # Only log first 3
                q_preview = sample.question[:60] + "..." if len(sample.question) > 60 else sample.question
                log(f"    Q{i+1}/{len(samples)} (ID: {sample.id}): {q_preview}")

        log(f"[STAGE 1/4] ✓ Prompt formatting complete")

    except Exception as e:
        log(f"[STAGE 1/4] ❌ ERROR in prompt formatting: {type(e).__name__}: {e}")
        raise

    # Generate batch with complete artifacts
    log(f"[STAGE 2/4 PID={os.getpid()}] Generating answers for {len(samples)} questions...")
    gen_start = time.time()

    try:
        # Set max input length based on dataset format (MATH needs longer prompts)
        max_input_length = 824 if dataset_format in ("math", "math-500") else 250
        log(f"  Dataset format: {dataset_format}, Max input length: {max_input_length} tokens")

        outputs = adapter.generate_batch_with_complete_artifacts(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            gold_answer_token_ids=gold_token_ids,
            max_input_length=max_input_length,
        )

        gen_time = time.time() - gen_start
        log(f"[STAGE 2/4] ✓ Generation complete ({gen_time:.1f}s, {gen_time/len(samples):.1f}s per question)")

    except Exception as e:
        gen_time = time.time() - gen_start
        log(f"[STAGE 2/4] ❌ ERROR in generation after {gen_time:.1f}s")
        log(f"  Error type: {type(e).__name__}")
        log(f"  Error message: {e}")
        raise

    # Post-process each output
    log(f"[STAGE 3/4 PID={os.getpid()}] Post-processing {len(samples)} questions...")
    post_start = time.time()

    try:
        # Detect dataset format from sample metadata
        dataset_format = "gsm8k"  # Default
        if samples and hasattr(samples[0], 'metadata'):
            # Check if it's MATH-500 format
            if 'subject' in samples[0].metadata or 'level' in samples[0].metadata:
                dataset_format = "math-500"

        results = []
        for i, (sample, output, gold_answer) in enumerate(zip(samples, outputs, gold_answers)):
            sample_post_start = time.time()

            # Post-process
            output = process_complete_generation(
                output,
                gold_answer=gold_answer,
                tokenizer=adapter._tokenizer,
                window_config=window_config,
                dataset_format=dataset_format,
            )

            sample_post_time = time.time() - sample_post_start
            if i == 0:  # Log first sample
                log(f"    Sample {i+1} post-processing took {sample_post_time:.1f}s")

            # Extract and evaluate answer
            predicted = output.produced_answer

            # Handle None gracefully
            if predicted is None:
                log(f"    WARNING: Failed to extract answer for Q{i+1} (ID: {sample.id})")
                log(f"    Generated text: {output.produced_text[:200]}..." if output.produced_text else "    (No text generated)")

            # Use appropriate evaluation function based on dataset format
            if predicted and gold_answer:
                if dataset_format in ("math", "math-500"):
                    is_correct = evaluate_math_answer(predicted, gold_answer)
                else:
                    is_correct = evaluate_answer(predicted, gold_answer)
            else:
                is_correct = False

            result = {
                "id": sample.id,
                "question": sample.question,
                "predicted": predicted,  # Keep None for proper extraction metrics
                "ground_truth": gold_answer,  # Keep None for proper extraction metrics
                "is_correct": is_correct,
                "dp1_idx": output.dp1_idx,
                "dp2_idx": output.dp2_idx,
                "reasoning_length": output.reasoning_length,
                "generated_tokens": len(output.timestep_artifacts) if output.timestep_artifacts else 0,
                "extraction_failed": predicted is None,  # Flag for analysis
                "trajectory_type": output.metadata.get("trajectory_type"),  # I1-I8 classification
            }

            results.append((output, result))

            # Log individual question result
            if i < 3:  # Only log first 3
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                pred_display = predicted if predicted is not None else "None"
                log(f"    Q{i+1} (ID: {sample.id}): {status} | Pred: {pred_display} | Gold: {gold_answer}")

        post_time = time.time() - post_start
        log(f"[STAGE 3/4] ✓ Post-processing complete ({post_time:.1f}s, {post_time/len(samples):.1f}s per question)")
        log(f"[STAGE 4/4] ✓ Batch processing complete - returning {len(results)} results")

        return results

    except Exception as e:
        log(f"[STAGE 3/4] ❌ ERROR in post-processing: {type(e).__name__}: {e}")
        raise


def main():
    """Run TRUE batch inference with complete artifact capture"""
    parser = argparse.ArgumentParser(description="Batch inference with complete artifacts")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for parallel processing")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instruct", help="Model name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--save-arrays", action="store_true", help="Save hidden_states and logits (WARNING: very large files)")
    parser.add_argument("--output-subdir", type=str, default=None, help="Subdirectory for output")

    # Multi-GPU support: dataset sharding
    parser.add_argument("--shard-id", type=int, default=None, help="Shard ID for multi-GPU processing (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards for multi-GPU processing")
    parser.add_argument("--batch-assignments", type=str, default=None, help="Path to JSON file with custom batch assignments (list of [shard_id, batch_idx] tuples)")

    args = parser.parse_args()

    # Immediate startup message with flush (important for log files)
    import sys
    import os
    print("=" * 80, flush=True)
    print("TRUE Batch Inference with Complete Artifact Capture", flush=True)
    print("=" * 80, flush=True)
    print(f"\nSTARTUP: Process started", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print(f"  Shard: {args.shard_id}/{args.num_shards}" if args.shard_id is not None else "  Mode: Single GPU", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Dataset: {args.dataset}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print("Memory management: Save and free after each batch", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    # Check if CUDA_LAUNCH_BLOCKING is set for debugging
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        print("\nNOTE: For more precise CUDA error locations, run with:", flush=True)
        print("  CUDA_LAUNCH_BLOCKING=1 python examples/batch_inference_complete.py [args]", flush=True)

    # Load configuration
    print(f"\nLoading configuration...", flush=True)
    config = load_config()
    settings = config.get_settings()
    print(f"  Configuration loaded", flush=True)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...", flush=True)
    max_samples = args.max_samples or settings.get("max_samples")
    full_dataset = prepare_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=max_samples,
    )
    print(f"  Dataset loaded: {len(full_dataset)} samples", flush=True)

    # Detect dataset format for later metrics calculation
    dataset_format = "gsm8k"  # Default
    if full_dataset and hasattr(full_dataset[0], 'metadata'):
        # Check if it's MATH-500 format
        if 'subject' in full_dataset[0].metadata or 'level' in full_dataset[0].metadata:
            dataset_format = "math-500"
    print(f"  Dataset format detected: {dataset_format}", flush=True)

    # ===================================================================
    # BATCH ASSIGNMENT MODE SELECTION
    # ===================================================================

    custom_batch_assignments = None
    if args.batch_assignments is not None:
        # Load custom batch assignments
        print(f"\nLoading custom batch assignments from: {args.batch_assignments}", flush=True)
        with open(args.batch_assignments, "r") as f:
            custom_batch_assignments = json.load(f)

        print(f"  Loaded {len(custom_batch_assignments)} batch assignments", flush=True)
        print(f"  Mode: Custom work redistribution", flush=True)

        # In custom mode, we process specific (shard_id, batch_idx) pairs
        # We still need the full dataset to extract samples
        samples = full_dataset
        total_samples = len(custom_batch_assignments) * args.batch_size  # Approximate
        num_batches = len(custom_batch_assignments)

    else:
        # Standard shard-based assignment
        if args.shard_id is not None and args.num_shards is not None:
            total_samples_before_shard = len(full_dataset)
            # Shard the dataset
            samples = [s for i, s in enumerate(full_dataset) if i % args.num_shards == args.shard_id]
            print(f"\n[SHARD {args.shard_id}/{args.num_shards}]", flush=True)
            print(f"  Total dataset size: {total_samples_before_shard}", flush=True)
            print(f"  This shard size: {len(samples)}", flush=True)
        else:
            samples = full_dataset

        total_samples = len(samples)
        num_batches = (total_samples + args.batch_size - 1) // args.batch_size

    print(f"Total samples to process: {total_samples}", flush=True)
    print(f"Number of batches: {num_batches}", flush=True)

    # Initialize model adapter
    print(f"\nInitializing model adapter: {args.model}...", flush=True)
    sys.stdout.flush()
    adapter = get_model_adapter(args.model)
    print(f"  Adapter created", flush=True)

    print(f"Loading model weights...", flush=True)
    sys.stdout.flush()
    adapter.load()
    print(f"  Model loaded successfully", flush=True)
    sys.stdout.flush()

    # Print device information
    try:
        import torch
        print(f"\nModel device information:")
        print(f"  Primary device: {adapter._model.device}")
        print(f"  Model dtype: {adapter._model.dtype}")

        # Check if model is distributed across devices
        if hasattr(adapter._model, 'hf_device_map'):
            print(f"  Device map: {adapter._model.hf_device_map}")
            devices = set(v for v in adapter._model.hf_device_map.values() if isinstance(v, int))
            if len(devices) > 1:
                print(f"  WARNING: Model is spread across {len(devices)} GPUs: {devices}")
                print(f"  This may cause synchronization issues. Consider using device_map='cuda:0'")

        # Check memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory after model loading:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                if allocated > 0 or reserved > 0:
                    print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    except Exception as e:
        print(f"  Warning: Could not get device information: {e}")

    # Set up output directory
    base_output_dir = config.get_output_dir("complete_artifacts")
    if args.output_subdir:
        base_output_dir = base_output_dir / args.output_subdir
    else:
        base_output_dir = base_output_dir / f"{args.dataset}_{args.split}"

    # Auto-add model suffix to output directory to prevent collisions between models
    if args.model is not None and args.model != "llama-3.1-8b-instruct":  # Only add suffix for non-default models
        model_short_name = args.model.split('-')[0]  # "deepseek-r1-..." → "deepseek"
        if model_short_name.lower() not in str(base_output_dir).lower():
            base_output_dir = Path(str(base_output_dir) + f"_{model_short_name}")
            print(f"\nAuto-renamed output dir to include model: {base_output_dir}")

    # Determine output mode based on custom assignments
    if custom_batch_assignments is not None:
        # Custom mode: will write to multiple shard directories
        # Create a work directory for this GPU's checkpoint
        output_dir = base_output_dir / f"gpu_{args.shard_id}_work"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCustom assignment mode - Work directory: {output_dir}")
        print(f"Results will be saved to original shard directories in: {base_output_dir}")

        # Checkpoint file for THIS GPU's work
        checkpoint_file = output_dir / "checkpoint.json"
    else:
        # Standard mode: single shard directory
        if args.shard_id is not None:
            output_dir = base_output_dir / f"shard_{args.shard_id}"
        else:
            output_dir = base_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        checkpoint_file = output_dir / "checkpoint.json"

    # Load existing checkpoint if available
    completed_batches = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                # Handle both int and string checkpoint keys
                completed_batches = set(checkpoint_data.get("completed_batches", []))
                print(f"\nResuming from checkpoint: {len(completed_batches)} batches already completed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            completed_batches = set()

    # In custom mode, also load checkpoints from all original shard directories
    if custom_batch_assignments is not None:
        print("\nLoading existing shard checkpoints to avoid re-processing...")
        for shard_id in range(args.num_shards):
            shard_dir = base_output_dir / f"shard_{shard_id}"
            shard_checkpoint = shard_dir / "checkpoint.json"
            if shard_checkpoint.exists():
                try:
                    with open(shard_checkpoint, "r") as f:
                        shard_data = json.load(f)
                        shard_completed = shard_data.get("completed_batches", [])
                        # Add these with shard prefix
                        for batch_idx in shard_completed:
                            completed_batches.add(f"shard_{shard_id}_batch_{batch_idx}")
                        if shard_completed:
                            print(f"  Shard {shard_id}: {len(shard_completed)} batches already completed")
                except Exception as e:
                    pass  # Ignore errors loading shard checkpoints

    # Window configuration (0%, 10%, ..., 100%)
    window_config = WindowConfig()

    # Run inference
    print("\n" + "=" * 80, flush=True)
    print("Running Batch Inference with Complete Artifact Capture", flush=True)
    print("=" * 80, flush=True)
    print(f"Starting batch processing loop with {num_batches} batches...", flush=True)
    sys.stdout.flush()

    all_results = []
    all_predictions = []
    all_ground_truths = []

    # Track consecutive failures for aggressive recovery
    consecutive_failures = 0
    max_consecutive_failures = 3

    # Process in batches
    import time
    print(f"Entering batch loop (batches: {num_batches})...", flush=True)
    sys.stdout.flush()

    for batch_iteration_idx in tqdm(range(num_batches), desc="Processing batches", ncols=100):
        # Determine which batch to process based on mode
        if custom_batch_assignments is not None:
            # Custom mode: get (shard_id, batch_idx) from assignments
            original_shard_id, original_batch_idx = custom_batch_assignments[batch_iteration_idx]

            # Use a composite checkpoint key: "shard_X_batch_Y"
            checkpoint_key = f"shard_{original_shard_id}_batch_{original_batch_idx}"

            # Skip if already completed
            if checkpoint_key in completed_batches or original_batch_idx in completed_batches:
                tqdm.write(f"Skipping batch {batch_iteration_idx + 1}/{num_batches} "
                          f"(shard {original_shard_id}, batch {original_batch_idx}) - already completed")
                continue

            # Extract samples from full dataset using original shard logic
            # Calculate which samples belong to this shard's batch
            samples_per_shard = (len(full_dataset) + args.num_shards - 1) // args.num_shards
            shard_samples = [s for i, s in enumerate(full_dataset) if i % args.num_shards == original_shard_id]

            # Get samples for this specific batch within the shard
            batch_start = original_batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(shard_samples))
            batch_samples = shard_samples[batch_start:batch_end]

            batch_start_time = time.time()
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Batch {batch_iteration_idx + 1}/{num_batches} | "
                      f"Shard {original_shard_id} Batch {original_batch_idx} | "
                      f"Questions {batch_start+1}-{batch_end}/{len(shard_samples)}")
            tqdm.write(f"{'='*80}")

        else:
            # Standard mode: sequential batch processing from sharded dataset
            batch_idx = batch_iteration_idx

            # Skip already completed batches
            if batch_idx in completed_batches:
                tqdm.write(f"Skipping batch {batch_idx + 1}/{num_batches} (already completed)")
                continue

            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_samples)
            batch_samples = samples[start_idx:end_idx]

            checkpoint_key = batch_idx

            batch_start_time = time.time()
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Batch {batch_idx + 1}/{num_batches} | Questions {start_idx+1}-{end_idx}/{total_samples}")
            tqdm.write(f"{'='*80}")

        max_retries = 2
        retry_count = 0
        batch_results = None

        while retry_count <= max_retries:
            try:
                # Process batch with detailed logging
                batch_results = process_batch(
                    batch_samples,
                    adapter,
                    args.model,
                    max_new_tokens=1024,
                    prompt_template="cot",
                    window_config=window_config,
                    log_fn=tqdm.write,
                )
                break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg

                if is_cuda_error and retry_count < max_retries:
                    retry_count += 1
                    # Determine display name based on mode
                    if custom_batch_assignments is not None:
                        display_name = f"shard {original_shard_id} batch {original_batch_idx}"
                    else:
                        display_name = f"batch {batch_idx + 1}"

                    tqdm.write(f"CUDA error in {display_name}, attempt {retry_count}/{max_retries + 1}: {error_msg}")
                    tqdm.write("Attempting recovery...")

                    # Aggressive recovery
                    try:
                        import torch
                        import time

                        # Try to clear any pending CUDA errors
                        try:
                            torch.cuda.synchronize()
                        except:
                            pass  # Ignore errors in sync during recovery

                        # Clear all caches
                        torch.cuda.empty_cache()
                        gc.collect()

                        # If we've had consecutive failures, try more aggressive recovery
                        if consecutive_failures >= 2:
                            tqdm.write("Multiple consecutive failures detected. Attempting aggressive GPU reset...")
                            # Try to reset all GPU statistics
                            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                                torch.cuda.reset_peak_memory_stats()
                            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                                torch.cuda.reset_accumulated_memory_stats()

                            # Longer wait for GPU to stabilize
                            time.sleep(5)
                        else:
                            time.sleep(2)

                    except Exception as recovery_error:
                        tqdm.write(f"Recovery error: {recovery_error}")

                    continue  # Retry the batch
                else:
                    # Non-CUDA error or max retries exceeded
                    # Determine display name based on mode
                    if custom_batch_assignments is not None:
                        display_name = f"shard {original_shard_id} batch {original_batch_idx}"
                    else:
                        display_name = f"batch {batch_idx + 1}"

                    tqdm.write(f"Error in {display_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    break  # Give up on this batch

        # Skip if no results after retries
        if batch_results is None:
            consecutive_failures += 1
            # Determine display name based on mode
            if custom_batch_assignments is not None:
                display_name = f"shard {original_shard_id} batch {original_batch_idx}"
            else:
                display_name = f"batch {batch_idx + 1}"

            tqdm.write(f"Skipping {display_name} after {retry_count} failed attempts")

            # Aggressive cleanup after failure
            try:
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass  # GPU may be in bad state
                    torch.cuda.empty_cache()
            except:
                pass

            # Check if we've had too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                tqdm.write(f"\n{'='*80}")
                tqdm.write(f"ERROR: {consecutive_failures} consecutive batch failures detected!")
                tqdm.write("GPU may be in an unrecoverable state. Consider:")
                tqdm.write("1. Restarting the script (it will resume from checkpoint)")
                tqdm.write("2. Checking GPU health with 'nvidia-smi'")
                tqdm.write("3. Reducing batch size")
                tqdm.write("4. Checking for hardware issues")
                tqdm.write(f"{'='*80}\n")
                # Continue anyway to try remaining batches
            continue

        # Success! Reset consecutive failure counter
        consecutive_failures = 0

        # Calculate batch accuracy before saving (for summary)
        batch_correct = sum(1 for _, r in batch_results if r["is_correct"])
        batch_total = len(batch_results)

        # Determine target directory for saving results
        if custom_batch_assignments is not None:
            # Custom mode: save to original shard directory
            target_shard_dir = base_output_dir / f"shard_{original_shard_id}"
            target_shard_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Standard mode: save to current output_dir
            target_shard_dir = output_dir

        # Save each output immediately and accumulate results
        for (output, result) in batch_results:
            # Save individual output to appropriate directory
            output_path = target_shard_dir / f"{result['id']}.json"
            save_generation_output(output, output_path, include_arrays=args.save_arrays)

            # Accumulate results
            all_results.append(result)
            all_predictions.append(result["predicted"])  # Keep None for proper extraction metrics
            all_ground_truths.append(result["ground_truth"])  # Keep None for proper extraction metrics

        # Aggressive memory cleanup after saving
        del batch_results
        del batch_samples

        # Force Python garbage collection
        gc.collect()

        # CUDA memory cleanup
        try:
            import torch
            # Synchronize to ensure all operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Empty cache
                torch.cuda.empty_cache()
                # Reset peak memory stats
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
        except Exception as cleanup_error:
            tqdm.write(f"Warning: CUDA cleanup error: {cleanup_error}")

        # Mark batch as completed and save checkpoint
        completed_batches.add(checkpoint_key)
        try:
            # Save our local checkpoint (tracks all work done by this GPU)
            checkpoint_list = [k if isinstance(k, int) else k for k in completed_batches]
            with open(checkpoint_file, "w") as f:
                json.dump({"completed_batches": checkpoint_list}, f)

            # In custom mode, also update the original shard's checkpoint
            if custom_batch_assignments is not None:
                shard_checkpoint_file = target_shard_dir / "checkpoint.json"
                shard_completed = set()

                # Load existing shard checkpoint
                if shard_checkpoint_file.exists():
                    try:
                        with open(shard_checkpoint_file, "r") as f:
                            shard_data = json.load(f)
                            shard_completed = set(shard_data.get("completed_batches", []))
                    except:
                        pass

                # Add this batch to shard's completed batches
                shard_completed.add(original_batch_idx)

                # Save updated shard checkpoint
                with open(shard_checkpoint_file, "w") as f:
                    json.dump({"completed_batches": list(shard_completed)}, f)

        except Exception as checkpoint_error:
            tqdm.write(f"Warning: Could not save checkpoint: {checkpoint_error}")

        # Print batch summary
        batch_time = time.time() - batch_start_time
        batch_acc = batch_correct / batch_total if batch_total > 0 else 0
        overall_acc = sum(r["is_correct"] for r in all_results) / len(all_results) if all_results else 0

        tqdm.write(f"  Batch complete: {batch_time:.1f}s | Batch accuracy: {batch_acc:.1%} | Overall: {overall_acc:.1%}")
        tqdm.write(f"  Progress: {len(all_results)}/{total_samples} questions ({len(all_results)/total_samples:.1%})")

    # Calculate and print overall metrics
    print("\n" + "=" * 80)

    # Check for extraction failures
    extraction_failures = sum(1 for r in all_results if r.get("extraction_failed", False))
    if extraction_failures > 0:
        print(f"\n⚠️  WARNING: {extraction_failures}/{len(all_results)} questions had answer extraction failures")
        print("These are counted as incorrect. Check logs above for details.")

    if all_predictions and all_ground_truths:
        metrics = calculate_metrics(all_predictions, all_ground_truths, task=dataset_format)
        print_metrics(metrics, title="Overall Results")

    # Save aggregated results
    results_file = output_dir / "aggregated_results.json"

    output_data = {
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "batch_size": args.batch_size,
            "total_samples": total_samples,
            "max_new_tokens": 1024,
            "save_arrays": args.save_arrays,
        },
        "metrics": metrics if all_predictions else {},
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failures / len(all_results) if all_results else 0,
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAggregated results saved to {results_file}")
    print(f"Individual outputs saved to {output_dir}/*.json")

    if not args.save_arrays:
        print("\nNote: hidden_states and logits_per_layer were NOT saved (use --save-arrays to include)")

    print("\n" + "=" * 80)
    print("Batch Inference Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
