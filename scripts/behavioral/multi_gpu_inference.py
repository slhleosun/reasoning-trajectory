"""Multi-GPU Batch Inference Launcher

This script launches multiple instances of batch_inference_complete.py in parallel,
each on a separate GPU, processing different shards of the dataset.

This achieves true parallel inference across multiple GPUs using data parallelism,
which is much more efficient than model parallelism for inference workloads.
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

def get_available_gpus():
    """Get list of available GPU IDs"""
    import torch
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def launch_shard_process(
    gpu_id,
    shard_id,
    num_shards,
    script_args,
    log_dir,
    batch_assignments_file=None,
):
    """Launch a single shard inference process on a specific GPU

    Args:
        gpu_id: GPU device ID to use
        shard_id: Shard ID (0-indexed)
        num_shards: Total number of shards
        script_args: Dictionary of arguments to pass to batch_inference_complete.py
        log_dir: Directory to save logs
        batch_assignments_file: Optional path to JSON file with specific batch assignments

    Returns:
        subprocess.Popen object
    """
    # Build command
    script_path = Path(__file__).parent / "batch_inference_complete.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--shard-id", str(shard_id),
        "--num-shards", str(num_shards),
    ]

    # Add batch assignments file if provided
    if batch_assignments_file is not None:
        cmd.extend(["--batch-assignments", str(batch_assignments_file)])

    # Add other arguments
    for key, value in script_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # Set up environment with single GPU visible
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set up log file
    log_file = log_dir / f"shard_{shard_id}_gpu_{gpu_id}.log"
    log_f = open(log_file, "w")

    print(f"[GPU {gpu_id}] Launching shard {shard_id}/{num_shards}")
    if batch_assignments_file:
        print(f"[GPU {gpu_id}] Using custom batch assignments: {batch_assignments_file}")
    print(f"[GPU {gpu_id}] Log file: {log_file}")
    print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")

    # Launch process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process, log_f


def merge_shard_results(output_dir, num_shards):
    """Merge results from all shards into a single aggregated results file

    Args:
        output_dir: Base output directory
        num_shards: Number of shards to merge
    """
    print("\n" + "=" * 80)
    print("Merging Results from All Shards")
    print("=" * 80)

    all_results = []
    combined_metrics = {
        "accuracy": 0,
        "total": 0,
    }

    # Load results from each shard
    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        results_file = shard_dir / "aggregated_results.json"

        if not results_file.exists():
            print(f"Warning: Results file not found for shard {shard_id}: {results_file}")
            continue

        print(f"Loading shard {shard_id}...")
        with open(results_file, "r") as f:
            shard_data = json.load(f)

        shard_results = shard_data.get("results", [])
        shard_metrics = shard_data.get("metrics", {})

        all_results.extend(shard_results)

        # Accumulate metrics
        if "accuracy" in shard_metrics:
            combined_metrics["accuracy"] += shard_metrics["accuracy"] * len(shard_results)
        combined_metrics["total"] += len(shard_results)

        print(f"  Shard {shard_id}: {len(shard_results)} results, accuracy: {shard_metrics.get('accuracy', 0):.2%}")

    # Calculate combined metrics
    if combined_metrics["total"] > 0:
        combined_metrics["accuracy"] /= combined_metrics["total"]

    # Save merged results
    merged_file = output_dir / "merged_results.json"
    merged_data = {
        "num_shards": num_shards,
        "total_results": len(all_results),
        "metrics": combined_metrics,
        "results": all_results,
    }

    with open(merged_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Merged Results Summary")
    print(f"{'=' * 80}")
    print(f"Total results: {len(all_results)}")
    print(f"Overall accuracy: {combined_metrics['accuracy']:.2%}")
    print(f"Merged results saved to: {merged_file}")
    print(f"{'=' * 80}\n")


def get_shard_progress(output_dir, shard_id):
    """Get progress information from a shard's checkpoint file

    Args:
        output_dir: Base output directory
        shard_id: Shard ID

    Returns:
        (completed_batches, total_batches) tuple, or (0, 0) if not available
    """
    try:
        shard_dir = output_dir / f"shard_{shard_id}"
        checkpoint_file = shard_dir / "checkpoint.json"

        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
                completed = len(data.get("completed_batches", []))
                return completed
        return 0
    except:
        return 0


def analyze_work_distribution(output_dir, num_shards, total_samples, batch_size):
    """Analyze work distribution across shards and detect imbalance

    Args:
        output_dir: Base output directory
        num_shards: Number of shards
        total_samples: Total samples in dataset
        batch_size: Batch size

    Returns:
        Tuple of (is_imbalanced, shard_stats, total_remaining_batches)
        where shard_stats is list of (shard_id, completed, total, remaining)
    """
    samples_per_shard = (total_samples + num_shards - 1) // num_shards
    batches_per_shard = (samples_per_shard + batch_size - 1) // batch_size

    shard_stats = []
    total_remaining = 0
    shards_with_work = 0

    for shard_id in range(num_shards):
        completed = get_shard_progress(output_dir, shard_id)
        remaining = batches_per_shard - completed

        shard_stats.append({
            'shard_id': shard_id,
            'completed': completed,
            'total': batches_per_shard,
            'remaining': remaining,
            'progress_pct': (completed / batches_per_shard * 100) if batches_per_shard > 0 else 100
        })

        total_remaining += remaining
        if remaining > 0:
            shards_with_work += 1

    # Detect imbalance: if work is concentrated in < 50% of shards
    is_imbalanced = False
    if total_remaining > 0 and shards_with_work < num_shards / 2:
        is_imbalanced = True

    return is_imbalanced, shard_stats, total_remaining


def collect_remaining_batches(output_dir, num_shards, total_samples, batch_size):
    """Collect all remaining batch indices across all shards

    Returns:
        List of (shard_id, batch_idx) tuples for all incomplete batches
    """
    samples_per_shard = (total_samples + num_shards - 1) // num_shards
    batches_per_shard = (samples_per_shard + batch_size - 1) // batch_size

    remaining_batches = []

    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        checkpoint_file = shard_dir / "checkpoint.json"

        completed_batches = set()
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)
                    completed_batches = set(data.get("completed_batches", []))
            except:
                pass

        # Collect incomplete batches
        for batch_idx in range(batches_per_shard):
            if batch_idx not in completed_batches:
                remaining_batches.append((shard_id, batch_idx))

    return remaining_batches


def redistribute_batches_to_gpus(remaining_batches, num_gpus):
    """Redistribute remaining batches evenly across all GPUs

    Args:
        remaining_batches: List of (shard_id, batch_idx) tuples
        num_gpus: Number of GPUs to distribute across

    Returns:
        Dict mapping gpu_idx -> list of (shard_id, batch_idx) assignments
    """
    assignments = {i: [] for i in range(num_gpus)}

    # Round-robin assignment
    for idx, batch_info in enumerate(remaining_batches):
        gpu_idx = idx % num_gpus
        assignments[gpu_idx].append(batch_info)

    return assignments


def monitor_processes(processes, log_files, output_dir, total_samples, batch_size, num_shards):
    """Monitor running processes and show progress

    Args:
        processes: List of (shard_id, gpu_id, Popen) tuples
        log_files: List of log file handles
        output_dir: Output directory to check for progress
        total_samples: Total number of samples in dataset
        batch_size: Batch size per GPU
        num_shards: Number of shards
    """
    print("\n" + "=" * 80)
    print("Multi-GPU Inference Progress")
    print("=" * 80)

    start_time = time.time()
    last_total_completed = 0
    samples_per_shard = (total_samples + num_shards - 1) // num_shards
    batches_per_shard = (samples_per_shard + batch_size - 1) // batch_size

    # Wait for all processes to complete
    while True:
        time.sleep(10)  # Check every 10 seconds

        running = []
        completed = []
        failed = []

        # Get process status
        for shard_id, gpu_id, process in processes:
            if process.poll() is None:
                running.append((shard_id, gpu_id))
            else:
                if process.returncode == 0:
                    completed.append((shard_id, gpu_id))
                else:
                    failed.append((shard_id, gpu_id, process.returncode))

        # Get actual progress from checkpoint files
        shard_progress = []
        total_completed_batches = 0
        for shard_id, gpu_id in running + completed:
            completed_batches = get_shard_progress(output_dir, shard_id)
            shard_progress.append((shard_id, gpu_id, completed_batches, batches_per_shard))
            total_completed_batches += completed_batches

        # Calculate statistics
        total_batches = batches_per_shard * num_shards
        total_completed_samples = total_completed_batches * batch_size
        # Don't count more than actual samples
        total_completed_samples = min(total_completed_samples, total_samples)

        progress_pct = (total_completed_samples / total_samples * 100) if total_samples > 0 else 0
        elapsed = time.time() - start_time

        # Calculate speed and ETA
        if total_completed_samples > 0 and elapsed > 0:
            samples_per_sec = total_completed_samples / elapsed
            remaining_samples = total_samples - total_completed_samples
            eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
            eta_min = eta_seconds / 60
        else:
            samples_per_sec = 0
            eta_min = 0

        # Only print if there's progress or it's been a while
        if total_completed_samples > last_total_completed or elapsed < 15:
            print(f"\rProgress: {total_completed_samples}/{total_samples} samples ({progress_pct:.1f}%) | "
                  f"Speed: {samples_per_sec:.1f} samples/s | "
                  f"Elapsed: {elapsed/60:.1f} min | "
                  f"ETA: {eta_min:.1f} min | "
                  f"Running: {len(running)}/{num_shards} GPUs", end="", flush=True)
            last_total_completed = total_completed_samples

        if failed:
            print(f"\n  ⚠ Failed shards: {[(f'GPU {g} shard {s}') for s, g, c in failed]}")

        # All done?
        if not running:
            print()  # New line after progress bar
            break

    # Close log files
    for log_f in log_files:
        log_f.close()

    # Final summary
    print("\n" + "=" * 80)
    print("All Processes Completed")
    print("=" * 80)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total samples: {total_samples}")
    print(f"Average speed: {total_samples/total_time:.1f} samples/second")
    print(f"Shards completed: {len(completed)}/{num_shards}")

    if failed:
        print(f"Shards failed: {len(failed)}/{num_shards}")
        for shard_id, gpu_id, code in failed:
            print(f"  - Shard {shard_id} (GPU {gpu_id}): exit code {code}")

    return len(failed) == 0


def main():
    """Main multi-GPU inference launcher"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Batch Inference Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all available GPUs
  python examples/multi_gpu_inference.py --dataset gsm8k --model llama-3.1-8b-instruct

  # Use specific GPUs
  python examples/multi_gpu_inference.py --gpus 0 1 2 3 --dataset gsm8k

  # Specify batch size and other parameters
  python examples/multi_gpu_inference.py --batch-size 32 --dataset gsm8k
        """
    )

    # Multi-GPU specific args
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: all available)")

    # Arguments to pass to batch_inference_complete.py
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for parallel processing")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Dataset name")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instruct",
                        help="Model name")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--save-arrays", action="store_true",
                        help="Save hidden_states and logits (WARNING: very large files)")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Subdirectory for output")

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-GPU Batch Inference Launcher")
    print("=" * 80)

    # Get GPUs to use
    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        print("Error: No GPUs available!")
        return 1

    num_gpus = len(gpu_ids)
    print(f"\nUsing {num_gpus} GPUs: {gpu_ids}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch size per GPU: {args.batch_size}")

    # Set up log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"multi_gpu_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Prepare arguments to pass to each shard
    script_args = {
        "batch-size": args.batch_size,
        "max-samples": args.max_samples,
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "save-arrays": args.save_arrays,
        "output-subdir": args.output_subdir,
    }

    # Launch processes
    print("\n" + "=" * 80)
    print("Launching Processes")
    print("=" * 80)

    processes = []
    log_files = []

    # Get dataset info for progress tracking
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src import load_config, prepare_dataset

    config = load_config()
    settings = config.get_settings()

    # Load dataset to get total sample count
    print("\nLoading dataset to determine size...")
    max_samples = args.max_samples or settings.get("max_samples")
    samples = prepare_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=max_samples,
    )
    total_samples = len(samples)
    print(f"Dataset size: {total_samples} samples")

    # Determine output directory (same logic as batch_inference_complete.py)
    base_output_dir = config.get_output_dir("complete_artifacts")
    if args.output_subdir:
        output_dir = base_output_dir / args.output_subdir
    else:
        output_dir = base_output_dir / f"{args.dataset}_{args.split}"

    # Auto-add model suffix to output directory to prevent collisions between models
    if args.model is not None and args.model != "llama-3.1-8b-instruct":  # Only add suffix for non-default models
        model_short_name = args.model.split('-')[0]  # "deepseek-r1-..." → "deepseek"
        if model_short_name.lower() not in str(output_dir).lower():
            output_dir = Path(str(output_dir) + f"_{model_short_name}")
            print(f"\nAuto-renamed output dir to include model: {output_dir}")

    # ===================================================================
    # CHECK FOR WORK IMBALANCE AND REDISTRIBUTE IF NECESSARY
    # ===================================================================

    print("\n" + "=" * 80)
    print("Analyzing Work Distribution")
    print("=" * 80)

    is_imbalanced, shard_stats, total_remaining = analyze_work_distribution(
        output_dir=output_dir,
        num_shards=num_gpus,
        total_samples=total_samples,
        batch_size=args.batch_size,
    )

    # Print shard progress
    print("\nCurrent shard progress:")
    for stat in shard_stats:
        print(f"  Shard {stat['shard_id']}: {stat['completed']}/{stat['total']} batches "
              f"({stat['progress_pct']:.1f}%) - {stat['remaining']} remaining")

    print(f"\nTotal remaining batches: {total_remaining}")

    # Decide whether to redistribute
    batch_assignment_files = None
    if total_remaining > 0 and is_imbalanced:
        print("\n⚠️  WORK IMBALANCE DETECTED!")
        print(f"   Work is concentrated in few shards. Redistributing across all {num_gpus} GPUs...")

        # Collect all remaining batches
        remaining_batches = collect_remaining_batches(
            output_dir=output_dir,
            num_shards=num_gpus,
            total_samples=total_samples,
            batch_size=args.batch_size,
        )

        print(f"   Collected {len(remaining_batches)} incomplete batches")

        # Redistribute across all GPUs
        gpu_assignments = redistribute_batches_to_gpus(remaining_batches, num_gpus)

        # Show redistribution
        print("\nRedistributed work:")
        for gpu_idx in range(num_gpus):
            num_batches = len(gpu_assignments[gpu_idx])
            print(f"  GPU {gpu_ids[gpu_idx]}: {num_batches} batches")

        # Save batch assignments to files
        batch_assignment_files = []
        assignments_dir = log_dir / "batch_assignments"
        assignments_dir.mkdir(exist_ok=True)

        for gpu_idx in range(num_gpus):
            assignment_file = assignments_dir / f"gpu_{gpu_idx}_assignments.json"
            with open(assignment_file, "w") as f:
                json.dump(gpu_assignments[gpu_idx], f, indent=2)
            batch_assignment_files.append(assignment_file)

        print(f"\nBatch assignments saved to: {assignments_dir}")

    elif total_remaining == 0:
        print("\n✓ All work completed! Nothing to process.")
        print("\n" + "=" * 80)
        print("Multi-GPU Inference Complete!")
        print("=" * 80)

        # Merge results if not already done
        merged_file = output_dir / "merged_results.json"
        if not merged_file.exists():
            merge_shard_results(output_dir, num_gpus)
        else:
            print(f"Results already merged: {merged_file}")

        return 0
    else:
        print("\n✓ Work distribution is balanced. Using standard shard assignment.")

    # ===================================================================
    # LAUNCH PROCESSES
    # ===================================================================

    for i, gpu_id in enumerate(gpu_ids):
        assignment_file = batch_assignment_files[i] if batch_assignment_files else None

        process, log_f = launch_shard_process(
            gpu_id=gpu_id,
            shard_id=i,
            num_shards=num_gpus,
            script_args=script_args,
            log_dir=log_dir,
            batch_assignments_file=assignment_file,
        )
        processes.append((i, gpu_id, process))
        log_files.append(log_f)

    # Monitor processes
    success = monitor_processes(
        processes=processes,
        log_files=log_files,
        output_dir=output_dir,
        total_samples=total_samples,
        batch_size=args.batch_size,
        num_shards=num_gpus,
    )

    if not success:
        print("\nSome processes failed. Check log files for details.")
        return 1

    merge_shard_results(output_dir, num_gpus)

    print("\n" + "=" * 80)
    print("Multi-GPU Inference Complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
