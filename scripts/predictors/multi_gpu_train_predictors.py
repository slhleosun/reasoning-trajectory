#!/usr/bin/env python
"""Multi-GPU launcher for predictor training

Distributes classifier training across multiple GPUs using subprocess isolation.

Usage:
    # Train all classifiers across 4 GPUs
    python scripts/predictors/multi_gpu_train_predictors.py \
        --mode all \
        --num-gpus 4

    # Quick mode (odd layers only) with specific GPUs
    python scripts/predictors/multi_gpu_train_predictors.py \
        --mode all \
        --quick \
        --gpus 0 1 2 3
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
):
    """Launch a single shard process on a specific GPU

    Each shard will train a subset of classifiers determined by:
    shard_id and num_shards (distributed round-robin)
    """
    script_path = Path(__file__).parent / "train_predictors.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--shard-id", str(shard_id),
        "--num-shards", str(num_shards),
    ]

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
    print(f"[GPU {gpu_id}] Log file: {log_file}")

    # Launch process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process, log_f


def monitor_processes(processes, log_files, num_shards):
    """Monitor running processes and show progress"""
    print("\n" + "=" * 80)
    print("Multi-GPU Predictor Training Progress")
    print("=" * 80)

    start_time = time.time()

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

        elapsed = time.time() - start_time

        # Print progress
        print(f"\rElapsed: {elapsed/60:.1f} min | "
              f"Completed: {len(completed)}/{num_shards} | "
              f"Running: {len(running)} GPUs", end="", flush=True)

        if failed:
            print(f"\n  ⚠ Failed shards: {[(f'GPU {g} shard {s} (exit code {c})') for s, g, c in failed]}")

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
    print(f"Shards completed: {len(completed)}/{num_shards}")

    if failed:
        print(f"Shards failed: {len(failed)}/{num_shards}")
        for shard_id, gpu_id, code in failed:
            print(f"  - Shard {shard_id} (GPU {gpu_id}): exit code {code}")

    return len(failed) == 0


def merge_shard_results(output_dir: Path, num_shards: int) -> bool:
    """Merge all shard results into a single summary file (appending to existing)

    Args:
        output_dir: Output directory containing shard subdirectories
        num_shards: Number of shards to merge
    """
    print("\n" + "=" * 80)
    print("MERGING SHARD RESULTS")
    print("=" * 80)

    # Load existing summary if it exists
    merged_summary_path = output_dir / "summary.json"
    existing_summary = []

    if merged_summary_path.exists():
        try:
            with open(merged_summary_path, 'r') as f:
                existing_summary = json.load(f)
            print(f"\n📝 Found existing summary with {len(existing_summary)} entries")
        except (json.JSONDecodeError, IOError) as e:
            print(f"\n⚠ Could not load existing summary ({e}), starting fresh")
            existing_summary = []

    # Collect new results from all shards
    print(f"\nCollecting results from {num_shards} shards...")
    new_results = []

    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        summary_path = shard_dir / "summary.json"

        if not summary_path.exists():
            print(f"  ⚠ Warning: Shard {shard_id} summary not found: {summary_path}")
            continue

        print(f"  Loading shard {shard_id}...")
        with open(summary_path, "r") as f:
            shard_data = json.load(f)

        new_results.extend(shard_data)
        print(f"    → {len(shard_data)} classifiers")

    if not new_results:
        print("\n❌ Error: No results found in any shard!")
        return False

    # Create a set of new keys for efficient lookup
    new_keys = {entry['key'] for entry in new_results}

    # Keep existing entries that are NOT being updated in this run
    preserved_entries = [entry for entry in existing_summary if entry['key'] not in new_keys]

    # Combine preserved + new entries
    all_results = preserved_entries + new_results

    # Sort results by key
    all_results.sort(key=lambda x: x['key'])

    # Save merged summary
    print(f"\nSaving merged summary to {merged_summary_path}...")
    with open(merged_summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Merged summary saved!")
    print(f"  Total classifiers: {len(all_results)} ({len(preserved_entries)} existing + {len(new_results)} new/updated)")

    # Merge NPZ files from each shard
    print("\nMerging individual classifier NPZ files...")
    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        if not shard_dir.exists():
            continue

        # Move all NPZ files to main output dir
        for npz_file in shard_dir.glob("*.npz"):
            dest_file = output_dir / npz_file.name
            if not dest_file.exists():
                npz_file.rename(dest_file)

    print(f"✓ NPZ files merged!")

    # Cleanup shard directories
    print("\nCleaning up shard directories...")
    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        if shard_dir.exists():
            # Remove directory and contents
            import shutil
            shutil.rmtree(shard_dir)
            print(f"  Deleted shard_{shard_id}/")

    print(f"✓ All shard directories deleted")

    print("\n" + "=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU predictor training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all classifiers across 4 GPUs
  python scripts/predictors/multi_gpu_train_predictors.py --mode all --num-gpus 4

  # Quick mode (odd layers only) with specific GPUs
  python scripts/predictors/multi_gpu_train_predictors.py --mode all --quick --gpus 0 1 2 3

  # Single feature set across multiple GPUs
  python scripts/predictors/multi_gpu_train_predictors.py \\
      --mode single --feature-set step1_step2 --label-type correctness --num-gpus 4
        """
    )

    # Multi-GPU specific args
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: all available)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (alternative to --gpus)")

    # Training args (passed to train_predictors.py)
    parser.add_argument("--npz-path", type=Path,
                        default=Path("output/steering_vectors/steering_vectors_llama-3.1-8b-instruct_8000.npz"),
                        help="Path to steering vectors NPZ file")
    parser.add_argument("--error-json", type=Path,
                        default=Path("output/error_annotations/gsm8k_train_errors.json"),
                        help="Path to error annotations JSON")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/predictor_results"),
                        help="Output directory for results")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "single"],
                        help="Training mode: 'all' or 'single'")
    parser.add_argument("--feature-set", type=str, default="step1_step2",
                        choices=["step1_step2", "step2_minus_step1", "step1_step2_step3", "step_diffs", "hash_only", "hash_minus_last", "hash_pca", "hash_last_diffs_pca", "hash_last_diffs_pca_joint", "pca_concat", "pca_diff"],
                        help="Feature set (for single mode)")
    parser.add_argument("--label-type", type=str, default="correctness",
                        choices=["correctness", "error_type"],
                        help="Label type (for single mode)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to train (for single mode)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Test set size (default: 0.1)")
    parser.add_argument("--model-type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Model architecture: 'linear' for logistic regression (default) or 'mlp' for small MLP")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only train on odd-numbered layers")

    args = parser.parse_args()

    print("=" * 80)
    print("Multi-GPU Predictor Training Launcher")
    print("=" * 80)

    # Get GPUs to use
    if args.gpus is not None:
        gpu_ids = args.gpus
    elif args.num_gpus is not None:
        available_gpus = get_available_gpus()
        gpu_ids = available_gpus[:args.num_gpus]
    else:
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        print("Error: No GPUs available!")
        return 1

    num_gpus = len(gpu_ids)
    print(f"\nUsing {num_gpus} GPUs: {gpu_ids}")
    print(f"Mode: {args.mode}")
    print(f"Model type: {args.model_type.upper()} ({'Logistic Regression' if args.model_type == 'linear' else 'MLP (1 hidden layer, 128 units)'})")
    if args.quick:
        print(f"Quick mode: ENABLED (odd layers only)")
    if args.mode == "single":
        print(f"Feature set: {args.feature_set}")
        print(f"Label type: {args.label_type}")

    # Set up log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"multi_gpu_predictors_{args.mode}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Prepare arguments to pass to each shard
    # Note: We DON'T pass num-workers here - each shard runs single-threaded on its GPU
    script_args = {
        "npz-path": args.npz_path,
        "error-json": args.error_json,
        "output-dir": args.output_dir,
        "mode": args.mode,
        "test-size": args.test_size,
        "model-type": args.model_type,
        "num-workers": 1,  # Each shard uses 1 worker (single GPU)
    }

    # Add mode-specific args
    if args.mode == "single":
        script_args["feature-set"] = args.feature_set
        if args.layer is not None:
            script_args["layer"] = args.layer

    # IMPORTANT: Pass label-type for ALL modes (not just single)
    # In "all" mode, this specifies which label type to train for all feature sets
    script_args["label-type"] = args.label_type

    if args.quick:
        script_args["quick"] = True

    # Launch processes
    print("\n" + "=" * 80)
    print("Launching Processes")
    print("=" * 80)

    processes = []
    log_files = []

    for i, gpu_id in enumerate(gpu_ids):
        process, log_f = launch_shard_process(
            gpu_id=gpu_id,
            shard_id=i,
            num_shards=num_gpus,
            script_args=script_args,
            log_dir=log_dir,
        )
        processes.append((i, gpu_id, process))
        log_files.append(log_f)

    # Monitor processes
    success = monitor_processes(
        processes=processes,
        log_files=log_files,
        num_shards=num_gpus,
    )

    if not success:
        print("\n⚠ Some processes failed. Check log files for details:")
        for i, gpu_id in enumerate(gpu_ids):
            log_file = log_dir / f"shard_{i}_gpu_{gpu_id}.log"
            print(f"  {log_file}")
        return 1

    # Merge shard results
    merge_success = merge_shard_results(
        output_dir=args.output_dir,
        num_shards=num_gpus,
    )

    if not merge_success:
        print("\n❌ Failed to merge shard results. Shard files preserved.")
        return 1

    print("\n" + "=" * 80)
    print("Multi-GPU Training Complete!")
    print("=" * 80)
    print(f"\nLogs saved to: {log_dir}")
    print(f"Results saved to: {args.output_dir}/summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
