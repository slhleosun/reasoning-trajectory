#!/usr/bin/env python
"""Multi-GPU launcher for collecting steering vectors

Distributes question collection across multiple GPUs and merges results.

Usage:
    python scripts/steering/multi_gpu_collect_steering.py --num-questions 8000 --merged-dir output/complete_artifacts/gsm8k_train_r1_deepseek/merged --num-gpus 7 --model deepseek-r1-distill-llama-8b --output reasoning-traj-sync/output/steering_vectors/steering_vectors_r1-distill-llama-8b_8000.npz
"""

import sys
import os
import argparse
import subprocess
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np


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
    """Launch a single shard process on a specific GPU"""
    script_path = Path(__file__).parent / "collect_steering_vectors.py"

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
    print("Multi-GPU Steering Vector Collection Progress")
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
              f"Running: {len(running)}/{num_shards} GPUs", end="", flush=True)

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


def merge_shard_results(output_dir: Path, num_shards: int, output_path: Path, shard_filename: str = "steering_vectors.npz") -> bool:
    """Merge all shard results into a single file"""
    print("\n" + "=" * 80)
    print("MERGING SHARD RESULTS")
    print("=" * 80)

    all_steering_vectors = []
    all_step_activations = None
    all_hash_activations = None
    total_stats = {
        "total_questions_attempted": 0,
        "successful_questions": 0,
        "failed_no_json": 0,
        "failed_missing_fields": 0,
        "failed_no_positions": 0,
        "total_differences": 0
    }

    num_layers = None
    hidden_dim = None
    merged_task = "gsm8k"
    merged_qid_to_stem = {}

    # Collect data from all shards
    print(f"\nCollecting data from {num_shards} shards...")
    print(f"Looking for: {shard_filename}")
    for shard_id in range(num_shards):
        shard_path = output_dir / f"shard_{shard_id}" / shard_filename

        if not shard_path.exists():
            print(f"  ⚠ Warning: Shard {shard_id} not found: {shard_path}")
            continue

        print(f"  Loading shard {shard_id}...")
        data = np.load(shard_path, allow_pickle=True)

        # Get metadata
        if num_layers is None:
            num_layers = int(data["num_layers"])
            hidden_dim = int(data["hidden_dim"])

            # Initialize activation lists
            all_step_activations = [[] for _ in range(num_layers)]
            all_hash_activations = [[] for _ in range(num_layers)]
            all_step_numbers = [[] for _ in range(num_layers)]
            all_question_ids_step = [[] for _ in range(num_layers)]
            all_question_ids_hash = [[] for _ in range(num_layers)]
            all_is_correct_step = [[] for _ in range(num_layers)]
            all_is_correct_hash = [[] for _ in range(num_layers)]

        # Collect steering vectors (we'll recompute the trimmed mean later)
        # For now, we need to collect all the raw differences, but we don't have them
        # So instead, we'll collect all the raw activations and recompute

        # Load activations
        step_acts = data["step_activations"]
        hash_acts = data["hash_activations"]
        step_nums = data.get("step_numbers", None)  # May not exist in old files
        question_ids_step = data.get("question_ids_step", None)
        question_ids_hash = data.get("question_ids_hash", None)
        is_correct_step = data.get("is_correct_step", None)
        is_correct_hash = data.get("is_correct_hash", None)

        for layer_idx in range(num_layers):
            if len(step_acts[layer_idx]) > 0:
                all_step_activations[layer_idx].append(step_acts[layer_idx])
            if len(hash_acts[layer_idx]) > 0:
                all_hash_activations[layer_idx].append(hash_acts[layer_idx])
            if step_nums is not None and len(step_nums[layer_idx]) > 0:
                all_step_numbers[layer_idx].append(step_nums[layer_idx])
            if question_ids_step is not None and len(question_ids_step[layer_idx]) > 0:
                all_question_ids_step[layer_idx].append(question_ids_step[layer_idx])
            if question_ids_hash is not None and len(question_ids_hash[layer_idx]) > 0:
                all_question_ids_hash[layer_idx].append(question_ids_hash[layer_idx])
            if is_correct_step is not None and len(is_correct_step[layer_idx]) > 0:
                all_is_correct_step[layer_idx].append(is_correct_step[layer_idx])
            if is_correct_hash is not None and len(is_correct_hash[layer_idx]) > 0:
                all_is_correct_hash[layer_idx].append(is_correct_hash[layer_idx])

        # Merge stats
        if "stats" in data:
            stats = json.loads(str(data["stats"]))
            for key in total_stats.keys():
                if key in stats:
                    total_stats[key] += stats[key]

        # Collect task and question_id_to_stem
        if "task" in data:
            merged_task = str(data["task"])
        if "question_id_to_stem" in data:
            try:
                shard_mapping = json.loads(str(data["question_id_to_stem"]))
                merged_qid_to_stem.update(shard_mapping)
            except (json.JSONDecodeError, TypeError):
                pass

        print(f"    → Loaded activations")

    if num_layers is None:
        print("\n❌ Error: No data found in any shard!")
        return False

    # Concatenate activations (optimized with list comprehensions)
    print(f"\nConcatenating activations...")
    all_step_activations = [
        np.concatenate(all_step_activations[i], axis=0) if all_step_activations[i] else np.array([])
        for i in range(num_layers)
    ]
    all_hash_activations = [
        np.concatenate(all_hash_activations[i], axis=0) if all_hash_activations[i] else np.array([])
        for i in range(num_layers)
    ]
    all_step_numbers = [
        np.concatenate(all_step_numbers[i], axis=0) if all_step_numbers[i] else np.array([], dtype=np.int32)
        for i in range(num_layers)
    ]
    all_question_ids_step = [
        np.concatenate(all_question_ids_step[i], axis=0) if all_question_ids_step[i] else np.array([], dtype=np.int64)
        for i in range(num_layers)
    ]
    all_question_ids_hash = [
        np.concatenate(all_question_ids_hash[i], axis=0) if all_question_ids_hash[i] else np.array([], dtype=np.int64)
        for i in range(num_layers)
    ]
    all_is_correct_step = [
        np.concatenate(all_is_correct_step[i], axis=0) if all_is_correct_step[i] else np.array([], dtype=np.bool_)
        for i in range(num_layers)
    ]
    all_is_correct_hash = [
        np.concatenate(all_is_correct_hash[i], axis=0) if all_is_correct_hash[i] else np.array([], dtype=np.bool_)
        for i in range(num_layers)
    ]

    # Recompute steering vectors from all activations (on GPU)
    print(f"\nRecomputing steering vectors with trimmed mean on GPU...")
    import torch

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    def trimmed_mean_by_norm_gpu(vectors_tensor, trim_proportion=0.1):
        """Compute trimmed mean on GPU using PyTorch"""
        if len(vectors_tensor) == 0:
            return torch.zeros(hidden_dim, device=device)
        if len(vectors_tensor) == 1:
            return vectors_tensor[0]

        # Compute norms
        norms = torch.norm(vectors_tensor, dim=1)
        sorted_indices = torch.argsort(norms)

        n_vectors = len(vectors_tensor)
        n_trim = int(n_vectors * trim_proportion)

        if n_trim > 0:
            keep_indices = sorted_indices[n_trim:-n_trim]
        else:
            keep_indices = sorted_indices

        # Trim and compute mean
        trimmed_vectors = vectors_tensor[keep_indices]
        return torch.mean(trimmed_vectors, dim=0)

    steering_vectors = np.zeros((num_layers, hidden_dim), dtype=np.float32)

    # Pre-build question ID mapping once (shared across all layers)
    print(f"  Building question ID mappings...")
    step_qids = all_question_ids_step[0]  # Same for all layers
    hash_qids = all_question_ids_hash[0]  # Same for all layers

    # Build hash_qid -> index mapping (NumPy-based for speed)
    hash_dict = {}
    for idx, qid in enumerate(hash_qids):
        if qid not in hash_dict:
            hash_dict[qid] = idx  # Use first occurrence only

    # Collect matching pairs as NumPy arrays
    step_indices = []
    hash_indices = []
    for step_idx, step_qid in enumerate(step_qids):
        if step_qid in hash_dict:
            step_indices.append(step_idx)
            hash_indices.append(hash_dict[step_qid])

    step_indices = np.array(step_indices, dtype=np.int64)
    hash_indices = np.array(hash_indices, dtype=np.int64)

    print(f"  Found {len(step_indices)} matching step-hash pairs")

    if len(step_indices) == 0:
        print("  ⚠ Warning: No matching pairs found!")
        return False

    # Convert indices to torch once
    step_indices_gpu = torch.from_numpy(step_indices).to(device)
    hash_indices_gpu = torch.from_numpy(hash_indices).to(device)

    # Process all layers on GPU
    print(f"  Processing {num_layers} layers on GPU...")
    for layer_idx in range(num_layers):
        step_acts = all_step_activations[layer_idx]
        hash_acts = all_hash_activations[layer_idx]

        if len(step_acts) == 0 or len(hash_acts) == 0:
            continue

        # Move to GPU and index
        step_acts_gpu = torch.from_numpy(step_acts).to(device)
        hash_acts_gpu = torch.from_numpy(hash_acts).to(device)

        # Compute differences on GPU (vectorized)
        matched_steps = step_acts_gpu[step_indices_gpu]
        matched_hashes = hash_acts_gpu[hash_indices_gpu]
        differences = matched_hashes - matched_steps

        # Compute trimmed mean on GPU
        steering_vec = trimmed_mean_by_norm_gpu(differences, trim_proportion=0.1)

        # Move back to CPU
        steering_vectors[layer_idx] = steering_vec.cpu().numpy()

        # Print progress
        if (layer_idx + 1) % 8 == 0 or (layer_idx + 1) == num_layers:
            print(f"    → {layer_idx + 1}/{num_layers} layers processed")

    print(f"  ✓ All {num_layers} layers processed")

    # Print statistics
    print("\n" + "=" * 80)
    print("MERGED STATISTICS")
    print("=" * 80)
    print(f"Questions attempted: {total_stats['total_questions_attempted']}")
    print(f"Successful questions: {total_stats['successful_questions']} ({total_stats['successful_questions']/max(total_stats['total_questions_attempted'], 1)*100:.1f}%)")
    print(f"\nFailure breakdown:")
    print(f"  No JSON file: {total_stats['failed_no_json']}")
    print(f"  Missing fields: {total_stats['failed_missing_fields']}")
    print(f"  No Step/#### positions: {total_stats['failed_no_positions']}")
    print(f"\nComputed data:")
    print(f"  Total differences: {total_stats['total_differences']}")
    print(f"  Steering vectors shape: {steering_vectors.shape}")

    # Save merged data
    print(f"\nSaving merged results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        steering_vectors=steering_vectors,
        step_activations=all_step_activations,
        hash_activations=all_hash_activations,
        step_numbers=all_step_numbers,
        question_ids_step=all_question_ids_step,
        question_ids_hash=all_question_ids_hash,
        is_correct_step=all_is_correct_step,
        is_correct_hash=all_is_correct_hash,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        stats=json.dumps(total_stats),
        task=merged_task,
        question_id_to_stem=json.dumps(merged_qid_to_stem)
    )

    print(f"✓ Merged data saved successfully!")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Cleanup shards
    print("\n" + "-" * 80)
    print("CLEANING UP SHARD FILES")
    print("-" * 80)

    for shard_id in range(num_shards):
        shard_dir = output_dir / f"shard_{shard_id}"
        if shard_dir.exists():
            print(f"  Deleting shard_{shard_id}...")
            shutil.rmtree(shard_dir)

    print(f"✓ All shard directories deleted")

    print("\n" + "=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU steering vector collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 4 GPUs, 100 questions
  python scripts/multi_gpu_collect_steering.py --num-questions 100 --num-gpus 4

  # Use specific GPUs
  python scripts/multi_gpu_collect_steering.py --gpus 0 1 2 3 --num-questions 100
        """
    )

    # Multi-GPU specific args
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: all available)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (alternative to --gpus)")

    # Arguments to pass to collect_steering_vectors.py
    parser.add_argument("--merged-dir", type=Path,
                        default=Path("output/complete_artifacts/gsm8k_train/merged"))
    parser.add_argument("--model", type=str, default=None,
                        help="Model key from config (e.g., 'deepseek-r1-distill-llama-8b'). If not specified, uses default from config.")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("output/steering_vectors.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", type=str, default="hash", choices=["hash", "final"],
                        help="Template mode: 'hash' for #### (default) or 'final' for Final Answer:")

    args = parser.parse_args()

    # Auto-add model suffix to output filename to prevent collisions
    if args.model is not None:
        # Extract model name from model key (e.g., "deepseek-r1-distill-llama-8b" -> "deepseek")
        model_short_name = args.model.split('-')[0] if '-' in args.model else args.model
        # Check if output filename already has model name
        if model_short_name.lower() not in args.output.stem.lower():
            # Add model suffix before extension
            args.output = args.output.parent / f"{args.output.stem}_{model_short_name}{args.output.suffix}"
            print(f"Auto-renamed output to include model: {args.output}")

    # Auto-adjust paths for final template
    if args.template == "final":
        # Add _final suffix to merged-dir if using default and not already present
        default_merged_dir = Path("output/complete_artifacts/gsm8k_train/merged")
        if args.merged_dir == default_merged_dir:
            args.merged_dir = Path("output/complete_artifacts/gsm8k_train_final/merged")

        # Add _final suffix to output path if not already present
        if "_final" not in str(args.output):
            args.output = args.output.parent / (args.output.stem + "_final" + args.output.suffix)

    marker_name = "####" if args.template == "hash" else "Final Answer:"

    print("=" * 80)
    print(f"Multi-GPU Steering Vector Collection Launcher - {marker_name}")
    print("=" * 80)
    print(f"Template mode: {args.template} ({marker_name})")

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
    print(f"Total questions: {args.num_questions}")
    print(f"Questions per GPU: ~{args.num_questions // num_gpus}")

    # Set up log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"multi_gpu_steering_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Prepare arguments to pass to each shard
    script_args = {
        "merged-dir": args.merged_dir,
        "num-questions": args.num_questions,
        "output": args.output,
        "seed": args.seed,
        "template": args.template,
    }

    # Add model argument if specified
    if args.model is not None:
        script_args["model"] = args.model

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
    # Determine shard filename based on template
    shard_filename = args.output.name  # This will be "steering_vectors_final.npz" if template is final
    merge_success = merge_shard_results(
        output_dir=args.output.parent,
        num_shards=num_gpus,
        output_path=args.output,
        shard_filename=shard_filename,
    )

    if not merge_success:
        print("\n❌ Failed to merge shard results. Shard files preserved.")
        return 1

    print("\n" + "=" * 80)
    print("Multi-GPU Collection Complete!")
    print("=" * 80)
    print(f"\nLogs saved to: {log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())