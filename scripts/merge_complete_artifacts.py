#!/usr/bin/env python
"""Merge individual JSON files from shard directories into a single merged directory

This script copies all JSON files from multiple shard directories
into a single merged/ directory for use with steering vector collection.

Usage:
    # Merge from gsm8k_train_final shards (default pattern matches all .json files)
    python scripts/merge_complete_artifacts.py --input output/complete_artifacts/gsm8k_train_final

    # Merge from math-500_test shards
    python scripts/merge_complete_artifacts.py --input output/complete_artifacts/math-500_test --delete-shards

    # Merge with specific pattern (for backwards compatibility)
    python scripts/merge_complete_artifacts.py --input output/complete_artifacts/gsm8k_train --pattern "gsm8k_*.json"
"""

import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def collect_shard_files(base_dir: Path, pattern: str = "*.json"):
    """Collect all JSON files from shard directories

    Args:
        base_dir: Base directory containing shard_N subdirectories
        pattern: Glob pattern for files to collect (default: "*.json")

    Returns:
        List of (source_path, filename) tuples
    """
    files_to_merge = []

    # Find all shard directories
    shard_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("shard_")])

    if not shard_dirs:
        print(f"Warning: No shard directories found in {base_dir}")
        return files_to_merge

    print(f"\nFound {len(shard_dirs)} shard directories:")
    for shard_dir in shard_dirs:
        print(f"  {shard_dir.name}")

    # Collect files from each shard
    print(f"\nScanning for JSON files (pattern: {pattern})...")
    for shard_dir in shard_dirs:
        # Look for files matching the pattern
        json_files = list(shard_dir.glob(pattern))

        for json_file in json_files:
            files_to_merge.append((json_file, json_file.name))

        print(f"  {shard_dir.name}: {len(json_files)} files")

    return files_to_merge


def merge_files(files_to_merge, merged_dir: Path, overwrite: bool = False):
    """Copy files to merged directory

    Args:
        files_to_merge: List of (source_path, filename) tuples
        merged_dir: Target merged directory
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (copied, skipped, errors) counts
    """
    # Create merged directory
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMerging {len(files_to_merge)} files to {merged_dir}...")

    copied = 0
    skipped = 0
    errors = 0

    for source_path, filename in tqdm(files_to_merge, desc="Copying files", ncols=100):
        target_path = merged_dir / filename

        # Check if file already exists
        if target_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            shutil.copy2(source_path, target_path)
            copied += 1
        except Exception as e:
            print(f"\nError copying {source_path}: {e}")
            errors += 1

    print(f"\n{'='*80}")
    print("MERGE SUMMARY")
    print(f"{'='*80}")
    print(f"Total files found: {len(files_to_merge)}")
    print(f"Files copied: {copied}")
    print(f"Files skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nMerged directory: {merged_dir}")
    print(f"{'='*80}\n")

    return copied, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Merge individual JSON files from shard directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge GSM8K Final Answer version (all .json files)
  python scripts/merge_complete_artifacts.py --input output/complete_artifacts/gsm8k_train_final

  # Merge MATH-500 test shards and delete after merge
  python scripts/merge_complete_artifacts.py --input output/complete_artifacts/math-500_test --delete-shards

  # Merge with specific pattern
  python scripts/merge_complete_artifacts.py --input output/complete_artifacts/gsm8k_train --pattern "gsm8k_*.json"

  # Overwrite existing files
  python scripts/merge_complete_artifacts.py --input output/complete_artifacts/gsm8k_train_final --overwrite
        """
    )

    parser.add_argument("--input", type=Path, required=True,
                        help="Base directory containing shard_N subdirectories")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output merged directory (default: <input>/merged)")
    parser.add_argument("--pattern", type=str, default="*.json",
                        help="Glob pattern for files to collect (default: *.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files in merged directory")
    parser.add_argument("--delete-shards", action="store_true",
                        help="Delete original shard directories after successful merge")

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    if not args.input.is_dir():
        print(f"Error: Input path is not a directory: {args.input}")
        return 1

    # Set output directory
    if args.output is None:
        args.output = args.input / "merged"

    print("=" * 80)
    print("MERGE COMPLETE ARTIFACTS")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"File pattern: {args.pattern}")
    print(f"Overwrite mode: {args.overwrite}")
    print(f"Delete shards after merge: {args.delete_shards}")

    # Collect files from shards
    files_to_merge = collect_shard_files(args.input, pattern=args.pattern)

    if not files_to_merge:
        print("\nNo files found to merge!")
        return 1

    # Merge files
    copied, skipped, errors = merge_files(files_to_merge, args.output, overwrite=args.overwrite)

    # Delete shard directories if requested and merge was successful
    if args.delete_shards:
        if errors > 0:
            print(f"\n⚠️  WARNING: {errors} errors occurred during merge.")
            print("Shard directories will NOT be deleted to prevent data loss.")
            print("Please check the errors above and re-run after fixing issues.")
        else:
            print("\n" + "=" * 80)
            print("DELETING SHARD DIRECTORIES")
            print("=" * 80)

            # Find all shard directories
            shard_dirs = sorted([d for d in args.input.iterdir()
                               if d.is_dir() and d.name.startswith("shard_")])

            print(f"\nFound {len(shard_dirs)} shard directories to delete:")
            for shard_dir in shard_dirs:
                print(f"  {shard_dir.name}")

            print(f"\nDeleting shard directories...")
            deleted = 0
            delete_errors = 0

            for shard_dir in tqdm(shard_dirs, desc="Deleting shards", ncols=100):
                try:
                    shutil.rmtree(shard_dir)
                    deleted += 1
                except Exception as e:
                    print(f"\nError deleting {shard_dir}: {e}")
                    delete_errors += 1

            print(f"\n{'='*80}")
            print("DELETION SUMMARY")
            print(f"{'='*80}")
            print(f"Shard directories deleted: {deleted}")
            print(f"Deletion errors: {delete_errors}")
            print(f"{'='*80}\n")

            if delete_errors > 0:
                print("⚠️  Some shard directories could not be deleted. Check errors above.")

    print("✓ Merge complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
