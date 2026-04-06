#!/usr/bin/env python
"""Analyze linear probe transfer results

Reads JSON files from apply_linear_probes.py and creates comprehensive reports
showing average and best accuracy for each step across different probe transfers.

Usage:
    python scripts/predictors/analyze_probe_transfer_results.py \
        --input-dir output/linear_probe_transfer
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


def load_results(json_path: Path) -> List[Dict]:
    """Load results from JSON file

    Returns:
        List of result dicts with keys: target, layer_idx, accuracy, f1, auc, n_samples, etc.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_single_file(json_path: Path) -> Dict:
    """Analyze a single probe transfer result file

    Returns:
        Dict with analysis results for each target (step_1, ..., step_5, hash)
    """
    results = load_results(json_path)

    # Organize by target
    target_results = defaultdict(list)
    for result in results:
        target = result['target']
        target_results[target].append(result)

    # Compute statistics for each target
    analysis = {}

    for target in ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']:
        if target not in target_results:
            analysis[target] = {
                'avg_accuracy': None,
                'best_accuracy': None,
                'best_layer': None,
                'num_layers': 0
            }
            continue

        target_data = target_results[target]
        accuracies = [r['accuracy'] for r in target_data]

        # Find best
        best_idx = np.argmax(accuracies)
        best_result = target_data[best_idx]

        analysis[target] = {
            'avg_accuracy': np.mean(accuracies),
            'best_accuracy': best_result['accuracy'],
            'best_layer': best_result['layer_idx'],
            'num_layers': len(target_data)
        }

    return analysis


def parse_filename(filename: str) -> Tuple[str, str]:
    """Parse filename to extract probe_from and eval_on model names

    Args:
        filename: e.g., "base_on_instruct.json" or "probe_transfer_results_base_on_inst.json"

    Returns:
        (probe_from, eval_on) tuple
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Remove common prefixes
    name = name.replace('probe_transfer_results_', '')

    # Split by '_on_'
    if '_on_' in name:
        parts = name.split('_on_')
        return parts[0], parts[1]
    else:
        return name, "unknown"


def print_comprehensive_report(results_dir: Path):
    """Print comprehensive report for all probe transfer results

    Args:
        results_dir: Directory containing JSON files
    """
    # Find all JSON files
    json_files = sorted(results_dir.glob('*.json'))

    if len(json_files) == 0:
        print(f"\nNo JSON files found in {results_dir}")
        return

    print("\n" + "="*120)
    print("LINEAR PROBE TRANSFER ANALYSIS")
    print("="*120)
    print(f"Results directory: {results_dir}")
    print(f"Number of transfer experiments: {len(json_files)}")
    print("="*120)

    # Analyze each file
    all_analyses = {}

    for json_file in json_files:
        probe_from, eval_on = parse_filename(json_file.name)
        print(f"\nAnalyzing: {json_file.name}")
        print(f"  Probe From: {probe_from}")
        print(f"  Eval. On:   {eval_on}")

        analysis = analyze_single_file(json_file)
        all_analyses[json_file.name] = {
            'probe_from': probe_from,
            'eval_on': eval_on,
            'analysis': analysis
        }

    # Print detailed results for each transfer
    print("\n" + "="*120)
    print("DETAILED RESULTS BY TRANSFER")
    print("="*120)

    targets = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'hash']
    target_names = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Final Answer']

    for json_file in json_files:
        entry = all_analyses[json_file.name]
        probe_from = entry['probe_from']
        eval_on = entry['eval_on']
        analysis = entry['analysis']

        print(f"\n{'='*120}")
        print(f"Probe From: {probe_from} | Eval. On: {eval_on}")
        print(f"{'='*120}")

        # Print table header
        header = f"{'Target':<20}{'Avg Accuracy':<20}{'Best Accuracy':<20}{'Best Layer':<15}{'# Layers':<15}"
        print(header)
        print("-" * 120)

        # Print each target
        for target, target_name in zip(targets, target_names):
            stats = analysis[target]

            if stats['avg_accuracy'] is not None:
                avg_acc = f"{stats['avg_accuracy']:.4f}"
                best_acc = f"{stats['best_accuracy']:.4f}"
                best_layer = f"{stats['best_layer']}"
                num_layers = f"{stats['num_layers']}"
            else:
                avg_acc = "N/A"
                best_acc = "N/A"
                best_layer = "N/A"
                num_layers = "0"

            row = f"{target_name:<20}{avg_acc:<20}{best_acc:<20}{best_layer:<15}{num_layers:<15}"
            print(row)

        # Print overall average
        valid_avgs = [stats['avg_accuracy'] for stats in analysis.values()
                     if stats['avg_accuracy'] is not None]
        if valid_avgs:
            overall_avg = np.mean(valid_avgs)
            print("-" * 120)
            print(f"{'OVERALL AVERAGE':<20}{overall_avg:<20.4f}")

    # Print comparison table (all transfers side by side)
    print("\n" + "="*120)
    print("COMPARISON TABLE: AVERAGE ACCURACY ACROSS ALL TRANSFERS")
    print("="*120)

    # Create header with short names
    header = f"{'Target':<20}"
    file_names = [all_analyses[f]['probe_from'] + '→' + all_analyses[f]['eval_on']
                  for f in [jf.name for jf in json_files]]

    for fname in file_names:
        header += f"{fname:<15}"
    print(header)
    print("-" * 120)

    # Print each target row
    for target, target_name in zip(targets, target_names):
        row = f"{target_name:<20}"

        for json_file in json_files:
            analysis = all_analyses[json_file.name]['analysis']
            stats = analysis[target]

            if stats['avg_accuracy'] is not None:
                row += f"{stats['avg_accuracy']:<15.4f}"
            else:
                row += f"{'N/A':<15}"

        print(row)

    # Print overall average row
    print("-" * 120)
    overall_row = f"{'OVERALL':<20}"

    for json_file in json_files:
        analysis = all_analyses[json_file.name]['analysis']
        valid_avgs = [stats['avg_accuracy'] for stats in analysis.values()
                     if stats['avg_accuracy'] is not None]

        if valid_avgs:
            overall_avg = np.mean(valid_avgs)
            overall_row += f"{overall_avg:<15.4f}"
        else:
            overall_row += f"{'N/A':<15}"

    print(overall_row)
    print("="*120)

    # Print comparison table for BEST accuracy
    print("\n" + "="*120)
    print("COMPARISON TABLE: BEST ACCURACY ACROSS ALL TRANSFERS")
    print("="*120)

    # Create header
    header = f"{'Target':<20}"
    for fname in file_names:
        header += f"{fname:<15}"
    print(header)
    print("-" * 120)

    # Print each target row
    for target, target_name in zip(targets, target_names):
        row = f"{target_name:<20}"

        for json_file in json_files:
            analysis = all_analyses[json_file.name]['analysis']
            stats = analysis[target]

            if stats['best_accuracy'] is not None:
                row += f"{stats['best_accuracy']:<15.4f}"
            else:
                row += f"{'N/A':<15}"

        print(row)

    # Print overall best average row
    print("-" * 120)
    overall_row = f"{'OVERALL':<20}"

    for json_file in json_files:
        analysis = all_analyses[json_file.name]['analysis']
        valid_bests = [stats['best_accuracy'] for stats in analysis.values()
                      if stats['best_accuracy'] is not None]

        if valid_bests:
            overall_best_avg = np.mean(valid_bests)
            overall_row += f"{overall_best_avg:<15.4f}"
        else:
            overall_row += f"{'N/A':<15}"

    print(overall_row)
    print("="*120)

    # Print best layers table
    print("\n" + "="*120)
    print("BEST LAYER FOR EACH TARGET ACROSS ALL TRANSFERS")
    print("="*120)

    # Create header
    header = f"{'Target':<20}"
    for fname in file_names:
        header += f"{fname:<15}"
    print(header)
    print("-" * 120)

    # Print each target row
    for target, target_name in zip(targets, target_names):
        row = f"{target_name:<20}"

        for json_file in json_files:
            analysis = all_analyses[json_file.name]['analysis']
            stats = analysis[target]

            if stats['best_layer'] is not None:
                layer_str = f"L{stats['best_layer']:02d}"
                row += f"{layer_str:<15}"
            else:
                row += f"{'N/A':<15}"

        print(row)

    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze linear probe transfer results"
    )
    parser.add_argument("--input-dir", type=Path,
                       default=Path("output/linear_probe_transfer"),
                       help="Directory containing JSON files from apply_linear_probes.py")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"\nError: Input directory not found: {args.input_dir}")
        print("\nPlease run apply_linear_probes.py first to generate the results.")
        return 1

    print_comprehensive_report(args.input_dir)

    print("\n" + "="*120)
    print("ANALYSIS COMPLETE!")
    print("="*120 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
