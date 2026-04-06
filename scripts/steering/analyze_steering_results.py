#!/usr/bin/env python
"""Analyze steering intervention results and create plots

Plots accuracy, reasoning length, and num_steps vs alpha for each intervention mode.

IMPORTANT: This script RE-EXTRACTS answers and RE-EVALUATES correctness using robust methods
instead of trusting the values in the JSON files. It also recalculates step counts using
the updated method (counting only before #### marker).
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import evaluate_answer

# Set up plotting style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8


def extract_answer_after_hash(text: str) -> Optional[str]:
    """Extract answer after #### marker - ROBUST VERSION

    Rules (as specified):
    1. Find first #### occurrence in generated response
    2. Look for first number after ####
    3. If number has spaces between digits (like "118 000"), concatenate them
    4. If first number is followed by operators (+-*/), look for first number after "=" instead
    5. If number after = is still followed by operators, recursively look for next =
    6. FALLBACK: If no #### found OR extraction fails, extract last number from entire response

    Examples:
    - "#### 24" → "24"
    - "#### 118 000" → "118000"
    - "#### $240 + $56 = $296" → "296"
    - "#### -3 degrees" → "-3"
    - "#### The final answer is 260" → "260"
    - "The answer is 42 dollars" (no ####) → "42"
    - "#### (no number)" → last number in entire text

    Args:
        text: Generated text

    Returns:
        Extracted answer (always returns a number if any exist in text)
    """
    if not text:
        return None

    # Helper function: Extract last number from text as fallback
    def extract_last_number(txt: str) -> Optional[str]:
        all_numbers = re.findall(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', txt)
        if all_numbers:
            return all_numbers[-1].replace(' ', '').replace(',', '')
        return None

    if "####" not in text:
        # FALLBACK: No #### found, extract last number from entire text
        return extract_last_number(text)

    # Step 1: Find first #### occurrence
    parts = text.split("####", 1)
    if len(parts) < 2:
        # FALLBACK: Split failed, extract last number from entire text
        return extract_last_number(text)

    answer_text = parts[1].strip()

    # Remove leading currency symbols
    answer_text = re.sub(r'^[\$£€¥₹]+\s*', '', answer_text)

    # Step 2: Find first number after ####
    # Pattern matches numbers with optional negative, spaces/commas as separators
    # Handles: "24", "118 000", "-3", "1,234.56"
    first_number_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', answer_text)

    if not first_number_match:
        # FALLBACK: No number found after ####, extract last number from entire text
        return extract_last_number(text)

    # Check what comes immediately after the first number (skip whitespace)
    first_number_end = first_number_match.end()
    after_number = answer_text[first_number_end:].lstrip()

    # Step 4: If first number is followed by operators, it's an equation
    # Recursively look for the number after "=" that's NOT followed by operators
    if after_number and after_number[0] in '+-*/':
        # It's an equation, find all = signs and check each one
        remaining_text = answer_text
        while '=' in remaining_text:
            equals_parts = remaining_text.split('=', 1)
            if len(equals_parts) < 2:
                break

            result_text = equals_parts[1].strip()
            # Remove currency symbols from result
            result_text = re.sub(r'^[\$£€¥₹]+\s*', '', result_text)
            # Extract first number after this =
            result_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', result_text)

            if result_match:
                # Check what comes after this number
                result_number_end = result_match.end()
                after_result = result_text[result_number_end:].lstrip()

                # If NOT followed by operator, this is the final answer
                if not after_result or after_result[0] not in '+-*/':
                    # Step 3: Concatenate numbers with spaces (remove spaces and commas)
                    return result_match.group(1).replace(' ', '').replace(',', '')

                # Still followed by operator, look for next =
                # Move to the part after this =
                remaining_text = equals_parts[1]
            else:
                break

        # FALLBACK: Equation processing didn't find final answer, use last number from entire text
        return extract_last_number(text)

    # Not an equation, return the first number found
    # Step 3: Concatenate numbers with spaces (remove spaces and commas)
    return first_number_match.group(1).replace(' ', '').replace(',', '')


def count_reasoning_steps(text: str) -> int:
    """Count number of reasoning steps in generated text

    NEW METHOD: Counts "Step" tokens ONLY in the reasoning portion (before #### marker).
    This ensures we don't count spurious steps that may appear after the final answer.

    Algorithm:
    1. Split text by "####" marker
    2. Count "Step N:" or "Step N." patterns only in the part BEFORE ####
    3. Use case-insensitive matching

    Examples:
        "Step 1: ... Step 6: ... #### $18. Step 7: ..." → 6 (not 7)
        "Step 1: ... Step 2: ..." (no ####) → 2
        "No steps #### 42" → 0

    Args:
        text: Generated text

    Returns:
        Number of reasoning steps (int >= 0)
    """
    # Split by #### and only count steps in the reasoning part (before ####)
    if "####" in text:
        reasoning_part = text.split("####")[0]
    else:
        reasoning_part = text

    # Match "Step N:" or "Step N." patterns (case-insensitive)
    pattern = r'\bStep\s+\d+\s*[:.]'
    matches = re.findall(pattern, reasoning_part, re.IGNORECASE)
    return len(matches)


def load_results(base_dir: Path, mode: str) -> Dict[float, dict]:
    """Load all results for a given mode and RE-COMPUTE metrics using robust extraction

    IMPORTANT: This function re-extracts answers and re-evaluates correctness using
    robust methods instead of trusting the values in the JSON files.

    Returns:
        Dict mapping alpha -> summary data (with re-computed metrics)
    """
    mode_dir = base_dir / mode
    if not mode_dir.exists():
        return {}

    results = {}
    for json_file in sorted(mode_dir.glob("results_alpha*.json")):
        # Extract alpha from filename
        alpha_str = json_file.stem.replace("results_alpha", "")
        alpha = float(alpha_str)

        with open(json_file, "r") as f:
            data = json.load(f)

        # Re-compute summary statistics from raw results using robust extraction
        if "results" in data:
            summary = compute_summary_with_robust_extraction(data["results"])
            results[alpha] = summary
        elif "summary" in data:
            # Fallback: if no raw results, use existing summary (with warning)
            print(f"  WARNING: No raw results found in {json_file.name}, using existing summary")
            results[alpha] = data["summary"]

    return results


def compute_summary_with_robust_extraction(raw_results: List[dict]) -> dict:
    """Compute summary statistics from raw results using ROBUST extraction methods

    Re-extracts answers, re-evaluates correctness, and recalculates steps using
    the updated robust methods.

    Args:
        raw_results: List of result dictionaries with baseline/intervened generated text

    Returns:
        Summary dict with same structure as original summaries
    """
    n_total = len(raw_results)

    # Initialize counters
    baseline_correct = 0
    intervened_correct = 0

    baseline_lengths = []
    intervened_lengths = []

    baseline_steps = []
    intervened_steps = []

    # Flip tracking
    stayed_correct = 0
    correct_to_wrong = 0
    wrong_to_correct = 0
    stayed_wrong = 0

    for result in raw_results:
        gold_answer = result["gold_answer"]

        # Extract text from nested structure: result["baseline"]["produced_text"]
        baseline_text = result.get("baseline", {}).get("produced_text", "")
        intervened_text = result.get("intervened", {}).get("produced_text", "")

        # RE-EXTRACT answers using robust method
        baseline_predicted = extract_answer_after_hash(baseline_text)
        intervened_predicted = extract_answer_after_hash(intervened_text)

        # RE-EVALUATE correctness
        baseline_is_correct = evaluate_answer(baseline_predicted, gold_answer) if baseline_predicted else False
        intervened_is_correct = evaluate_answer(intervened_predicted, gold_answer) if intervened_predicted else False

        # Count correct
        if baseline_is_correct:
            baseline_correct += 1
        if intervened_is_correct:
            intervened_correct += 1

        # Track flips
        if baseline_is_correct and intervened_is_correct:
            stayed_correct += 1
        elif baseline_is_correct and not intervened_is_correct:
            correct_to_wrong += 1
        elif not baseline_is_correct and intervened_is_correct:
            wrong_to_correct += 1
        else:
            stayed_wrong += 1

        # Get lengths from nested structure: result["baseline"]["reasoning_length"]
        baseline_lengths.append(result.get("baseline", {}).get("reasoning_length", 0))
        intervened_lengths.append(result.get("intervened", {}).get("reasoning_length", 0))

        # RE-CALCULATE steps using updated method (count only before ####)
        baseline_steps.append(count_reasoning_steps(baseline_text))
        intervened_steps.append(count_reasoning_steps(intervened_text))

    # Compute statistics
    baseline_acc = baseline_correct / n_total * 100
    intervened_acc = intervened_correct / n_total * 100
    acc_change = intervened_acc - baseline_acc

    baseline_avg_len = np.mean(baseline_lengths)
    intervened_avg_len = np.mean(intervened_lengths)
    len_change = intervened_avg_len - baseline_avg_len

    baseline_avg_steps = np.mean(baseline_steps)
    intervened_avg_steps = np.mean(intervened_steps)
    steps_change = intervened_avg_steps - baseline_avg_steps

    # Create summary dict matching original structure
    summary = {
        "accuracy": {
            "baseline": baseline_acc,
            "intervened": intervened_acc,
            "change": acc_change
        },
        "reasoning_length": {
            "baseline_avg": baseline_avg_len,
            "intervened_avg": intervened_avg_len,
            "avg_change": len_change,
            "baseline_std": np.std(baseline_lengths),
            "intervened_std": np.std(intervened_lengths)
        },
        "num_steps": {
            "baseline_avg": baseline_avg_steps,
            "intervened_avg": intervened_avg_steps,
            "avg_change": steps_change,
            "baseline_std": np.std(baseline_steps),
            "intervened_std": np.std(intervened_steps)
        },
        "flips": {
            "stayed_correct": stayed_correct,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
            "stayed_wrong": stayed_wrong
        },
        "n_samples": n_total
    }

    return summary


def print_mode_summary(mode: str, results: Dict[float, dict]):
    """Print summary statistics for a mode"""
    print(f"\n{'='*80}")
    print(f"{mode} SUMMARY (with ROBUST RE-EXTRACTION & RE-EVALUATION)")
    print(f"{'='*80}\n")

    if not results:
        print("  No results found.\n")
        return

    print(f"Found results for {len(results)} alpha values: {sorted(results.keys())}")
    print(f"Metrics computed using robust extraction (handles spaces, equations, negatives, fallbacks)\n")

    # Create table
    print(f"{'Alpha':<8} {'Baseline Acc':<14} {'Interv Acc':<14} {'Acc Δ':<10} "
          f"{'Length Δ':<12} {'Steps Δ':<10}")
    print("-" * 80)

    for alpha in sorted(results.keys()):
        summary = results[alpha]

        baseline_acc = summary["accuracy"]["baseline"]
        interv_acc = summary["accuracy"]["intervened"]
        acc_change = summary["accuracy"]["change"]

        length_change = summary["reasoning_length"]["avg_change"]
        steps_change = summary["num_steps"]["avg_change"]

        # Format with colors (using + prefix for increases)
        acc_delta_str = f"{acc_change:+.2f}%"
        length_delta_str = f"{length_change:+.1f}"
        steps_delta_str = f"{steps_change:+.2f}"

        print(f"{alpha:<8.1f} {baseline_acc:<14.2f} {interv_acc:<14.2f} "
              f"{acc_delta_str:<10} {length_delta_str:<12} {steps_delta_str:<10}")

    print()


def plot_metric_comparison(
    modes: List[str],
    mode_results: Dict[str, Dict[float, dict]],
    metric_key: str,
    metric_name: str,
    ylabel: str,
    output_path: Path
):
    """Plot a single metric across modes

    Args:
        modes: List of mode names
        mode_results: Dict mapping mode -> (alpha -> summary)
        metric_key: Key in summary dict (e.g., "accuracy", "reasoning_length")
        metric_name: Human-readable metric name
        ylabel: Y-axis label
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'BOOST_LAST_N': '#2E86AB',  # Blue
        'SUPPRESS_LAST_N': '#A23B72',  # Purple
    }

    for mode in modes:
        results = mode_results.get(mode, {})
        if not results:
            continue

        alphas = sorted(results.keys())

        # Extract baseline and intervened values
        baseline_vals = []
        intervened_vals = []
        changes = []

        for alpha in alphas:
            summary = results[alpha]

            if metric_key == "accuracy":
                baseline_vals.append(summary["accuracy"]["baseline"])
                intervened_vals.append(summary["accuracy"]["intervened"])
                changes.append(summary["accuracy"]["change"])
            elif metric_key == "reasoning_length":
                baseline_vals.append(summary["reasoning_length"]["baseline_avg"])
                intervened_vals.append(summary["reasoning_length"]["intervened_avg"])
                changes.append(summary["reasoning_length"]["avg_change"])
            elif metric_key == "num_steps":
                baseline_vals.append(summary["num_steps"]["baseline_avg"])
                intervened_vals.append(summary["num_steps"]["intervened_avg"])
                changes.append(summary["num_steps"]["avg_change"])

        color = colors.get(mode, '#000000')

        # Left plot: Baseline and Intervened
        ax1.plot(alphas, baseline_vals, 'o--', color=color, alpha=0.5,
                label=f'{mode} (Baseline)')
        ax1.plot(alphas, intervened_vals, 'o-', color=color,
                label=f'{mode} (Intervened)')

        # Right plot: Change (Δ)
        ax2.plot(alphas, changes, 'o-', color=color, label=mode)

    # Left plot formatting
    ax1.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric_name}: Baseline vs Intervened', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Right plot formatting
    ax2.set_xlabel('Alpha (Steering Strength)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Δ {ylabel}', fontsize=12, fontweight='bold')
    ax2.set_title(f'{metric_name}: Change from Baseline', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_flip_analysis(
    modes: List[str],
    mode_results: Dict[str, Dict[float, dict]],
    output_path: Path
):
    """Plot flip analysis using pie charts showing baseline categories and transitions"""

    for mode in modes:
        results = mode_results.get(mode, {})
        if not results:
            continue

        alphas = sorted(results.keys())

        # Create a grid of pie charts (one per alpha)
        n_alphas = len(alphas)
        n_cols = 5  # 5 columns
        n_rows = (n_alphas + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{mode.replace("_", " ")} - Answer Correctness Transitions',
                    fontsize=16, fontweight='bold', y=0.995)

        # Color scheme: shades showing original state and transitions
        # Dark green: stayed correct (good)
        # Light orange: correct→wrong (bad flip)
        # Light green: wrong→correct (good flip)
        # Dark red: stayed wrong (bad)
        colors = {
            'stayed_correct': '#2D6A4F',      # Dark green
            'correct_to_wrong': '#E76F51',    # Orange-red (bad flip)
            'wrong_to_correct': '#52B788',    # Light green (good flip)
            'stayed_wrong': '#9D0208',        # Dark red
        }

        for idx, alpha in enumerate(alphas):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            summary = results[alpha]
            flips = summary.get("flips", {})

            # Get counts
            stayed_correct = flips.get('stayed_correct', 0)
            correct_to_wrong = flips.get('correct_to_wrong', 0)
            wrong_to_correct = flips.get('wrong_to_correct', 0)
            stayed_wrong = flips.get('stayed_wrong', 0)

            total = stayed_correct + correct_to_wrong + wrong_to_correct + stayed_wrong

            # Data for pie chart
            sizes = [stayed_correct, correct_to_wrong, wrong_to_correct, stayed_wrong]
            labels = [
                f'Stayed Correct\n{stayed_correct} ({stayed_correct/total*100:.1f}%)',
                f'Correct→Wrong\n{correct_to_wrong} ({correct_to_wrong/total*100:.1f}%)',
                f'Wrong→Correct\n{wrong_to_correct} ({wrong_to_correct/total*100:.1f}%)',
                f'Stayed Wrong\n{stayed_wrong} ({stayed_wrong/total*100:.1f}%)'
            ]
            pie_colors = [colors['stayed_correct'], colors['correct_to_wrong'],
                         colors['wrong_to_correct'], colors['stayed_wrong']]

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=pie_colors,
                autopct='',
                startangle=90,
                textprops={'fontsize': 8, 'weight': 'bold'}
            )

            # Make percentage text white for dark colors
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)

            ax.set_title(f'α = {alpha:.1f}', fontsize=11, fontweight='bold', pad=10)

        # Remove empty subplots
        for idx in range(n_alphas, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        # Save mode-specific pie chart
        mode_output = output_path.parent / f"flip_analysis_pie_{mode}.png"
        plt.savefig(mode_output, dpi=300, bbox_inches='tight')
        print(f"  Saved: {mode_output}")
        plt.close()

    # Create a combined legend figure
    fig_legend = plt.figure(figsize=(10, 3))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')

    # Create legend patches
    legend_elements = [
        mpatches.Patch(facecolor='#2D6A4F', edgecolor='black', label='Stayed Correct (Baseline Correct → Still Correct)'),
        mpatches.Patch(facecolor='#E76F51', edgecolor='black', label='Correct → Wrong (Baseline Correct → Now Wrong)'),
        mpatches.Patch(facecolor='#52B788', edgecolor='black', label='Wrong → Correct (Baseline Wrong → Now Correct)'),
        mpatches.Patch(facecolor='#9D0208', edgecolor='black', label='Stayed Wrong (Baseline Wrong → Still Wrong)'),
    ]

    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, frameon=True,
                    title='Color Legend', title_fontsize=14)

    legend_path = output_path.parent / "flip_analysis_pie_legend.png"
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {legend_path}")
    plt.close()


def create_summary_table(
    modes: List[str],
    mode_results: Dict[str, Dict[float, dict]],
    output_path: Path
):
    """Create a comprehensive summary table plot"""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')

    # Prepare table data
    table_data = []
    header = ['Mode', 'Alpha', 'Baseline Acc', 'Interv Acc', 'Acc Δ',
              'Baseline Len', 'Interv Len', 'Len Δ',
              'Baseline Steps', 'Interv Steps', 'Steps Δ']
    table_data.append(header)

    for mode in modes:
        results = mode_results.get(mode, {})
        if not results:
            continue

        for alpha in sorted(results.keys()):
            summary = results[alpha]

            row = [
                mode.replace('_', ' '),
                f"{alpha:.1f}",
                f"{summary['accuracy']['baseline']:.2f}",
                f"{summary['accuracy']['intervened']:.2f}",
                f"{summary['accuracy']['change']:+.2f}",
                f"{summary['reasoning_length']['baseline_avg']:.1f}",
                f"{summary['reasoning_length']['intervened_avg']:.1f}",
                f"{summary['reasoning_length']['avg_change']:+.1f}",
                f"{summary['num_steps']['baseline_avg']:.2f}",
                f"{summary['num_steps']['intervened_avg']:.2f}",
                f"{summary['num_steps']['avg_change']:+.2f}",
            ]
            table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.06] + [0.09]*9)

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')

    plt.title('Steering Intervention Results Summary',
             fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def analyze_intervention_set(base_dir: Path, modes: List[str], output_dir: Path, set_name: str):
    """Analyze a set of intervention results and create plots

    Args:
        base_dir: Base directory containing mode subdirectories
        modes: List of mode names to analyze
        output_dir: Where to save plots
        set_name: Name of this intervention set (for display)
    """
    print("\n" + "="*80)
    print(f"{set_name.upper()} ANALYSIS")
    print("="*80)
    print(f"\nBase directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\n⚠️  USING ROBUST RE-EXTRACTION:")
    print(f"   - Re-extracting answers from generated text (handles spaces, equations, negatives)")
    print(f"   - Re-evaluating correctness using standardized methods")
    print(f"   - Recalculating steps (counting only before #### marker)")
    print(f"   - NOT trusting values from JSON files\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results for all modes
    mode_results = {}

    for mode in modes:
        results = load_results(base_dir, mode)
        mode_results[mode] = results
        print_mode_summary(mode, results)

    # Skip plotting if no results found
    if not any(mode_results.values()):
        print(f"\nNo results found for {set_name}. Skipping plots.\n")
        return

    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")

    # Accuracy plot
    plot_metric_comparison(
        modes, mode_results,
        metric_key="accuracy",
        metric_name="Accuracy",
        ylabel="Accuracy (%)",
        output_path=output_dir / "accuracy_comparison.png"
    )

    # Reasoning length plot
    plot_metric_comparison(
        modes, mode_results,
        metric_key="reasoning_length",
        metric_name="Reasoning Length",
        ylabel="Reasoning Length (tokens)",
        output_path=output_dir / "reasoning_length_comparison.png"
    )

    # Number of steps plot
    plot_metric_comparison(
        modes, mode_results,
        metric_key="num_steps",
        metric_name="Number of Reasoning Steps",
        ylabel="Number of Steps",
        output_path=output_dir / "num_steps_comparison.png"
    )

    # Flip analysis plot
    plot_flip_analysis(
        modes, mode_results,
        output_path=output_dir / "flip_analysis.png"
    )

    # Summary table
    create_summary_table(
        modes, mode_results,
        output_path=output_dir / "summary_table.png"
    )

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80 + "\n")

    # Analyze key findings
    for mode in modes:
        results = mode_results.get(mode, {})
        if not results:
            continue

        print(f"\n{mode}:")
        print("-" * 40)

        # Find best and worst alpha for accuracy
        acc_changes = {alpha: summary["accuracy"]["change"]
                      for alpha, summary in results.items()}

        best_alpha = max(acc_changes.keys(), key=lambda k: acc_changes[k])
        worst_alpha = min(acc_changes.keys(), key=lambda k: acc_changes[k])

        print(f"  Best α for accuracy: {best_alpha:.1f} ({acc_changes[best_alpha]:+.2f}%)")
        print(f"  Worst α for accuracy: {worst_alpha:.1f} ({acc_changes[worst_alpha]:+.2f}%)")

        # Average effects across all alphas
        avg_acc_change = np.mean(list(acc_changes.values()))
        avg_len_change = np.mean([s["reasoning_length"]["avg_change"]
                                 for s in results.values()])
        avg_steps_change = np.mean([s["num_steps"]["avg_change"]
                                   for s in results.values()])

        print(f"\n  Average effects across all α:")
        print(f"    Accuracy Δ: {avg_acc_change:+.2f}%")
        print(f"    Reasoning Length Δ: {avg_len_change:+.1f} tokens")
        print(f"    Num Steps Δ: {avg_steps_change:+.2f} steps")

    print("\n" + "="*80)
    print(f"{set_name.upper()} ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        print(f"  - {plot_file.name}")
    print()


def main():
    print("\n" + "="*100)
    print("STEERING INTERVENTION ANALYSIS - COMPREHENSIVE (with ROBUST RE-EXTRACTION)")
    print("="*100)
    print("\n⚠️  IMPORTANT: This analysis RE-COMPUTES all metrics using robust methods:")
    print("   - Answers are re-extracted from generated text (not trusted from JSON)")
    print("   - Correctness is re-evaluated using standardized evaluation")
    print("   - Steps are recalculated using updated method (counts only before ####)")
    print("   - All plots reflect the re-computed metrics")
    print("="*100)

    # ========== ORIGINAL MODE: Compare multiple modes together ==========
    # Hash-Step Steering Interventions (original)
    # original_base_dir = Path("output/steering_interventions")
    original_base_dir = Path("output/steering_interventions_test")
    if original_base_dir.exists():
        original_modes = ["BOOST_LAST_N", "SUPPRESS_LAST_N", "BOOST_MID_N", "SUPPRESS_MID_N"]
        original_output_dir = original_base_dir / "analysis_plots"

        analyze_intervention_set(
            base_dir=original_base_dir,
            modes=original_modes,  # Multiple modes compared together
            output_dir=original_output_dir,
            set_name="Hash-Step Steering Interventions"
        )

    # ========== NEW MODE: Analyze each mode separately + combined comparison ==========
    # step2_step1_base_dir = Path("output/step1_step2_interventions")
    # if step2_step1_base_dir.exists():
    #     # Discover all mode subdirectories
    #     discovered_modes = []
    #     for mode_dir in step2_step1_base_dir.iterdir():
    #         if mode_dir.is_dir() and not mode_dir.name.startswith('.'):
    #             discovered_modes.append(mode_dir.name)

    #     if discovered_modes:
    #         print(f"\n\nDiscovered Step2-Step1 modes: {discovered_modes}")

    #         # 1. Analyze each mode separately (plots saved to mode subfolder)
    #         for mode in discovered_modes:
    #             mode_dir = step2_step1_base_dir / mode
    #             mode_output_dir = mode_dir  # Save plots IN the mode directory

    #             analyze_intervention_set(
    #                 base_dir=step2_step1_base_dir,
    #                 modes=[mode],  # Analyze one mode at a time
    #                 output_dir=mode_output_dir,
    #                 set_name=f"Step2-Step1 - {mode}"
    #             )

    #         # 2. Also create combined comparison plots (all modes together)
    #         combined_output_dir = step2_step1_base_dir / "analysis_plots"
    #         analyze_intervention_set(
    #             base_dir=step2_step1_base_dir,
    #             modes=discovered_modes,  # All modes compared together
    #             output_dir=combined_output_dir,
    #             set_name="Step2-Step1 - Combined Comparison"
    #         )

    print("\n" + "="*100)
    print("ALL ANALYSIS COMPLETE!")
    print("="*100)
    print()


if __name__ == "__main__":
    sys.exit(main())
