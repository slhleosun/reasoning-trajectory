"""Utility functions"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to JSONL file

    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def last_boxed_only_string(text: str) -> Optional[str]:
    """Extract the last \\boxed{} expression from text (Hendrycks MATH standard)

    This is the canonical extractor used in:
    - Original MATH dataset (Hendrycks et al.)
    - lm-evaluation-harness
    - Minerva
    - Math-Verify

    Args:
        text: Text containing LaTeX \\boxed{} expressions

    Returns:
        Content of the last \\boxed{} expression, or None if not found
    """
    import re

    # Pattern for \boxed{} or \fbox{}
    # Use a simple pattern first for non-nested cases
    simple_pattern = r'\\(?:boxed|fbox)\{([^{}]+)\}'
    matches = list(re.finditer(simple_pattern, text))

    if matches:
        # Return content of last match
        return matches[-1].group(1).strip()

    # If no simple match, try to handle nested braces with brace counting
    # This handles cases like \boxed{\frac{a}{b}}
    boxed_starts = list(re.finditer(r'\\(?:boxed|fbox)\{', text))

    if not boxed_starts:
        return None

    # Process from the last \boxed{ occurrence
    last_start = boxed_starts[-1]
    start_pos = last_start.end()

    # Count braces to find matching closing brace
    depth = 1
    end_pos = start_pos

    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end_pos = i
                break

    if depth == 0:
        content = text[start_pos:end_pos].strip()
        return content if content else None

    return None


def get_gold_answer_math(
    gold_answer_field: Optional[str] = None,
    gold_solution_text: Optional[str] = None,
) -> Optional[str]:
    """Get gold answer for MATH dataset in a dataset-agnostic way

    Args:
        gold_answer_field: Direct answer field (MATH-500 HuggingFace format)
        gold_solution_text: Solution text with \\boxed{} answer (original MATH format)

    Returns:
        Extracted gold answer string
    """
    # MATH-500 HuggingFace format: use answer field directly
    if gold_answer_field is not None:
        return gold_answer_field.strip()

    # Original Hendrycks MATH format: extract from solution using \\boxed{}
    if gold_solution_text is not None:
        ans = last_boxed_only_string(gold_solution_text)
        if ans:
            return ans

    return None


def extract_answer_before_hash_r1(text: str, task: str = "gsm8k") -> Optional[str]:
    """Extract answer that appears BEFORE #### marker (R1-specific pattern)

    R1 models often put the final answer right before ####, then regenerate after.
    Examples:
    - "Total time = 16 hours.\n####" → "16"
    - "The answer is \\boxed{72}.\n####" → "72"

    Args:
        text: Generated text containing #### marker
        task: Task type (gsm8k or math/math-500)

    Returns:
        Extracted answer from before ####, or None
    """
    import re

    if '####' not in text:
        return None

    # Get text before the last ####
    last_hash_idx = text.rfind('####')
    before_hash = text[:last_hash_idx].strip()

    if not before_hash:
        return None

    if task in ("math", "math-500"):
        # MATH: Look for \boxed{} before ####
        boxed_answer = last_boxed_only_string(before_hash)
        if boxed_answer:
            return boxed_answer

        # Fallback: extract from last line before ####
        lines = before_hash.split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            if line and not line.startswith('Step'):
                # Try to extract LaTeX expression or number
                # Remove common prefixes
                for prefix in ['So,', 'Therefore,', 'Thus,', 'Hence,', 'The answer is', 'Final answer:']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break

                # Clean and return if reasonable length
                line = line.rstrip('.')
                if line and len(line) < 100:
                    return line

        return None

    else:  # GSM8K
        # Look for the first number before ####, then extend backwards
        # Get last 200 chars before #### to avoid scanning entire text
        snippet = before_hash[-200:] if len(before_hash) > 200 else before_hash

        # Find all numbers in the snippet
        # Pattern: optional $, then number with optional commas/spaces
        number_pattern = r'\$?\s*(-?\d+(?:[,\s]\d+)*(?:\.\d+)?)'
        matches = list(re.finditer(number_pattern, snippet))

        if not matches:
            return None

        # Get the last number before ####
        last_match = matches[-1]
        number_str = last_match.group(1).replace(',', '').replace(' ', '')

        # Now look backwards from this number to find the start of the answer phrase
        # This helps extract "16 hours" instead of just "16"
        match_start = last_match.start()
        phrase_start = match_start

        # Look back up to 50 chars or until we hit punctuation/newline
        lookback_start = max(0, match_start - 50)
        for i in range(match_start - 1, lookback_start - 1, -1):
            char = snippet[i]
            if char in '.!?\n':
                phrase_start = i + 1
                break
            elif char in '=':
                # Include the part after = (e.g., "= 16")
                phrase_start = i
                break

        # Extract the phrase containing the number
        phrase = snippet[phrase_start:last_match.end()].strip()

        # Clean up the phrase
        phrase = phrase.lstrip('=').strip()
        phrase = phrase.lstrip('$').strip()

        # For GSM8K, we typically want just the number
        # But first try to extract it cleanly from the phrase
        clean_number = re.search(r'(-?\d+(?:\.\d+)?)', phrase)
        if clean_number:
            return clean_number.group(1)

        return number_str


def extract_answer(
    text: str,
    task: str = "gsm8k",
    use_r1_fallback: bool = False,
    gold_answer: Optional[str] = None
) -> Optional[str]:
    """Extract final answer from generated text (ROBUST version with MATH-specific improvements)

    Handles:
    - GSM8K format: "#### 36 + 3 = 39" → "39"
    - MATH format: "\\boxed{42}" or "\\boxed{x \\in [-2,7]}" → "42" or "x \\in [-2,7]"
    - MATH with ####: "#### $answer$" or "#### answer"
    - R1-specific: Answer before #### when use_r1_fallback=True
    - Comma-separated numbers: "#### 1,234" → "1234"
    - Currency symbols: "#### $70" → "70"
    - Negative numbers: "#### -42" → "-42"
    - Spaces between digits: "#### 118 000" → "118000"
    - Text after number: "#### 70 dollars" → "70"
    - Recursive operators: "#### 36 + 3 = 39 + 1 = 40" → "40"
    - Repetition detection: Stops before repetitive patterns
    - Fallback: Extracts from last coherent step

    Args:
        text: Generated text
        task: Task/dataset name (gsm8k, math-500, math, etc.)
        use_r1_fallback: If True, try extracting answer before #### when after-#### extraction
                        fails or is incorrect (for R1 models)
        gold_answer: Optional gold answer for R1 fallback validation

    Returns:
        Extracted answer or None
    """
    if not text:
        return None

    import re

    # Validate and normalize task name
    task = task.lower().strip()
    if task not in ("gsm8k", "math", "math-500"):
        # Treat unknown tasks as generic numeric extraction
        task = "gsm8k"

    # Helper function: Detect repetitive pattern and truncate
    def remove_repetition(txt: str) -> str:
        """Remove repetitive patterns from end of text"""
        if len(txt) < 100:
            return txt

        # Check for exact repetition of substrings at the end
        for pattern_len in [20, 30, 40, 50]:
            if len(txt) < pattern_len * 3:
                continue
            pattern = txt[-pattern_len:]
            # Count how many times this pattern repeats at the end
            count = 1
            pos = len(txt) - pattern_len * 2
            while pos >= 0 and txt[pos:pos+pattern_len] == pattern:
                count += 1
                pos -= pattern_len

            if count >= 3:  # If pattern repeats 3+ times, truncate
                return txt[:pos + pattern_len]

        # Check for "Step N: ####" repetition (common failure mode)
        if txt.count('Step ') > 10:
            # Find where "Step N: ####" pattern starts repeating
            matches = list(re.finditer(r'Step \d+: ####', txt))
            if len(matches) >= 5:  # If we see this pattern 5+ times
                # Truncate at the first occurrence of this pattern
                return txt[:matches[0].start()]

        return txt

    # Helper function: Extract last number from text as fallback
    def extract_last_number(txt: str) -> Optional[str]:
        all_numbers = re.findall(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', txt)
        if all_numbers:
            return all_numbers[-1].replace(' ', '').replace(',', '')
        return None

    # Helper function: Extract answer from #### marker (MATH-aware)
    def extract_from_hash_marker(txt: str) -> Optional[str]:
        """Extract answer after #### marker, handling MATH-specific formats"""
        if '####' not in txt:
            return None

        # Split by #### and get the LAST occurrence (most recent answer)
        parts = txt.split('####')
        answer_part = parts[-1].strip()
        before_hash = parts[-2] if len(parts) >= 2 else ""

        # Check if answer_part is actually useful
        # It's not useful if: empty, too long, starts with "Step", or is a full sentence without clear answer
        answer_looks_like_sentence = (
            answer_part and
            len(answer_part.split()) > 5 and  # More than 5 words
            not answer_part[0].isdigit() and  # Doesn't start with a number
            not answer_part.startswith('$') and  # Doesn't start with LaTeX
            ('the' in answer_part.lower() or 'is' in answer_part.lower())  # Contains common sentence words
        )

        if not answer_part or len(answer_part) > 200 or answer_part.startswith('Step') or answer_looks_like_sentence:
            # If nothing useful after ####, or it's too long (likely paragraph), or starts with Step, or looks like a sentence
            # Try to extract from the text BEFORE #### (look for "Final Answer:" or similar)
            if before_hash:
                # Look for common answer indicators in the last 300 chars before ####
                before_snippet = before_hash[-300:] if len(before_hash) > 300 else before_hash

                # Try to find "Final Answer:", "Answer:", etc.
                answer_patterns = [
                    r'Final Answer:\s*([^\n]+)',
                    r'Answer:\s*([^\n]+)',
                    r'Therefore,?\s+(?:the answer is|we get|we have)\s*:?\s*([^\n]+)',
                    r'(?:Thus|Hence|So),?\s+(?:the answer is|we get|we have)\s*:?\s*([^\n]+)',
                ]

                for pattern in answer_patterns:
                    match = re.search(pattern, before_snippet, re.IGNORECASE)
                    if match:
                        potential_answer = match.group(1).strip()
                        # Clean up the extracted answer
                        potential_answer = potential_answer.rstrip('.,;:')
                        # Extract just the number/value if it's a sentence
                        words = potential_answer.split()
                        if len(words) > 3:  # If it's a phrase, extract the first number/value
                            # Look for a number at the start
                            number_match = re.match(r'^(-?\d+(?:\.\d+)?)', potential_answer)
                            if number_match:
                                return number_match.group(1)
                        return potential_answer

        # Continue with normal after-#### processing
        if not answer_part:
            # If last #### has nothing after it, try second-to-last
            if len(parts) >= 2:
                answer_part = parts[-2].strip()
            if not answer_part:
                return None

        # Skip if it starts with "Step" (incomplete generation)
        if answer_part.startswith('Step'):
            # Try to find a #### that's NOT followed by Step
            for i in range(len(parts) - 1, 0, -1):
                candidate = parts[i].strip()
                if candidate and not candidate.startswith('Step'):
                    answer_part = candidate
                    break
            else:
                return None

        # Format 1: #### $latex_expression$
        # Extract content between first $ and last $ on same line
        first_line = answer_part.split('\n')[0].strip()
        if first_line.startswith('$') and first_line.count('$') >= 2:
            # Extract content between first and last $
            dollar_content = first_line[1:]  # Remove first $
            if '$' in dollar_content:
                dollar_content = dollar_content[:dollar_content.rfind('$')]  # Remove last $
                dollar_content = dollar_content.strip()

                # If it's an equation like "f(x)=5", extract the value after =
                if '=' in dollar_content:
                    # Split by = and get the rightmost part
                    eq_parts = dollar_content.split('=')
                    # Get last non-empty part
                    for part in reversed(eq_parts):
                        part = part.strip()
                        if part:
                            return part

                return dollar_content

        # Format 2: #### plain answer (number or expression)
        # Take first line only, clean it
        first_line = answer_part.split('\n')[0].strip()

        # Remove trailing/leading $ if present
        first_line = first_line.strip('$')

        # Remove markdown formatting (**, *, etc.)
        first_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', first_line)  # **text** → text
        first_line = re.sub(r'\*([^*]+)\*', r'\1', first_line)      # *text* → text
        first_line = first_line.strip()

        # Remove common prefixes that indicate an answer
        prefixes_to_remove = [
            'The final answer is',
            'The answer is',
            'Final Answer:',
            'Final Answer',
            'Answer:',
            'Answer',
            'Therefore,',
            'Thus,',
            'So,',
            'Hence,',
        ]

        for prefix in prefixes_to_remove:
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                # Remove trailing punctuation and colons after removing prefix
                first_line = first_line.lstrip(':').strip().rstrip('.,;:')
                break

        # Remove common suffixes like explanations
        first_line = re.split(r'\n|Explanation:|Note:|Solution:', first_line)[0].strip()

        # If what remains is a sentence (contains many words) OR has text after a number, extract just the number
        # This handles cases like "3 treeks", "70 dollars", "The combined weight of three treeks equals the weight of one squig."
        words = first_line.split()
        if len(words) >= 2:  # Has multiple words (number + text), try to extract just the value
            # Look for numbers in the text
            numbers = re.findall(r'-?\d+(?:\.\d+)?', first_line)
            if numbers:
                # If there's only one number, return it
                if len(numbers) == 1:
                    return numbers[0]
                # If multiple numbers, try to find the most likely answer (usually the last meaningful one)
                return numbers[-1]
            # If no numbers, try to extract key mathematical value
            # Look for pattern like "X equals Y" or "X = Y" or "has X units"
            if '=' in first_line or 'equals' in first_line.lower() or 'has' in first_line.lower():
                split_parts = re.split(r'=|equals|has', first_line, flags=re.IGNORECASE)
                if len(split_parts) >= 2:
                    last_part = split_parts[-1].strip().rstrip('.,;:')
                    # Try to extract number or short expression
                    cleaned = re.sub(r'\b(the|a|an|one|of|to|is|are)\b', '', last_part, flags=re.IGNORECASE).strip()
                    # Extract first number from cleaned part
                    num_match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
                    if num_match:
                        return num_match.group(0)
                    if cleaned and len(cleaned) < 20:
                        return cleaned

        return first_line if first_line else None

    # Helper function: Extract from last coherent step
    def extract_from_last_step(txt: str) -> Optional[str]:
        """Extract answer from the last coherent step before repetition"""
        # Find all Step N: patterns
        step_matches = list(re.finditer(r'Step \d+:', txt))
        if not step_matches:
            return None

        # Get the last few steps
        for match in reversed(step_matches[-5:]):  # Check last 5 steps
            start = match.end()
            # Find end of this step (next Step or end of text)
            next_match_idx = step_matches.index(match) + 1
            if next_match_idx < len(step_matches):
                end = step_matches[next_match_idx].start()
            else:
                end = len(txt)

            step_content = txt[start:end].strip()

            # Look for equations or expressions ending with $ on a line by itself
            lines = step_content.split('\n')
            for line in reversed(lines):
                line = line.strip()
                # Check if line contains LaTeX expression
                if '$' in line and not line.startswith('Step'):
                    # Extract from $...$
                    if line.count('$') >= 2:
                        dollar_parts = line.split('$')
                        for part in reversed(dollar_parts):
                            part = part.strip()
                            if part and not part.startswith('Step'):
                                # Check if this looks like an answer (not a full sentence)
                                if len(part) < 50 and ('=' in part or '\\' in part or part.replace('.', '').replace('-', '').replace(',', '').replace(' ', '').replace('/', '').isalnum()):
                                    # Extract the value after = if present
                                    if '=' in part:
                                        after_eq = part.split('=')[-1].strip()
                                        if after_eq:
                                            return after_eq
                                    return part

        return None

    if task in ("math-500", "math"):
        # MATH format: For gold answers, the text IS the answer (may contain LaTeX)
        # For generated text, use multi-strategy extraction

        # Remove repetitive patterns first
        text = remove_repetition(text)

        # Strategy 1: Look for \boxed{} notation (official MATH format, highest priority)
        # Use the canonical last_boxed_only_string extractor
        boxed_answer = last_boxed_only_string(text)
        if boxed_answer:
            result = boxed_answer
        # Note: If boxed_answer is None, this is a formatting failure for MATH.
        # Upstream code may want to track n_with_boxed vs n_without_boxed
        # to distinguish formatting quality from reasoning quality.

        # Strategy 2: Look for #### marker (GSM8K-style, but model sometimes uses it)
        elif '####' in text:
            hash_answer = extract_from_hash_marker(text)
            if hash_answer:
                result = hash_answer
            else:
                result = None
        # Strategy 3: If text looks like a direct answer (gold answer case)
        # This handles gold answers like "\frac{14}{3}", "x \\in [-2,7]", or "p - q"
        elif not ("Step" in text or len(text) > 200):
            # Likely a gold answer - return as-is after light cleanup
            result = text.strip()
        else:
            # Strategy 4: Extract from last coherent step
            step_answer = extract_from_last_step(text)
            if step_answer:
                result = step_answer
            else:
                # Strategy 5: For MATH, do NOT default to last numeric token
                result = None

        # R1 FALLBACK for MATH: Check if we should try before-hash extraction
        if use_r1_fallback and '####' in text:
            should_try_r1 = False
            if result is None:
                should_try_r1 = True
            elif gold_answer is not None and not evaluate_math_answer(result, gold_answer):
                should_try_r1 = True

            if should_try_r1:
                r1_answer = extract_answer_before_hash_r1(text, task)
                if r1_answer and (gold_answer is None or evaluate_math_answer(r1_answer, gold_answer)):
                    return r1_answer

        return result

    if task == "gsm8k":
        # GSM8K format: answer is after ####
        # Use the shared extract_from_hash_marker helper for consistency
        from_hash = extract_from_hash_marker(text)

        if from_hash is None:
            # No #### found or extraction failed, try last number
            result = extract_last_number(text)
        else:
            # For GSM8K, enforce "single number" semantics
            # Normalize to extract just the numeric value
            num = normalize_answer(from_hash)
            result = num if num != "" else extract_last_number(text)

        # R1 FALLBACK for GSM8K: Check if we should try before-hash extraction
        if use_r1_fallback and '####' in text:
            should_try_r1 = False
            if result is None:
                should_try_r1 = True
            elif gold_answer is not None and not evaluate_answer(result, gold_answer):
                should_try_r1 = True

            if should_try_r1:
                r1_answer = extract_answer_before_hash_r1(text, task)
                if r1_answer and (gold_answer is None or evaluate_answer(r1_answer, gold_answer)):
                    return r1_answer

        return result

    # Generic extraction: try to find last number in entire text
    return extract_last_number(text)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison

    Removes spaces, commas, currency symbols and converts to standard number format.
    Extracts first number if answer contains non-numeric text.

    Args:
        answer: Answer string

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    import re

    # First try to extract just the number (handles "70 dollars", etc.)
    # Remove leading currency symbols
    answer = re.sub(r'^[\$£€¥₹]+\s*', '', answer.strip())

    # Extract first number (with optional decimal point and commas)
    number_match = re.match(r'(-?[\d,]+(?:\.\d+)?)', answer)
    if number_match:
        answer = number_match.group(1)

    # Remove remaining spaces and commas
    answer = re.sub(r'[\s,]', '', answer)

    # Try to convert to number and normalize
    try:
        num = float(answer)
        if num.is_integer():
            return str(int(num))
        else:
            return str(num)
    except (ValueError, TypeError):
        return answer.strip()


def answers_match(answer1: str, answer2: str) -> bool:
    """Check if two answers match after normalization

    Args:
        answer1: First answer
        answer2: Second answer

    Returns:
        True if answers match after normalization
    """
    return normalize_answer(answer1) == normalize_answer(answer2)


def evaluate_answer(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-5,
) -> bool:
    """Evaluate if predicted answer matches ground truth (robust version)

    Uses normalization to handle:
    - Different spacing: "1 000" vs "1000"
    - Commas: "1,000" vs "1000"
    - Currency: "$50" vs "50"
    - Trailing text: "72 clips" vs "72"
    - Decimal vs integer: "72.0" vs "72"

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance (not used with normalization, kept for API compatibility)

    Returns:
        True if answers match
    """
    if not predicted or not ground_truth:
        return False

    # Use robust normalization and matching
    return answers_match(predicted, ground_truth)


def math_normalize_answer(answer: str) -> str:
    """Normalize MATH dataset answer for comparison

    Handles:
    - LaTeX expressions: \\frac{a}{b}, \\sqrt{x}, etc.
    - Intervals: [a,b], (a,b), x \\in [a,b]
    - Sets: \\{a, b, c\\}
    - Matrices and vectors
    - Numerical values with various formats
    - Whitespace normalization

    Args:
        answer: Answer string (may contain LaTeX)

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    import re

    # Strip outer whitespace
    answer = answer.strip()

    # Remove LaTeX display mode markers
    answer = answer.replace('$', '')
    answer = answer.replace('\\[', '').replace('\\]', '')
    answer = answer.replace('\\(', '').replace('\\)', '')

    # Normalize whitespace around common delimiters
    answer = re.sub(r'\s*,\s*', ',', answer)  # "a, b" → "a,b"
    answer = re.sub(r'\s*=\s*', '=', answer)  # "x = 5" → "x=5"
    answer = re.sub(r'\s+', ' ', answer)      # Multiple spaces → single space

    # Normalize LaTeX fractions: \frac{a}{b} → frac(a,b) for comparison
    # This allows \frac{1}{2} to match \frac{2}{4} after simplification
    def normalize_frac(match):
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"frac({num},{den})"
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', normalize_frac, answer)

    # Normalize common LaTeX commands (preserve but standardize)
    latex_commands = {
        '\\sqrt': 'sqrt',
        '\\pi': 'pi',
        '\\infty': 'inf',
        '\\in': ' in ',
        '\\cup': ' cup ',
        '\\cap': ' cap ',
        '\\subset': ' subset ',
        '\\subseteq': ' subseteq ',
        '\\emptyset': 'emptyset',
        '\\varnothing': 'emptyset',
        '\\left': '',
        '\\right': '',
        # Note: \\{ and \\} handled separately below for clarity
        '\\ldots': '...',
        '\\cdots': '...',
        '\\le': '<=',
        '\\ge': '>=',
        '\\leq': '<=',
        '\\geq': '>=',
        '\\ne': '!=',
        '\\neq': '!=',
        '\\times': '*',
        '\\cdot': '*',
        '\\div': '/',
    }

    for latex_cmd, replacement in latex_commands.items():
        answer = answer.replace(latex_cmd, replacement)

    # Handle escaped braces separately (avoid duplicate with dict)
    answer = answer.replace('\\{', '{')
    answer = answer.replace('\\}', '}')

    # Normalize interval notation: remove extra spaces inside brackets
    answer = re.sub(r'\[\s*', '[', answer)
    answer = re.sub(r'\s*\]', ']', answer)
    answer = re.sub(r'\(\s*', '(', answer)
    answer = re.sub(r'\s*\)', ')', answer)

    # Try to simplify fractions if both numerator and denominator are integers
    def simplify_frac(match):
        import math
        try:
            num = int(match.group(1))
            den = int(match.group(2))
            gcd = math.gcd(abs(num), abs(den))
            return f"frac({num//gcd},{den//gcd})"
        except:
            return match.group(0)
    answer = re.sub(r'frac\((-?\d+),(-?\d+)\)', simplify_frac, answer)

    # Normalize numbers: remove commas, standardize decimals
    def normalize_number(match):
        num_str = match.group(0).replace(',', '')
        try:
            num = float(num_str)
            if num.is_integer():
                return str(int(num))
            else:
                # Round to 6 decimal places to avoid floating point issues
                return f"{num:.6f}".rstrip('0').rstrip('.')
        except:
            return match.group(0)

    answer = re.sub(r'-?\d+(?:,\d{3})*(?:\.\d+)?', normalize_number, answer)

    # Final cleanup
    answer = answer.strip()

    return answer


def canonical_non_numeric(ans: str) -> Optional[str]:
    """Map various verbal answers to a canonical label

    Handles non-numeric MATH answers like:
    - "no solution"
    - "does not exist"
    - "infinitely many solutions"
    - "all real numbers"

    Args:
        ans: Answer string

    Returns:
        Canonical label if recognized, None otherwise
    """
    import re

    a = ans.strip().lower()
    # Strip punctuation
    a = re.sub(r'[.,;:!?\s]+', ' ', a).strip()

    # Common patterns
    if any(kw in a for kw in ["no solution", "no real solution", "no real solutions", "no solutions"]):
        return "NO_SOLUTION"

    if any(kw in a for kw in ["does not exist", "dne", "undefined"]):
        return "DNE"

    if any(kw in a for kw in ["infinitely many", "infinite number of solutions", "infinite solutions"]):
        return "INFINITELY_MANY"

    if any(kw in a for kw in ["all real numbers", "any real number", "for all real x", "all reals"]):
        return "ALL_REALS"

    return None


def math_answers_match(answer1: str, answer2: str) -> bool:
    """Check if two MATH dataset answers match after normalization

    Uses multiple strategies:
    1. Exact string match after normalization
    2. Numeric equality for pure numbers
    3. SymPy-based symbolic equivalence (algebraic, fraction, etc.)
    4. Whitespace-insensitive string comparison

    Args:
        answer1: First answer
        answer2: Second answer

    Returns:
        True if answers match after normalization
    """
    if not answer1 or not answer2:
        return False

    import re

    norm1 = math_normalize_answer(answer1)
    norm2 = math_normalize_answer(answer2)

    # 0. Non-numeric canonical forms (check before other strategies)
    # Handles "no solution", "DNE", "infinitely many", etc.
    canon1 = canonical_non_numeric(norm1)
    canon2 = canonical_non_numeric(norm2)
    if canon1 is not None or canon2 is not None:
        return canon1 == canon2

    # 1. Direct string match after normalization
    if norm1 == norm2:
        return True

    # 2. Try to evaluate as numbers if possible
    try:
        # Extract just numbers for comparison (handles cases like "4" vs "4.0")
        num1_match = re.search(r'^-?\d+(?:\.\d+)?$', norm1)
        num2_match = re.search(r'^-?\d+(?:\.\d+)?$', norm2)

        if num1_match and num2_match:
            num1 = float(norm1)
            num2 = float(norm2)
            return abs(num1 - num2) < 1e-6
    except:
        pass

    # 3. Try SymPy-based symbolic equivalence
    # This handles algebraic equivalence, fraction simplification, etc.
    try:
        import sympy as sp

        def try_sympy_parse(expr: str) -> Optional[sp.Expr]:
            """Attempt to parse expression as SymPy object"""
            try:
                expr_clean = expr

                # Convert frac(a,b) back to a/b
                expr_clean = re.sub(r'frac\(([^,]+),([^)]+)\)', r'(\1)/(\2)', expr_clean)

                # Basic cleanup: remove stray spaces around operators
                expr_clean = re.sub(r'\s+', ' ', expr_clean)

                # Locals mapping: let sympify know about sqrt, pi, etc.
                # Don't inject sp. into the string; pass locals dict instead
                locals_dict = {
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                    "inf": sp.oo,
                    "e": sp.E,
                }

                return sp.sympify(expr_clean, locals=locals_dict, rational=True)
            except Exception:
                return None

        expr1 = try_sympy_parse(norm1)
        expr2 = try_sympy_parse(norm2)

        if expr1 is not None and expr2 is not None:
            try:
                # Check symbolic equality
                diff = sp.simplify(expr1 - expr2)
                if diff == 0:
                    return True

                # Also try expanding and simplifying both sides
                if sp.simplify(expr1) == sp.simplify(expr2):
                    return True

                # For fractions, check if they're equivalent
                if isinstance(expr1, sp.Rational) and isinstance(expr2, sp.Rational):
                    return expr1 == expr2

            except Exception:
                pass

    except ImportError:
        # SymPy not available, skip symbolic checking
        pass

    # 4. Check if they differ only in equivalent representations
    # E.g., "x in [-2,7]" vs "x in [ -2 , 7 ]"
    compact1 = re.sub(r'\s+', '', norm1).lower()
    compact2 = re.sub(r'\s+', '', norm2).lower()

    return compact1 == compact2


def evaluate_math_answer(
    predicted: str,
    ground_truth: str,
) -> bool:
    """Evaluate if predicted MATH answer matches ground truth

    Uses MATH-specific normalization that handles:
    - LaTeX expressions
    - Fractions and simplification
    - Intervals and sets
    - Multiple equivalent representations

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if answers match
    """
    if not predicted or not ground_truth:
        return False

    return math_answers_match(predicted, ground_truth)


def format_prompt(
    question: str,
    template: str = "default",
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Format prompt for model input

    Args:
        question: Input question
        template: Prompt template to use
        few_shot_examples: Optional few-shot examples

    Returns:
        Formatted prompt
    """
    if template == "default":
        prompt = f"Question: {question}\n\nAnswer:"
    elif template == "cot":
        # Structured prompt with clear instructions for GSM8K format
        prompt = f"""You are a helpful assistant that solves problems step by step with each step signified by "Step [step_number]: ".
Always provide your final answer after #### at the end.

Question: {question}

Please solve this step by step, putting each step after "Step [step_number]: " and always provide your final answer after ####.

Solution:

"""
    elif template == "cot_final":
        # Chain of thought with "Final Answer:" marker
        prompt = f"""You are a helpful assistant that solves problems step by step with each step signified by "Step [step_number]: ".
Always provide your final answer after "Final Answer:" at the end.

Question: {question}

Please solve this step by step, putting each step after "Step [step_number]: " and always provide your final answer after "Final Answer:".

Solution:

"""
    elif template == "math_cot":
        # Chain of thought for MATH dataset with \boxed{} formatting
        prompt = f"""You are a helpful assistant that solves olympiad-style math problems step by step.
At the end, ALWAYS put your final answer in LaTeX form inside \\boxed{{...}} on its own line.

Question: {question}

Please solve this step by step, and put ONLY the final result (no explanation) inside \\boxed{{...}} on the last line.

Solution:

"""
    elif template == "direct":
        prompt = question
    else:
        prompt = question

    # Add few-shot examples if provided
    if few_shot_examples:
        examples_text = ""
        for ex in few_shot_examples:
            examples_text += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt = examples_text + prompt

    return prompt


def calculate_metrics(
    predictions: List[Optional[str]],
    ground_truths: List[Optional[str]],
    task: str = "gsm8k",
) -> Dict[str, float]:
    """Calculate evaluation metrics with extraction tracking

    Distinguishes between extraction failures and reasoning errors:
    - accuracy: standard accuracy over all prompts (correct / total)
    - conditional_accuracy: accuracy given extraction succeeded (correct / extractable)
    - extraction_rate: fraction of prompts with extractable answers

    This allows distinguishing formatting quality from reasoning quality.

    Args:
        predictions: List of predicted answers (None if extraction failed)
        ground_truths: List of ground truth answers
        task: Task format ("gsm8k" or "math" / "math-500")

    Returns:
        Dictionary of metrics including accuracy, conditional_accuracy, extraction_rate
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions and ground truths must match")

    total_prompts = len(predictions)
    valid_total = 0
    n_extractable = 0
    n_correct = 0

    for pred, gt in zip(predictions, ground_truths):
        # Skip if ground truth is None (bad gold answer)
        if gt is None:
            continue

        valid_total += 1

        # Track extraction success
        if pred is None:
            continue

        n_extractable += 1

        # Dispatch to appropriate evaluation function based on task
        if task in ("math", "math-500"):
            is_correct = evaluate_math_answer(pred, gt)
        else:
            is_correct = evaluate_answer(pred, gt)

        if is_correct:
            n_correct += 1

    overall_acc = n_correct / valid_total if valid_total > 0 else 0.0
    cond_acc = n_correct / n_extractable if n_extractable > 0 else 0.0
    extract_rate = n_extractable / valid_total if valid_total > 0 else 0.0

    return {
        "accuracy": overall_acc,                # correct / valid_total
        "conditional_accuracy": cond_acc,       # correct / extractable
        "extraction_rate": extract_rate,        # extractable / valid_total
        "correct": n_correct,
        "extractable": n_extractable,
        "total": valid_total,                   # valid examples (excludes None golds)
        "total_prompts": total_prompts,         # all prompts including invalid
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Results") -> None:
    """Pretty print metrics

    Args:
        metrics: Dictionary of metrics
        title: Title for output
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")

    print(f"{'='*50}\n")


def create_directory_structure(base_dir: Optional[str] = None) -> None:
    """Create standard directory structure

    Args:
        base_dir: Base directory (defaults to current directory)
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    directories = [
        "data",
        "data/gsm8k",
        "models",
        "models/cache",
        "output",
        "output/results",
        "output/trajectories",
        "output/checkpoints",
        "output/logs",
        "output/cache",
        "config",
        "src",
        "examples",
    ]

    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at {base_dir}")
