"""Span detection utilities for dp1/dp2 and answer extraction"""

import re
from typing import List, Optional
from transformers import PreTrainedTokenizer


def find_subseq(container_ids: List[int], subseq_ids: List[int]) -> Optional[int]:
    """Find first occurrence of subsequence in container

    Args:
        container_ids: Full sequence of token IDs
        subseq_ids: Subsequence to find

    Returns:
        Start index of first occurrence, or None if not found
    """
    if not subseq_ids or not container_ids:
        return None

    subseq_len = len(subseq_ids)
    for i in range(len(container_ids) - subseq_len + 1):
        if container_ids[i : i + subseq_len] == subseq_ids:
            return i

    return None


def ids_of_append(
    prefix_ids: List[int],
    append_text: str,
    tokenizer: PreTrainedTokenizer
) -> List[int]:
    """Return token IDs that get appended when text follows prefix_ids

    This ensures context-aware tokenization - the same tokenization as if
    the text was generated after the prefix.

    Args:
        prefix_ids: Existing token IDs (e.g., prompt tokens)
        append_text: Text to append (e.g., " #### 50")
        tokenizer: Tokenizer

    Returns:
        Token IDs for the appended text only
    """
    if not append_text:
        return []

    # Decode prefix to get text
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

    # Tokenize the combined text
    combined_ids = tokenizer.encode(
        prefix_text + append_text,
        add_special_tokens=False
    )

    # Return only the new tokens (after prefix)
    return combined_ids[len(prefix_ids):]


def tokenize_answer_marker(
    tokenizer: PreTrainedTokenizer, answer_text: str
) -> List[int]:
    """Tokenize answer with #### marker for span detection

    Args:
        tokenizer: Tokenizer
        answer_text: Answer string (without ####)

    Returns:
        Token IDs for " #### answer"
    """
    # Create the marker + answer string
    full_text = f" #### {answer_text}"

    # Tokenize (without special tokens)
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)

    return token_ids


def extract_answer_after_hash(text: str) -> Optional[str]:
    """Extract numerical answer after FIRST #### marker in generated text - ROBUST VERSION

    Rules (as specified):
    1. Find first #### occurrence in generated response
    2. Look for first number after ####
    3. If number has spaces between digits (like "118 000"), concatenate them
    4. If first number is followed by operators (+-*/), look for first number after "=" instead
    5. If number after = is still followed by operators, recursively look for next =
    6. FALLBACK: If no #### found OR extraction fails, extract last number from entire response

    Handles:
    - Mathematical expressions: "#### 36 + 3 = 39" -> "39"
    - Comma-separated numbers: "#### 1,234" -> "1234"
    - Currency symbols: "#### $70" -> "70"
    - Negative numbers: "#### -42" -> "-42"
    - Spaces between digits: "#### 118 000" -> "118000"
    - Recursive operators: "#### 36 + 3 = 39 + 1 = 40" -> "40"
    - Fallback: If no ####, extracts last number from entire text

    Args:
        text: Generated text (should NOT include prompt)

    Returns:
        Extracted numerical answer string (normalized, commas and spaces removed), or None
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
        return extract_last_number(text)

    parts = text.split("####", 1)
    if len(parts) < 2:
        return extract_last_number(text)

    answer_text = parts[1].strip()
    answer_text = re.sub(r'^[\$\u00a3\u20ac\u00a5\u20b9]+\s*', '', answer_text)

    # Find first number (handles spaces between digits, commas, negative numbers)
    first_number_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', answer_text)
    if not first_number_match:
        return extract_last_number(text)

    first_number_end = first_number_match.end()
    after_number = answer_text[first_number_end:].lstrip()

    # Check if number is followed by operators (+-*/)
    if after_number and after_number[0] in '+-*/':
        # Look for result after = sign recursively
        remaining_text = answer_text
        while '=' in remaining_text:
            equals_parts = remaining_text.split('=', 1)
            if len(equals_parts) < 2:
                break

            result_text = equals_parts[1].strip()
            result_text = re.sub(r'^[\$\u00a3\u20ac\u00a5\u20b9]+\s*', '', result_text)
            result_match = re.search(r'(-?\d+(?:[\s,]\d+)*(?:\.\d+)?)', result_text)

            if result_match:
                result_number_end = result_match.end()
                after_result = result_text[result_number_end:].lstrip()

                # If result is NOT followed by operators, this is our answer
                if not after_result or after_result[0] not in '+-*/':
                    return result_match.group(1).replace(' ', '').replace(',', '')

                # Otherwise, continue to next = sign
                remaining_text = equals_parts[1]
            else:
                break

        # No valid result found, fallback to last number
        return extract_last_number(text)

    # No operators after first number, return it
    return first_number_match.group(1).replace(' ', '').replace(',', '')


def find_subseq_reverse(container_ids: List[int], subseq_ids: List[int]) -> Optional[int]:
    """Find LAST occurrence of subsequence in container (search backwards)

    Args:
        container_ids: Full sequence of token IDs
        subseq_ids: Subsequence to find

    Returns:
        Start index of last occurrence, or None if not found
    """
    if not subseq_ids or not container_ids:
        return None

    subseq_len = len(subseq_ids)
    # Search backwards from end
    for i in range(len(container_ids) - subseq_len, -1, -1):
        if container_ids[i : i + subseq_len] == subseq_ids:
            return i

    return None


def detect_dp2_index(
    full_seq_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    dp1_idx: int,
    produced_text: Optional[str] = None,
    produced_answer: Optional[str] = None,
) -> Optional[int]:
    """Detect dp2 (position of first token of the produced answer) in token sequence

    Uses context-aware tokenization to find the exact answer token, following the
    reference implementation from the trajectory binning script.

    Args:
        full_seq_ids: Complete sequence IDs (prompt + generated)
        tokenizer: Tokenizer
        dp1_idx: Start of reasoning (first generated token after prompt)
        produced_text: Generated text (used for validation)
        produced_answer: Extracted answer string (e.g., "6")

    Returns:
        Index of first token of produced answer, or None if not found
    """
    if not produced_answer:
        return None

    # Extract prompt IDs (everything up to dp1_idx, inclusive)
    prompt_ids = full_seq_ids[:dp1_idx + 1]

    # Get context-aware tokenization of " #### {answer}" appended to prompt
    prod_span = ids_of_append(prompt_ids, f" #### {produced_answer}", tokenizer)

    if not prod_span or len(prod_span) < 3:
        return None

    # The answer token is typically at index 2 in the span:
    # [0]: tokens for " ####" (could be multiple)
    # [1]: space or first part of answer
    # [2]: answer token itself
    # But this can vary, so we'll search for it

    # Try to find which index in prod_span is the actual answer token
    # by checking if it appears in the generated sequence
    gen_region = full_seq_ids[dp1_idx + 1:]

    # Try multiple strategies to find the answer token
    ans_start_rel = None

    # Strategy 1: Try single-token match for prod_span[2]
    if len(prod_span) > 2:
        ans_start_rel = find_subseq(gen_region, [prod_span[2]])

    # Strategy 2: Try with preceding token (prod_span[1:3])
    if ans_start_rel is None and len(prod_span) > 2:
        ans_start_rel = find_subseq(gen_region, [prod_span[1], prod_span[2]])

    # Strategy 3: Brute-force search for prod_span[2]
    if ans_start_rel is None and len(prod_span) > 2:
        for j, tid in enumerate(gen_region):
            if tid == prod_span[2]:
                ans_start_rel = j
                break

    if ans_start_rel is not None:
        # Convert relative index to absolute
        dp2_idx = (dp1_idx + 1) + ans_start_rel
        return dp2_idx
    else:
        return None


def get_gold_answer_first_token(
    gold_answer: str,
    tokenizer: PreTrainedTokenizer,
    prompt_ids: Optional[List[int]] = None
) -> Optional[int]:
    """Get first token ID of gold answer (for ranking features)

    Uses context-aware tokenization if prompt_ids are provided, otherwise
    uses standalone tokenization as fallback.

    Args:
        gold_answer: Gold answer string
        tokenizer: Tokenizer
        prompt_ids: Optional prompt token IDs for context-aware tokenization

    Returns:
        First token ID of answer, or None
    """
    if not gold_answer:
        return None

    if prompt_ids is not None:
        # Context-aware tokenization (reference implementation approach)
        gold_span = ids_of_append(prompt_ids, f" #### {gold_answer}", tokenizer)

        if not gold_span or len(gold_span) < 3:
            # Fallback to standalone tokenization
            prompt_ids = None
        else:
            # Return the answer token (typically index 2)
            # [0]: " ####" tokens
            # [1]: space or part of answer
            # [2]: answer token itself
            return gold_span[2]

    # Fallback: Standalone tokenization (old behavior)
    if prompt_ids is None:
        answer_marker_ids = tokenize_answer_marker(tokenizer, gold_answer)

        if not answer_marker_ids:
            return None

        # Return first token after #### marker
        hash_marker_ids = tokenizer.encode(" ####", add_special_tokens=False)
        if len(answer_marker_ids) > len(hash_marker_ids):
            return answer_marker_ids[len(hash_marker_ids)]
        else:
            return answer_marker_ids[-1] if answer_marker_ids else None

    return None
