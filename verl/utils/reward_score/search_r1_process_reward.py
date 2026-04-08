# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Process-based reward function that encourages retrieval usage.

This reward function combines:
1. Final answer correctness (outcome reward)
2. Search usage bonus (process reward)
3. Structure penalties for malformed outputs
"""

import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from <answer> tags."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if len(matches) < 1:
        return None
    
    return matches[-1].group(1).strip()


def count_tags(text, tag):
    """Count opening and closing tags."""
    opening_tags = text.count(f"<{tag}>")
    closing_tags = text.count(f"</{tag}>")
    return opening_tags, closing_tags


def has_information_tags(text):
    """Check if the trajectory contains <information> tags (evidence of retrieval)."""
    return "<information>" in text and "</information>" in text


def count_search_queries(text):
    """Count the number of valid search queries."""
    search_pattern = r"<search>(.*?)</search>"
    matches = re.findall(search_pattern, text, re.DOTALL)
    # Filter out empty searches
    valid_searches = [m.strip() for m in matches if m.strip()]
    return len(valid_searches)


def compute_score(
    solution_str, 
    ground_truth, 
    method="strict", 
    format_score=0.0, 
    score=1.0,
    **kwargs
):
    """
    Simplified scoring function: 0 reward if no search+information pair exists.
    
    Args:
        solution_str: The full trajectory including think/search/answer tags
        ground_truth: The ground truth answer
        method: Extraction method (unused, for compatibility)
        format_score: Score for malformed outputs (default 0.0)
        score: Score for correct final answer (default 1.0)
    
    Returns:
        Float score: 0.0 if no retrieval, otherwise standard EM score
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_tags(solution_str, "answer")
    search_count = count_search_queries(solution_str)
    has_retrieval = has_information_tags(solution_str)
    
    do_print = random.randint(1, 64) == 1
    
    # Support multiple ground truth formats
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", None)
        if targets is None:
            targets = ground_truth.get("targets", None)
        if targets is None:
            targets = [str(ground_truth)]
    elif isinstance(ground_truth, (list, tuple, set)):
        targets = list(ground_truth)
    else:
        targets = ground_truth
    
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {targets}")
        print(f"Extracted answer: {answer}")
        print(f"Search count: {search_count}")
        print(f"Has retrieval: {has_retrieval}")
        print(f"Solution string (first 500 chars): {solution_str[:500]}")
    
    # HARD CONSTRAINT: No search+information pair = 0 reward
    if not has_retrieval or search_count < 1:
        if do_print:
            print("❌ NO SEARCH+INFORMATION PAIR - Reward = 0")
            print(f"   has_retrieval={has_retrieval}, search_count={search_count}")
            print("--------------------------------")
        return 0.0
    
    # Standard EM scoring (only reached if retrieval was used)
    if answer is None:
        final_score = 0.0
    elif em_check(answer, targets):
        # Penalize excessive answer tags (likely formatting issue)
        if open_count > 2 or close_count > 2:
            final_score = score / 10
        else:
            final_score = score
    else:
        final_score = format_score
    
    if do_print:
        print(f"✅ RETRIEVAL DETECTED")
        print(f"   Answer correct: {em_check(answer, targets) if answer else False}")
        print(f"   Final score: {final_score:.3f}")
        print("--------------------------------")
    
    return final_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, **kwargs):
    """Substring exact match version (for compatibility)."""
    # For subem, we use the same process reward logic
    return compute_score(solution_str, ground_truth, method, format_score, score, **kwargs)
