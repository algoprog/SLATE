"""
Exact-match reward wrapper for standard Search-R1 training mode.

Delegates to verl's built-in search_r1_like_qa_em reward.
"""

from verl.utils.reward_score.search_r1_like_qa_em import (
    compute_score,
    compute_score_subem,
    extract_solution,
    em_check,
    normalize_answer,
)

__all__ = [
    "compute_score",
    "compute_score_subem",
    "extract_solution",
    "em_check",
    "normalize_answer",
]
