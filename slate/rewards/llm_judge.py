"""
LLM-as-judge reward model for SLATE.

Provides decomposed ternary rewards {-1, 0, +1} for:
- Thinking/reasoning quality (r_think)
- Query generation quality (r_query)
- Final answer correctness (r_answer)

Adapted from search-reasoner/reward.py with async batching support.
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class LLMJudgeReward:
    """
    LLM-as-judge reward model using decomposed ternary scoring.

    Communicates with a served LLM (via HTTP API) to evaluate
    reasoning quality, query quality, and answer correctness
    on a {-1, 0, +1} scale.
    """

    def __init__(self, api_url: str = "http://localhost:9000/generate", temperature: float = 0.0):
        self.api_url = api_url
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API synchronously."""
        try:
            params = {
                "prompts": [prompt],
                "temperature": self.temperature,
            }
            response = requests.post(self.api_url, json=params, timeout=60)
            response.raise_for_status()
            return response.json()["responses"][0]
        except Exception as e:
            logger.warning(f"LLM judge API call failed: {e}")
            return ""

    def _call_llm_batch(self, prompts: List[str]) -> List[str]:
        """Call the LLM API with a batch of prompts."""
        if not prompts:
            return []
        try:
            params = {
                "prompts": prompts,
                "temperature": self.temperature,
            }
            response = requests.post(self.api_url, json=params, timeout=120)
            response.raise_for_status()
            return response.json()["responses"]
        except Exception as e:
            logger.warning(f"LLM judge batch API call failed: {e}")
            return [""] * len(prompts)

    async def _call_llm_async(self, prompt: str) -> str:
        """Call the LLM API asynchronously using asyncio executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_llm, prompt)

    async def _call_llm_batch_async(self, prompts: List[str]) -> List[str]:
        """Call the LLM API with a batch of prompts asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_llm_batch, prompts)

    @staticmethod
    def extract_score(response: str) -> float:
        """Extract numerical score from LLM response."""
        # Try <score>...</score> tags first
        score_match = re.search(r"<score>\s*(-?\d+\.?\d*)\s*</score>", response, re.DOTALL)
        if score_match:
            return float(score_match.group(1))

        # Fallback patterns
        for pattern in [r"Score:\s*(-?\d+\.?\d*)", r"score:\s*(-?\d+\.?\d*)",
                        r"Rating:\s*(-?\d+\.?\d*)", r"rating:\s*(-?\d+\.?\d*)"]:
            match = re.search(pattern, response.strip(), re.MULTILINE)
            if match:
                return float(match.group(1))

        # Last resort: first number
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            return float(numbers[0])

        return 0.0

    # ------------------------------------------------------------------ #
    # Reward prompts (from paper Section 3.3)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _thinking_prompt(thinking: str, context: str) -> str:
        return f"""Evaluate the quality of the following reasoning step in a search-based question answering system.

Context: {context if context else "None"}

Current Thinking Step: {thinking}

The reasoning should be based on the previous context and the question, nothing else.

Evaluate this thinking step on these criteria:
1. Relevance: Does it address the question appropriately?
2. Clarity: Is the reasoning clear and logical?
3. Specificity: Does it identify concrete information needs?
4. Progress: Does it move toward answering the question?
5. Faithfulness: Does it accurately reflect the information in the previous context? Is there any out-of-context information?

Provide a score using EXACTLY one of these three values:
- +1: GOOD - Clear, relevant reasoning that identifies specific information needs and moves toward answering the question
- 0: ACCEPTABLE - Reasoning is somewhat relevant but vague, lacks specificity, or makes only minimal progress
- -1: BAD - Irrelevant, misleading, or counterproductive reasoning that does not help answer the question

First provide your reasoning, then the score. Use this exact format:
<explanation>
Your reasoning here
</explanation>
<score>numerical score</score>"""

    @staticmethod
    def _query_prompt(thinking: str, query: str, context: str) -> str:
        return f"""Evaluate the quality of the following search query for a question answering system.

Context: {context if context else "None"}

Thinking before this query: {thinking}

Generated Query: {query}

IMPORTANT: This is a multi-step reasoning system. The query does NOT need to directly answer the final question in one step. Instead, evaluate whether it makes good progress toward the answer by retrieving useful intermediate information.

Evaluate this query on these criteria:
1. Relevance: Will it retrieve information that makes progress toward answering the question? (Intermediate steps are valuable!)
2. Specificity: Is it specific enough to get useful results? Eg. a single generic word or phrase might be too vague and return too many results.
3. Searchability: Is it well-formed for a search engine with appropriate keywords? Good queries combine multiple relevant terms.
4. Alignment: Does it align with the thinking step that preceded it?
5. Novelty: Does it explore new information (not redundant with the context)? If the context already contains the answer to what the query is searching for, the query is redundant and unhelpful.

Provide a score using EXACTLY one of these three values:
- +1: GOOD - Specific, well-formed query that will retrieve useful information to make progress (even if intermediate). Has clear keywords and good searchability. Combines multiple relevant terms or uses specific names/concepts.
- 0: ACCEPTABLE - Query has some specificity but could be improved. May lack context-specific keywords or be somewhat generic, but shows reasonable attempt at targeting the information need.
- -1: BAD - Single generic word without context (e.g., just "singer", "perfume", "city"), completely irrelevant to the question, redundant with information already in the context, or so poorly formed it will return millions of unhelpful results.

First provide your reasoning, then the score. Use this exact format:
<explanation>
Your reasoning here
</explanation>
<score>numerical score</score>"""

    @staticmethod
    def _answer_prompt(predicted_answer: str, ground_truth: str, context: str) -> str:
        return f"""Evaluate if the predicted answer correctly answers the question.

Context: {context if context else "None"}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted_answer}

Compare the predicted answer to the ground truth. They don't need to be word-for-word identical, but the predicted answer should convey the same core information.

Provide a score using EXACTLY one of these three values:
- +1: CORRECT - The predicted answer conveys the same core information as the ground truth
- 0: PARTIALLY CORRECT - The answer is incomplete, ambiguous, or contains minor inaccuracies
- -1: INCORRECT - The answer is wrong or contradicts the ground truth

First provide your reasoning, then the score. Use this exact format:
<explanation>
Your reasoning here
</explanation>
<score>numerical score</score>"""

    # ------------------------------------------------------------------ #
    # Single-evaluation methods
    # ------------------------------------------------------------------ #

    def reward_thinking(self, thinking: str, context: str = "") -> float:
        """Evaluate reasoning quality. Returns score in {-1, 0, +1}."""
        prompt = self._thinking_prompt(thinking, context)
        response = self._call_llm(prompt)
        return self.extract_score(response)

    def reward_query(self, thinking: str, query: str, context: str = "") -> float:
        """Evaluate query quality (before seeing retrieval results). Returns score in {-1, 0, +1}."""
        prompt = self._query_prompt(thinking, query, context)
        response = self._call_llm(prompt)
        return self.extract_score(response)

    def reward_answer(self, predicted: str, ground_truth: str, context: str = "") -> float:
        """Evaluate answer correctness. Returns score in {-1, 0, +1}."""
        prompt = self._answer_prompt(predicted, ground_truth, context)
        response = self._call_llm(prompt)
        return self.extract_score(response)

    # ------------------------------------------------------------------ #
    # Composite reward (paper Eq. 3 and 4)
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_step_reward_search(r_think: float, r_query: float) -> float:
        """Composite reward for a search step (Eq. 3)."""
        return r_think + r_query

    @staticmethod
    def compute_step_reward_answer(
        r_think: float, r_answer: float,
        step: int, max_budget: int, lambda_bonus: float = 0.1,
    ) -> float:
        """Composite reward for an answer step (Eq. 4)."""
        bonus = lambda_bonus * (max_budget - step) / max_budget
        return r_think + r_answer + bonus

    # ------------------------------------------------------------------ #
    # Batched evaluation for k candidates at a step
    # ------------------------------------------------------------------ #

    def evaluate_candidates_batch(
        self,
        candidates: List[Dict],
        context: str,
        ground_truth: str,
        step: int,
        max_budget: int,
        lambda_bonus: float = 0.1,
    ) -> List[float]:
        """
        Evaluate k candidate actions at a single step, returning composite rewards.

        Each candidate dict has keys:
            - 'thinking': str
            - 'action_type': 'search' or 'answer'
            - 'action_content': query text or answer text

        Returns list of k composite reward scores.
        """
        prompts = []
        candidate_types = []

        for cand in candidates:
            thinking = cand["thinking"]
            action_type = cand["action_type"]
            content = cand["action_content"]

            # Always evaluate thinking
            prompts.append(self._thinking_prompt(thinking, context))

            if action_type == "search":
                prompts.append(self._query_prompt(thinking, content, context))
                candidate_types.append("search")
            else:
                prompts.append(self._answer_prompt(content, ground_truth, context))
                candidate_types.append("answer")

        # Batch call to LLM judge
        responses = self._call_llm_batch(prompts)

        # Parse scores and compute composite rewards
        rewards = []
        idx = 0
        for i, cand_type in enumerate(candidate_types):
            r_think = self.extract_score(responses[idx])
            idx += 1
            r_action = self.extract_score(responses[idx])
            idx += 1

            if cand_type == "search":
                reward = self.compute_step_reward_search(r_think, r_action)
            else:
                reward = self.compute_step_reward_answer(
                    r_think, r_action, step, max_budget, lambda_bonus
                )
            rewards.append(reward)

        return rewards

    async def evaluate_candidates_batch_async(
        self,
        candidates: List[Dict],
        context: str,
        ground_truth: str,
        step: int,
        max_budget: int,
        lambda_bonus: float = 0.1,
    ) -> List[float]:
        """Async version of evaluate_candidates_batch."""
        prompts = []
        candidate_types = []

        for cand in candidates:
            thinking = cand["thinking"]
            action_type = cand["action_type"]
            content = cand["action_content"]

            prompts.append(self._thinking_prompt(thinking, context))

            if action_type == "search":
                prompts.append(self._query_prompt(thinking, content, context))
                candidate_types.append("search")
            else:
                prompts.append(self._answer_prompt(content, ground_truth, context))
                candidate_types.append("answer")

        responses = await self._call_llm_batch_async(prompts)

        rewards = []
        idx = 0
        for i, cand_type in enumerate(candidate_types):
            r_think = self.extract_score(responses[idx])
            idx += 1
            r_action = self.extract_score(responses[idx])
            idx += 1

            if cand_type == "search":
                reward = self.compute_step_reward_search(r_think, r_action)
            else:
                reward = self.compute_step_reward_answer(
                    r_think, r_action, step, max_budget, lambda_bonus
                )
            rewards.append(reward)

        return rewards


def compute_score(solution_str, ground_truth, **kwargs):
    """
    Verl-compatible reward function using LLM judge.

    This is a wrapper that can be loaded as a custom reward function
    in the verl training config. Falls back to EM if judge is unavailable.
    """
    import re as _re

    judge_url = kwargs.get("judge_url", "http://localhost:9000/generate")
    judge = LLMJudgeReward(api_url=judge_url)

    # Extract answer from solution
    answer_match = _re.search(r"<answer>(.*?)</answer>", solution_str, _re.DOTALL)
    if not answer_match:
        return 0.0

    predicted = answer_match.group(1).strip()

    # Get ground truth target
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", ground_truth.get("targets", [str(ground_truth)]))
    elif isinstance(ground_truth, (list, tuple)):
        targets = list(ground_truth)
    else:
        targets = ground_truth

    gt_str = targets[0] if isinstance(targets, list) else str(targets)

    score = judge.reward_answer(predicted, gt_str, context=solution_str)
    # Map {-1, 0, +1} to {0, 0.5, 1} for compatibility with verl's reward system
    return (score + 1.0) / 2.0
