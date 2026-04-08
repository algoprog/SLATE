"""
SLATE Agent Loop: Truncated Step-Level Sampling with Dense LLM-Judge Rewards.

Implements Algorithm 1 from the SLATE paper. At each step t:
1. Generate k candidate actions (think + search/answer) from a shared prefix
2. Evaluate each candidate with the LLM judge (decomposed ternary rewards)
3. Compute step-level GRPO advantages
4. Select the winning action via reward-weighted sampling
5. Extend the prefix and repeat

All k candidates and their rewards/advantages are stored for training.
"""

import asyncio
import copy
import logging
import math
import os
import random
import re
import requests
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig

from slate.rewards.llm_judge import LLMJudgeReward
from slate.training.step_level_algos import compute_step_level_advantage, reward_weighted_sampling

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("slate_agent")
class SLATEAgentLoop(AgentLoopBase):
    """
    SLATE agent loop implementing truncated step-level sampling.

    At each step, generates k candidate actions from a shared prefix,
    evaluates them with the LLM judge, computes step-level GRPO advantages,
    and selects the best action to extend the trajectory.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SLATEAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor

        # Standard rollout config
        cls.max_turns = config.get("max_turns", 4)
        cls.max_start_length = config.data.get("max_start_length", 2048)
        cls.max_prompt_length = config.data.max_prompt_length
        cls.max_response_length = config.data.max_response_length
        cls.max_obs_length = config.data.get("max_obs_length", 500)
        cls.no_think_rl = config.algorithm.get("no_think_rl", False)

        # SLATE-specific config
        slate_cfg = config.get("slate", {})
        cls.k = slate_cfg.get("k", 5)
        cls.eta = slate_cfg.get("eta", 0.7)
        cls.max_budget = slate_cfg.get("max_budget", 4)
        cls.lambda_bonus = slate_cfg.get("lambda_bonus", 0.1)
        cls.judge_url = slate_cfg.get("judge_url", "http://localhost:9000/generate")
        cls.judge_temperature = slate_cfg.get("judge_temperature", 0.0)

        # Retrieval config
        cls.search_url = None
        cls.topk = 3
        if hasattr(config, "retriever") and config.retriever:
            cls.search_url = config.retriever.get("url", "http://127.0.0.1:8000/retrieve")
            cls.topk = config.retriever.get("topk", 3)
        elif "search_url" in config:
            cls.search_url = config.search_url
            cls.topk = config.get("topk", 3)
        else:
            cls.search_url = "http://127.0.0.1:8000/retrieve"

        # Initialize LLM judge
        cls.judge = LLMJudgeReward(api_url=cls.judge_url, temperature=cls.judge_temperature)

        # Initialize tensor helper
        cls.tensor_helper = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=cls.max_prompt_length,
            max_obs_length=cls.max_obs_length,
            max_start_length=cls.max_start_length,
        ))

        print(f"SLATEAgentLoop initialized:")
        print(f"  k={cls.k}, eta={cls.eta}, max_budget={cls.max_budget}")
        print(f"  lambda_bonus={cls.lambda_bonus}")
        print(f"  judge_url={cls.judge_url}")
        print(f"  search_url={cls.search_url}, topk={cls.topk}")

    def _perform_search(self, query: str) -> str:
        """Call retrieval service."""
        try:
            response = requests.post(
                self.search_url,
                json={"query": query, "topk": self.topk},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()
            if isinstance(results, list):
                return "\n".join([str(r) for r in results[:self.topk]])
            elif isinstance(results, dict) and "results" in results:
                return "\n".join([str(r) for r in results["results"][:self.topk]])
            return str(results)
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return "No results found."

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the SLATE truncated step-level sampling loop."""
        messages = list(kwargs["raw_prompt"])
        ground_truth = kwargs.get("reward_model", {})
        if isinstance(ground_truth, dict):
            gt_str = ground_truth.get("target", [""])[0] if isinstance(ground_truth.get("target"), list) else str(ground_truth.get("target", ""))
        elif isinstance(ground_truth, (list, tuple)):
            gt_str = str(ground_truth[0]) if ground_truth else ""
        else:
            gt_str = str(ground_truth)

        metrics = {}
        request_id = uuid4().hex

        # Tags
        THINK_OPEN = "<think>"
        THINK_CLOSE = "</think>"
        SEARCH_OPEN = "<search>"
        SEARCH_CLOSE = "</search>"
        ANSWER_OPEN = "<answer>"
        ANSWER_CLOSE = "</answer>"

        def _sampling_params_with_stop(stops: list[str]) -> dict[str, Any]:
            sp = dict(sampling_params)
            sp["stop"] = stops
            return sp

        async def _encode(text: str) -> list[int]:
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
            )

        async def _decode(token_ids: list[int]) -> str:
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True)
            )

        async def _generate_segment(
            *,
            prompt_token_ids: list[int],
            stop: list[str],
            sp_override: Optional[dict[str, Any]] = None,
        ):
            sp = _sampling_params_with_stop(stop)
            if sp_override:
                sp.update(sp_override)
            out = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_token_ids,
                sampling_params=sp,
                image_data=None,
            )
            seg_str = await _decode(out.token_ids)
            stop_included = any(s in seg_str for s in stop) if stop else True
            return out, seg_str, stop_included

        def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
            if close_tag in text:
                text = text.split(close_tag)[0]
            if open_tag in text:
                text = text.split(open_tag, 1)[1]
            return text.strip()

        # Build initial prompt
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        initial_prompt_ids = torch.tensor([prompt_ids[-self.max_start_length:]], dtype=torch.long)

        # Extract question text for context building
        question_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                question_text = msg.get("content", "")

        # State tracking
        all_response_ids = []
        all_response_mask = []
        all_info_mask = []

        # SLATE-specific: store per-step candidate data for training
        step_data = []  # List of dicts per step

        current_prompt_ids = initial_prompt_ids.clone()
        context = f"Question: {question_text}"
        active = True
        turn_count = 0

        def _append_with_budget(
            token_ids: list[int],
            *,
            is_llm_token: int,
            is_trainable: int,
        ) -> list[int]:
            remaining = self.max_response_length - len(all_response_ids)
            if remaining <= 0:
                return []
            token_ids = token_ids[:remaining]
            all_response_ids.extend(token_ids)
            all_response_mask.extend([is_llm_token] * len(token_ids))
            all_info_mask.extend([is_trainable] * len(token_ids))
            return token_ids

        with simple_timer("slate_agent_loop", metrics):
            for turn in range(self.max_budget):
                if not active:
                    break
                turn_count += 1

                # --------------------------------------------------------
                # Generate k candidate actions from the shared prefix
                # Each candidate: <think>...</think>\n<search>query</search>
                #              or <think>...</think>\n<answer>text</answer>
                # --------------------------------------------------------

                candidates = []
                candidate_tasks = []

                for j in range(self.k):
                    candidate_tasks.append(
                        self._generate_single_candidate(
                            current_prompt_ids=current_prompt_ids.clone(),
                            sampling_params=sampling_params,
                            turn=turn,
                            candidate_idx=j,
                        )
                    )

                # Run k candidate generations concurrently
                with simple_timer(f"generate_k_candidates_turn_{turn}", metrics):
                    candidate_results = await asyncio.gather(*candidate_tasks)

                for result in candidate_results:
                    candidates.append(result)

                # --------------------------------------------------------
                # Evaluate all k candidates with LLM judge
                # --------------------------------------------------------
                judge_candidates = []
                for cand in candidates:
                    judge_candidates.append({
                        "thinking": cand["thinking"],
                        "action_type": cand["action_type"],
                        "action_content": cand["action_content"],
                    })

                with simple_timer(f"judge_evaluate_turn_{turn}", metrics):
                    rewards = await self.judge.evaluate_candidates_batch_async(
                        candidates=judge_candidates,
                        context=context,
                        ground_truth=gt_str,
                        step=turn,
                        max_budget=self.max_budget,
                        lambda_bonus=self.lambda_bonus,
                    )

                # --------------------------------------------------------
                # Compute step-level GRPO advantages
                # --------------------------------------------------------
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                advantages = compute_step_level_advantage(rewards_tensor)

                # --------------------------------------------------------
                # Select winning action via reward-weighted sampling
                # --------------------------------------------------------
                selected_idx = reward_weighted_sampling(advantages, temperature=self.eta)

                # Store step data for training
                step_data.append({
                    "step": turn,
                    "candidates": candidates,
                    "rewards": rewards,
                    "advantages": advantages.tolist(),
                    "selected_idx": selected_idx,
                })

                # --------------------------------------------------------
                # Append the selected candidate's tokens to the trajectory
                # --------------------------------------------------------
                selected = candidates[selected_idx]

                # Append the selected candidate's tokens
                appended = _append_with_budget(
                    selected["token_ids"], is_llm_token=1, is_trainable=1
                )
                if len(appended) < len(selected["token_ids"]):
                    active = False
                    break

                current_prompt_ids = torch.cat(
                    [current_prompt_ids, torch.tensor([appended], dtype=torch.long)], dim=1
                )
                if current_prompt_ids.shape[1] > self.max_prompt_length:
                    current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

                # Update context for next step's judge evaluation
                if selected["thinking"]:
                    context += f"\nThinking: {selected['thinking']}"

                if selected["action_type"] == "answer":
                    active = False
                    break

                if selected["action_type"] == "search":
                    query = selected["action_content"]
                    context += f"\nQuery: {query}"

                    # Perform retrieval
                    with simple_timer(f"search_turn_{turn}", metrics):
                        observation = await self.loop.run_in_executor(
                            None, lambda q=query: self._perform_search(q)
                        )

                    # Truncate observation
                    obs_content = observation.strip()
                    wrapper_str = "\n<information></information>\n"
                    wrapper_ids = await _encode(wrapper_str)
                    max_content_length = self.max_obs_length - len(wrapper_ids)

                    obs_content_ids = await _encode(obs_content)
                    if len(obs_content_ids) > max_content_length:
                        obs_content_ids = obs_content_ids[:max_content_length]
                        obs_content = await _decode(obs_content_ids)

                    obs_str = f"\n<information>{obs_content}</information>\n"
                    obs_ids = await _encode(obs_str)

                    appended_obs = _append_with_budget(obs_ids, is_llm_token=0, is_trainable=0)
                    if len(appended_obs) < len(obs_ids):
                        active = False
                        break

                    context += f"\nRetrieved: {obs_content[:300]}"

                    # Add <think> for next turn
                    think_open_ids = await _encode(THINK_OPEN)
                    appended_think = _append_with_budget(think_open_ids, is_llm_token=1, is_trainable=1)
                    if len(appended_think) < len(think_open_ids):
                        active = False
                        break

                    current_prompt_ids = torch.cat(
                        [
                            current_prompt_ids,
                            torch.tensor([appended_obs], dtype=torch.long),
                            torch.tensor([appended_think], dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if current_prompt_ids.shape[1] > self.max_prompt_length:
                        current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

            # If still active after max turns, force an answer
            if active:
                forced_nl = await _encode("\n")
                _append_with_budget(forced_nl, is_llm_token=1, is_trainable=1)
                forced_answer_open_ids = await _encode(ANSWER_OPEN)
                appended_open = _append_with_budget(forced_answer_open_ids, is_llm_token=1, is_trainable=1)
                if appended_open:
                    current_prompt_ids = torch.cat(
                        [current_prompt_ids, torch.tensor([appended_open], dtype=torch.long)], dim=1
                    )
                final_out, final_str, final_stop_included = await _generate_segment(
                    prompt_token_ids=current_prompt_ids[0].tolist(),
                    stop=[ANSWER_CLOSE],
                )
                _append_with_budget(final_out.token_ids, is_llm_token=1, is_trainable=1)
                if not final_stop_included:
                    forced_close = await _encode(ANSWER_CLOSE)
                    _append_with_budget(forced_close, is_llm_token=1, is_trainable=1)

        metrics["total_turns"] = turn_count
        metrics["k_candidates_per_step"] = self.k

        total_tokens = len(all_response_mask)
        trainable_tokens = sum(all_info_mask)
        metrics["state_masking/total_tokens"] = total_tokens
        metrics["state_masking/trainable_tokens"] = trainable_tokens
        metrics["state_masking/masked_tokens"] = total_tokens - trainable_tokens

        # Build output
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            num_turns=turn_count,
            metrics=metrics,
            extra_fields={
                "info_mask": all_info_mask,
                "step_data": step_data,
                "training_mode": "slate",
            },
        )
        return output

    async def _generate_single_candidate(
        self,
        current_prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        turn: int,
        candidate_idx: int,
    ) -> dict:
        """
        Generate a single candidate action from the current prefix.

        Returns a dict with:
            - thinking: str
            - action_type: "search" or "answer"
            - action_content: query or answer text
            - token_ids: list of all generated token ids for this candidate
        """
        THINK_OPEN = "<think>"
        THINK_CLOSE = "</think>"
        SEARCH_OPEN = "<search>"
        SEARCH_CLOSE = "</search>"
        ANSWER_OPEN = "<answer>"
        ANSWER_CLOSE = "</answer>"

        async def _encode(text):
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
            )

        async def _decode(token_ids):
            return await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True)
            )

        def _sampling_params_with_stop(stops):
            sp = dict(sampling_params)
            sp["stop"] = stops
            return sp

        # Use a unique request_id per candidate for independent sampling
        req_id = uuid4().hex
        all_token_ids = []
        working_prompt = current_prompt_ids.clone()

        # If this is the first turn, force <think> open tag
        if turn == 0:
            think_open_ids = await _encode(THINK_OPEN)
            all_token_ids.extend(think_open_ids)
            working_prompt = torch.cat(
                [working_prompt, torch.tensor([think_open_ids], dtype=torch.long)], dim=1
            )

        # Stage A: Generate thinking until </think>
        sp = _sampling_params_with_stop([THINK_CLOSE, ANSWER_CLOSE])
        think_out = await self.server_manager.generate(
            request_id=req_id,
            prompt_ids=working_prompt[0].tolist(),
            sampling_params=sp,
            image_data=None,
        )
        think_str = await _decode(think_out.token_ids)
        all_token_ids.extend(think_out.token_ids)

        # Check if model produced answer prematurely inside think
        if ANSWER_CLOSE in think_str:
            # Extract thinking and answer
            thinking = think_str.split(ANSWER_CLOSE)[0]
            answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", think_str, re.DOTALL)
            answer_text = answer_match.group(1).strip() if answer_match else thinking.strip()
            return {
                "thinking": thinking.strip(),
                "action_type": "answer",
                "action_content": answer_text,
                "token_ids": all_token_ids,
            }

        # Ensure </think> is included
        if THINK_CLOSE not in think_str:
            close_ids = await _encode(THINK_CLOSE)
            all_token_ids.extend(close_ids)
            think_str += THINK_CLOSE

        thinking = _extract_think(think_str, THINK_OPEN, THINK_CLOSE)

        working_prompt = torch.cat(
            [working_prompt, torch.tensor([think_out.token_ids], dtype=torch.long)], dim=1
        )
        if working_prompt.shape[1] > self.max_prompt_length:
            working_prompt = working_prompt[:, -self.max_prompt_length:]

        # Add newline
        nl_ids = await _encode("\n")
        all_token_ids.extend(nl_ids)
        working_prompt = torch.cat(
            [working_prompt, torch.tensor([nl_ids], dtype=torch.long)], dim=1
        )

        # Stage B: Force "<" then decide search vs answer
        lt_ids = await _encode("<")
        all_token_ids.extend(lt_ids)
        working_prompt = torch.cat(
            [working_prompt, torch.tensor([lt_ids], dtype=torch.long)], dim=1
        )

        search_word_ids = await _encode("search")
        answer_word_ids = await _encode("answer")
        gt_ids = await _encode(">")

        # Score "search>" vs "answer>" from the same prefix
        base_prefix = working_prompt[0].tolist()

        async def _score_seq(token_seq):
            score = 0.0
            prefix = list(base_prefix)
            for tok in token_seq:
                sp_peek = _sampling_params_with_stop([])
                sp_peek["logprobs"] = 20
                sp_peek["max_tokens"] = 1
                out = await self.server_manager.generate(
                    request_id=req_id,
                    prompt_ids=prefix,
                    sampling_params=sp_peek,
                    image_data=None,
                )
                if out.top_logprobs:
                    lp = out.top_logprobs[0].get(int(tok), -math.inf)
                    if not math.isfinite(lp):
                        return -math.inf
                    score += float(lp)
                else:
                    return -math.inf
                prefix.append(int(tok))
            return score

        s_score = await _score_seq(search_word_ids + gt_ids)
        a_score = await _score_seq(answer_word_ids + gt_ids)

        # Sample proportionally
        temp = float(sampling_params.get("temperature", 1.0))
        scores = {"search": s_score, "answer": a_score}
        if temp <= 0:
            action = max(scores, key=scores.get)
        else:
            best = max(scores.values())
            weights = {
                k: (math.exp((v - best) / temp) if math.isfinite(v) else 0.0)
                for k, v in scores.items()
            }
            z = sum(weights.values())
            if z <= 0:
                action = "search"
            else:
                r = random.random() * z
                acc = 0.0
                action = "search"
                for k, w in weights.items():
                    acc += w
                    if r <= acc:
                        action = k
                        break

        forced_word_ids = search_word_ids if action == "search" else answer_word_ids
        forced_open_ids = forced_word_ids + gt_ids
        all_token_ids.extend(forced_open_ids)
        working_prompt = torch.cat(
            [working_prompt, torch.tensor([forced_open_ids], dtype=torch.long)], dim=1
        )

        # Stage C: Generate content for chosen action
        if action == "search":
            sp = _sampling_params_with_stop([SEARCH_CLOSE])
            search_out = await self.server_manager.generate(
                request_id=req_id,
                prompt_ids=working_prompt[0].tolist(),
                sampling_params=sp,
                image_data=None,
            )
            search_str = await _decode(search_out.token_ids)
            all_token_ids.extend(search_out.token_ids)

            if SEARCH_CLOSE not in search_str:
                close_ids = await _encode(SEARCH_CLOSE)
                all_token_ids.extend(close_ids)
                search_str += SEARCH_CLOSE

            query = search_str.replace(SEARCH_CLOSE, "").strip()

            return {
                "thinking": thinking,
                "action_type": "search",
                "action_content": query,
                "token_ids": all_token_ids,
            }

        else:  # answer
            sp = _sampling_params_with_stop([ANSWER_CLOSE])
            answer_out = await self.server_manager.generate(
                request_id=req_id,
                prompt_ids=working_prompt[0].tolist(),
                sampling_params=sp,
                image_data=None,
            )
            answer_str = await _decode(answer_out.token_ids)
            all_token_ids.extend(answer_out.token_ids)

            if ANSWER_CLOSE not in answer_str:
                close_ids = await _encode(ANSWER_CLOSE)
                all_token_ids.extend(close_ids)
                answer_str += ANSWER_CLOSE

            answer_text = answer_str.replace(ANSWER_CLOSE, "").strip()

            return {
                "thinking": thinking,
                "action_type": "answer",
                "action_content": answer_text,
                "token_ids": all_token_ids,
            }


def _extract_think(text: str, open_tag: str, close_tag: str) -> str:
    """Extract content between think tags."""
    if close_tag in text:
        text = text.split(close_tag)[0]
    if open_tag in text:
        text = text.split(open_tag, 1)[1]
    return text.strip()
