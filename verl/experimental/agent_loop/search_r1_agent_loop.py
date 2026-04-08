# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Custom agent loop that integrates search_r1 logic into the new verl framework.
This reuses the logic from search_r1/llm_agent/generation.py for compatibility.
"""

import asyncio
import copy
import logging
import os
import re
import requests
import math
import random
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    """
    Agent loop that uses the search_r1 logic for multi-turn search-based reasoning.
    
    This agent loop:
    1. Generates model responses
    2. Parses <search>query</search> or <answer>text</answer> tags
    3. Calls retrieval service for search actions
    4. Formats observations and continues the loop
    5. Stops when <answer> is generated or max_turns reached
    """
    
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SearchR1AgentLoop initialization")
        
        cls.tokenizer = tokenizer
        cls.processor = processor
        
        # Get configuration from rollout config
        cls.max_turns = config.get("max_turns", 10)
        cls.max_start_length = config.data.get("max_start_length", 2048)
        cls.max_prompt_length = config.data.max_prompt_length
        cls.max_response_length = config.data.max_response_length
        cls.max_obs_length = config.data.get("max_obs_length", 500)
        cls.no_think_rl = config.algorithm.get("no_think_rl", False)
        
        # Get retrieval configuration - check multiple possible locations
        cls.search_url = None
        cls.topk = 3
        
        # Try to get from retriever config (old style)
        if hasattr(config, "retriever") and config.retriever:
            cls.search_url = config.retriever.get("url", "http://127.0.0.1:8000/retrieve")
            cls.topk = config.retriever.get("topk", 3)
        # Try to get from custom config
        elif "search_url" in config:
            cls.search_url = config.search_url
            cls.topk = config.get("topk", 3)
        else:
            # Default to localhost
            cls.search_url = "http://127.0.0.1:8000/retrieve"
            cls.topk = 3
            logger.warning(f"No retrieval URL configured, using default: {cls.search_url}")
        
        print(f"SearchR1AgentLoop initialized with:")
        print(f"  max_turns: {cls.max_turns}")
        print(f"  max_prompt_length: {cls.max_prompt_length}")
        print(f"  max_response_length: {cls.max_response_length}")
        print(f"  search_url: {cls.search_url}")
        print(f"  topk: {cls.topk}")
        
        # Initialize tensor helper
        cls.tensor_helper = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=cls.max_prompt_length,
            max_obs_length=cls.max_obs_length,
            max_start_length=cls.max_start_length
        ))
    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the search_r1 agent loop."""
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex

        # ---------------------------------------------------------------------
        # Strict trajectory format enforcement (prevents invalid rollouts):
        #
        # 1) Force "<think>" immediately after the original user query.
        # 2) Generate thinking until "</think>" (stop sequence, inclusive).
        # 3) Force "\n" then choose exactly one of "<search>" or "<answer>".
        # 4) If search: generate until "</search>", then inject "\n<information>...</information>\n<think>".
        # 5) If answer: generate until "</answer>" and end.
        # ---------------------------------------------------------------------
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
            return await self.loop.run_in_executor(None, lambda: self.tokenizer.encode(text, add_special_tokens=False))

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
            # Some engines do not include the stop string in output; track this so callers can
            # forcibly append the close tag to keep trajectories well-formed.
            stop_included = any(s in seg_str for s in stop) if stop else True
            return out, seg_str, stop_included

        def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
            # Best-effort extractor; assumes close_tag is present.
            if close_tag in text:
                text = text.split(close_tag)[0]
            if open_tag in text:
                text = text.split(open_tag, 1)[1]
            return text.strip()

        def _sample_action_from_toplogprobs(
            *,
            top_logprobs_step0: dict[int, float],
            search_token_id: int,
            answer_token_id: int,
        ) -> str:
            # Sample between the two tokens proportional to exp(logprob).
            ls = top_logprobs_step0.get(search_token_id, -math.inf)
            la = top_logprobs_step0.get(answer_token_id, -math.inf)
            if not math.isfinite(ls) and not math.isfinite(la):
                # Fallback if neither is in top-k.
                return "search"
            # log-sum-exp for numerical stability
            m = max(ls, la)
            ps = math.exp(ls - m) if math.isfinite(ls) else 0.0
            pa = math.exp(la - m) if math.isfinite(la) else 0.0
            z = ps + pa
            if z <= 0.0:
                return "search"
            r = random.random()
            return "search" if (ps / z) >= r else "answer"

        async def _peek_next_token_toplogprobs(
            *,
            prompt_token_ids: list[int],
            topk: int = 20,
        ) -> Optional[dict[int, float]]:
            # Request 1 token with top-logprobs for constrained branching decisions.
            out, _, _ = await _generate_segment(
                prompt_token_ids=prompt_token_ids,
                stop=[],  # no stop; `max_tokens=1` ensures short output
                sp_override={"logprobs": int(topk), "max_tokens": 1},
            )
            if not out.top_logprobs:
                return None
            return out.top_logprobs[0]

        async def _score_token_sequence(
            *,
            prompt_token_ids: list[int],
            token_seq: list[int],
            topk: int = 20,
        ) -> float:
            # Approximate log-probability of emitting `token_seq` from `prompt_token_ids`
            # under the current sampling distribution, using repeated 1-token peeks.
            score = 0.0
            prefix = list(prompt_token_ids)
            for tok in token_seq:
                top_lp = await _peek_next_token_toplogprobs(prompt_token_ids=prefix, topk=topk)
                if top_lp is None:
                    return -math.inf
                lp = top_lp.get(int(tok), -math.inf)
                if not math.isfinite(lp):
                    return -math.inf
                score += float(lp)
                prefix.append(int(tok))
            return score

        def _sample_from_scores(scores: dict[str, float], *, temperature: float) -> str:
            # Sample proportional to exp(score / temperature) (stable).
            # This makes the branch decision follow the same temperature behavior as token sampling.
            if temperature is None:
                temperature = 1.0
            if temperature <= 0:
                # Greedy / deterministic
                return max(scores.items(), key=lambda kv: kv[1])[0]
            best = max(scores.values())
            weights = {
                k: (math.exp((v - best) / float(temperature)) if math.isfinite(v) else 0.0)
                for k, v in scores.items()
            }
            z = sum(weights.values())
            if z <= 0.0:
                return "search"
            r = random.random() * z
            acc = 0.0
            for k, w in weights.items():
                acc += w
                if r <= acc:
                    return k
            return next(iter(weights.keys()))
        
        # Get initial prompt token IDs
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        
        # Truncate to max_start_length for initial input
        initial_prompt_ids = torch.tensor([prompt_ids[-self.max_start_length:]], dtype=torch.long)
        
        # Initialize state tracking
        all_response_ids = []
        all_response_mask = []
        all_info_mask = []  # Track which tokens are observations (to mask from loss)
        active = True
        turn_count = 0
        skip_think_open = False  # Track if <think> was already added after </information>

        def _append_with_budget(
            token_ids: list[int],
            *,
            is_llm_token: int,
            is_trainable: int,
        ) -> list[int]:
            """Append tokens to trajectory buffers under the global max_response_length budget.

            Returns the possibly-truncated token list actually appended.
            """
            remaining = self.max_response_length - len(all_response_ids)
            if remaining <= 0:
                return []
            token_ids = token_ids[:remaining]
            all_response_ids.extend(token_ids)
            all_response_mask.extend([is_llm_token] * len(token_ids))
            all_info_mask.extend([is_trainable] * len(token_ids))
            return token_ids
        
        # Current working prompt that grows with each turn
        current_prompt_ids = initial_prompt_ids.clone()
        
        with simple_timer("search_r1_agent_loop", metrics):
            # Main multi-turn loop (each turn strictly follows: think -> (search|answer))
            for turn in range(self.max_turns):
                if not active:
                    break
                
                turn_count += 1

                # Force "<think>" at the start of the turn (unless already added after </information>)
                if not skip_think_open:
                    forced_think_open_ids = await _encode(THINK_OPEN)
                    appended_forced_think = _append_with_budget(
                        forced_think_open_ids, is_llm_token=1, is_trainable=1
                    )
                    if len(appended_forced_think) < len(forced_think_open_ids):
                        active = False
                        break
                    current_prompt_ids = torch.cat(
                        [current_prompt_ids, torch.tensor([appended_forced_think], dtype=torch.long)], dim=1
                    )
                    if current_prompt_ids.shape[1] > self.max_prompt_length:
                        current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]
                else:
                    # <think> was already added after </information>, just reset the flag
                    skip_think_open = False

                # Stage A: generate thinking text until "</think>" (inclusive).
                # Also stop on "</answer>" in case the model generates an answer prematurely inside <think>.
                with simple_timer(f"generate_think_{turn}", metrics):
                    think_out, think_str, think_stop_included = await _generate_segment(
                        prompt_token_ids=current_prompt_ids[0].tolist(),
                        stop=[THINK_CLOSE, ANSWER_CLOSE],
                    )
                
                # Check if model generated </answer> prematurely inside <think>
                if ANSWER_CLOSE in think_str:
                    # Model generated an answer - append tokens and terminate
                    appended_think_ids = _append_with_budget(think_out.token_ids, is_llm_token=1, is_trainable=1)
                    if len(appended_think_ids) < len(think_out.token_ids):
                        active = False
                        break
                    if not think_stop_included:
                        forced_answer_close_ids = await _encode(ANSWER_CLOSE)
                        appended_close = _append_with_budget(forced_answer_close_ids, is_llm_token=1, is_trainable=1)
                        if len(appended_close) < len(forced_answer_close_ids):
                            active = False
                            break
                        appended_think_ids = appended_think_ids + appended_close
                    active = False
                    break
                
                # Normal case: model generated </think>
                appended_think_ids = _append_with_budget(think_out.token_ids, is_llm_token=1, is_trainable=1)
                if len(appended_think_ids) < len(think_out.token_ids):
                    active = False
                    break
                if not think_stop_included:
                    forced_think_close_ids = await _encode(THINK_CLOSE)
                    appended_close = _append_with_budget(forced_think_close_ids, is_llm_token=1, is_trainable=1)
                    if len(appended_close) < len(forced_think_close_ids):
                        active = False
                        break
                    appended_think_ids = appended_think_ids + appended_close
                current_prompt_ids = torch.cat(
                    [current_prompt_ids, torch.tensor([appended_think_ids], dtype=torch.long)], dim=1
                )
                if current_prompt_ids.shape[1] > self.max_prompt_length:
                    current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

                # Force newline after "</think>".
                forced_newline_ids = await _encode("\n")
                appended_nl = _append_with_budget(forced_newline_ids, is_llm_token=1, is_trainable=1)
                if len(appended_nl) < len(forced_newline_ids):
                    active = False
                    break
                current_prompt_ids = torch.cat(
                    [current_prompt_ids, torch.tensor([appended_nl], dtype=torch.long)], dim=1
                )
                if current_prompt_ids.shape[1] > self.max_prompt_length:
                    current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

                # Stage B: choose exactly one of "<search>" or "<answer>".
                # We force '<' then sample between the *words* "search" and "answer" (plus the closing '>'),
                # because the full tags are often multi-token.
                lt_ids = await _encode("<")
                appended_lt = _append_with_budget(lt_ids, is_llm_token=1, is_trainable=1)
                if len(appended_lt) < len(lt_ids):
                    active = False
                    break
                current_prompt_ids = torch.cat(
                    [current_prompt_ids, torch.tensor([appended_lt], dtype=torch.long)], dim=1
                )
                if current_prompt_ids.shape[1] > self.max_prompt_length:
                    current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

                search_word_ids = await _encode("search")
                answer_word_ids = await _encode("answer")
                gt_ids = await _encode(">")
                # Score "search>" vs "answer>" from the same prefix (after the forced '<').
                with simple_timer(f"sample_action_{turn}", metrics):
                    base_prefix = current_prompt_ids[0].tolist()
                    scores = {
                        "search": await _score_token_sequence(
                            prompt_token_ids=base_prefix, token_seq=search_word_ids + gt_ids, topk=20
                        ),
                        "answer": await _score_token_sequence(
                            prompt_token_ids=base_prefix, token_seq=answer_word_ids + gt_ids, topk=20
                        ),
                    }
                    action = _sample_from_scores(scores, temperature=float(sampling_params.get("temperature", 1.0)))

                forced_word_ids = search_word_ids if action == "search" else answer_word_ids
                forced_open_ids = forced_word_ids + gt_ids
                appended_open = _append_with_budget(forced_open_ids, is_llm_token=1, is_trainable=1)
                if len(appended_open) < len(forced_open_ids):
                    active = False
                    break
                current_prompt_ids = torch.cat(
                    [current_prompt_ids, torch.tensor([appended_open], dtype=torch.long)], dim=1
                )
                if current_prompt_ids.shape[1] > self.max_prompt_length:
                    current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]

                # Stage C: generate the content for the chosen action (inclusive close tag).
                if action == "search":
                    with simple_timer(f"generate_search_{turn}", metrics):
                        search_out, search_str, search_stop_included = await _generate_segment(
                            prompt_token_ids=current_prompt_ids[0].tolist(),
                            stop=[SEARCH_CLOSE],
                        )
                    appended_search_ids = _append_with_budget(search_out.token_ids, is_llm_token=1, is_trainable=1)
                    if len(appended_search_ids) < len(search_out.token_ids):
                        active = False
                        break
                    if not search_stop_included:
                        forced_search_close_ids = await _encode(SEARCH_CLOSE)
                        appended_close = _append_with_budget(forced_search_close_ids, is_llm_token=1, is_trainable=1)
                        if len(appended_close) < len(forced_search_close_ids):
                            active = False
                            break
                        appended_search_ids = appended_search_ids + appended_close
                        search_str = search_str + SEARCH_CLOSE

                    # Extract query and perform search.
                    query = _extract_between(f"{SEARCH_OPEN}{search_str}", SEARCH_OPEN, SEARCH_CLOSE)
                    with simple_timer(f"search_turn_{turn}", metrics):
                        observation = await self.loop.run_in_executor(None, lambda: self._perform_search(query))

                    # Requirement (3): results are always inside <information>...</information>
                    # with a newline before and after.
                    # Truncate the observation content BEFORE adding tags to ensure tags are never cut off
                    observation_content = observation.strip()
                    
                    # Pre-encode to check length, leaving room for the wrapper tags
                    wrapper_str = "\n<information></information>\n"
                    wrapper_ids = await _encode(wrapper_str)
                    max_content_length = self.max_obs_length - len(wrapper_ids)
                    
                    # Encode observation and truncate if needed
                    obs_content_ids = await _encode(observation_content)
                    if len(obs_content_ids) > max_content_length:
                        logger.warning(
                            f"Observation too long: {len(obs_content_ids)} > {max_content_length}, truncating content"
                        )
                        obs_content_ids = obs_content_ids[:max_content_length]
                        observation_content = await _decode(obs_content_ids)
                    
                    # Now add the tags after truncation
                    obs_str = f"\n<information>{observation_content}</information>\n"
                    obs_ids = await _encode(obs_str)

                    appended_obs_ids = _append_with_budget(obs_ids, is_llm_token=0, is_trainable=0)
                    if len(appended_obs_ids) < len(obs_ids):
                        active = False
                        break

                    # Add <think> immediately after </information> (trainable)
                    forced_think_after_info_ids = await _encode(THINK_OPEN)
                    appended_think_after_info = _append_with_budget(
                        forced_think_after_info_ids, is_llm_token=1, is_trainable=1
                    )
                    if len(appended_think_after_info) < len(forced_think_after_info_ids):
                        active = False
                        break

                    # Update prompt: include generated search content + observation + <think>
                    current_prompt_ids = torch.cat(
                        [
                            current_prompt_ids,
                            torch.tensor([appended_search_ids], dtype=torch.long),
                            torch.tensor([appended_obs_ids], dtype=torch.long),
                            torch.tensor([appended_think_after_info], dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if current_prompt_ids.shape[1] > self.max_prompt_length:
                        current_prompt_ids = current_prompt_ids[:, -self.max_prompt_length:]
                    
                    # Set flag since <think> was already added after </information>
                    skip_think_open = True
                    continue

                # action == "answer"
                with simple_timer(f"generate_answer_{turn}", metrics):
                    answer_out, answer_str, answer_stop_included = await _generate_segment(
                        prompt_token_ids=current_prompt_ids[0].tolist(),
                        stop=[ANSWER_CLOSE],
                    )
                appended_answer_ids = _append_with_budget(answer_out.token_ids, is_llm_token=1, is_trainable=1)
                if len(appended_answer_ids) < len(answer_out.token_ids):
                    active = False
                    break
                if not answer_stop_included:
                    forced_answer_close_ids = await _encode(ANSWER_CLOSE)
                    appended_close = _append_with_budget(forced_answer_close_ids, is_llm_token=1, is_trainable=1)
                    if len(appended_close) < len(forced_answer_close_ids):
                        active = False
                        break
                    appended_answer_ids = appended_answer_ids + appended_close
                # Requirement (4): end generation when "</answer>" is generated.
                active = False
                break
            
            # If still active after max turns, generate final response
            if active:
                # Fallback: if we ran out of turns without producing an answer, force an answer stage.
                forced_nl = await _encode("\n")
                _append_with_budget(forced_nl, is_llm_token=1, is_trainable=1)
                forced_answer_open_ids = await _encode(ANSWER_OPEN)
                appended_open = _append_with_budget(forced_answer_open_ids, is_llm_token=1, is_trainable=1)
                if appended_open:
                    current_prompt_ids = torch.cat(
                        [current_prompt_ids, torch.tensor([appended_open], dtype=torch.long)], dim=1
                    )
                with simple_timer("generate_final_answer", metrics):
                    final_out, final_str, final_stop_included = await _generate_segment(
                        prompt_token_ids=current_prompt_ids[0].tolist(),
                        stop=[ANSWER_CLOSE],
                    )
                _append_with_budget(final_out.token_ids, is_llm_token=1, is_trainable=1)
                if not final_stop_included:
                    forced_answer_close_ids = await _encode(ANSWER_CLOSE)
                    _append_with_budget(forced_answer_close_ids, is_llm_token=1, is_trainable=1)
                active = False
        
        metrics['total_turns'] = turn_count
        
        # Calculate state masking statistics
        total_tokens = len(all_response_mask)
        trainable_tokens = sum(all_info_mask)
        masked_tokens = total_tokens - trainable_tokens
        metrics['state_masking/total_tokens'] = total_tokens
        metrics['state_masking/trainable_tokens'] = trainable_tokens
        metrics['state_masking/masked_tokens'] = masked_tokens
        metrics['state_masking/trainable_ratio'] = trainable_tokens / total_tokens if total_tokens > 0 else 0.0
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            response_logprobs=None,
            metrics=metrics,
            extra_fields={},
        )
        
        # Add info_mask to extra_fields for state masking during training
        output.extra_fields['info_mask'] = all_info_mask
        
        return output
    
    def _postprocess_response(self, response_str: str) -> str:
        """Post-process response to stop at search or answer operation."""
        if '</search>' in response_str:
            return response_str.split('</search>')[0] + '</search>'
        elif '</answer>' in response_str:
            return response_str.split('</answer>')[0] + '</answer>'
        return response_str
    
    def _parse_action(self, response_str: str) -> Tuple[str, str]:
        """
        Parse action and content from response string.
        
        Returns:
            Tuple of (action, content) where action is 'search', 'answer', or None
        """
        pattern = r'<(search|answer)>(.*?)</\1>'
        match = re.search(pattern, response_str, re.DOTALL)
        
        if match:
            action = match.group(1)  # 'search' or 'answer'
            content = match.group(2).strip()
            return action, content
        
        return None, ""
    
    def _perform_search(self, query: str) -> str:
        """
        Perform search using the retrieval service.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results as string
        """
        try:
            payload = {
                "queries": [query],
                "topk": self.topk,
                "return_scores": True
            }
            
            response = requests.post(self.search_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Format the results
            if 'result' in result and len(result['result']) > 0:
                return self._format_passages(result['result'][0])
            else:
                return "No results found."
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search error: {str(e)}"
    
    def _format_passages(self, retrieval_result: List[dict]) -> str:
        """
        Format retrieval results into a readable string.
        
        Args:
            retrieval_result: List of retrieved documents
            
        Returns:
            Formatted string with document titles and contents
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        
        return format_reference.strip()
