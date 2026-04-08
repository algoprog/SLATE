"""
Trajectory parsing utilities for SLATE.

Parses multi-turn search-augmented reasoning trajectories
into structured step-level representations.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajectoryStep:
    """A single step in a search-augmented reasoning trajectory."""
    thinking: str = ""
    action_type: str = ""  # "search" or "answer"
    action_content: str = ""  # query text or answer text
    information: Optional[str] = None  # retrieved docs (None for answer steps)
    step_index: int = 0


@dataclass
class Trajectory:
    """A complete multi-step trajectory."""
    question: str = ""
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    ground_truth: str = ""

    @property
    def num_search_steps(self) -> int:
        return sum(1 for s in self.steps if s.action_type == "search")

    @property
    def num_steps(self) -> int:
        return len(self.steps)


def parse_trajectory_text(text: str) -> Trajectory:
    """
    Parse a trajectory string with <think>, <search>, <information>, <answer> tags
    into a structured Trajectory object.

    Expected format:
        <think>reasoning</think>
        <search>query</search>
        <information>docs</information>
        <think>reasoning</think>
        ...
        <answer>final answer</answer>
    """
    traj = Trajectory()

    # Extract final answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        traj.final_answer = answer_match.group(1).strip()

    # Tokenize into segments by XML tags
    # Pattern captures tag name and content
    tag_pattern = re.compile(
        r"<(think|search|information|answer)>(.*?)</\1>",
        re.DOTALL,
    )

    segments = [(m.group(1), m.group(2).strip()) for m in tag_pattern.finditer(text)]

    current_step = TrajectoryStep(step_index=0)
    step_idx = 0

    for tag, content in segments:
        if tag == "think":
            if current_step.action_type:
                # Previous step is complete, start a new one
                traj.steps.append(current_step)
                step_idx += 1
                current_step = TrajectoryStep(step_index=step_idx)
            current_step.thinking = content

        elif tag == "search":
            current_step.action_type = "search"
            current_step.action_content = content

        elif tag == "information":
            current_step.information = content

        elif tag == "answer":
            current_step.action_type = "answer"
            current_step.action_content = content
            traj.steps.append(current_step)
            break

    # Append last step if it has an action but wasn't added
    if current_step.action_type and (not traj.steps or traj.steps[-1] is not current_step):
        traj.steps.append(current_step)

    return traj


def build_context_at_step(question: str, steps: List[TrajectoryStep], up_to_step: int) -> str:
    """
    Build the accumulated context string up to (but not including) step `up_to_step`.
    This context is passed to the LLM judge for evaluation.
    """
    context = f"Question: {question}"

    for step in steps[:up_to_step]:
        if step.thinking:
            context += f"\nThinking: {step.thinking}"
        if step.action_type == "search" and step.action_content:
            context += f"\nQuery: {step.action_content}"
        if step.information:
            context += f"\nRetrieved: {step.information[:500]}"

    return context


def extract_thinking_and_action(text: str) -> Tuple[str, str, str]:
    """
    Extract thinking, action type, and action content from a single
    candidate's generated text.

    Returns:
        (thinking, action_type, action_content)
        action_type is "search" or "answer"
    """
    thinking = ""
    action_type = ""
    action_content = ""

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    search_match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if search_match:
        action_type = "search"
        action_content = search_match.group(1).strip()
    elif answer_match:
        action_type = "answer"
        action_content = answer_match.group(1).strip()

    return thinking, action_type, action_content
