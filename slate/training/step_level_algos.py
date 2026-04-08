"""
Step-level algorithms for SLATE training.

Implements:
- Step-level GRPO advantage estimation (Eq. 5 from paper)
- Reward-weighted sampling for action selection (Section 3.2)
- Best-of-k selection (greedy alternative)
"""

import torch
import numpy as np
from typing import List, Optional


def compute_step_level_advantage(
    step_rewards: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute step-level group-relative advantages for k candidates at a single step.

    Implements Eq. 5 from the paper:
        A_t^(j) = (r_t^(j) - mean(r_t)) / (std(r_t) + eps)

    Args:
        step_rewards: Tensor of shape (k,) with rewards for k candidates.
        epsilon: Small constant for numerical stability.

    Returns:
        Tensor of shape (k,) with normalized advantages.
    """
    mean_r = step_rewards.mean()
    std_r = step_rewards.std()
    return (step_rewards - mean_r) / (std_r + epsilon)


def reward_weighted_sampling(
    advantages: torch.Tensor,
    temperature: float = 0.7,
) -> int:
    """
    Sample an action index proportional to softmax(advantages / temperature).

    This is the reward-weighted sampling strategy from Section 3.2 of the paper,
    balancing exploitation of high-reward actions with exploration.

    Args:
        advantages: Tensor of shape (k,) with step-level advantages.
        temperature: Sampling temperature eta. Lower = more greedy.

    Returns:
        Selected action index (int).
    """
    if temperature <= 0:
        # Greedy / deterministic
        return int(advantages.argmax().item())

    logits = advantages / temperature
    probs = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item())


def best_of_k(rewards: torch.Tensor) -> int:
    """
    Pure exploitation: select the candidate with the highest reward.

    Args:
        rewards: Tensor of shape (k,) with step-level rewards.

    Returns:
        Selected action index (int).
    """
    return int(rewards.argmax().item())


def compute_step_rewards_batch(
    raw_rewards: List[List[float]],
    epsilon: float = 1e-6,
) -> List[torch.Tensor]:
    """
    Compute step-level advantages for a batch of steps.

    Args:
        raw_rewards: List of lists, each inner list has k rewards for one step.
        epsilon: Numerical stability constant.

    Returns:
        List of tensors, each of shape (k,) with normalized advantages.
    """
    advantages = []
    for step_rewards in raw_rewards:
        rewards_t = torch.tensor(step_rewards, dtype=torch.float32)
        adv = compute_step_level_advantage(rewards_t, epsilon)
        advantages.append(adv)
    return advantages


def aggregate_step_objectives(
    step_advantages: List[torch.Tensor],
    step_log_ratios: List[torch.Tensor],
    clip_eps: float = 0.2,
    kl_coef: float = 0.001,
    step_kl_divs: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute the full SLATE training objective by aggregating over steps and candidates.

    Implements Eq. 7 from the paper:
        J_SLATE(theta) = E_x [ sum_t (1/k) sum_j J_t^(j)(theta) - beta * KL ]

    Args:
        step_advantages: List of T tensors, each (k,) with step-level advantages.
        step_log_ratios: List of T tensors, each (k,) with per-token mean log(pi/pi_old).
        clip_eps: PPO clipping epsilon.
        kl_coef: KL regularization coefficient beta.
        step_kl_divs: Optional list of T tensors with per-step KL divergences.

    Returns:
        Scalar loss tensor.
    """
    total_loss = torch.tensor(0.0)
    num_steps = len(step_advantages)

    for t in range(num_steps):
        adv = step_advantages[t]  # (k,)
        log_ratio = step_log_ratios[t]  # (k,)
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        step_obj = torch.min(surr1, surr2).mean()

        total_loss = total_loss + step_obj

        # KL penalty
        if step_kl_divs is not None and t < len(step_kl_divs):
            total_loss = total_loss - kl_coef * step_kl_divs[t].mean()

    return total_loss / max(num_steps, 1)
