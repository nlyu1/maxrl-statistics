from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def tailrl_advantages(
    *,
    rewards: Float[Tensor, "batch rollout"],
    survival_floor: float,
) -> Float[Tensor, "batch rollout"]:
    count = rewards.shape[-1]
    if count < 2:
        raise ValueError("TailRL requires at least two rollouts.")
    sorted_rewards, indices = rewards.float().sort(dim=-1, stable=True)
    previous = torch.cat(
        (torch.zeros_like(sorted_rewards[..., :1]), sorted_rewards[..., :-1]),
        dim=-1,
    )
    survivors = torch.arange(count, 0, -1, device=rewards.device, dtype=torch.float32)
    increments = (sorted_rewards - previous) / survivors
    increments = increments * (survivors / count >= survival_floor)
    weights = increments.cumsum(dim=-1) * count
    weights = weights - weights.mean(dim=-1, keepdim=True)
    return torch.zeros_like(weights).scatter(dim=-1, index=indices, src=weights)


@torch.no_grad()
def binary_maxrl_advantages(
    *,
    successes: Float[Tensor, "batch rollout"],
) -> Float[Tensor, "batch rollout"]:
    count = successes.shape[-1]
    num_successes = successes.sum(dim=-1, keepdim=True)
    weights = count * successes / num_successes.clamp_min(1) - 1
    return torch.where(num_successes > 0, weights, torch.zeros_like(weights))


@torch.no_grad()
def rloo_advantages(
    *,
    rewards: Float[Tensor, "batch rollout"],
) -> Float[Tensor, "batch rollout"]:
    count = rewards.shape[-1]
    if count < 2:
        raise ValueError("RLOO requires at least two rollouts.")
    return count / (count - 1) * (rewards - rewards.mean(dim=-1, keepdim=True))


@torch.no_grad()
def grpo_advantages(
    *,
    rewards: Float[Tensor, "batch rollout"],
    epsilon: float,
    standard_deviation_floor: float,
) -> Float[Tensor, "batch rollout"]:
    standard_deviation = rewards.std(dim=-1, keepdim=True)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    return torch.where(
        standard_deviation >= standard_deviation_floor,
        centered / (standard_deviation + epsilon),
        torch.zeros_like(centered),
    )
