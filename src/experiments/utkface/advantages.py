"""Last-dimension estimators adapted from Zanette-Labs/TailRL; see SOURCES.md."""

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor


def batched_advantages(
    *,
    rewards: Float[Tensor, "*batch rollouts"],
    estimator: Literal["grpo", "rloo", "binary_maxrl", "tailrl"],
    survival_floor: float,
    std_floor: float,
) -> Float[Tensor, "*batch rollouts"]:
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("At least two rollouts are required")
    if not 0 <= survival_floor < 1 or std_floor <= 0:
        raise ValueError("Invalid survival or standard-deviation floor")
    values = rewards.double() if rewards.dtype == torch.float64 else rewards.float()
    count = values.shape[-1]
    centered = values - values.mean(dim=-1, keepdim=True)
    if estimator == "rloo":
        return centered * (count / (count - 1))
    if estimator == "grpo":
        std = values.std(dim=-1, keepdim=True)
        return torch.where(std >= std_floor, centered / std.clamp_min(std_floor), 0.0)
    if estimator == "binary_maxrl":
        success = values == 1
        successes = success.sum(dim=-1, keepdim=True).to(values.dtype)
        result = torch.where(
            success, (count - successes) / successes.clamp_min(1), -1.0
        )
        return torch.where(successes > 0, result, 0.0)
    if estimator == "tailrl":
        ordered, indices = values.sort(dim=-1, stable=True)
        gaps = ordered - torch.cat(
            (torch.zeros_like(ordered[..., :1]), ordered[..., :-1]), dim=-1
        )
        survivors = torch.arange(count, 0, -1, device=values.device, dtype=values.dtype)
        increments = gaps / survivors * (survivors / count >= survival_floor)
        weights = increments.cumsum(dim=-1) * count
        weights = weights - weights.mean(dim=-1, keepdim=True)
        return torch.zeros_like(values).scatter(dim=-1, index=indices, src=weights)
    raise ValueError(f"Unknown estimator: {estimator}")
