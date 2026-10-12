from __future__ import annotations

# ruff: noqa: UP037
import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from src.experiments.utkface.config import ObjectiveConfig, Reward
from src.experiments.utkface.estimators import (
    binary_maxrl_advantages,
    grpo_advantages,
    rloo_advantages,
    tailrl_advantages,
)
from src.maxrl_estimator import MaxRLEstimatorConfig


def age_errors(*, targets: Int[Tensor, "batch"]) -> Float[Tensor, "batch age"]:
    ages = torch.arange(100, device=targets.device, dtype=torch.float32)
    return ages.unsqueeze(0) - targets.unsqueeze(1)


def age_rewards(
    *,
    errors: Float[Tensor, "batch age"],
    reward: Reward,
) -> Float[Tensor, "batch age"]:
    if reward == "mae":
        return 1 - errors.abs() / 99
    if reward == "mse":
        return 1 - (errors / 99).square()
    if reward == "binary":
        return (errors == 0).float()
    raise ValueError(f"Unknown reward: {reward}")


def statistical_log_kernel(
    *,
    errors: Float[Tensor, "batch age"],
    squared: bool,
    temperature_years: float,
) -> Float[Tensor, "batch age"]:
    scaled = errors / temperature_years
    return -0.5 * scaled.square() if squared else -scaled.abs()


def tailrl_population_loss(
    *,
    log_probs: Float[Tensor, "batch age"],
    rewards: Float[Tensor, "batch age"],
    survival_floor: float,
) -> Float[Tensor, ""]:
    sorted_rewards, indices = rewards.sort(dim=-1, descending=True, stable=True)
    sorted_log_probs = log_probs.gather(dim=-1, index=indices)
    log_survival = sorted_log_probs.logcumsumexp(dim=-1)
    if survival_floor > 0:
        log_survival = log_survival.clamp_min(math.log(survival_floor))
    following = torch.cat(
        (sorted_rewards[..., 1:], torch.zeros_like(sorted_rewards[..., :1])),
        dim=-1,
    )
    gaps = sorted_rewards - following
    return -(gaps * log_survival).sum(dim=-1).mean()


def population_grpo_loss(
    *,
    log_probs: Float[Tensor, "batch age"],
    rewards: Float[Tensor, "batch age"],
    epsilon: float,
    standard_deviation_floor: float,
) -> Float[Tensor, ""]:
    with torch.no_grad():
        probabilities = log_probs.exp()
        mean = (probabilities * rewards).sum(dim=-1, keepdim=True)
        variance = (probabilities * (rewards - mean).square()).sum(dim=-1, keepdim=True)
        deviation = variance.sqrt()
        advantages = torch.where(
            deviation >= standard_deviation_floor,
            (rewards - mean) / (deviation + epsilon),
            torch.zeros_like(rewards),
        )
    return -(probabilities * advantages * log_probs).sum(dim=-1).mean()


def compute_loss(
    *,
    logits: Float[Tensor, "batch age"],
    targets: Int[Tensor, "batch"],
    config: ObjectiveConfig,
) -> Float[Tensor, ""]:
    if logits.shape[-1] != 100:
        raise ValueError("The UTKFace policy must have exactly 100 age logits.")
    logits = (
        logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    )
    log_probs = F.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    errors = age_errors(targets=targets).to(log_probs.dtype)
    rewards = age_rewards(errors=errors, reward=config.reward)
    method = config.method
    if method == "mse":
        ages = torch.arange(100, device=logits.device, dtype=log_probs.dtype)
        mean = (probabilities * ages).sum(dim=-1)
        return ((mean - targets) / 99).square().mean()
    if method in ("ce", "maxrl_binary_population"):
        return F.cross_entropy(logits, targets)
    if method == "rloo_population":
        return -(probabilities * rewards).sum(dim=-1).mean()
    if method == "grpo_population":
        return population_grpo_loss(
            log_probs=log_probs,
            rewards=rewards,
            epsilon=config.standard_deviation_epsilon,
            standard_deviation_floor=config.standard_deviation_floor,
        )
    if method == "tailrl_population":
        return tailrl_population_loss(
            log_probs=log_probs,
            rewards=rewards,
            survival_floor=config.survival_floor,
        )
    if method in ("stat_maxrl_mse_population", "stat_maxrl_mae_population"):
        log_kernel = statistical_log_kernel(
            errors=errors,
            squared=method == "stat_maxrl_mse_population",
            temperature_years=config.temperature_years,
        )
        return -(log_probs + log_kernel).logsumexp(dim=-1).mean()
    with torch.no_grad():
        samples = torch.multinomial(
            probabilities, config.num_rollouts, replacement=True
        )
        sampled_rewards = rewards.gather(dim=-1, index=samples)
        if method == "tailrl":
            advantages = tailrl_advantages(
                rewards=sampled_rewards,
                survival_floor=config.survival_floor,
            )
        elif method == "maxrl_binary":
            successes = (samples == targets.unsqueeze(1)).float()
            advantages = binary_maxrl_advantages(successes=successes)
        elif method == "grpo":
            advantages = grpo_advantages(
                rewards=sampled_rewards,
                epsilon=config.standard_deviation_epsilon,
                standard_deviation_floor=config.standard_deviation_floor,
            )
        elif method == "rloo":
            advantages = rloo_advantages(rewards=sampled_rewards)
        elif method in ("stat_maxrl_mse", "stat_maxrl_mae"):
            log_kernel = statistical_log_kernel(
                errors=errors,
                squared=method == "stat_maxrl_mse",
                temperature_years=config.temperature_years,
            ).gather(dim=-1, index=samples)
            estimator = MaxRLEstimatorConfig.initialize(
                degree=config.degree,
                log_sup_likelihood=0.0,
                subtract_baseline=True,
            )
            advantages = estimator.compute_score_weights(log_likelihoods=log_kernel)
        else:
            raise ValueError(f"Unknown sampled method: {method}")
    sampled_log_probs = log_probs.gather(dim=-1, index=samples)
    return -(advantages.detach() * sampled_log_probs).mean()
