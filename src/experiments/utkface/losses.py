import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from src.experiments.utkface.advantages import batched_advantages
from src.experiments.utkface.config import LossConfig

NUM_BINS = 100
MAX_AGE = NUM_BINS - 1


def log_policy(*, logits: Float[Tensor, "batch 100"]) -> Float[Tensor, "batch 100"]:
    if logits.ndim != 2 or logits.shape[-1] != NUM_BINS:
        raise ValueError("Expected logits shaped (batch, 100)")
    values = logits.double() if logits.dtype == torch.float64 else logits.float()
    return F.log_softmax(values, dim=-1)


def age_errors(
    *, log_probs: Float[Tensor, "batch 100"], targets: Int[Tensor, "batch"]
) -> Float[Tensor, "batch 100"]:
    bins = torch.arange(NUM_BINS, device=log_probs.device, dtype=log_probs.dtype)
    return bins.unsqueeze(0) - targets.unsqueeze(-1)


def log_window_probabilities(
    *,
    log_probs: Float[Tensor, "batch 100"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, "batch 99"]:
    errors = age_errors(log_probs=log_probs, targets=targets).abs()
    radii = torch.arange(MAX_AGE, device=log_probs.device)
    windows = errors[:, None, :] <= radii[None, :, None]
    return torch.logsumexp(
        log_probs[:, None, :].masked_fill(~windows, -torch.inf), dim=-1
    )


def policy_loss(
    *,
    logits: Float[Tensor, "batch 100"],
    targets: Int[Tensor, "batch"],
    config: LossConfig,
) -> Float[Tensor, ""]:
    log_probs = log_policy(logits=logits)
    probs = log_probs.exp()
    errors = age_errors(log_probs=log_probs, targets=targets)
    method = config.method
    if method in {"mse", "mae"}:
        mean_error = errors.mul(probs).sum(dim=-1) / MAX_AGE
        return (mean_error.square() if method == "mse" else mean_error.abs()).mean()
    if method in {"ce", "binary_maxrl_population"}:
        return F.nll_loss(log_probs, targets)
    if method == "tailrl_population":
        log_windows = log_window_probabilities(log_probs=log_probs, targets=targets)
        if config.survival_floor > 0:
            log_windows = log_windows.clamp_min(math.log(config.survival_floor))
        return -log_windows.mean()
    if method.startswith("stat_maxrl_"):
        scaled = errors / config.kernel_scale_years
        log_kernel = -0.5 * scaled.square() if "_mse" in method else -scaled.abs()
        if method.endswith("_population"):
            return -torch.logsumexp(log_probs + log_kernel, dim=-1).mean()
        with torch.no_grad():
            actions = torch.multinomial(probs, config.num_rollouts, replacement=True)
            advantages = normalized_kernel_advantages(
                log_kernel=log_kernel.gather(-1, actions)
            )
        return -(advantages * log_probs.gather(-1, actions)).mean()
    reward = 1 - (
        (errors / MAX_AGE).square() if "_mse" in method else errors.abs() / MAX_AGE
    )
    if method.endswith("_population"):
        mean_reward = (probs * reward).sum(dim=-1)
        if method.startswith("rloo_"):
            return -mean_reward.mean()
        if method.startswith("grpo_"):
            variance = (probs * (reward - mean_reward[:, None]).square()).sum(dim=-1)
            std = variance.detach().sqrt()
            return -torch.where(
                std >= config.std_floor,
                mean_reward / std.clamp_min(config.std_floor),
                0.0,
            ).mean()
        raise ValueError(f"Unsupported population method: {method}")
    with torch.no_grad():
        actions = torch.multinomial(probs, config.num_rollouts, replacement=True)
        sampled_reward = (
            (actions == targets[:, None]).to(probs.dtype)
            if method == "binary_maxrl"
            else reward.gather(-1, actions)
        )
        estimator = (
            method.split("_")[0] if method.startswith(("grpo_", "rloo_")) else method
        )
        advantages = batched_advantages(
            rewards=sampled_reward,
            estimator=estimator,
            survival_floor=config.survival_floor,
            std_floor=config.std_floor,
        )
    return -(advantages * log_probs.gather(-1, actions)).mean()


def normalized_kernel_advantages(*, log_kernel: Tensor) -> Tensor:
    if log_kernel.shape[-1] < 2:
        raise ValueError("At least two rollouts are required")
    return log_kernel.softmax(-1) * log_kernel.shape[-1] - 1


def scalar_loss(*, predictions: Tensor, targets: Tensor, config: LossConfig) -> Tensor:
    values = (
        predictions.double()
        if predictions.dtype == torch.float64
        else predictions.float()
    )
    error = (values.flatten() - targets) / MAX_AGE
    if config.method == "mse":
        return error.square().mean()
    if config.method == "mae":
        return error.abs().mean()
    raise ValueError("Scalar regression supports MAE and MSE only")
