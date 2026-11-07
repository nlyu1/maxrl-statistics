import torch
import torch.nn.functional as F
from torch import Tensor

from src.experiments.bernoulli_calibration.config import SUPERVISED, RunConfig
from src.experiments.utkface.advantages import batched_advantages


def grouped_advantages(*, rewards: Tensor, labels: Tensor, config: RunConfig) -> Tensor:
    estimator = "binary_maxrl" if config.method == "maxrl" else config.method
    if config.protocol == "shared":
        return batched_advantages(
            rewards=rewards,
            estimator=estimator,
            survival_floor=0.0,
            std_floor=config.std_floor,
        )
    result = torch.zeros_like(rewards)
    for label in (0, 1):
        mask = labels == label
        count = mask.sum(-1, keepdim=True)
        successes = (rewards * mask).sum(-1, keepdim=True)
        mean = successes / count.clamp_min(1)
        centered = rewards - mean
        if config.method == "rloo":
            advantage = torch.where(
                count > 1, centered * count / (count - 1).clamp_min(1), rewards
            )
        elif config.method == "grpo":
            variance = (centered.square() * mask).sum(-1, keepdim=True)
            std = (variance / (count - 1).clamp_min(1)).sqrt()
            advantage = torch.where(
                (count > 1) & (std >= config.std_floor),
                centered / std.clamp_min(config.std_floor),
                0.0,
            )
        elif config.method == "maxrl":
            advantage = torch.where(
                successes > 0, rewards * count / successes.clamp_min(1) - 1, 0.0
            )
        else:
            raise ValueError(f"Unsupported policy method: {config.method}")
        result += advantage * mask
    return result


def training_loss(
    *,
    logits: Tensor,
    probabilities: Tensor,
    label_generator: torch.Generator,
    action_generator: torch.Generator,
    config: RunConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    values = logits.float()
    label_count = 1 if config.protocol == "shared" else config.num_rollouts
    labels = (
        torch.rand(
            (len(values), label_count), device=values.device, generator=label_generator
        )
        < probabilities[:, None]
    ).float()
    expanded = values[:, None].expand_as(labels)
    if config.method in SUPERVISED:
        if config.method == "ce":
            loss = F.binary_cross_entropy_with_logits(expanded, labels)
        else:
            error = values.sigmoid()[:, None] - labels
            loss = (error.square() if config.method == "mse" else error.abs()).mean()
        return loss, {"sampled_label_one_fraction": labels.mean()}
    with torch.no_grad():
        actions = (
            torch.rand(
                (len(values), config.num_rollouts),
                device=values.device,
                generator=action_generator,
            )
            < values.sigmoid()[:, None]
        ).float()
        rewards = (actions == labels).float()
        labels = labels.expand_as(rewards)
        advantages = grouped_advantages(rewards=rewards, labels=labels, config=config)
        if config.protocol == "shared":
            counts = torch.full((len(values), 1), config.num_rollouts)
            successes = rewards.sum(-1, keepdim=True)
        else:
            counts = torch.stack(
                [(labels == label).sum(-1) for label in (0, 1)], dim=-1
            )
            successes = torch.stack(
                [(rewards * (labels == label)).sum(-1) for label in (0, 1)], dim=-1
            )
        active = counts > 0
        diagnostic = {
            "sampled_label_one_fraction": labels.mean(),
            "sampled_reward": rewards.mean(),
            "nonzero_advantage_fraction": (advantages != 0).float().mean(),
            "empty_group_fraction": (counts == 0).float().mean(),
            "singleton_group_fraction": (counts == 1).float().mean(),
            "no_success_group_fraction": ((successes == 0) & active).sum()
            / active.sum(),
        }
    log_probabilities = -F.binary_cross_entropy_with_logits(
        values[:, None].expand_as(actions), actions, reduction="none"
    )
    return -(advantages * log_probabilities).mean(), diagnostic


def truncated_log(*, probability: Tensor, degree: int) -> Tensor:
    powers = torch.arange(
        1, degree + 1, device=probability.device, dtype=probability.dtype
    )
    terms = torch.exp(torch.log1p(-probability)[:, None] * powers)
    return -(terms / powers).sum(-1)


def native_losses(*, logits: Tensor, targets: Tensor, config: RunConfig) -> Tensor:
    probability = logits.sigmoid()
    if config.method == "ce":
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if config.method == "mse":
        return (probability - targets).square() + targets * (1 - targets)
    if config.method in {"mae", "rloo", "grpo"}:
        return targets * (1 - probability) + (1 - targets) * probability
    if config.method == "maxrl":
        one = probability if config.protocol == "shared" else targets * probability
        zero = (
            1 - probability
            if config.protocol == "shared"
            else (1 - targets) * (1 - probability)
        )
        return -(
            targets * truncated_log(probability=one, degree=config.num_rollouts - 1)
            + (1 - targets)
            * truncated_log(probability=zero, degree=config.num_rollouts - 1)
        )
    raise ValueError(f"Unsupported method: {config.method}")
