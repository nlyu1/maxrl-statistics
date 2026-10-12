from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.experiments.utkface.config import UTKFaceConfig
from src.experiments.utkface.distributed import (
    Runtime,
    execution_context,
    sum_across_ranks,
    unwrap,
)
from src.experiments.utkface.objectives import age_errors, statistical_log_kernel


@torch.no_grad()
def batch_metrics(
    *, logits: Tensor, targets: Tensor, temperature_years: float
) -> dict[str, Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    ages = torch.arange(100, device=logits.device, dtype=torch.float32)
    errors = age_errors(targets=targets)
    mean = (probabilities * ages).sum(dim=-1)
    greedy = logits.argmax(dim=-1)
    median = (probabilities.cumsum(dim=-1) >= 0.5).long().argmax(dim=-1)
    rewards = 1 - errors.abs() / 99
    sorted_rewards, indices = rewards.sort(dim=-1, descending=True, stable=True)
    log_survival = log_probs.gather(dim=-1, index=indices).logcumsumexp(dim=-1)
    following = torch.cat(
        (sorted_rewards[:, 1:], torch.zeros_like(sorted_rewards[:, :1])), dim=-1
    )
    gaps = sorted_rewards - following
    result = {
        "mean_mae": (mean - targets).abs(),
        "mean_mse": (mean - targets).square(),
        "greedy_mae": (greedy - targets).abs().float(),
        "median_mae": (median - targets).abs().float(),
        "expected_mae": (probabilities * errors.abs()).sum(dim=-1),
        "expected_mse": (probabilities * errors.square()).sum(dim=-1),
        "expected_reward": (probabilities * rewards).sum(dim=-1),
        "exact_probability": probabilities.gather(
            dim=-1, index=targets[:, None]
        ).squeeze(-1),
        "greedy_accuracy": (greedy == targets).float(),
        "within_5_probability": (probabilities * (errors.abs() <= 5)).sum(dim=-1),
        "entropy": -(probabilities * log_probs).sum(dim=-1),
        "ce": -log_probs.gather(dim=-1, index=targets[:, None]).squeeze(-1),
        "tailrl": -(gaps * log_survival).sum(dim=-1),
    }
    for squared, name in ((True, "stat_maxrl_mse"), (False, "stat_maxrl_mae")):
        kernel = statistical_log_kernel(
            errors=errors,
            squared=squared,
            temperature_years=temperature_years,
        )
        result[name] = -(log_probs + kernel).logsumexp(dim=-1)
    survival = log_survival.exp().clamp(0, 1)
    for count in (1, 8, 64):
        probability = -torch.expm1(count * torch.log1p(-survival))
        result[f"best_of_{count}_reward"] = (gaps * probability).sum(dim=-1)
    return result


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: UTKFaceConfig,
    runtime: Runtime,
) -> dict[str, float]:
    model = unwrap(model=model)
    model.eval()
    names = tuple(
        batch_metrics(
            logits=torch.zeros(1, 100, device=runtime.device),
            targets=torch.zeros(1, device=runtime.device, dtype=torch.long),
            temperature_years=config.objective.temperature_years,
        )
    )
    totals = torch.zeros(len(names) + 1, device=runtime.device, dtype=torch.float64)
    subgroup_names = (
        "easy",
        "medium",
        "hard",
        *(f"decade_{decade}" for decade in range(10)),
    )
    subgroup_metrics = ("mean_mae", "greedy_mae", "expected_mae")
    subgroup_totals = torch.zeros(
        len(subgroup_names), 4, device=runtime.device, dtype=torch.float64
    )
    for images, targets in loader:
        images = images.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)
        with execution_context(runtime=runtime, precision=config.training.precision):
            logits = model(images)
        metrics = batch_metrics(
            logits=logits,
            targets=targets,
            temperature_years=config.objective.temperature_years,
        )
        totals[:-1] += torch.stack([metrics[name].double().sum() for name in names])
        totals[-1] += targets.numel()
        easy = (targets >= 20) & (targets < 40)
        medium = (targets < 5) | ((targets >= 60) & (targets < 80))
        masks = (
            easy,
            medium,
            ~(easy | medium),
            *(targets // 10 == decade for decade in range(10)),
        )
        for index, mask in enumerate(masks):
            subgroup_totals[index, :3] += torch.stack([
                metrics[name][mask].double().sum() for name in subgroup_metrics
            ])
            subgroup_totals[index, 3] += mask.sum()
    totals = sum_across_ranks(values=totals, runtime=runtime)
    subgroup_totals = sum_across_ranks(values=subgroup_totals, runtime=runtime).cpu()
    if totals[-1].item() == 0:
        raise ValueError("Evaluation split is empty.")
    values = (totals[:-1] / totals[-1]).cpu().tolist()
    result = dict(zip(names, values, strict=True))
    for name, group in zip(subgroup_names, subgroup_totals, strict=True):
        result[f"{name}/count"] = group[3].item()
        if group[3] > 0:
            result.update({
                f"{name}/{metric}": (group[index] / group[3]).item()
                for index, metric in enumerate(subgroup_metrics)
            })
    return result
