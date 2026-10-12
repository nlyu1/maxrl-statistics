import torch
from torch import Tensor

from src.experiments.utkface.losses import (
    age_errors,
    log_policy,
    log_window_probabilities,
)

METRIC_NAMES = (
    "mae_mean",
    "mse_mean",
    "mae_median",
    "mae_mode",
    "expected_mae",
    "expected_mse",
    "ce",
    "tailrl",
    "entropy",
    "mode_accuracy",
    "mass_within_5",
    "oracle_best_of_8_mae",
    "oracle_best_of_64_mae",
)


def batch_metrics(*, logits: Tensor, targets: Tensor) -> Tensor:
    log_probs = log_policy(logits=logits)
    probs = log_probs.exp()
    errors = age_errors(log_probs=log_probs, targets=targets)
    mean_error = (probs * errors).sum(-1)
    median = (probs.cumsum(-1) < 0.5).sum(-1).clamp_max(99)
    mode = probs.argmax(-1)
    windows = log_window_probabilities(log_probs=log_probs, targets=targets)
    complement = (-windows.expm1()).clamp(0, 1)
    return torch.stack(
        (
            mean_error.abs(),
            mean_error.square(),
            (median - targets).abs(),
            (mode - targets).abs(),
            (probs * errors.abs()).sum(-1),
            (probs * errors.square()).sum(-1),
            -log_probs.gather(-1, targets[:, None]).squeeze(-1),
            -windows.mean(-1),
            -(probs * log_probs).sum(-1),
            (mode == targets).float(),
            (probs * (errors.abs() <= 5)).sum(-1),
            complement.pow(8).sum(-1),
            complement.pow(64).sum(-1),
        ),
        dim=-1,
    )


def metric_summary(*, totals: Tensor) -> dict:
    count = int(totals[-1].item())
    if count == 0:
        raise ValueError("Cannot evaluate an empty split")
    means = (totals[:-1] / count).cpu().tolist()
    result = dict(zip(METRIC_NAMES, means, strict=True))
    result["rmse_mean"] = result["mse_mean"] ** 0.5
    result["count"] = count
    return result
