import torch
from scipy.stats import spearmanr
from torch import Tensor

from src.experiments.utkface.losses import NUM_BINS, log_policy
from src.experiments.utkface.metrics import (
    METRIC_NAMES,
    batch_metrics,
    median_predictions,
)

_POINT_NAMES = ("mean", "median", "mode")
_CATEGORICAL_NAMES = METRIC_NAMES[4:]


def prediction_values(*, outputs: Tensor, scalar: bool) -> Tensor:
    if scalar:
        if outputs.ndim not in (1, 2) or (outputs.ndim == 2 and outputs.shape[1] != 1):
            raise ValueError("Scalar outputs must be shaped (batch,) or (batch, 1)")
        values = outputs.flatten()
        return values[:, None].expand(-1, 3)
    if outputs.ndim != 2 or outputs.shape[-1] != NUM_BINS:
        raise ValueError("Categorical outputs must be shaped (batch, 100)")
    log_probs = log_policy(logits=outputs)
    probs = log_probs.exp()
    ages = torch.arange(NUM_BINS, dtype=probs.dtype, device=probs.device)
    mean = (probs * ages).sum(-1)
    median = median_predictions(logits=outputs, scalar=False).to(probs.dtype)
    mode = probs.argmax(-1).to(probs.dtype)
    return torch.stack((mean, median, mode), dim=-1)


def _spearman(*, predictions: Tensor, targets: Tensor) -> float | None:
    if len(predictions.unique()) == 1 or len(targets.unique()) == 1:
        return None
    value = float(spearmanr(predictions, targets).statistic)
    return value if torch.isfinite(torch.tensor(value)) else None


def evaluate_outputs(
    *,
    outputs: Tensor,
    targets: Tensor,
    scalar: bool,
) -> dict[str, Tensor | dict[str, int | float | bool | None]]:
    targets = targets.detach()
    if targets.ndim == 1:
        target_values = targets
    elif targets.ndim == 2 and targets.shape[1] == 1:
        target_values = targets[:, 0]
    else:
        raise ValueError("Targets must be shaped (batch,) or (batch, 1)")
    target_values = target_values.float()
    if not torch.isfinite(target_values).all():
        raise ValueError("Targets must be finite")
    if not torch.equal(target_values, target_values.round()):
        raise ValueError("Targets must contain integer ages")
    if len(target_values) == 0:
        raise ValueError("Cannot evaluate an empty batch")

    outputs = outputs.detach()
    if not torch.isfinite(outputs).all():
        raise ValueError("Outputs must be finite")
    predictions = prediction_values(outputs=outputs, scalar=scalar)
    if predictions.shape[0] != len(target_values):
        raise ValueError("Output and target batch sizes differ")

    target_values = target_values.to(device=predictions.device, dtype=predictions.dtype)
    errors = predictions - target_values[:, None]
    mae = errors.abs().mean(0)
    mse = errors.square().mean(0)
    point_values = torch.stack((mae, mse)).double().cpu().tolist()
    cpu_predictions = predictions.double().cpu()
    cpu_targets = target_values.double().cpu()
    result: dict[str, int | float | bool | None] = {"count": len(target_values)}
    for index, name in enumerate(_POINT_NAMES):
        value = _spearman(
            predictions=cpu_predictions[:, index],
            targets=cpu_targets,
        )
        result[f"mae_{name}"] = point_values[0][index]
        result[f"mse_{name}"] = point_values[1][index]
        result[f"rmse_{name}"] = point_values[1][index] ** 0.5
        result[f"spearman_{name}"] = value
        result[f"spearman_{name}_defined"] = value is not None

    if not scalar:
        chunks = [
            batch_metrics(
                logits=outputs[start : start + 256],
                targets=targets[start : start + 256]
                .flatten()
                .to(device=outputs.device, dtype=torch.int64),
            )
            for start in range(0, len(outputs), 256)
        ]
        categorical = torch.cat(chunks)
        means = categorical.double().mean(0).tolist()
        result.update({
            name: float(value)
            for name, value in zip(_CATEGORICAL_NAMES, means[4:], strict=True)
        })
    return {"predictions": predictions, "metrics": result}
