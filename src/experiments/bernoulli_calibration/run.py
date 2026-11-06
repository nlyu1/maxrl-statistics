import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from src.experiments.bernoulli_calibration.config import (
    RunConfig,
    canonical_dataset,
)
from src.experiments.bernoulli_calibration.data import Dataset, load_dataset
from src.experiments.bernoulli_calibration.model import BagProbabilityModel
from src.experiments.bernoulli_calibration.objectives import (
    native_losses,
    training_loss,
)
from src.experiments.utils import set_seeds
from src.experiments.utkface.run import write_json


@torch.no_grad()
def evaluate(
    *, model: BagProbabilityModel, dataset: Dataset, split: str, config: RunConfig
) -> tuple[dict, Tensor]:
    model.eval()
    tokens, targets = dataset.tokens[split], dataset.probabilities[split]
    logits = torch.cat([
        model(tokens=tokens[start : start + config.eval_batch_size]).float()
        for start in range(0, len(tokens), config.eval_batch_size)
    ])
    probability = logits.sigmoid()
    difference = probability - targets
    centered_q = probability - probability.mean()
    centered_p = targets - targets.mean()
    denominator = centered_q.square().sum().sqrt() * centered_p.square().sum().sqrt()
    defined = denominator > 1e-12
    pearson = (centered_q * centered_p).sum() / denominator.clamp_min(1e-12)
    nll = F.binary_cross_entropy_with_logits(logits, targets)
    entropy = -(
        probability * F.logsigmoid(logits) + (1 - probability) * F.logsigmoid(-logits)
    ).mean()
    true_entropy = -(
        targets * targets.log() + (1 - targets) * torch.log1p(-targets)
    ).mean()
    metrics = {
        "native_loss": native_losses(
            logits=logits, targets=targets, config=config
        ).mean(),
        "mae": difference.abs().mean(),
        "rmse": difference.square().mean().sqrt(),
        "pearson": pearson,
        "label_nll": nll,
        "excess_nll": nll - true_entropy,
        "entropy": entropy,
        "true_entropy": true_entropy,
        "expected_reward": (
            targets * probability + (1 - targets) * (1 - probability)
        ).mean(),
    }
    names = tuple(metrics)
    numbers = torch.stack([metrics[name] for name in names]).cpu().tolist()
    if not np.isfinite(numbers).all():
        raise FloatingPointError(
            f"Nonfinite evaluation metrics: {dict(zip(names, numbers))}"
        )
    result = dict(zip(names, numbers, strict=True))
    result["pearson_defined"] = bool(defined.item())
    return result, probability.cpu()


def save_curves(*, output: Path, history: list[dict]) -> None:
    write_json(path=output / "history.json", payload=history)
    flattened = []
    for row in history:
        flattened.append({
            **{key: value for key, value in row.items() if not isinstance(value, dict)},
            **{
                f"{split}_{metric}": value
                for split in ("train", "validation", "diagnostics")
                for metric, value in row[split].items()
            },
        })
    pl.DataFrame(flattened, infer_schema_length=None).write_parquet(
        output / "history.parquet"
    )


def train(*, config: RunConfig) -> dict:
    output = config.output_dir
    result_path = output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["config"] != config.model_dump(mode="json"):
            raise ValueError(f"Completed run configuration mismatch: {output}")
        return result
    if not config.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Training requires an explicit CUDA device")
    torch.set_num_threads(1)
    device = torch.device(config.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    started = time.monotonic()
    history = []
    with (
        torch.cuda.device(device),
        torch.device(device),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        torch.cuda.reset_peak_memory_stats(device)
        set_seeds(config.seed)
        dataset = load_dataset(
            path=config.root / "dataset.pt", device=device, train_size=config.train_size
        )
        model = BagProbabilityModel(data=canonical_dataset(), config=config)
        digest = hashlib.sha256()
        for tensor in model.state_dict().values():
            digest.update(tensor.detach().cpu().numpy().tobytes())
        initial_fingerprint = digest.hexdigest()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        generators = {
            name: torch.Generator(device=device).manual_seed(config.seed + offset)
            for name, offset in (
                ("contexts", 1000),
                ("labels", 2000),
                ("actions", 3000),
            )
        }
        best_native_loss = float("inf")
        best_step = 0
        best_state = copy.deepcopy(model.state_dict())
        diagnostic_totals: dict[str, Tensor] = {}
        batches_since_evaluation = 0
        permutation = torch.empty(0, dtype=torch.long)
        cursor = 0
        for step in range(config.steps + 1):
            if step % config.eval_interval == 0 or step == config.steps:
                train_metrics, _ = evaluate(
                    model=model, dataset=dataset, split="train", config=config
                )
                validation, _ = evaluate(
                    model=model, dataset=dataset, split="validation", config=config
                )
                diagnostics = {
                    name: (value / batches_since_evaluation).item()
                    for name, value in diagnostic_totals.items()
                }
                history.append({
                    "step": step,
                    "epochs": step * config.batch_size / config.train_size,
                    "context_presentations": step * config.batch_size,
                    "label_draws": step
                    * config.batch_size
                    * (1 if config.protocol == "shared" else config.num_rollouts),
                    "policy_rollouts": 0
                    if config.method in {"ce", "mse", "mae"}
                    else step * config.batch_size * config.num_rollouts,
                    "seconds": time.monotonic() - started,
                    "train": train_metrics,
                    "validation": validation,
                    "diagnostics": diagnostics,
                })
                if validation["native_loss"] < best_native_loss:
                    best_native_loss = validation["native_loss"]
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                save_curves(output=output, history=history)
                diagnostic_totals = {}
                batches_since_evaluation = 0
            if step == config.steps:
                break
            if cursor + config.batch_size > len(permutation):
                permutation = torch.randperm(
                    config.train_size, generator=generators["contexts"]
                )
                cursor = 0
            index = permutation[cursor : cursor + config.batch_size]
            cursor += config.batch_size
            model.train()
            logits = model(tokens=dataset.tokens["train"][index])
            loss, diagnostics = training_loss(
                logits=logits,
                probabilities=dataset.probabilities["train"][index],
                label_generator=generators["labels"],
                action_generator=generators["actions"],
                config=config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for name, value in {"surrogate_loss": loss.detach(), **diagnostics}.items():
                diagnostic_totals[name] = (
                    diagnostic_totals.get(name, torch.zeros(())) + value
                )
            batches_since_evaluation += 1
        result = {
            "config": config.model_dump(mode="json"),
            "family": config.family,
            "dataset_fingerprint": dataset.fingerprint,
            "initial_model_fingerprint": initial_fingerprint,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "precision": "bf16",
            "best_native_step": best_step,
            "final": history[-1],
            "best_native": min(
                history, key=lambda row: row["validation"]["native_loss"]
            ),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        if config.stage == "main":
            result["test_final"], final_predictions = evaluate(
                model=model, dataset=dataset, split="test", config=config
            )
            torch.save(model.state_dict(), output / "final.pt")
            model.load_state_dict(best_state)
            result["test_best_native"], best_predictions = evaluate(
                model=model, dataset=dataset, split="test", config=config
            )
            torch.save(best_state, output / "best-native.pt")
            pl.DataFrame({
                "f": dataset.probabilities["test"].cpu().numpy(),
                "q_final": final_predictions.numpy(),
                "q_best_native": best_predictions.numpy(),
            }).write_parquet(output / "test-predictions.parquet")
        torch.cuda.synchronize(device)
        result["seconds"] = time.monotonic() - started
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        write_json(path=result_path, payload=result)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = train(config=RunConfig.model_validate_json(arguments.config.read_text()))
    print(
        json.dumps({
            "family": result["family"],
            "seconds": result["seconds"],
            "final": result["final"]["validation"],
        })
    )


if __name__ == "__main__":
    main()
