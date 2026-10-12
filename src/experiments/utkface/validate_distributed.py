from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.utkface.canonical import get_canonical_config
from src.experiments.utkface.config import (
    POPULATION_METHODS,
    SAMPLED_METHODS,
    ObjectiveConfig,
    UTKFaceConfig,
)
from src.experiments.utkface.data import EvaluationSampler
from src.experiments.utkface.distributed import (
    Runtime,
    execution_context,
    finish_runtime,
    initialize_runtime,
    synchronize,
    wrap_distributed,
)
from src.experiments.utkface.evaluate import evaluate
from src.experiments.utkface.model import create_policy
from src.experiments.utkface.objectives import compute_loss
from src.experiments.utkface.train import atomic_json


def validate_gradients(*, runtime: Runtime) -> dict[str, float]:
    torch.manual_seed(17)
    initial = nn.Linear(7, 100)
    reference = copy.deepcopy(initial).to(runtime.device)
    distributed = wrap_distributed(model=initial, runtime=runtime)
    inputs = torch.randn(4 * runtime.world_size, 7, device="cpu").to(runtime.device)
    targets = (torch.arange(len(inputs), device=runtime.device) * 17) % 100
    local = slice(4 * runtime.rank, 4 * (runtime.rank + 1))
    errors = {}
    for method in POPULATION_METHODS:
        config = get_canonical_config(
            method=method,
            data_directory=Path("/tmp/unused"),
            output_directory=Path("/tmp/unused"),
            seed=17,
        ).objective
        reference.zero_grad(set_to_none=True)
        distributed.zero_grad(set_to_none=True)
        with execution_context(runtime=runtime, precision="fp32"):
            full_loss = compute_loss(
                logits=reference(inputs), targets=targets, config=config
            )
            local_loss = compute_loss(
                logits=distributed(inputs[local]), targets=targets[local], config=config
            )
        full_loss.backward()
        local_loss.backward()
        deviations = []
        for expected, actual in zip(
            reference.parameters(), distributed.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-5)
            deviations.append((actual.grad - expected.grad).abs().max().item())
        errors[method] = max(deviations)
    return errors


def validate_evaluation(*, runtime: Runtime) -> float:
    torch.manual_seed(19)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 100))
    model = wrap_distributed(model=model, runtime=runtime)
    dataset = TensorDataset(
        torch.randn(23, 3, 32, 32, device="cpu"), torch.arange(23, device="cpu")
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=EvaluationSampler(
            size=len(dataset), rank=runtime.rank, world_size=runtime.world_size
        ),
    )
    config = get_canonical_config(
        method="ce",
        data_directory=Path("/tmp/unused"),
        output_directory=Path("/tmp/unused"),
        seed=19,
    )
    payload = config.model_dump()
    payload["training"].update(device=runtime.device.type, precision="fp32")
    config = UTKFaceConfig.model_validate(payload)
    actual = evaluate(model=model, loader=loader, config=config, runtime=runtime)
    single = Runtime(
        rank=0, world_size=1, local_rank=runtime.local_rank, device=runtime.device
    )
    expected = evaluate(
        model=model,
        loader=DataLoader(dataset, batch_size=4),
        config=config,
        runtime=single,
    )
    for name, value in expected.items():
        torch.testing.assert_close(
            torch.tensor(actual[name]), torch.tensor(value), atol=1e-5, rtol=1e-6
        )
    return max(abs(actual[name] - value) for name, value in expected.items())


def validate_resnet(*, runtime: Runtime, precision: str) -> dict[str, float]:
    torch.manual_seed(23)
    model = wrap_distributed(
        model=create_policy(pretrained=False, seed=23), runtime=runtime
    )
    model.train()
    inputs = torch.randn(2, 3, 32, 32, device=runtime.device)
    targets = torch.tensor([10 + runtime.rank, 70], device=runtime.device)
    results = {}
    for method in (*POPULATION_METHODS, *SAMPLED_METHODS):
        canonical = get_canonical_config(
            method=method,
            data_directory=Path("/tmp/unused"),
            output_directory=Path("/tmp/unused"),
            seed=23,
        ).objective
        config = ObjectiveConfig.model_validate({
            **canonical.model_dump(),
            "num_rollouts": 8,
            "degree": 4,
        })
        model.zero_grad(set_to_none=True)
        with execution_context(runtime=runtime, precision=precision):
            logits = model(inputs)
            with torch.autocast(device_type=runtime.device.type, enabled=False):
                loss = compute_loss(logits=logits, targets=targets, config=config)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            model.parameters(), 10.0, error_if_nonfinite=True
        )
        assert torch.isfinite(loss)
        results[method] = norm.item()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDP global-gradient, evaluation, and ResNet smoke validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    runtime = initialize_runtime(device=args.device)
    try:
        torch.set_num_threads(4)
        report = {
            "world_size": runtime.world_size,
            "device": str(runtime.device),
            "precision": args.precision,
            "gradient_max_absolute_errors": validate_gradients(runtime=runtime),
            "evaluation_max_absolute_error": validate_evaluation(runtime=runtime),
            "resnet_gradient_norms": validate_resnet(
                runtime=runtime, precision=args.precision
            ),
        }
        if runtime.primary:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path=args.output, value=report)
            print(report, flush=True)
        synchronize(runtime=runtime)
    finally:
        finish_runtime(runtime=runtime)


if __name__ == "__main__":
    main()
