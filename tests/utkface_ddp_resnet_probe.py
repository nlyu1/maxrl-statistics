"""Check full ResNet/SyncBatchNorm gradients against a global-batch reference."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.experiments.utkface.config import canonical_config
from src.experiments.utkface.distributed import distributed_context
from src.experiments.utkface.losses import policy_loss
from src.experiments.utkface.model import build_model


def main() -> None:
    with (
        distributed_context(device="cuda") as context,
        torch.cuda.device(context.device),
        torch.device(context.device),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False),
    ):
        if context.world_size != 2:
            raise ValueError("This probe requires exactly two ranks")
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(97)
        reference = build_model(pretrained=False, head="categorical").double()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            build_model(pretrained=False, head="categorical").double()
        )
        model.load_state_dict(reference.state_dict())
        wrapped = DistributedDataParallel(model, device_ids=[context.device.index])
        inputs = torch.randn(4, 3, 224, 224, dtype=torch.float64)
        targets = torch.tensor([10, 35, 60, 90])
        base = canonical_config(manifest=Path("unused"), output_dir=Path("unused")).loss
        rows = []
        for method in (
            "mse",
            "ce",
            "grpo_mae_population",
            "grpo_mse_population",
            "rloo_mae_population",
            "rloo_mse_population",
            "tailrl_population",
            "stat_maxrl_mae_population",
            "stat_maxrl_mse_population",
        ):
            config = base.model_copy(update={"method": method})
            reference.zero_grad(set_to_none=True)
            wrapped.zero_grad(set_to_none=True)
            policy_loss(
                logits=reference(inputs), targets=targets, config=config
            ).backward()
            policy_loss(
                logits=wrapped(inputs[context.rank :: 2]),
                targets=targets[context.rank :: 2],
                config=config,
            ).backward()
            errors, scales, maxima = [], [], []
            for actual, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                error = actual.grad - expected.grad
                errors.append(error.double().square().sum())
                scales.append(expected.grad.double().square().sum())
                maxima.append(error.abs().max())
            relative = (
                (torch.stack(errors).sum() / torch.stack(scales).sum()).sqrt().item()
            )
            maximum = torch.stack(maxima).max().item()
            if relative > 1e-7:
                raise AssertionError(f"{method}: relative gradient error {relative}")
            rows.append({
                "method": method,
                "relative_gradient_error": relative,
                "max_absolute_gradient_error": maximum,
            })
        if context.primary:
            print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
