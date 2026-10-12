"""Execute with torchrun to verify distributed gradients against one global batch."""

import json

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.experiments.utkface.distributed import distributed_context


def main() -> None:
    with (
        distributed_context(device="cuda") as context,
        torch.cuda.device(context.device),
        torch.device(context.device),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False),
    ):
        if context.world_size != 2:
            raise ValueError("This probe requires exactly two ranks")
        torch.manual_seed(71)
        reference = nn.Linear(7, 100, dtype=torch.float64)
        model = nn.Linear(7, 100, dtype=torch.float64)
        model.load_state_dict(reference.state_dict())
        wrapped = DistributedDataParallel(model, device_ids=[context.device.index])
        inputs = torch.randn(8, 7, dtype=torch.float64)
        targets = torch.arange(8) * 11
        nn.functional.cross_entropy(reference(inputs), targets).backward()
        nn.functional.cross_entropy(
            wrapped(inputs[context.rank :: 2]), targets[context.rank :: 2]
        ).backward()
        differences = []
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=1e-10, atol=1e-12
            )
            differences.append((actual.grad - expected.grad).abs().max().item())
        counts = torch.tensor([len(range(context.rank, 65, 2))], dtype=torch.int64)
        assert context.sum(values=counts).item() == 65
        if context.primary:
            print(
                json.dumps({
                    "max_gradient_difference": max(differences),
                    "evaluation_count": 65,
                    "world_size": 2,
                })
            )


if __name__ == "__main__":
    main()
