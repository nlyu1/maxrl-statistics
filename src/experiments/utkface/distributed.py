import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def sum(self, *, values: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

    def gather_pairs(
        self, *, pairs: list[tuple[float, int]]
    ) -> list[tuple[float, int]]:
        if self.world_size == 1:
            return pairs
        outputs = [None for _ in range(self.world_size)]
        dist.all_gather_object(outputs, pairs)
        return [pair for output in outputs for pair in output]


@contextmanager
def distributed_context(*, device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selected = torch.device(
        f"cuda:{local_rank}" if world_size > 1 and device.startswith("cuda") else device
    )
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This experiment requires a GPU supporting bfloat16")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if selected.type == "cuda" else "gloo",
            timeout=timedelta(minutes=10),
            device_id=selected if selected.type == "cuda" else None,
        )
    try:
        yield DistributedContext(rank=rank, world_size=world_size, device=selected)
    finally:
        if world_size > 1:
            dist.destroy_process_group()
