from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(kw_only=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize_runtime(*, device: Literal["cuda", "cpu"]) -> Runtime:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.cuda.set_device(local_rank)
        selected_device = torch.device("cuda", local_rank)
    else:
        selected_device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device == "cuda" else "gloo",
            timeout=timedelta(minutes=30),
        )
    return Runtime(
        rank=rank, world_size=world_size, local_rank=local_rank, device=selected_device
    )


def wrap_distributed(*, model: nn.Module, runtime: Runtime) -> nn.Module:
    model = model.to(runtime.device)
    if runtime.world_size == 1:
        return model
    if runtime.device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return DistributedDataParallel(model, device_ids=[runtime.local_rank])
    return DistributedDataParallel(model)


def unwrap(*, model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


@contextmanager
def execution_context(
    *,
    runtime: Runtime,
    precision: Literal["bf16", "fp32"],
) -> Iterator[None]:
    cuda = (
        torch.cuda.device(runtime.device)
        if runtime.device.type == "cuda"
        else nullcontext()
    )
    with (
        cuda,
        torch.device(runtime.device),
        torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ),
    ):
        yield


def sum_across_ranks(*, values: Tensor, runtime: Runtime) -> Tensor:
    if runtime.world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def synchronize(*, runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.barrier()


def finish_runtime(*, runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.destroy_process_group()
