"""Shared helpers for experiment single-run scripts and sweep orchestrators."""

import gc
import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    torch.compiler.reset()


def sweep_root_name(*, project: str, method: str, dataset_suffix: str = "") -> str:
    """Top-level artifacts folder name for a sweep.

    Examples:
        sweep_root_name(project="corpus-regression", method="sl") -> "corpus-regression-sl-sweep"
        sweep_root_name(project="bow", method="maxrl", dataset_suffix="-row-het") -> "bow-maxrl-row-het-sweep"
    """
    return f"{project}-{method}{dataset_suffix}-sweep"


def stride_4_top_down(n: int) -> list[int]:
    return [i for offset in range(4) for i in range(offset, n, 4)]


def stride_4_bottom_up(n: int) -> list[int]:
    return list(reversed(stride_4_top_down(n)))
