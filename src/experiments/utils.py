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
