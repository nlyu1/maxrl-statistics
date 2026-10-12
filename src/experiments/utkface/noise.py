import math

import numpy as np
import torch
from torch import Tensor

from src.experiments.utkface.config import NoiseConfig


def noisy_labels(*, ages: list[int], config: NoiseConfig, epoch: int) -> Tensor:
    clean = np.asarray(ages, dtype=np.float64)
    if config.mode == "none":
        return torch.from_numpy(clean.astype(np.int64))
    generator = np.random.default_rng(
        np.random.SeedSequence([config.seed, epoch if config.mode == "fresh" else 0])
    )
    noisy = clean + generator.normal(0, config.std_years, len(clean))
    return torch.from_numpy(np.rint(noisy).clip(0, 99).astype(np.int64))


def noise_summary(*, clean: Tensor, noisy: Tensor) -> dict:
    signal = clean.double().var(correction=0).item()
    errors = noisy.double() - clean
    mse = errors.square().mean().item()
    return {
        "signal_variance_years2": signal,
        "noise_mse_years2": mse,
        "noise_rmse_years": math.sqrt(mse),
        "noise_bias_years": errors.mean().item(),
        "noise_std_years": errors.std(correction=0).item(),
        "snr_db": 10 * math.log10(signal / mse) if signal > 0 and mse > 0 else None,
    }
