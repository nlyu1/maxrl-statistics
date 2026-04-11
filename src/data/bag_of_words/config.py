from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl

from src.config.base import BaseConfig

canonical_bags: dict[int, list[str]] = {
    5: ["terrible", "bad", "okay", "good", "wonderful"],
    7: ["terrible", "awful", "bad", "okay", "good", "great", "wonderful"],
    11: [
        "terrible",
        "dreadful",
        "awful",
        "bad",
        "poor",
        "okay",
        "decent",
        "good",
        "great",
        "excellent",
        "wonderful",
    ],
    15: [
        "abominable",
        "terrible",
        "dreadful",
        "awful",
        "bad",
        "poor",
        "mediocre",
        "okay",
        "decent",
        "fair",
        "good",
        "great",
        "excellent",
        "superb",
        "wonderful",
    ],
    21: [
        "execrable",
        "abominable",
        "terrible",
        "dreadful",
        "awful",
        "bad",
        "poor",
        "mediocre",
        "subpar",
        "lukewarm",
        "okay",
        "decent",
        "fair",
        "satisfactory",
        "good",
        "great",
        "excellent",
        "superb",
        "wonderful",
        "magnificent",
        "sublime",
    ],
}


def _power_law_density(*, num_words: int, decay_power: float) -> list[float]:
    """Normalized power-law weights, peaked at the center (neutral) word."""
    if num_words < 3 or num_words % 2 != 1:
        raise ValueError("word_assignments must have odd length >= 3")
    center = num_words // 2
    w = [(1 + abs(i - center)) ** -decay_power for i in range(num_words)]
    s = sum(w)
    return [x / s for x in w]


def _validate_inputs(
    *,
    snr: float,
    num_train_samples: int,
    num_val_samples: int,
    prompt_length: int,
    word_assignments: tuple[str, ...],
    word_decay_power: float,
) -> None:
    if not isfinite(snr) or snr < 0:
        raise ValueError("snr must be finite and non-negative")
    if num_train_samples < 0 or num_val_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if not isfinite(word_decay_power) or word_decay_power < 0:
        raise ValueError("word_decay_power must be finite and non-negative")
    if len(word_assignments) < 3 or len(word_assignments) % 2 != 1:
        raise ValueError("word_assignments must have odd length >= 3")
    if len(set(word_assignments)) != len(word_assignments):
        raise ValueError("word_assignments must be unique")


def _derived_fields(
    *,
    snr: float,
    word_assignments: tuple[str, ...],
    word_decay_power: float,
) -> tuple[dict[str, float], dict[str, int], float, float, float]:
    n = len(word_assignments)
    half = n // 2
    word_values = {w: i - half for i, w in enumerate(word_assignments)}
    word_density = dict(
        zip(
            word_assignments,
            _power_law_density(num_words=n, decay_power=word_decay_power),
        )
    )
    mean = sum(word_density[w] * word_values[w] for w in word_assignments)
    std = (
        sum(word_density[w] * word_values[w] ** 2 for w in word_assignments) - mean**2
    ) ** 0.5
    return word_density, word_values, mean, std, snr / (1 + snr**2) ** 0.5


class BagOfWordsDatasetConfig(BaseConfig):
    """
    Synthetic regression dataset. Each sample is a prompt of `prompt_length` words
    drawn from `word_density`. The regression target has unit variance at the
    specified signal-to-noise ratio.

    SNR = sig / noise;  R² = snr² / (1 + snr²)
    """

    snr: float
    num_train_samples: int
    num_val_samples: int
    prompt_length: int
    word_assignments: tuple[str, ...]  # ordered negative → positive
    word_decay_power: float

    # Derived — computed in initialize, stored for round-trip serialization
    word_density: dict[str, float]
    word_values: dict[str, int]
    unnormalized_signal_mean: float
    unnormalized_signal_std: float
    target_signal_std: float  # snr / sqrt(1 + snr²)

    @property
    def rsq(self) -> float:
        return self.snr**2 / (1 + self.snr**2)

    @classmethod
    def initialize(
        cls,
        *,
        snr: float,
        num_train_samples: int,
        num_val_samples: int,
        prompt_length: int,
        word_assignments: list[str],
        word_decay_power: float,
    ) -> "BagOfWordsDatasetConfig":
        word_assignments_tuple = tuple(word_assignments)
        _validate_inputs(
            snr=snr,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            prompt_length=prompt_length,
            word_assignments=word_assignments_tuple,
            word_decay_power=word_decay_power,
        )
        word_density, word_values, mean, std, target_std = _derived_fields(
            snr=snr,
            word_assignments=word_assignments_tuple,
            word_decay_power=word_decay_power,
        )
        return cls(
            snr=snr,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            prompt_length=prompt_length,
            word_assignments=word_assignments_tuple,
            word_decay_power=word_decay_power,
            word_density=word_density,
            word_values=word_values,
            unnormalized_signal_mean=mean,
            unnormalized_signal_std=std,
            target_signal_std=target_std,
        )

    def write_to(self, *, folder: Path):
        """
        Writes train.parquet, val.parquet, and config.json to folder.
        Each parquet has columns: "prompt" (space-separated words), "target" (float).
        Signal normalization: raw = sum of prompt word values;
            signal = (raw - L·μ) / (√L·σ) · target_signal_std
            target = signal + N(0, noise_std)
        """
        folder.mkdir(parents=True, exist_ok=True)
        words = list(self.word_assignments)
        probs = [self.word_density[w] for w in words]
        vals = np.array([self.word_values[w] for w in words])
        rng = np.random.default_rng()
        noise_std = (1 - self.target_signal_std**2) ** 0.5

        for split, n in [
            ("train", self.num_train_samples),
            ("val", self.num_val_samples),
        ]:
            idx = rng.choice(len(words), size=(n, self.prompt_length), p=probs)
            raw = vals[idx].sum(axis=1).astype(float)
            signal = (
                (raw - self.prompt_length * self.unnormalized_signal_mean)
                / ((self.prompt_length**0.5) * self.unnormalized_signal_std)
                * self.target_signal_std
            )
            targets = signal + rng.normal(0, noise_std, size=n)
            prompts = [" ".join(np.array(words)[row]) for row in idx]
            pl.DataFrame(
                {"prompt": prompts, "target": targets, "signal": signal}
            ).write_parquet(folder / f"{split}.parquet")

        (folder / "config.json").write_text(self.model_dump_json())

    @classmethod
    def load_from(cls, path: Path) -> "BagOfWordsDatasetConfig":
        """Restores config from a folder written by write_to."""
        return cls.model_validate_json((path / "config.json").read_text())
