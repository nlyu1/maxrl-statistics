from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl

from src.config.base import BaseConfig
from src.data.common import aux_words

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
    """Normalized power-law weights over semantic words only."""
    if num_words < 3 or num_words % 2 != 1:
        raise ValueError("word_assignments must have odd length >= 3")
    center = num_words // 2
    weights = [(1 + abs(i - center)) ** -decay_power for i in range(num_words)]
    total = sum(weights)
    return [weight / total for weight in weights]


def _validate_inputs(
    *,
    snr: float,
    num_train_samples: int,
    num_val_samples: int,
    prompt_length: int,
    word_assignments: tuple[str, ...],
    aux_words_ratio: float,
    word_decay_power: float,
) -> None:
    if not isfinite(snr) or snr < 0:
        raise ValueError("snr must be finite and non-negative")
    if num_train_samples < 0 or num_val_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if not isfinite(aux_words_ratio) or not 0 <= aux_words_ratio < 1:
        raise ValueError("aux_words_ratio must be finite and in [0, 1)")
    if not isfinite(word_decay_power) or word_decay_power < 0:
        raise ValueError("word_decay_power must be finite and non-negative")
    if len(word_assignments) < 3 or len(word_assignments) % 2 != 1:
        raise ValueError("word_assignments must have odd length >= 3")
    if len(set(word_assignments)) != len(word_assignments):
        raise ValueError("word_assignments must be unique")
    if len(set(aux_words)) != len(aux_words):
        raise ValueError("canonical aux_words must be unique")
    if set(word_assignments) & set(aux_words):
        raise ValueError("word_assignments and canonical aux_words must be disjoint")
    if aux_words_ratio > 0 and not aux_words:
        raise ValueError(
            "canonical aux_words must be non-empty when aux_words_ratio > 0"
        )


def _derived_fields(
    *,
    snr: float,
    word_assignments: tuple[str, ...],
    aux_words_ratio: float,
    word_decay_power: float,
) -> tuple[dict[str, float], dict[str, int], float, float, float]:
    semantic_density_values = _power_law_density(
        num_words=len(word_assignments),
        decay_power=word_decay_power,
    )
    half = len(word_assignments) // 2

    word_values = {word: i - half for i, word in enumerate(word_assignments)}
    word_values.update({word: 0 for word in aux_words})

    semantic_mass = 1.0 - aux_words_ratio
    aux_mass = aux_words_ratio

    word_density: dict[str, float] = {
        word: semantic_mass * prob
        for word, prob in zip(word_assignments, semantic_density_values)
    }
    if aux_words:
        aux_prob = aux_mass / len(aux_words)
        word_density.update({word: aux_prob for word in aux_words})

    mean = sum(word_density[word] * word_values[word] for word in word_density)
    second_moment = sum(
        word_density[word] * (word_values[word] ** 2) for word in word_density
    )
    variance = second_moment - mean**2
    if variance <= 0:
        raise ValueError(
            "Per-token signal variance must be positive; decrease aux_words_ratio."
        )

    std = variance**0.5
    target_signal_std = snr / (1 + snr**2) ** 0.5
    return word_density, word_values, mean, std, target_signal_std


class BagOfWordsDatasetConfig(BaseConfig):
    """
    Synthetic regression dataset. Each sample is a prompt of `prompt_length` words
    drawn iid from `word_density`.

    Semantic words come from `word_assignments` and contribute ordered values
    from negative to positive. Auxiliary words are drawn from the canonical
    `src.data.common.aux_words` bank, contribute zero to the raw score, and
    occupy total probability mass `aux_words_ratio`. `word_decay_power` shapes
    only the semantic-word distribution.

    SNR = sig / noise;  R² = snr² / (1 + snr²)
    """

    snr: float
    num_train_samples: int
    num_val_samples: int
    prompt_length: int
    word_assignments: tuple[str, ...]  # ordered negative → positive
    aux_words_ratio: float
    word_decay_power: float

    # Derived — computed in initialize, stored for round-trip serialization
    word_density: dict[str, float]  # full token distribution over semantic + aux
    word_values: dict[str, int]  # aux words have value 0
    unnormalized_signal_mean: float  # per-token mean under full token distribution
    unnormalized_signal_std: float  # per-token std under full token distribution
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
        aux_words_ratio: float,
        word_decay_power: float,
    ) -> "BagOfWordsDatasetConfig":
        word_assignments_tuple = tuple(word_assignments)
        _validate_inputs(
            snr=snr,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            prompt_length=prompt_length,
            word_assignments=word_assignments_tuple,
            aux_words_ratio=aux_words_ratio,
            word_decay_power=word_decay_power,
        )
        word_density, word_values, mean, std, target_std = _derived_fields(
            snr=snr,
            word_assignments=word_assignments_tuple,
            aux_words_ratio=aux_words_ratio,
            word_decay_power=word_decay_power,
        )
        return cls(
            snr=snr,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            prompt_length=prompt_length,
            word_assignments=word_assignments_tuple,
            aux_words_ratio=aux_words_ratio,
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

        Tokens are sampled iid from the full mixture `word_density`. Auxiliary
        words contribute zero to `raw`, so normalization uses the full per-token
        mixture moments:
            raw = sum of prompt word values
            signal = (raw - L·μ) / (sqrt(L)·σ) · target_signal_std
            target = signal + N(0, noise_std)
        """
        folder.mkdir(parents=True, exist_ok=True)
        words = list(self.word_density)
        probs = np.array([self.word_density[word] for word in words], dtype=float)
        vals = np.array([self.word_values[word] for word in words], dtype=float)
        vocab = np.asarray(words, dtype=object)

        rng = np.random.default_rng()
        noise_std = (1 - self.target_signal_std**2) ** 0.5

        for split, n in [
            ("train", self.num_train_samples),
            ("val", self.num_val_samples),
        ]:
            idx = rng.choice(len(words), size=(n, self.prompt_length), p=probs)
            raw = vals[idx].sum(axis=1)
            signal = (
                (raw - self.prompt_length * self.unnormalized_signal_mean)
                / ((self.prompt_length**0.5) * self.unnormalized_signal_std)
                * self.target_signal_std
            )
            targets = signal + rng.normal(0, noise_std, size=n)
            prompts = [" ".join(vocab[row]) for row in idx]
            pl.DataFrame(
                {"prompt": prompts, "target": targets, "signal": signal}
            ).write_parquet(folder / f"{split}.parquet")

        (folder / "config.json").write_text(self.model_dump_json())

    @classmethod
    def load_from(cls, path: Path) -> "BagOfWordsDatasetConfig":
        """Restores config from a folder written by write_to."""
        return cls.model_validate_json((path / "config.json").read_text())

    @classmethod
    def init_or_load_from(
        cls,
        *,
        folder: Path,
        snr: float,
        num_train_samples: int,
        num_val_samples: int,
        prompt_length: int,
        word_assignments: list[str],
        aux_words_ratio: float,
        word_decay_power: float,
    ) -> "BagOfWordsDatasetConfig":
        config = cls.initialize(
            snr=snr,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            prompt_length=prompt_length,
            word_assignments=word_assignments,
            aux_words_ratio=aux_words_ratio,
            word_decay_power=word_decay_power,
        )

        config_path = folder / "config.json"
        required_paths = (folder / "train.parquet", folder / "val.parquet")
        if config_path.exists():
            saved = cls.load_from(folder)
            if saved == config and all(path.exists() for path in required_paths):
                return saved
            if saved != config:
                print("Cached config mismatch — re-generating dataset.")
            else:
                print("Cached dataset files missing — re-generating dataset.")

        config.write_to(folder=folder)
        return config
