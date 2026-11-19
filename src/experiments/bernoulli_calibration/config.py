from collections import Counter, defaultdict
from functools import cache
from math import exp, log, log1p, sqrt
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.config.base import BaseConfig

Method = Literal["ce", "mse", "mae", "rloo", "grpo", "maxrl"]
Protocol = Literal["shared", "split"]
SUPERVISED = ("ce", "mse", "mae")
METHODS = (*SUPERVISED, "rloo", "grpo", "maxrl")
ROLLOUTS = (2, 4, 8, 16, 64, 256)
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LR_EXTENSION_FACTOR = 3.0
LR_EXTENSION_ROUNDS = 2
SEEDS = (42, 43, 44)
TRAIN_SIZES = (8192, 16384, 32768)
EARLY_SNAPSHOT_STEPS = (1, 2, 4, 8, 16)


@cache
def population_scores(
    *, word_values: tuple[float, ...], length: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    weights = {
        value: count / len(word_values) for value, count in Counter(word_values).items()
    }
    distribution = {0.0: 1.0}
    for _ in range(length):
        following: dict[float, float] = defaultdict(float)
        for total, mass in distribution.items():
            for value, probability in weights.items():
                following[total + value] += mass * probability
        distribution = dict(following)
    ordered = sorted(distribution.items())
    return tuple(value / sqrt(length) for value, _ in ordered), tuple(
        mass for _, mass in ordered
    )


def population_mean(
    *, word_values: tuple[float, ...], length: int, logit_bias: float
) -> float:
    scores, masses = population_scores(word_values=word_values, length=length)
    total = 0.0
    for score, mass in zip(scores, masses, strict=True):
        value = score + logit_bias
        probability = (
            1 / (1 + exp(-value)) if value >= 0 else exp(value) / (1 + exp(value))
        )
        total += mass * probability
    return total


def teacher_bias(
    *, word_values: tuple[float, ...], length: int, target_mean: float
) -> float:
    if not 0 < target_mean < 1:
        raise ValueError(
            "Target probability mean must be strictly between zero and one"
        )
    scores, _ = population_scores(word_values=word_values, length=length)
    target_logit = log(target_mean) - log1p(-target_mean)
    low, high = target_logit - max(scores), target_logit - min(scores)
    for _ in range(80):
        middle = (low + high) / 2
        mean = population_mean(
            word_values=word_values, length=length, logit_bias=middle
        )
        if mean == target_mean:
            return middle
        if mean < target_mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class DatasetConfig(BaseConfig):
    seed: int
    vocabulary: tuple[str, ...]
    word_values: tuple[float, ...]
    length: int = Field(ge=1)
    train_capacity: int = Field(ge=1)
    validation_size: int = Field(ge=2)
    test_size: int = Field(ge=2)
    target_mean: float = Field(gt=0, lt=1)
    logit_bias: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_vocabulary(self) -> "DatasetConfig":
        if len(self.vocabulary) != len(self.word_values):
            raise ValueError("Each vocabulary item requires a teacher value")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary words must be unique")
        mean = population_mean(
            word_values=self.word_values, length=self.length, logit_bias=self.logit_bias
        )
        if abs(mean - self.target_mean) > 1e-10:
            raise ValueError("Teacher bias does not match the declared population mean")
        return self


class RunConfig(BaseConfig):
    root: Path
    identifier: str
    stage: Literal["smoke", "pilot", "main"]
    method: Method
    protocol: Protocol
    num_rollouts: int = Field(ge=2)
    train_size: int = Field(ge=1)
    seed: int
    learning_rate: float = Field(gt=0)
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    eval_interval: int = Field(ge=1)
    eval_batch_size: int = Field(ge=1)
    embedding_dim: int = Field(ge=1)
    hidden_dim: int = Field(ge=1)
    std_floor: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    device: str

    @property
    def output_dir(self) -> Path:
        return self.root / "runs" / self.identifier

    @property
    def family(self) -> str:
        count = (
            "labels1"
            if self.method in SUPERVISED and self.protocol == "shared"
            else f"n{self.num_rollouts}"
        )
        return f"{self.method}-{self.protocol}-{count}"


def canonical_dataset(*, target_mean: float) -> DatasetConfig:
    scored = (
        "awful",
        "grim",
        "terrible",
        "dreadful",
        "poor",
        "bad",
        "weak",
        "rough",
        "good",
        "nice",
        "solid",
        "pleasant",
        "great",
        "excellent",
        "wonderful",
        "perfect",
    )
    fillers = (
        "the",
        "a",
        "and",
        "with",
        "in",
        "on",
        "at",
        "by",
        "from",
        "to",
        "of",
        "for",
        "as",
        "it",
        "this",
        "that",
    )
    values = (-2.0,) * 4 + (-1.0,) * 4 + (1.0,) * 4 + (2.0,) * 4 + (0.0,) * 16
    return DatasetConfig(
        seed=20261107,
        vocabulary=scored + fillers,
        word_values=values,
        length=16,
        train_capacity=32768,
        validation_size=8192,
        test_size=8192,
        target_mean=target_mean,
        logit_bias=teacher_bias(word_values=values, length=16, target_mean=target_mean),
    )


def canonical_run(*, root: Path) -> RunConfig:
    return RunConfig(
        root=root,
        identifier="smoke",
        stage="smoke",
        method="ce",
        protocol="shared",
        num_rollouts=16,
        train_size=8192,
        seed=42,
        learning_rate=1e-3,
        steps=256,
        batch_size=256,
        eval_interval=32,
        eval_batch_size=2048,
        embedding_dim=128,
        hidden_dim=32,
        std_floor=1e-8,
        weight_decay=0.0,
        device="cuda:0",
    )
