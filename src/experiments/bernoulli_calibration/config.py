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


class DatasetConfig(BaseConfig):
    seed: int
    vocabulary: tuple[str, ...]
    word_values: tuple[float, ...]
    length: int = Field(ge=1)
    train_capacity: int = Field(ge=1)
    validation_size: int = Field(ge=2)
    test_size: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_vocabulary(self) -> "DatasetConfig":
        if len(self.vocabulary) != len(self.word_values):
            raise ValueError("Each vocabulary item requires a teacher value")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Vocabulary words must be unique")
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


def canonical_dataset() -> DatasetConfig:
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
    return DatasetConfig(
        seed=20261107,
        vocabulary=scored + fillers,
        word_values=(-2.0,) * 4 + (-1.0,) * 4 + (1.0,) * 4 + (2.0,) * 4 + (0.0,) * 16,
        length=16,
        train_capacity=32768,
        validation_size=8192,
        test_size=8192,
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
