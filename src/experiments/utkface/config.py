from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.model.optimizer import LRScheduleConfig

Method = Literal[
    "mse",
    "ce",
    "grpo_population",
    "rloo_population",
    "maxrl_binary_population",
    "tailrl_population",
    "stat_maxrl_mse_population",
    "stat_maxrl_mae_population",
    "grpo",
    "rloo",
    "maxrl_binary",
    "tailrl",
    "stat_maxrl_mse",
    "stat_maxrl_mae",
]
Reward = Literal["mae", "mse", "binary"]

POPULATION_METHODS: tuple[Method, ...] = (
    "mse",
    "ce",
    "grpo_population",
    "rloo_population",
    "maxrl_binary_population",
    "tailrl_population",
    "stat_maxrl_mse_population",
    "stat_maxrl_mae_population",
)
SAMPLED_METHODS: tuple[Method, ...] = (
    "grpo",
    "rloo",
    "maxrl_binary",
    "tailrl",
    "stat_maxrl_mse",
    "stat_maxrl_mae",
)


class ObjectiveConfig(BaseConfig):
    method: Method
    reward: Reward
    temperature_years: float = Field(gt=0)
    num_rollouts: int = Field(ge=2)
    degree: int = Field(ge=1)
    survival_floor: float = Field(ge=0, lt=1)
    standard_deviation_epsilon: float = Field(gt=0)
    standard_deviation_floor: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_degree(self) -> Self:
        if self.degree > self.num_rollouts:
            raise ValueError("Statistical MaxRL degree must not exceed num_rollouts.")
        if (
            self.method in ("maxrl_binary", "maxrl_binary_population")
            and self.reward != "binary"
        ):
            raise ValueError(
                "Discrete MaxRL requires reward='binary' (exact age matches)."
            )
        return self


class DataConfig(BaseConfig):
    directory: Path
    input_size: int = Field(ge=32)
    split_seed: int = Field(ge=0)
    max_train_samples: int = Field(ge=0)
    max_eval_samples: int = Field(ge=0)
    num_workers: int = Field(ge=0)


class TrainingConfig(BaseConfig):
    epochs: int = Field(ge=1)
    global_batch_size: int = Field(ge=2)
    eval_batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    momentum: float = Field(ge=0, lt=1)
    gradient_clip: float = Field(gt=0)
    schedule: LRScheduleConfig
    precision: Literal["bf16", "fp32"]
    device: Literal["cuda", "cpu"]
    checkpoint_epochs: tuple[int, ...]

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if self.schedule.lr_schedule != "cosine":
            raise ValueError("UTKFace requires an explicit cosine schedule.")
        if not 0 <= self.schedule.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if not 0 <= self.schedule.lr_min_ratio <= 1:
            raise ValueError("lr_min_ratio must be in [0, 1].")
        if any(epoch < 1 for epoch in self.checkpoint_epochs):
            raise ValueError("Checkpoint epochs must be positive.")
        return self


class UTKFaceConfig(BaseConfig):
    objective: ObjectiveConfig
    data: DataConfig
    training: TrainingConfig
    seed: int = Field(ge=0)
    pretrained: bool
    output_directory: Path

    def scientific_settings(self) -> dict:
        return self.model_dump(mode="json", exclude={"output_directory"})
