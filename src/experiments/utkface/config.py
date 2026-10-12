from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.model.optimizer import LRScheduleConfig

Method = Literal[
    "mse",
    "ce",
    "grpo_mae_population",
    "grpo_mse_population",
    "rloo_mae_population",
    "rloo_mse_population",
    "binary_maxrl_population",
    "tailrl_population",
    "stat_maxrl_mse_population",
    "stat_maxrl_mae_population",
    "grpo_mae",
    "grpo_mse",
    "rloo_mae",
    "rloo_mse",
    "binary_maxrl",
    "tailrl",
]


class LossConfig(BaseConfig):
    method: Method
    kernel_scale_years: float = Field(gt=0)
    num_rollouts: int = Field(ge=2)
    survival_floor: float = Field(ge=0, lt=1)
    std_floor: float = Field(gt=0)


class RunConfig(BaseConfig):
    manifest: Path
    output_dir: Path
    loss: LossConfig
    seed: int = Field(ge=0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=2)
    eval_batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    clip_grad_norm: float = Field(gt=0)
    schedule: LRScheduleConfig
    pretrained: bool
    device: str
    precision: Literal["bf16", "fp32"]
    train_limit: int = Field(ge=0)
    eval_limit: int = Field(ge=0)
    evaluate_test: bool
    save_checkpoint: bool
    deterministic: bool
    cpu_threads: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_device(self) -> "RunConfig":
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU verification requires precision=fp32")
        return self


def canonical_config(*, manifest: Path, output_dir: Path) -> RunConfig:
    return RunConfig(
        manifest=manifest,
        output_dir=output_dir,
        loss=LossConfig(
            method="ce",
            kernel_scale_years=5.0,
            num_rollouts=64,
            survival_floor=0.0,
            std_floor=1e-4,
        ),
        seed=42,
        epochs=20,
        batch_size=64,
        eval_batch_size=128,
        num_workers=4,
        learning_rate=3e-4,
        weight_decay=1e-4,
        clip_grad_norm=5.0,
        schedule=LRScheduleConfig(
            lr_schedule="cosine", warmup_ratio=0.05, lr_min_ratio=0.1
        ),
        pretrained=True,
        device="cuda:0",
        precision="bf16",
        train_limit=0,
        eval_limit=0,
        evaluate_test=False,
        save_checkpoint=True,
        deterministic=False,
        cpu_threads=4,
    )
