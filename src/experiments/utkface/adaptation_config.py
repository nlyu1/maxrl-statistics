from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.experiments.utkface.config import RunConfig


class TrainingControl(BaseConfig):
    schedule_epochs: int = Field(ge=1)
    extend_checkpoint: Path
    extend_epoch: int = Field(ge=0)
    keep_latest_checkpoints: int = Field(ge=1)
    model_milestones: tuple[int, ...]


class AdaptationConfig(BaseConfig):
    run: RunConfig
    pipeline: Literal["original", "fixed"]
    initialization: Literal["imagenet", "scratch", "checkpoint"]
    initial_checkpoint: Path
    backbone_frozen: bool
    use_feature_bank: bool
    feature_bank: Path
    global_start_epoch: int = Field(ge=0)
    view_seed: int = Field(ge=0)
    order_seed: int = Field(ge=0)
    checkpoint_every: int = Field(ge=1)
    backbone_learning_rate: float = Field(gt=0)
    schedule_kind: Literal["cosine", "warmup_constant"]
    control: TrainingControl

    @model_validator(mode="after")
    def validate_bank(self) -> "AdaptationConfig":
        if self.use_feature_bank != self.backbone_frozen:
            raise ValueError("Frozen jobs require a feature bank")
        if self.global_start_epoch and self.initialization != "checkpoint":
            raise ValueError("Continuations require checkpoint initialization")
        if self.control.extend_epoch:
            if (
                self.schedule_kind != "warmup_constant"
                or self.initialization != "scratch"
                or self.global_start_epoch
            ):
                raise ValueError(
                    "Horizon extensions require a scratch constant-LR trajectory"
                )
            if self.control.extend_epoch >= self.run.epochs:
                raise ValueError("Extension epoch must precede the target horizon")
        if self.control.schedule_epochs > self.run.epochs:
            raise ValueError("Schedule reference cannot exceed the run horizon")
        return self


class FeatureBankConfig(BaseConfig):
    run: RunConfig
    pipeline: Literal["original", "fixed"]
    initialization: Literal["imagenet", "checkpoint"]
    initial_checkpoint: Path
    global_start_epoch: int = Field(ge=0)
    view_seed: int = Field(ge=0)
    order_seed: int = Field(ge=0)

    @property
    def output_dir(self) -> Path:
        return self.run.output_dir


class AdaptationJob(BaseConfig):
    identifier: str
    kind: Literal["train", "bank"]
    config_path: Path
    output_dir: Path
    dependencies: tuple[str, ...]
    priority: int


class AdaptationPlan(BaseConfig):
    output_dir: Path
    manifest: Path
    jobs: tuple[AdaptationJob, ...]
    per_gpu_jobs: int = Field(ge=1, le=8)
    max_gpus: int = Field(default=0, ge=0)
    min_free_gib: float = Field(ge=0)
    poll_seconds: float = Field(gt=0, le=30)
    dispatch_lock: Path
