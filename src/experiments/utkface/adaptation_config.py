from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.experiments.utkface.config import RunConfig


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

    @model_validator(mode="after")
    def validate_bank(self) -> "AdaptationConfig":
        if self.use_feature_bank != self.backbone_frozen:
            raise ValueError("Frozen jobs require a feature bank")
        if self.global_start_epoch and self.initialization != "checkpoint":
            raise ValueError("Continuations require checkpoint initialization")
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
    max_jobs: int = Field(ge=1, le=80)
    poll_seconds: float = Field(gt=0, le=30)
    dispatch_lock: Path
