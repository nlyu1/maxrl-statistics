from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from src.config.base import BaseConfig
from src.experiments.utkface.adaptation_config import AdaptationConfig

MechanismLoss = Literal["mse", "mae", "ce", "tailrl_population"]
BinomialMode = Literal["fixed", "free", "affine"]
HeadPrecision = Literal["bf16", "fp32"]


class MechanismVariant(StrEnum):
    CATEGORICAL_MSE = "categorical_mse"
    CATEGORICAL_TAILRL = "categorical_tailrl"
    CATEGORICAL_CE = "categorical_ce"
    SCALAR_MSE = "scalar_mse"
    SCALAR_MAE = "scalar_mae"
    BINOMIAL_TAILRL = "binomial_tailrl"
    BINOMIAL_CE = "binomial_ce"
    BINOMIAL_TAILRL_FREE = "binomial_tailrl_free"
    BINOMIAL_CE_FREE = "binomial_ce_free"
    BINOMIAL_TAILRL_AFFINE = "binomial_tailrl_affine"
    BINOMIAL_CE_AFFINE = "binomial_ce_affine"
    HEAD_TAILRL_BACKBONE_MSE = "head_tailrl_backbone_mse"
    HEAD_MSE_BACKBONE_TAILRL = "head_mse_backbone_tailrl"
    CATEGORICAL_CE_BINOMIAL_INIT_FP32 = "categorical_ce_binomial_init_fp32"
    CATEGORICAL_TAILRL_BINOMIAL_INIT_FP32 = "categorical_tailrl_binomial_init_fp32"
    CATEGORICAL_CE_RANDOM_FP32 = "categorical_ce_random_fp32"
    CATEGORICAL_TAILRL_RANDOM_FP32 = "categorical_tailrl_random_fp32"
    BINOMIAL_CE_FREE_FP32 = "binomial_ce_free_fp32"
    BINOMIAL_TAILRL_FREE_FP32 = "binomial_tailrl_free_fp32"


OBJECTIVES: dict[MechanismVariant, tuple[MechanismLoss, MechanismLoss]] = {
    MechanismVariant.CATEGORICAL_MSE: ("mse", "mse"),
    MechanismVariant.CATEGORICAL_TAILRL: ("tailrl_population", "tailrl_population"),
    MechanismVariant.CATEGORICAL_CE: ("ce", "ce"),
    MechanismVariant.SCALAR_MSE: ("mse", "mse"),
    MechanismVariant.SCALAR_MAE: ("mae", "mae"),
    MechanismVariant.BINOMIAL_TAILRL: ("tailrl_population", "tailrl_population"),
    MechanismVariant.BINOMIAL_CE: ("ce", "ce"),
    MechanismVariant.BINOMIAL_TAILRL_FREE: ("tailrl_population", "tailrl_population"),
    MechanismVariant.BINOMIAL_CE_FREE: ("ce", "ce"),
    MechanismVariant.BINOMIAL_TAILRL_AFFINE: ("tailrl_population", "tailrl_population"),
    MechanismVariant.BINOMIAL_CE_AFFINE: ("ce", "ce"),
    MechanismVariant.HEAD_TAILRL_BACKBONE_MSE: ("tailrl_population", "mse"),
    MechanismVariant.HEAD_MSE_BACKBONE_TAILRL: ("mse", "tailrl_population"),
    MechanismVariant.CATEGORICAL_CE_BINOMIAL_INIT_FP32: ("ce", "ce"),
    MechanismVariant.CATEGORICAL_TAILRL_BINOMIAL_INIT_FP32: (
        "tailrl_population",
        "tailrl_population",
    ),
    MechanismVariant.CATEGORICAL_CE_RANDOM_FP32: ("ce", "ce"),
    MechanismVariant.CATEGORICAL_TAILRL_RANDOM_FP32: (
        "tailrl_population",
        "tailrl_population",
    ),
    MechanismVariant.BINOMIAL_CE_FREE_FP32: ("ce", "ce"),
    MechanismVariant.BINOMIAL_TAILRL_FREE_FP32: (
        "tailrl_population",
        "tailrl_population",
    ),
}

BINOMIAL_MODES: dict[MechanismVariant, BinomialMode] = {
    MechanismVariant.BINOMIAL_TAILRL: "fixed",
    MechanismVariant.BINOMIAL_CE: "fixed",
    MechanismVariant.BINOMIAL_TAILRL_FREE: "free",
    MechanismVariant.BINOMIAL_CE_FREE: "free",
    MechanismVariant.BINOMIAL_TAILRL_AFFINE: "affine",
    MechanismVariant.BINOMIAL_CE_AFFINE: "affine",
    MechanismVariant.BINOMIAL_CE_FREE_FP32: "free",
    MechanismVariant.BINOMIAL_TAILRL_FREE_FP32: "free",
}

FUSED_BINOMIAL_VARIANTS = {
    MechanismVariant.CATEGORICAL_CE_BINOMIAL_INIT_FP32,
    MechanismVariant.CATEGORICAL_TAILRL_BINOMIAL_INIT_FP32,
}
FP32_VARIANTS = FUSED_BINOMIAL_VARIANTS | {
    MechanismVariant.CATEGORICAL_CE_RANDOM_FP32,
    MechanismVariant.CATEGORICAL_TAILRL_RANDOM_FP32,
    MechanismVariant.BINOMIAL_CE_FREE_FP32,
    MechanismVariant.BINOMIAL_TAILRL_FREE_FP32,
}


class MechanismConfig(BaseConfig):
    adaptation: AdaptationConfig
    variant: MechanismVariant
    head_seed: int = Field(ge=0)

    @property
    def scalar(self) -> bool:
        return self.variant in {
            MechanismVariant.SCALAR_MSE,
            MechanismVariant.SCALAR_MAE,
        }

    @property
    def binomial(self) -> bool:
        return self.variant in BINOMIAL_MODES

    @property
    def binomial_mode(self) -> BinomialMode:
        return BINOMIAL_MODES[self.variant]

    @property
    def head_precision(self) -> HeadPrecision:
        return "fp32" if self.variant in FP32_VARIANTS else "bf16"

    @property
    def categorical_binomial_init(self) -> bool:
        return self.variant in FUSED_BINOMIAL_VARIANTS

    @property
    def head_initialization(
        self,
    ) -> Literal[
        "scalar_random",
        "score_zero_residual",
        "categorical_binomial_fused",
        "categorical_random",
    ]:
        if self.scalar:
            return "scalar_random"
        if self.binomial:
            return "score_zero_residual"
        return (
            "categorical_binomial_fused"
            if self.categorical_binomial_init
            else "categorical_random"
        )

    @property
    def head_method(self) -> MechanismLoss:
        return OBJECTIVES[self.variant][0]

    @property
    def backbone_method(self) -> MechanismLoss:
        return OBJECTIVES[self.variant][1]

    @model_validator(mode="after")
    def validate_experiment(self) -> "MechanismConfig":
        adaptation = self.adaptation
        run = adaptation.run
        expected_head = "scalar" if self.scalar else "categorical"
        if run.head != expected_head:
            raise ValueError(f"{self.variant} requires run.head={expected_head}")
        if run.loss.method != self.head_method:
            raise ValueError(f"{self.variant} requires loss.method={self.head_method}")
        if adaptation.initialization not in {"imagenet", "scratch"}:
            raise ValueError("Mechanism runs require fresh ImageNet or scratch models")
        if run.pretrained != (adaptation.initialization == "imagenet"):
            raise ValueError("run.pretrained must match adaptation.initialization")
        if adaptation.backbone_frozen or adaptation.use_feature_bank:
            raise ValueError("Mechanism runs train an unfrozen image encoder")
        if adaptation.global_start_epoch or adaptation.control.extend_epoch:
            raise ValueError("Mechanism runs do not support trajectory extensions")
        if run.eval_limit:
            raise ValueError(
                "Mechanism runs evaluate the full validation and test splits"
            )
        if not run.save_checkpoint:
            raise ValueError(
                "Mechanism runs require best, final, and latest checkpoints"
            )
        milestones = adaptation.control.model_milestones
        if tuple(sorted(set(milestones))) != milestones or any(
            epoch < 1 or epoch > run.epochs for epoch in milestones
        ):
            raise ValueError(
                "Model milestones must be unique increasing training epochs"
            )
        return self
