from __future__ import annotations

import json
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import polars as pl
import torch
from pydantic import ValidationError

from src import get_repo_base
from src.config.base import BaseConfig
from src.data.corpus_regression import (
    CorpusRegressionDataloadingConfig,
    CorpusRegressionDataset,
    CorpusRegressionDatasetConfig,
)
from src.model.minimal import CausalLMConfig
from src.model.optimizer import (
    CausalLMWithLinearHeadOptimizerConfig,
    LRScheduleConfig,
    build_lr_scheduler,
)

if TYPE_CHECKING:
    from src.experiments.corpus_regression.state import CorpusRegressionStudyBaseState


class CorpusRegressionStudyBaseConfig(BaseConfig):
    data: CorpusRegressionDatasetConfig
    dataset_folder: Path
    dataloading: CorpusRegressionDataloadingConfig

    model: CausalLMConfig
    optimizer: CausalLMWithLinearHeadOptimizerConfig

    train_steps: int
    val_every_n_steps: int
    study_folder: Path
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"

    @classmethod
    def canonical_kwargs(
        cls,
        *,
        dataset_base_folder: Path,
        study_base_folder: Path,
        num_lookforward_tokens: int,
        embedding_dim: int = 32,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        prefix_length: int = 128,
        num_samples: int = 100_000,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        batch_size: int = 64,
        eval_batch_size_multiple: int = 2,
        weight_decay: float = 0.0,
        lr_per_sample: float = 1e-5,
        backbone_lr_divisor: float = 6.66,
        lr_schedule: Literal["flat", "cosine"] = "flat",
        warmup_ratio: float = 0.05,
        lr_min_ratio: float = 0.1,
        train_steps: int = 10_000,
        val_every_n_steps: int = 2000,
        clip_grad_norm: float = 1.0,
        compile_model: bool = True,
        head_init_norm: float = 1.0,
        train_from_scratch: bool = False,
    ) -> dict:
        data_config = CorpusRegressionDatasetConfig(
            prefix_length=prefix_length,
            num_samples=num_samples,
            pretrained_tokenizer_model_name=model_name,
            num_lookforward_tokens=num_lookforward_tokens,
            embedding_dim=embedding_dim,
            label_type=label_type,
            normalize_labels=normalize_labels,
            label_range=label_range,
        )
        dataset_folder = data_config.get_canonical_folder(dataset_base_folder)
        # Build (or reuse) the on-disk dataset.
        data_config.init_or_load_from(folder=dataset_folder, **data_config.model_dump())

        dataloading = CorpusRegressionDataloadingConfig(
            train_batch_size=batch_size,
            eval_batch_size=batch_size * eval_batch_size_multiple,
            drop_last=True,
        )

        # Use post-validation embedding_dim (forced to 1 for token_id).
        effective_dim = data_config.embedding_dim
        # Per-sample (linear-scaling-rule) LR formula. The previous per-token
        # convention (× prefix_length) was unmotivated for the methods that
        # produce one prediction per sample at the last position only. The
        # × batch_size factor is retained so batch-size sweeps preserve the
        # same gradient-noise scale.
        head_lr = lr_per_sample * batch_size
        # From-scratch training conventionally uses the same LR for backbone
        # and head — there are no pretrained features to preserve, so the
        # fine-tuning divisor is dropped.
        effective_backbone_lr_divisor = (
            1.0 if train_from_scratch else backbone_lr_divisor
        )
        lr_schedule_config = LRScheduleConfig(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        sched_segment = lr_schedule_segment(
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            lr_min_ratio=lr_min_ratio,
        )
        bs_segment = batch_size_segment(batch_size=batch_size)
        study_folder = study_base_folder / dataset_folder.name
        if bs_segment is not None:
            study_folder = study_folder / bs_segment
        study_folder = (
            study_folder
            / f"lr_{lr_per_sample:.2e}"
            / f"steps-{train_steps:06d}"
        )
        if sched_segment is not None:
            study_folder = study_folder / sched_segment
        return dict(
            data=data_config,
            dataset_folder=dataset_folder,
            dataloading=dataloading,
            model=CausalLMConfig(
                pretrained_model=model_name,
                initial_output_norms=[head_init_norm] * effective_dim,
                train_from_scratch=train_from_scratch,
            ),
            optimizer=CausalLMWithLinearHeadOptimizerConfig(
                lr=head_lr / effective_backbone_lr_divisor,
                head_lr=head_lr,
                weight_decay=weight_decay,
                clip_grad_norm=clip_grad_norm,
                lr_schedule=lr_schedule_config,
            ),
            train_steps=train_steps,
            val_every_n_steps=val_every_n_steps,
            study_folder=study_folder,
            compile_model=compile_model,
        )

    @classmethod
    def get_canonical(cls, **kwargs: object) -> Self:
        config = cls(**cls.canonical_kwargs(**kwargs))
        config.save_config_json()
        return config

    @classmethod
    def has_study_started(cls, study_folder: Path) -> bool:
        return (study_folder / "train_metrics.parquet").exists() and (
            study_folder / "config.json"
        ).exists()

    @classmethod
    def is_study_complete(cls, study_folder: Path) -> bool:
        """True iff the study at *study_folder* has completed all its training steps."""
        if not cls.has_study_started(study_folder):
            return False
        train_steps = json.loads((study_folder / "config.json").read_text())[
            "train_steps"
        ]
        max_step = (
            pl
            .read_parquet(study_folder / "train_metrics.parquet")
            .select(pl.col("step").max())
            .item()
        )
        return max_step >= train_steps - 1

    @classmethod
    def prepare_study_folder(cls, *, study_folder: Path, tag: str) -> bool:
        """Return True if the study should run; False if it's already complete.

        Wipes any partial artifacts so a retry starts clean.
        """
        if cls.is_study_complete(study_folder):
            print(f"!!! {tag} already complete, skipping ({study_folder})")
            return False
        if cls.has_study_started(study_folder):
            print(f"=== {tag} partial → wiping {study_folder} ===")
            shutil.rmtree(study_folder)
        return True

    def save_config_json(self) -> None:
        """Create the study folder and persist `config.json`. If the file
        already exists, verify it matches `self` and warn on mismatch. If
        the saved JSON fails validation under the current schema (e.g. a
        new required field was added), warn and overwrite."""
        self.study_folder.mkdir(parents=True, exist_ok=True)
        config_path = self.study_folder / "config.json"
        config_json = self.model_dump_json(indent=2)
        if not config_path.exists():
            config_path.write_text(config_json)
            return

        try:
            saved = type(self).model_validate_json(config_path.read_text())
        except ValidationError as e:
            warnings.warn(
                f"{config_path} fails validation under the current schema; "
                f"overwriting.\n{e}"
            )
            config_path.write_text(config_json)
            return
        if saved != self:
            warnings.warn(
                f"{config_path} already exists with different config contents"
            )

    def get_state_cls(self) -> type[CorpusRegressionStudyBaseState]:
        from src.experiments.corpus_regression.state import (
            CorpusRegressionStudyBaseState,
        )

        return CorpusRegressionStudyBaseState

    def _build_common_state_kwargs(
        self,
        *,
        device: torch.device,
    ) -> dict:
        torch.set_float32_matmul_precision("medium")
        self.save_config_json()
        dataset = CorpusRegressionDataset.load_from(self.dataset_folder)
        train_dl = self.dataloading.get_train_dataloader(dataset)
        val_dl = self.dataloading.get_val_dataloader(dataset)

        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context:
            model = self.model.get_model().to(device=device, dtype=torch.bfloat16)
            optimizer = self.optimizer.get_optimizer(model)
            scheduler = build_lr_scheduler(
                optimizer,
                schedule_config=self.optimizer.lr_schedule,
                train_steps=self.train_steps,
            )
            if self.compile_model and device.type == "cuda":
                model = torch.compile(model, mode=self.compile_mode)

        return dict(
            config=self,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=dataset,
            train_dl=train_dl,
            val_dl=val_dl,
            device=device,
        )

    def _extra_state_kwargs(self) -> dict:
        """Subclass hook: extra kwargs appended to the state constructor call."""
        return {}

    def initialize(
        self,
        *,
        device: torch.device,
    ) -> CorpusRegressionStudyBaseState:
        common = self._build_common_state_kwargs(device=device)
        return self.get_state_cls()(**common, **self._extra_state_kwargs())


# Default training batch size. Path layouts written before `batch_size` was
# exposed as a sweep dimension all used this value. `batch_size_segment`
# returns `None` here so existing artifacts keep their byte-for-byte path.
DEFAULT_BATCH_SIZE: int = 64


def batch_size_segment(*, batch_size: int) -> str | None:
    """Optional path segment encoding the training batch size, inserted
    **between** ``{dataset_folder.name}`` and ``lr_<value>`` in study folders
    and orchestrator log paths.

    Returns ``None`` for the canonical default (`DEFAULT_BATCH_SIZE = 64`)
    so existing on-disk artifacts keep their byte-for-byte layout — exactly
    mirroring `lr_schedule_segment`'s ``"flat" → None`` convention. For any
    other value, returns ``f"bs-{batch_size}"``.

    Both write-side (this module's ``canonical_kwargs``, ``sl_ce.py``'s
    ``canonical_kwargs``) and read-side (``analysis.py`` ``from_*_sweep``
    loaders) **must** call this same helper so they cannot drift apart.
    """
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size!r}",
        )
    if batch_size == DEFAULT_BATCH_SIZE:
        return None
    return f"bs-{batch_size}"


def baseline_mode_folder(*, subtract_baseline: bool) -> str:
    return "subtract-baseline" if subtract_baseline else "no-subtract-baseline"


def likelihood_mode_folder(*, use_factorized_likelihoods: bool) -> str:
    return "factorized" if use_factorized_likelihoods else "joint"


def factorized_mode_folder(*, factorized: bool) -> str:
    return "factorized" if factorized else "joint"


def sigma_folder(*, gaussian_stdev: float) -> str:
    return f"sigma-{gaussian_stdev:.1f}"


def lr_schedule_segment(
    *,
    lr_schedule: Literal["flat", "cosine"],
    warmup_ratio: float,
    lr_min_ratio: float,
) -> str | None:
    """Optional path segment encoding the LR schedule, inserted **after** the
    `lr_<value>` segment in study folders and orchestrator log paths.

    Returns `None` for `"flat"` so the caller skips the segment entirely —
    this preserves byte-for-byte backward compatibility with all existing
    on-disk artifacts written under flat-LR training. Non-flat schedules are
    encoded as ``sched-warmup-{warmup_ratio:.3f}-cosine-{lr_min_ratio:.2f}``,
    e.g. ``sched-warmup-0.050-cosine-0.10``.

    Both write-side (``config.py`` / ``sl_ce.py`` ``canonical_kwargs``) and
    read-side (``analysis.py from_*_sweep`` loaders) **must** call this same
    helper so they cannot drift apart.
    """
    if lr_schedule == "flat":
        return None
    if lr_schedule != "cosine":
        raise ValueError(
            f"Unsupported lr_schedule={lr_schedule!r}; "
            "expected 'flat' or 'cosine'.",
        )
    return f"sched-warmup-{warmup_ratio:.3f}-cosine-{lr_min_ratio:.2f}"


def project_dir() -> Path:
    return get_repo_base() / "artifacts" / "corpus-regression"


def data_dir() -> Path:
    """Canonical dataset cache: <repo>/artifacts/corpus-regression/data/."""
    return project_dir() / "data"


def artifacts_dir() -> Path:
    """Canonical sweep-output root: <repo>/artifacts/corpus-regression/artifacts/."""
    return project_dir() / "artifacts"
