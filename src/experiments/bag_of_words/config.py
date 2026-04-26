from __future__ import annotations

import json
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, get_args

import polars as pl
import torch
from pydantic import ValidationError

from src.config.base import BaseConfig
from src.data.bag_of_words import BagOfWordsDatasetConfig, canonical_bags
from src.data.dataloading import DataloadingConfig
from src.data.heterogeneous_bag_of_words import (
    RowHeterogeneousBagOfWordsDatasetConfig,
    SignalHeterogeneousBagOfWordsDatasetConfig,
)
from src.data.parquet import TokenizedParquetDatasetConfig
from src.model.minimal import CausalLMConfig
from src.model.optimizer import CausalLMWithLinearHeadOptimizerConfig

# Each member inherits BaseConfig's extra='forbid', so exactly one class
# validates a given payload (the one whose required het fields match). Order
# is not load-bearing — a homoskedastic payload fails both het members on
# missing required fields, and a het payload fails the base on the forbidden
# extra field.
BagOfWordsDatasetConfigUnion = (
    BagOfWordsDatasetConfig
    | RowHeterogeneousBagOfWordsDatasetConfig
    | SignalHeterogeneousBagOfWordsDatasetConfig
)

if TYPE_CHECKING:
    from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState


DatasetKind = Literal["homoskedastic", "row_heteroskedastic", "word_heteroskedastic"]
DATASET_KINDS: tuple[str, ...] = get_args(DatasetKind)


# Canonical sweep-time defaults. Changing these changes the dataset folder
# name and invalidates every `bow-*-sweep` artifact layout.
CANONICAL_AUX_WORDS_RATIO: float = 0.5
_CANONICAL_ROW_HARDNESS_ETA: float = 8.0
_CANONICAL_SNR_HALFLIFE: float = 0.15


SWEEP_ROOT_SUFFIX_BY_DATASET: dict[str, str] = {
    "homoskedastic": "-hom-sweep",
    "row_heteroskedastic": "-row-het-sweep",
    "word_heteroskedastic": "-word-het-sweep",
}


def _ceil_to_multiple(*, value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _default_num_words(dataset: DatasetKind) -> int:
    return 7 if dataset == "homoskedastic" else 15


def _dataset_config_cls(dataset: DatasetKind) -> type[BagOfWordsDatasetConfig]:
    if dataset == "homoskedastic":
        return BagOfWordsDatasetConfig
    if dataset == "row_heteroskedastic":
        return RowHeterogeneousBagOfWordsDatasetConfig
    if dataset == "word_heteroskedastic":
        return SignalHeterogeneousBagOfWordsDatasetConfig
    raise ValueError(f"unknown dataset {dataset!r}")


def _dataset_extra_kwargs(
    *,
    dataset: DatasetKind,
    row_hardness_eta: float,
    snr_halflife_in_word_quantile: float,
) -> dict:
    if dataset == "homoskedastic":
        return {}
    if dataset == "row_heteroskedastic":
        return {"row_hardness_eta": row_hardness_eta}
    if dataset == "word_heteroskedastic":
        return {"snr_halflife_in_word_quantile": snr_halflife_in_word_quantile}
    raise ValueError(f"unknown dataset {dataset!r}")


def _dataset_folder_suffix(
    *,
    dataset: DatasetKind,
    row_hardness_eta: float,
    snr_halflife_in_word_quantile: float,
) -> str:
    if dataset == "homoskedastic":
        return ""
    if dataset == "row_heteroskedastic":
        return f"_eta-{row_hardness_eta}"
    if dataset == "word_heteroskedastic":
        return f"_hl-{snr_halflife_in_word_quantile}"
    raise ValueError(f"unknown dataset {dataset!r}")


def canonical_dataset_folder_name(
    *,
    dataset: DatasetKind,
    corr: float,
    aux_words_ratio: float,
    num_words: int | None = None,
    prompt_length: int = 128,
    word_decay_power: float = 1.0,
    row_hardness_eta: float = _CANONICAL_ROW_HARDNESS_ETA,
    snr_halflife_in_word_quantile: float = _CANONICAL_SNR_HALFLIFE,
) -> str:
    """Single source of truth for the dataset subfolder name. Shared by
    `canonical_kwargs` (run time) and analysis.py (discovery time)."""
    resolved_num_words = _default_num_words(dataset) if num_words is None else num_words
    suffix = _dataset_folder_suffix(
        dataset=dataset,
        row_hardness_eta=row_hardness_eta,
        snr_halflife_in_word_quantile=snr_halflife_in_word_quantile,
    )
    return (
        f"{resolved_num_words}-words_corr-{corr}_len-{prompt_length}"
        f"_pow-{word_decay_power}_ar-{aux_words_ratio}{suffix}"
    )


def sweep_root_name(*, method: str, dataset: DatasetKind) -> str:
    """Name of the top-level artifacts folder for a given (method, dataset).

    Example: method='maxrl', dataset='word_heteroskedastic' -> 'bow-maxrl-word-het-sweep'.
    Mirrors the layout used by single_run / orchestrate / analysis.
    """
    if dataset not in SWEEP_ROOT_SUFFIX_BY_DATASET:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of "
            f"{sorted(SWEEP_ROOT_SUFFIX_BY_DATASET)}"
        )
    return f"bow-{method}{SWEEP_ROOT_SUFFIX_BY_DATASET[dataset]}"


def baseline_mode_folder(*, subtract_baseline: bool) -> str:
    return "subtract-baseline" if subtract_baseline else "no-subtract-baseline"


def stride_4_top_down(n: int) -> list[int]:
    return [i for offset in range(4) for i in range(offset, n, 4)]


def stride_4_bottom_up(n: int) -> list[int]:
    return list(reversed(stride_4_top_down(n)))


def has_study_started(study_folder: Path) -> bool:
    return (study_folder / "metrics.parquet").exists() and (
        study_folder / "config.json"
    ).exists()


def is_study_complete(study_folder: Path) -> bool:
    """True iff the study at *study_folder* has completed all its training epochs."""
    if not has_study_started(study_folder):
        return False
    train_epochs = json.loads((study_folder / "config.json").read_text())[
        "train_epochs"
    ]
    max_epoch = (
        pl
        .read_parquet(study_folder / "metrics.parquet")
        .select(pl.col("epoch").max())
        .item()
    )
    return max_epoch >= train_epochs - 1


def prepare_study_folder(*, study_folder: Path, tag: str) -> bool:
    """Return True if the study should run; False if it's already complete.

    Wipes any partial artifacts so a retry starts clean.
    """
    if is_study_complete(study_folder):
        print(f"!!! {tag} already complete, skipping ({study_folder})")
        return False
    if has_study_started(study_folder):
        print(f"=== {tag} partial → wiping {study_folder} ===")
        shutil.rmtree(study_folder)
    return True


class BagOfWordsStudyBaseConfig(BaseConfig):
    data: BagOfWordsDatasetConfigUnion
    tokenization: TokenizedParquetDatasetConfig
    dataloading: DataloadingConfig

    model: CausalLMConfig
    optimizer: CausalLMWithLinearHeadOptimizerConfig

    train_epochs: int
    study_folder: Path
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"

    @classmethod
    def canonical_kwargs(
        cls,
        *,
        dataset: DatasetKind,
        dataset_base_folder: Path,
        study_base_folder: Path,
        corr: float,
        aux_words_ratio: float,
        num_samples: int = 50_000,
        num_words: int | None = None,
        prompt_length: int = 128,
        filter_samples_above_n_tokens: int = 384,
        word_decay_power: float = 1.0,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        eval_batch_size_multiple: int = 2,
        lr_per_token: float = 1e-6,
        backbone_lr_divisor: float = 6.66,
        pad_to_multiple: int = 8,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        train_epochs: int = 20,
        clip_grad_norm: float = 1.0,
        compile_model: bool = True,
        row_hardness_eta: float = _CANONICAL_ROW_HARDNESS_ETA,
        snr_halflife_in_word_quantile: float = _CANONICAL_SNR_HALFLIFE,
    ) -> dict:
        resolved_num_words = (
            _default_num_words(dataset) if num_words is None else num_words
        )
        if resolved_num_words not in canonical_bags:
            supported = ", ".join(str(n) for n in sorted(canonical_bags))
            raise ValueError(f"num_words must be one of {{{supported}}}")

        dataset_folder = dataset_base_folder / canonical_dataset_folder_name(
            dataset=dataset,
            corr=corr,
            aux_words_ratio=aux_words_ratio,
            num_words=resolved_num_words,
            prompt_length=prompt_length,
            word_decay_power=word_decay_power,
            row_hardness_eta=row_hardness_eta,
            snr_halflife_in_word_quantile=snr_halflife_in_word_quantile,
        )
        data = _dataset_config_cls(dataset).init_or_load_from(
            folder=dataset_folder,
            corr=corr,
            num_train_samples=num_samples,
            num_val_samples=num_samples,
            prompt_length=prompt_length,
            word_assignments=list(canonical_bags[resolved_num_words]),
            aux_words_ratio=aux_words_ratio,
            word_decay_power=word_decay_power,
            **_dataset_extra_kwargs(
                dataset=dataset,
                row_hardness_eta=row_hardness_eta,
                snr_halflife_in_word_quantile=snr_halflife_in_word_quantile,
            ),
        )

        tokenization = TokenizedParquetDatasetConfig(
            tokenizer_model_name=model_name,
            folder=dataset_folder,
            filter_samples_above_n_tokens=filter_samples_above_n_tokens,
            pad_to_multiple=pad_to_multiple,
        )
        dataloading = DataloadingConfig(
            train_batch_size=batch_size,
            eval_batch_size=batch_size * eval_batch_size_multiple,
            drop_last=True,  # This is explicitly ok. Synthetic dataset anyways
            world_size=1,
            rank=0,
        )

        head_lr = lr_per_token * batch_size * prompt_length
        return dict(
            data=data,
            tokenization=tokenization,
            dataloading=dataloading,
            model=CausalLMConfig(
                pretrained_model=model_name,
                initial_output_norms=[corr],
            ),
            optimizer=CausalLMWithLinearHeadOptimizerConfig(
                lr=head_lr / backbone_lr_divisor,
                head_lr=head_lr,
                weight_decay=weight_decay,
                clip_grad_norm=clip_grad_norm,
            ),
            train_epochs=train_epochs,
            study_folder=study_base_folder / dataset_folder.name,
            compile_model=compile_model,
        )

    @classmethod
    def get_canonical(cls, *, dataset: DatasetKind, **kwargs: object) -> Self:
        config = cls(**cls.canonical_kwargs(dataset=dataset, **kwargs))
        config.save_config_json()
        return config

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

    def get_state_cls(self) -> type[BagOfWordsStudyBaseState]:
        from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState

        return BagOfWordsStudyBaseState

    def _build_common_state_kwargs(
        self,
        *,
        device: torch.device,
    ) -> dict:
        torch.set_float32_matmul_precision("medium")
        self.save_config_json()
        dataset = self.tokenization.init_or_load_dataset()
        train_dl = self.dataloading.get_train_dataloader(dataset)
        val_dl = self.dataloading.get_val_dataloader(dataset)

        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context:
            model = self.model.get_model().to(device=device, dtype=torch.bfloat16)
            optimizer = self.optimizer.get_optimizer(model)
            if self.compile_model and device.type == "cuda":
                model = torch.compile(model, mode=self.compile_mode)

        return dict(
            config=self,
            model=model,
            optimizer=optimizer,
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
    ) -> BagOfWordsStudyBaseState:
        common = self._build_common_state_kwargs(device=device)
        return self.get_state_cls()(**common, **self._extra_state_kwargs())
