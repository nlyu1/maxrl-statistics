from __future__ import annotations

import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.config.base import BaseConfig
from src.data.bag_of_words import BagOfWordsDatasetConfig, canonical_bags
from src.data.dataloading import DataloadingConfig
from src.data.parquet import TokenizedParquetDatasetConfig
from src.model.minimal import CausalLMConfig
from src.model.optimizer import CausalLMWithLinearHeadOptimizerConfig

if TYPE_CHECKING:
    from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState


def _ceil_to_multiple(*, value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class BagOfWordsStudyBaseConfig(BaseConfig):
    data: BagOfWordsDatasetConfig
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
        dataset_base_folder: Path,
        study_base_folder: Path,
        corr: float,
        aux_words_ratio: float,
        num_words: int = 7,
        num_samples: int = 50_000,
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
    ) -> dict:
        if num_words not in canonical_bags:
            supported = ", ".join(str(n) for n in sorted(canonical_bags))
            raise ValueError(f"num_words must be one of {{{supported}}}")

        dataset_folder = dataset_base_folder / (
            f"{num_words}-words_corr-{corr}_len-{prompt_length}"
            f"_pow-{word_decay_power}_ar-{aux_words_ratio}"
        )
        data = BagOfWordsDatasetConfig.init_or_load_from(
            folder=dataset_folder,
            corr=corr,
            num_train_samples=num_samples,
            num_val_samples=num_samples,
            prompt_length=prompt_length,
            word_assignments=list(canonical_bags[num_words]),
            aux_words_ratio=aux_words_ratio,
            word_decay_power=word_decay_power,
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
    def get_canonical(cls, **kwargs: object) -> "BagOfWordsStudyBaseConfig":
        config = cls(**cls.canonical_kwargs(**kwargs))
        config.prepare_study_folder()
        return config

    def prepare_study_folder(self) -> None:
        self.study_folder.mkdir(parents=True, exist_ok=True)
        config_path = self.study_folder / "config.json"
        config_json = self.model_dump_json(indent=2)
        if not config_path.exists():
            config_path.write_text(config_json)
            return

        saved = type(self).model_validate_json(config_path.read_text())
        if saved != self:
            warnings.warn(
                f"{config_path} already exists with different config contents"
            )

    def get_state_cls(self) -> type[BagOfWordsStudyBaseState]:
        from src.experiments.bag_of_words.state import BagOfWordsStudyBaseState

        return BagOfWordsStudyBaseState

    def initialize(
        self,
        *,
        device: torch.device,
    ) -> BagOfWordsStudyBaseState:
        torch.set_float32_matmul_precision("medium")
        self.prepare_study_folder()
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

        return self.get_state_cls()(
            config=self,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            train_dl=train_dl,
            val_dl=val_dl,
            device=device,
        )
