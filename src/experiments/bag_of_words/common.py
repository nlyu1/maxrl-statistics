from src.config.base import BaseConfig
from src.data.bag_of_words import BagOfWordsDatasetConfig
from src.data.dataloading import DataloadingConfig
from src.data.parquet import TokenizedParquetDatasetConfig
from src.model.minimal import CausalLMConfig
from src.model.optimizer import CausalLMWithLinearHeadOptimizerConfig
from pydantic.dataclasses import dataclass
import torch


class BagOfWordsStudyBaseConfig(BaseConfig):
    # --- data pipeline ---
    data: BagOfWordsDatasetConfig
    tokenization: TokenizedParquetDatasetConfig
    dataloading: DataloadingConfig

    # --- model-side ---
    model: CausalLMConfig
    optimizer: CausalLMWithLinearHeadOptimizerConfig

    # --- Training logistics ---
    train_epochs: int
    study_folder: Path

    @classmethod
    def get_canonical(
        *,
        dataset_base_folder: Path,
        study_base_folder: Path,
        snr: float,
        aux_words_ratio: float,
        num_words: int = 7,
        num_samples: int = 50_000,
        prompt_length: int = 256,
        word_decay_power: float = 0.0,
        batch_size: int = 64,
        eval_batch_size_multiple: int = 2,
        lr_per_token: float = 2.3e-8,
        backbone_lr_divisor: float = 6.66,
        pad_to_multiple: int = 8,
        model_name: str ="HuggingFaceTB/SmolLM2-135M"
    ) -> "BagOfWordsStudyBaseConfig":
        pass

    def initialize(
        self,
        *,
        device: torch.device
    ) -> "BagOfWordsStudyBaseState":
        ...



@dataclass(kw_only=True)
class BagOfWordsStudyBaseState: