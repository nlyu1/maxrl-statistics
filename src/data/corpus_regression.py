from src.config.base import BaseConfig
"""
This aims to be a more standalone module.
There are some duplication with bag-of-words library code
due to minor semantic changes. It's a design choice to keep
them slightly separate.
"""

class CorpusRegressionDatasetConfig(BaseConfig):
    """
    Initializes `HuggingFaceFW/fineweb-edu`.
    Extracts `num_samples * 2` samples of length >=
        `prefix_length + num_lookforward_tokens`.
    Each sample consists of `prefix_length` tokens and
        `embedding_dim` number of pm-1.
    """

    prefix_length: int
    num_samples: int
    pretrained_tokenizer_model_name: str

    num_lookfoward_tokens: int # >= 1
    embedding_dim: int # dimensions of rademacher embedding

@pydantic dataclass
class CorpusRegressionDataset:
    config: CorpusRegressionDatasetConfig

    train_tokens: Int[Tensor, "batch seq_len"]
    train_labels: Float[Tensor, "batch dim"]

    val_tokens: Int[Tensor, "batch seq_len"]
    val_labels: Float[Tensor, "batch dim"]

    def write_to():...
        """
        Use pickles for the tokens & labels (cpu tensors).
        Use json for the configs
        """

    def load_from(self, folder: Path):...


class CorpusRegressionDataloadingConfig(BaseConfig):
    """
    Single-device, by default
    """
    train_batch_size: int
    eval_batch_size: int
    drop_last: bool

    def get_train_dataloader(self, ds: CorpusRegressionDataset) -> ...
        pass

    def get_val_dataloader(self, ds: CorpusRegressionDataset) -> ...