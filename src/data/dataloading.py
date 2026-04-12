from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.config.base import BaseConfig
from src.data.utils import token_length_distribution_plot


class TokenizedRegressionDatasetConfig(BaseConfig, ABC):
    tokenizer_model_name: str
    folder: Path

    filter_samples_above_n_tokens: int
    pad_to_multiple: int

    @abstractmethod
    def to_dataset(self) -> "TokenizedRegressionDataset": ...


@dataclass
class RegressionSplit:
    tokens: Int[Tensor, "num seq"]
    targets: Float[Tensor, "num"]
    ground_truth: Float[Tensor, "num"]

    @property
    def num_samples(self) -> int:
        return self.tokens.shape[0]


@dataclass
class TokenizedRegressionDataset:
    train: RegressionSplit
    val: RegressionSplit

    # Same length as original unfiltered parquets — horizontally concat-able
    unfiltered_train_token_lengths: Int[Tensor, "num_unfiltered"]
    unfiltered_val_token_lengths: Int[Tensor, "num_unfiltered"]

    @property
    def ceil_padded_seqlen(self) -> int:
        return self.train.tokens.shape[1]

    @property
    def num_train_samples(self) -> int:
        return self.train.num_samples

    @property
    def num_val_samples(self) -> int:
        return self.val.num_samples

    def token_lengths_plot(self, filter_threshold: int | None = None) -> go.Figure:
        return token_length_distribution_plot(
            self.unfiltered_train_token_lengths,
            self.unfiltered_val_token_lengths,
            train_num_filtered=self.num_train_samples,
            val_num_filtered=self.num_val_samples,
            filter_threshold=filter_threshold,
            pad_threshold=self.ceil_padded_seqlen,
        )


class DataloadingConfig(BaseConfig):
    train_batch_size: int
    eval_batch_size: int
    drop_last: bool  # Applies to both train & val
    world_size: int
    rank: int

    def _shard_split(self, split: RegressionSplit) -> RegressionSplit:
        return RegressionSplit(
            tokens=split.tokens[self.rank :: self.world_size],
            targets=split.targets[self.rank :: self.world_size],
            ground_truth=split.ground_truth[self.rank :: self.world_size],
        )

    def _get_dataloader(
        self,
        split: RegressionSplit,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        sharded_split = self._shard_split(split)
        return DataLoader(
            TensorDataset(
                sharded_split.tokens,
                sharded_split.targets,
                sharded_split.ground_truth,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
        )

    def get_train_dataloader(self, ds: TokenizedRegressionDataset) -> DataLoader:
        """Shards train split by rank::world_size, then shuffles."""
        return self._get_dataloader(
            ds.train,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def get_val_dataloader(self, ds: TokenizedRegressionDataset) -> DataLoader:
        """Shards val split by rank::world_size."""
        return self._get_dataloader(
            ds.val,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )
