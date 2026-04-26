from __future__ import annotations

"""
This aims to be a more standalone module.
There are some duplication with bag-of-words library code
due to minor semantic changes. It's a design choice to keep
them slightly separate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.config.base import BaseConfig

_DATASET = "HuggingFaceFW/fineweb-edu"
_DATASET_CONFIG = "sample-10BT"
_SEED = 42


class CorpusRegressionDatasetConfig(BaseConfig):
    """
    Initializes `HuggingFaceFW/fineweb-edu`.
    Extracts `num_samples * 2` samples of length >=
        `prefix_length + num_lookfoward_tokens`.
    Each sample consists of `prefix_length` tokens and
        `embedding_dim` ±1 Rademacher labels.

    Labels: element-wise product of per-token ±1 Rademacher vectors
    over the lookforward window. With num_lookfoward_tokens=1 this is
    the single next-token's Rademacher vector.
    """

    prefix_length: int
    num_samples: int
    pretrained_tokenizer_model_name: str

    num_lookfoward_tokens: int  # >= 1
    embedding_dim: int  # dimensions of rademacher embedding

    @classmethod
    def canonical_kwargs(cls) -> dict[str, Any]:
        return {
            "prefix_length": 128,
            "num_samples": 50_000,
            "pretrained_tokenizer_model_name": "HuggingFaceTB/SmolLM2-135M",
            "num_lookfoward_tokens": 1,
            "embedding_dim": 32,
        }

    @classmethod
    def get_canonical(cls) -> Self:
        return cls(**cls.canonical_kwargs())

    def build(self) -> "CorpusRegressionDataset":
        """
        Streams fineweb-edu, tokenizes, computes Rademacher labels,
        shuffles, and returns a train/val split.
        """
        min_length = self.prefix_length + self.num_lookfoward_tokens
        target_total = 2 * self.num_samples

        tok = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_model_name)
        vocab_size = len(tok)

        # Deterministic ±1 embedding: row i is the Rademacher vector for token i.
        # Seeded at 0 so labels are reproducible regardless of dataset shuffle seed.
        rad_rng = torch.Generator()
        rad_rng.manual_seed(0)
        rademacher: Int[Tensor, "vocab embedding_dim"] = (
            torch.randint(0, 2, (vocab_size, self.embedding_dim), generator=rad_rng) * 2 - 1
        ).to(torch.int8)

        ds_stream = load_dataset(
            _DATASET, name=_DATASET_CONFIG, split="train", streaming=True
        ).shuffle(seed=_SEED, buffer_size=10_000)

        prefix_list: list[list[int]] = []
        lookforward_list: list[list[int]] = []

        bar = tqdm(total=target_total, desc="collecting samples")
        for ex in ds_stream:
            text = ex["text"].strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < min_length:
                continue
            prefix_list.append(ids[: self.prefix_length])
            lookforward_list.append(
                ids[self.prefix_length : self.prefix_length + self.num_lookfoward_tokens]
            )
            bar.update(1)
            if len(prefix_list) >= target_total:
                break
        bar.close()

        total = len(prefix_list)

        # Labels: Rademacher vectors for each lookforward token, flattened.
        # Shape: (total, num_lookfoward_tokens * embedding_dim), values in {-1, +1}.
        lf_tensor = torch.tensor(lookforward_list, dtype=torch.long)  # (total, num_look)
        labels: Float[Tensor, "total label_dim"] = rademacher[lf_tensor].flatten(1).float()

        shuffle_rng = torch.Generator()
        shuffle_rng.manual_seed(_SEED)
        perm = torch.randperm(total, generator=shuffle_rng)

        all_tokens: Int[Tensor, "total prefix_length"] = torch.tensor(
            prefix_list, dtype=torch.int32
        )[perm]
        all_labels: Float[Tensor, "total label_dim"] = labels[perm]

        return CorpusRegressionDataset(
            config=self,
            train_tokens=all_tokens[: self.num_samples],
            train_labels=all_labels[: self.num_samples],
            val_tokens=all_tokens[self.num_samples :],
            val_labels=all_labels[self.num_samples :],
        )

    def build_or_load(self, folder: Path) -> "CorpusRegressionDataset":
        """Loads from `folder` if config matches; otherwise builds and saves."""
        config_path = folder / "config.json"
        if config_path.exists():
            saved = CorpusRegressionDatasetConfig.model_validate_json(
                config_path.read_text()
            )
            if saved == self:
                return CorpusRegressionDataset.load_from(folder)
            print("Cached config mismatch — re-building dataset.")
        ds = self.build()
        ds.write_to(folder)
        return ds


@dataclass(kw_only=True)
class CorpusRegressionDataset:
    config: CorpusRegressionDatasetConfig

    train_tokens: Int[Tensor, "batch seq_len"]
    train_labels: Float[Tensor, "batch dim"]

    val_tokens: Int[Tensor, "batch seq_len"]
    val_labels: Float[Tensor, "batch dim"]

    def write_to(self, folder: Path) -> None:
        """Pickles tensors to `folder`; writes config as json."""
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(self.config.model_dump_json())
        torch.save(self.train_tokens, folder / "train_tokens.pt")
        torch.save(self.train_labels, folder / "train_labels.pt")
        torch.save(self.val_tokens, folder / "val_tokens.pt")
        torch.save(self.val_labels, folder / "val_labels.pt")

    @classmethod
    def load_from(cls, folder: Path) -> "CorpusRegressionDataset":
        config = CorpusRegressionDatasetConfig.model_validate_json(
            (folder / "config.json").read_text()
        )
        return cls(
            config=config,
            train_tokens=torch.load(folder / "train_tokens.pt"),
            train_labels=torch.load(folder / "train_labels.pt"),
            val_tokens=torch.load(folder / "val_tokens.pt"),
            val_labels=torch.load(folder / "val_labels.pt"),
        )


class CorpusRegressionDataloadingConfig(BaseConfig):
    """Single-device dataloader config."""

    train_batch_size: int
    eval_batch_size: int
    drop_last: bool

    def get_train_dataloader(self, ds: CorpusRegressionDataset) -> DataLoader:
        return DataLoader(
            TensorDataset(ds.train_tokens, ds.train_labels),
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    def get_val_dataloader(self, ds: CorpusRegressionDataset) -> DataLoader:
        return DataLoader(
            TensorDataset(ds.val_tokens, ds.val_labels),
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
