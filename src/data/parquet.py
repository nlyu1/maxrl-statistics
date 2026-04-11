from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer

from src.config.base import BaseConfig

_BATCH_SIZE = 1024


class TokenizedParquetDatasetConfig(BaseConfig):
    tokenizer_model_name: str
    folder: Path

    filter_samples_above_n_tokens: int
    pad_to_multiple: int

    def to_dataset(self) -> "TokenizedParquetDataset":
        """
        Our working folder is folder/{tokenizer_model_name}
        1. Asserts that "folder" exists
        2. Reads train & val
        3. Instantiates the auto-tokenizer and vector-tokenizes the full dataset.
            - Filter out samples above n tokens
            - Calculates the max train & val num_tokens (uncanonical, I know), prints it,
                then computes a single ceil_padded_seqlen
        4. Instantiates tensors with padding
        """
        assert self.folder.exists(), f"Folder {self.folder} does not exist"

        train_df = pl.read_parquet(self.folder / "train.parquet")
        val_df = pl.read_parquet(self.folder / "val.parquet")

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def get_lengths(prompts: list[str]) -> list[int]:
            lengths: list[int] = []
            for start in range(0, len(prompts), _BATCH_SIZE):
                batch = prompts[start : start + _BATCH_SIZE]
                enc = tokenizer(
                    batch,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                )
                lengths.extend(len(ids) for ids in enc["input_ids"])
            return lengths

        train_prompts = train_df["prompt"].to_list()
        val_prompts = val_df["prompt"].to_list()

        train_lengths = get_lengths(train_prompts)
        val_lengths = get_lengths(val_prompts)

        unfiltered_train_token_lengths = torch.tensor(train_lengths, dtype=torch.int32)
        unfiltered_val_token_lengths = torch.tensor(val_lengths, dtype=torch.int32)

        train_mask = [n <= self.filter_samples_above_n_tokens for n in train_lengths]
        val_mask = [n <= self.filter_samples_above_n_tokens for n in val_lengths]

        filtered_train_df = train_df.filter(pl.Series(train_mask))
        filtered_val_df = val_df.filter(pl.Series(val_mask))
        filtered_train_prompts = [p for p, m in zip(train_prompts, train_mask) if m]
        filtered_val_prompts = [p for p, m in zip(val_prompts, val_mask) if m]
        filtered_train_lengths = [n for n, m in zip(train_lengths, train_mask) if m]
        filtered_val_lengths = [n for n, m in zip(val_lengths, val_mask) if m]

        max_train = max(filtered_train_lengths)
        max_val = max(filtered_val_lengths)
        print(f"Max token length (train): {max_train}")
        print(f"Max token length (val):   {max_val}")
        ceil_padded_seqlen = (
            math.ceil(max(max_train, max_val) / self.pad_to_multiple)
            * self.pad_to_multiple
        )
        print(f"ceil_padded_seqlen:       {ceil_padded_seqlen}")

        def tokenize_padded(prompts: list[str]) -> Tensor:
            chunks: list[Tensor] = []
            for start in tqdm(range(0, len(prompts), _BATCH_SIZE)):
                batch = prompts[start : start + _BATCH_SIZE]
                enc = tokenizer(
                    batch,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=ceil_padded_seqlen,
                    truncation=False,
                    return_tensors="pt",
                )
                chunks.append(enc["input_ids"].to(torch.int32))
            return torch.cat(chunks, dim=0)

        train_tokens = tokenize_padded(filtered_train_prompts)
        val_tokens = tokenize_padded(filtered_val_prompts)

        train_targets = torch.tensor(
            filtered_train_df["target"].to_list(), dtype=torch.float32
        )
        train_ground_truth = torch.tensor(
            filtered_train_df["signal"].to_list(), dtype=torch.float32
        )
        val_targets = torch.tensor(
            filtered_val_df["target"].to_list(), dtype=torch.float32
        )
        val_ground_truth = torch.tensor(
            filtered_val_df["signal"].to_list(), dtype=torch.float32
        )

        return TokenizedParquetDataset(
            config=self,
            train_tokens=train_tokens,
            train_targets=train_targets,
            train_ground_truth=train_ground_truth,
            val_tokens=val_tokens,
            val_targets=val_targets,
            val_ground_truth=val_ground_truth,
            unfiltered_train_token_lengths=unfiltered_train_token_lengths,
            unfiltered_val_token_lengths=unfiltered_val_token_lengths,
        )


@dataclass
class TokenizedParquetDataset:
    config: TokenizedParquetDatasetConfig

    train_tokens: Int[Tensor, "num seq"]
    train_targets: Float[Tensor, "num"]
    train_ground_truth: Float[Tensor, "num"]

    val_tokens: Int[Tensor, "num seq"]
    val_targets: Float[Tensor, "num"]
    val_ground_truth: Float[Tensor, "num"]

    # Same length as original unfiltered parquets — horizontally concat-able
    unfiltered_train_token_lengths: Int[Tensor, "num_unfiltered"]
    unfiltered_val_token_lengths: Int[Tensor, "num_unfiltered"]

    @property
    def ceil_padded_seqlen(self) -> int:
        return self.train_tokens.shape[1]

    @property
    def num_train_samples(self) -> int:
        return self.train_tokens.shape[0]

    @property
    def num_val_samples(self) -> int:
        return self.val_tokens.shape[0]

    @classmethod
    def from_folder(
        cls, *, tokenizer_model_name: str, folder: Path
    ) -> "TokenizedParquetDataset":
        work_dir = folder / tokenizer_model_name.replace("/", "--")
        assert work_dir.exists(), f"Folder {work_dir} does not exist"

        config = TokenizedParquetDatasetConfig.model_validate_json(
            (work_dir / "config.json").read_text()
        )

        def load_split(name: str) -> tuple[Tensor, Tensor, Tensor]:
            df = pl.read_parquet(work_dir / f"{name}.parquet")
            tokens = torch.tensor(df["tokens"].to_list(), dtype=torch.int32)
            targets = torch.tensor(df["target"].to_list(), dtype=torch.float32)
            ground_truth = torch.tensor(
                df["ground_truth"].to_list(), dtype=torch.float32
            )
            return tokens, targets, ground_truth

        train_tokens, train_targets, train_ground_truth = load_split("train")
        val_tokens, val_targets, val_ground_truth = load_split("val")

        unfiltered_train_token_lengths = torch.tensor(
            pl.read_parquet(work_dir / "unfiltered_train_lengths.parquet")[
                "token_length"
            ].to_list(),
            dtype=torch.int32,
        )
        unfiltered_val_token_lengths = torch.tensor(
            pl.read_parquet(work_dir / "unfiltered_val_lengths.parquet")[
                "token_length"
            ].to_list(),
            dtype=torch.int32,
        )

        return cls(
            config=config,
            train_tokens=train_tokens,
            train_targets=train_targets,
            train_ground_truth=train_ground_truth,
            val_tokens=val_tokens,
            val_targets=val_targets,
            val_ground_truth=val_ground_truth,
            unfiltered_train_token_lengths=unfiltered_train_token_lengths,
            unfiltered_val_token_lengths=unfiltered_val_token_lengths,
        )

    def to_folder(self, exists_ok: bool = False):
        """Serializes to folder. Prefer parquet serialization (ints)."""
        work_dir = self.config.folder / self.config.tokenizer_model_name.replace(
            "/", "--"
        )
        work_dir.mkdir(parents=True, exist_ok=exists_ok)

        (work_dir / "config.json").write_text(self.config.model_dump_json())

        for name, tokens, targets, ground_truth in [
            ("train", self.train_tokens, self.train_targets, self.train_ground_truth),
            ("val", self.val_tokens, self.val_targets, self.val_ground_truth),
        ]:
            pl.DataFrame(
                {
                    "tokens": tokens.tolist(),
                    "target": targets.tolist(),
                    "ground_truth": ground_truth.tolist(),
                }
            ).write_parquet(work_dir / f"{name}.parquet")

        pl.DataFrame(
            {"token_length": self.unfiltered_train_token_lengths.tolist()}
        ).write_parquet(work_dir / "unfiltered_train_lengths.parquet")

        pl.DataFrame(
            {"token_length": self.unfiltered_val_token_lengths.tolist()}
        ).write_parquet(work_dir / "unfiltered_val_lengths.parquet")
