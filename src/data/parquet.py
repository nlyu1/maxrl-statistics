from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch import Tensor
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer

from src.data.dataloading import (
    RegressionSplit,
    TokenizedRegressionDataset,
    TokenizedRegressionDatasetConfig,
)

_BATCH_SIZE = 1024


def _tokenizer_work_dir(*, folder: Path, tokenizer_model_name: str) -> Path:
    return folder / tokenizer_model_name.replace("/", "--")


def _load_raw_split(*, folder: Path, split_name: str) -> pl.DataFrame:
    return pl.read_parquet(folder / f"{split_name}.parquet")


def _load_tokenizer(*, tokenizer_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _get_token_lengths(*, tokenizer, prompts: list[str]) -> list[int]:
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


def _filter_by_token_length(
    *,
    df: pl.DataFrame,
    prompts: list[str],
    token_lengths: list[int],
    max_tokens: int,
) -> tuple[pl.DataFrame, list[str], list[int]]:
    keep_mask = [n <= max_tokens for n in token_lengths]
    filtered_df = df.filter(pl.Series(keep_mask))
    filtered_prompts = [prompt for prompt, keep in zip(prompts, keep_mask) if keep]
    filtered_lengths = [n for n, keep in zip(token_lengths, keep_mask) if keep]
    return filtered_df, filtered_prompts, filtered_lengths


def _get_ceil_padded_seqlen(
    *,
    train_token_lengths: list[int],
    val_token_lengths: list[int],
    pad_to_multiple: int,
) -> int:
    if not train_token_lengths:
        raise ValueError("No train samples remain after token-length filtering")
    if not val_token_lengths:
        raise ValueError("No val samples remain after token-length filtering")

    max_train = max(train_token_lengths)
    max_val = max(val_token_lengths)
    print(f"Max token length (train): {max_train}")
    print(f"Max token length (val):   {max_val}")

    ceil_padded_seqlen = (
        math.ceil(max(max_train, max_val) / pad_to_multiple) * pad_to_multiple
    )
    print(f"ceil_padded_seqlen:       {ceil_padded_seqlen}")
    return ceil_padded_seqlen


def _tokenize_padded(
    *,
    tokenizer,
    prompts: list[str],
    ceil_padded_seqlen: int,
    desc: str,
) -> Tensor:
    if not prompts:
        raise ValueError("Cannot tokenize an empty split")

    chunks: list[Tensor] = []
    for start in tqdm(range(0, len(prompts), _BATCH_SIZE), desc=desc):
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


def _build_tokenized_split_from_raw(
    *,
    df: pl.DataFrame,
    prompts: list[str],
    tokenizer,
    ceil_padded_seqlen: int,
    desc: str,
) -> RegressionSplit:
    return RegressionSplit(
        tokens=_tokenize_padded(
            tokenizer=tokenizer,
            prompts=prompts,
            ceil_padded_seqlen=ceil_padded_seqlen,
            desc=desc,
        ),
        targets=torch.tensor(df["target"].to_list(), dtype=torch.float32),
        ground_truth=torch.tensor(df["signal"].to_list(), dtype=torch.float32),
    )


def _load_tokenized_split(*, work_dir: Path, split_name: str) -> RegressionSplit:
    df = pl.read_parquet(work_dir / f"{split_name}.parquet")
    return RegressionSplit(
        tokens=torch.tensor(df["tokens"].to_list(), dtype=torch.int32),
        targets=torch.tensor(df["target"].to_list(), dtype=torch.float32),
        ground_truth=torch.tensor(df["ground_truth"].to_list(), dtype=torch.float32),
    )


def _write_tokenized_split(
    *,
    work_dir: Path,
    split_name: str,
    split: RegressionSplit,
) -> None:
    pl.DataFrame(
        {
            "tokens": split.tokens.tolist(),
            "target": split.targets.tolist(),
            "ground_truth": split.ground_truth.tolist(),
        }
    ).write_parquet(work_dir / f"{split_name}.parquet")


def _read_token_lengths(*, path: Path) -> Tensor:
    return torch.tensor(
        pl.read_parquet(path)["token_length"].to_list(),
        dtype=torch.int32,
    )


def _write_token_lengths(*, path: Path, token_lengths: Tensor) -> None:
    pl.DataFrame({"token_length": token_lengths.tolist()}).write_parquet(path)


class TokenizedParquetDatasetConfig(TokenizedRegressionDatasetConfig):
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

        train_df = _load_raw_split(folder=self.folder, split_name="train")
        val_df = _load_raw_split(folder=self.folder, split_name="val")
        tokenizer = _load_tokenizer(tokenizer_model_name=self.tokenizer_model_name)

        train_prompts = train_df["prompt"].to_list()
        val_prompts = val_df["prompt"].to_list()

        train_lengths = _get_token_lengths(tokenizer=tokenizer, prompts=train_prompts)
        val_lengths = _get_token_lengths(tokenizer=tokenizer, prompts=val_prompts)

        unfiltered_train_token_lengths = torch.tensor(train_lengths, dtype=torch.int32)
        unfiltered_val_token_lengths = torch.tensor(val_lengths, dtype=torch.int32)

        filtered_train_df, filtered_train_prompts, filtered_train_lengths = (
            _filter_by_token_length(
                df=train_df,
                prompts=train_prompts,
                token_lengths=train_lengths,
                max_tokens=self.filter_samples_above_n_tokens,
            )
        )
        filtered_val_df, filtered_val_prompts, filtered_val_lengths = (
            _filter_by_token_length(
                df=val_df,
                prompts=val_prompts,
                token_lengths=val_lengths,
                max_tokens=self.filter_samples_above_n_tokens,
            )
        )

        ceil_padded_seqlen = _get_ceil_padded_seqlen(
            train_token_lengths=filtered_train_lengths,
            val_token_lengths=filtered_val_lengths,
            pad_to_multiple=self.pad_to_multiple,
        )

        train = _build_tokenized_split_from_raw(
            df=filtered_train_df,
            prompts=filtered_train_prompts,
            tokenizer=tokenizer,
            ceil_padded_seqlen=ceil_padded_seqlen,
            desc="Tokenize train",
        )
        val = _build_tokenized_split_from_raw(
            df=filtered_val_df,
            prompts=filtered_val_prompts,
            tokenizer=tokenizer,
            ceil_padded_seqlen=ceil_padded_seqlen,
            desc="Tokenize val",
        )

        return TokenizedParquetDataset(
            config=self,
            train=train,
            val=val,
            unfiltered_train_token_lengths=unfiltered_train_token_lengths,
            unfiltered_val_token_lengths=unfiltered_val_token_lengths,
        )

    def init_or_load_dataset(self) -> "TokenizedParquetDataset":
        """Loads from cache if the folder exists and its config matches; otherwise re-generates."""
        work_dir = _tokenizer_work_dir(
            folder=self.folder,
            tokenizer_model_name=self.tokenizer_model_name,
        )
        config_path = work_dir / "config.json"
        if config_path.exists():
            saved = TokenizedParquetDatasetConfig.model_validate_json(
                config_path.read_text()
            )
            if saved == self:
                return TokenizedParquetDataset.from_folder(
                    tokenizer_model_name=self.tokenizer_model_name,
                    folder=self.folder,
                )
            print("Cached config mismatch — re-generating dataset.")
        ds = self.to_dataset()
        ds.to_folder(exists_ok=True)
        return ds


@dataclass
class TokenizedParquetDataset(TokenizedRegressionDataset):
    config: TokenizedParquetDatasetConfig

    @classmethod
    def from_folder(
        cls, *, tokenizer_model_name: str, folder: Path
    ) -> "TokenizedParquetDataset":
        work_dir = _tokenizer_work_dir(
            folder=folder,
            tokenizer_model_name=tokenizer_model_name,
        )
        assert work_dir.exists(), f"Folder {work_dir} does not exist"

        config = TokenizedParquetDatasetConfig.model_validate_json(
            (work_dir / "config.json").read_text()
        )

        train = _load_tokenized_split(work_dir=work_dir, split_name="train")
        val = _load_tokenized_split(work_dir=work_dir, split_name="val")

        unfiltered_train_token_lengths = _read_token_lengths(
            path=work_dir / "unfiltered_train_lengths.parquet"
        )
        unfiltered_val_token_lengths = _read_token_lengths(
            path=work_dir / "unfiltered_val_lengths.parquet"
        )

        return cls(
            config=config,
            train=train,
            val=val,
            unfiltered_train_token_lengths=unfiltered_train_token_lengths,
            unfiltered_val_token_lengths=unfiltered_val_token_lengths,
        )

    def to_folder(self, exists_ok: bool = False):
        """Serializes to folder. Prefer parquet serialization (ints)."""
        work_dir = _tokenizer_work_dir(
            folder=self.config.folder,
            tokenizer_model_name=self.config.tokenizer_model_name,
        )
        work_dir.mkdir(parents=True, exist_ok=exists_ok)

        (work_dir / "config.json").write_text(self.config.model_dump_json())

        _write_tokenized_split(work_dir=work_dir, split_name="train", split=self.train)
        _write_tokenized_split(work_dir=work_dir, split_name="val", split=self.val)
        _write_token_lengths(
            path=work_dir / "unfiltered_train_lengths.parquet",
            token_lengths=self.unfiltered_train_token_lengths,
        )
        _write_token_lengths(
            path=work_dir / "unfiltered_val_lengths.parquet",
            token_lengths=self.unfiltered_val_token_lengths,
        )
