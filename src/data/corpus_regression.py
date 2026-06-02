from __future__ import annotations

"""
This aims to be a more standalone module.
There are some duplication with bag-of-words library code
due to minor semantic changes. It's a design choice to keep
them slightly separate.

Also imported by `writeup/code/`; check usage there on updates.
"""

import fcntl
import html as _html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
from datasets import load_dataset
from jaxtyping import Float, Int
from pydantic import model_validator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.config.base import BaseConfig

_DATASET = "HuggingFaceFW/fineweb-edu"
_DATASET_CONFIG = "sample-10BT"
_SEED = 42


# Canonical sweep grid for the corpus-regression experiments. 8 lookforward
# horizons spanning next-token to 8-token skip-ahead.
candidate_lookforward_tokens: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8,
]

# Rollout counts for RL sweeps.
candidate_rollout_steps: list[int] = [4, 16, 128, 1024]

candidate_seeds: list[int] = [51, 61, 121, 153, 228]


def _stream_token_prefixes(
    *,
    tokenizer: Any,
    min_length: int,
    target_count: int,
    desc: str,
    overshoot: int = 0,
) -> list[list[int]]:
    """Stream fineweb-edu, return up to `target_count` token-id sequences. Each
    sequence is truncated to `min_length + overshoot` tokens (or the doc's full
    length if shorter than that). Docs with fewer than `min_length` tokens are
    skipped. Single source of truth for streaming used by `build` (overshoot=0)
    and `demonstrate` (overshoot>0 to show trailing context past the label token)."""
    stream = load_dataset(
        _DATASET, name=_DATASET_CONFIG, split="train", streaming=True
    ).shuffle(seed=_SEED, buffer_size=10_000)

    take = min_length + overshoot
    out: list[list[int]] = []
    bar = tqdm(total=target_count, desc=desc)
    for ex in stream:
        text = ex["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_length:
            continue
        out.append(ids[:take])
        bar.update(1)
        if len(out) >= target_count:
            break
    bar.close()
    return out


class CorpusRegressionDatasetConfig(BaseConfig):
    """
    Initializes `HuggingFaceFW/fineweb-edu`.
    Extracts `num_samples * 2` samples of length >=
        `prefix_length + num_lookforward_tokens`.
    Input: the first `prefix_length` tokens of each sample.
    Label: the `embedding_dim`-dim ±1 Rademacher vector of the
        final lookforward token (i.e. the token at index
        `prefix_length + num_lookforward_tokens - 1`).
    With `num_lookforward_tokens > 1` the intermediate tokens are
        not exposed to the model and do not contribute to the label —
        this is a skip-ahead target, not next-token prediction.
    """

    prefix_length: int
    num_samples: int
    pretrained_tokenizer_model_name: str

    num_lookforward_tokens: int  # >= 1
    embedding_dim: int  # dimensions of rademacher embedding
    label_type: Literal["rademacher", "token_id"] = "rademacher"
    normalize_labels: bool = False
    label_range: tuple[float, float] = (0.0, 1.0)

    @model_validator(mode="before")
    @classmethod
    def _enforce_token_id_dim(cls, data: dict[str, Any]) -> dict[str, Any]:
        """When label_type='token_id', force embedding_dim=1."""
        if data.get("label_type") == "token_id":
            data["embedding_dim"] = 1
        return data

    @classmethod
    def canonical_kwargs(
        cls,
        *,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
    ) -> dict[str, Any]:
        return {
            "prefix_length": 128,
            "num_samples": 100_000,
            "pretrained_tokenizer_model_name": "HuggingFaceTB/SmolLM2-135M",
            "num_lookforward_tokens": 1,
            "embedding_dim": 32,
            "label_type": label_type,
            "normalize_labels": normalize_labels,
            "label_range": label_range,
        }

    @classmethod
    def get_canonical(cls) -> Self:
        return cls(**cls.canonical_kwargs())

    def get_canonical_folder(self, base_path: Path) -> Path:
        """Resolve `base_path / <slug>` where the slug encodes every field that
        affects dataset bytes. Single source of truth for cache directory."""
        tokenizer_slug = (
            self.pretrained_tokenizer_model_name.split("/")[-1].lower().replace("-", "_")
        )
        slug = (
            f"fineweb_edu_{tokenizer_slug}"
            f"_{self.num_samples}x{self.prefix_length}"
            f"_look{self.num_lookforward_tokens}_dim{self.embedding_dim}"
        )
        # Only append label_type suffix for non-default values (backward-compatible
        # slug for existing Rademacher caches).
        if self.label_type != "rademacher":
            slug += f"_{self.label_type}"
        if self.normalize_labels:
            lo, hi = self.label_range
            slug += f"_norm{lo:.2g}_{hi:.2g}"
        return base_path / slug

    def _rademacher_matrix(self, vocab_size: int) -> Int[Tensor, "vocab embedding_dim"]:
        """Deterministic ±1 token embedding (seed=0). Row i is the Rademacher
        vector assigned to token i; reproducible across runs and configs."""
        rng = torch.Generator()
        rng.manual_seed(0)
        return (
            torch.randint(0, 2, (vocab_size, self.embedding_dim), generator=rng) * 2 - 1
        ).to(torch.int8)

    def build(self) -> "CorpusRegressionDataset":
        """Stream fineweb-edu, tokenize, compute labels, shuffle,
        and return a train/val split."""
        min_length = self.prefix_length + self.num_lookforward_tokens
        tok = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_model_name)

        collected = _stream_token_prefixes(
            tokenizer=tok,
            min_length=min_length,
            target_count=2 * self.num_samples,
            desc="collecting samples",
        )

        total = len(collected)
        full: Int[Tensor, "total min_length"] = torch.tensor(collected, dtype=torch.long)

        # Input is the prefix; label is derived from the final lookforward token.
        prefix_tokens: Int[Tensor, "total prefix_length"] = full[:, : self.prefix_length].to(torch.int32)

        if self.label_type == "rademacher":
            rademacher = self._rademacher_matrix(len(tok))
            labels: Float[Tensor, "total embedding_dim"] = rademacher[full[:, -1]].float()
        elif self.label_type == "token_id":
            labels: Float[Tensor, "total 1"] = full[:, -1].float().unsqueeze(1)
            if self.normalize_labels:
                vocab_size = len(tok)
                lo, hi = self.label_range
                labels = lo + (hi - lo) * (labels / vocab_size)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type!r}")

        shuffle_rng = torch.Generator()
        shuffle_rng.manual_seed(_SEED)
        perm = torch.randperm(total, generator=shuffle_rng)
        all_tokens: Int[Tensor, "total prefix_length"] = prefix_tokens[perm]
        all_labels: Float[Tensor, "total embedding_dim"] = labels[perm]

        return CorpusRegressionDataset(
            config=self,
            train_tokens=all_tokens[: self.num_samples],
            train_labels=all_labels[: self.num_samples],
            val_tokens=all_tokens[self.num_samples :],
            val_labels=all_labels[self.num_samples :],
        )

    def demonstrate(self, num_samples: int = 50):
        """Stream `num_samples` fresh docs and return a Solara browser for visual
        inspection: prefix (black) + lookahead window (blue) + final predicted token
        (bold blue) + the label. Independent of any cached dataset."""
        min_length = self.prefix_length + self.num_lookforward_tokens
        tok = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_model_name)

        collected = _stream_token_prefixes(
            tokenizer=tok,
            min_length=min_length,
            target_count=num_samples,
            desc="streaming demo samples",
            overshoot=16,
        )
        last = self.prefix_length + self.num_lookforward_tokens - 1

        if self.label_type == "rademacher":
            rademacher = self._rademacher_matrix(len(tok))
            samples = [
                {
                    "prefix_ids": ids[: self.prefix_length],
                    "middle_ids": ids[self.prefix_length : last],
                    "final_id": ids[last],
                    "tail_ids": ids[last + 1 :],
                    "label": rademacher[ids[last]].tolist(),
                }
                for ids in collected
            ]
        elif self.label_type == "token_id":
            samples = [
                {
                    "prefix_ids": ids[: self.prefix_length],
                    "middle_ids": ids[self.prefix_length : last],
                    "final_id": ids[last],
                    "tail_ids": ids[last + 1 :],
                    "label": [ids[last]],
                }
                for ids in collected
            ]
        else:
            raise ValueError(f"Unknown label_type: {self.label_type!r}")

        return _inspect_corpus_samples(tokenizer=tok, samples=samples)

    @classmethod
    def init_or_load_from(cls, *, folder: Path, **init_kwargs) -> Self:
        """Reuse the cached dataset at `folder` if its config matches, else (re)generate.
        kwargs forward to `cls(**init_kwargs)`. Tensors land on disk; materialize them
        with `CorpusRegressionDataset.load_from(folder)`.

        Uses a file lock to prevent concurrent processes from building the same
        dataset simultaneously (race on first cold-start when multiple GPU workers
        share the same dataset config).
        """
        config = cls(**init_kwargs)
        folder.mkdir(parents=True, exist_ok=True)
        lock_path = folder / ".build.lock"
        config_path = folder / "config.json"
        required_paths = (
            folder / "train_tokens.pt",
            folder / "train_labels.pt",
            folder / "val_tokens.pt",
            folder / "val_labels.pt",
        )

        with lock_path.open("w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                # Re-check under lock — another process may have built while we waited.
                if config_path.exists():
                    saved = cls.model_validate_json(config_path.read_text())
                    if saved == config and all(path.exists() for path in required_paths):
                        return saved
                    if saved != config:
                        print("Cached config mismatch — re-building dataset.")
                    else:
                        print("Cached dataset files missing — re-building dataset.")
                config.build().write_to(folder)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
        return config


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


##### Dataset visualization ####
# Pure UI for `CorpusRegressionDatasetConfig.demonstrate`. Knows nothing about
# fineweb-edu or Rademacher embeddings — just renders a tokenizer + pre-collected
# samples (`prefix_ids`, `middle_ids`, `final_id`, `label`).

_PREFIX_COLOR = "#111"
_LOOKAHEAD_COLOR = "#1f77b4"
_POS_COLOR = "#1f77b4"
_NEG_COLOR = "#d62728"


def _sample_body_html(*, tokenizer: Any, sample: dict[str, Any]) -> str:
    final_id = sample["final_id"]
    prefix = _html.escape(tokenizer.decode(sample["prefix_ids"]))
    middle = (
        _html.escape(tokenizer.decode(sample["middle_ids"]))
        if sample["middle_ids"]
        else ""
    )
    final = _html.escape(tokenizer.decode([final_id]))
    tail = (
        _html.escape(tokenizer.decode(sample["tail_ids"]))
        if sample.get("tail_ids")
        else ""
    )
    final_box = (
        f"color:#fff;background:{_LOOKAHEAD_COLOR};font-weight:bold;"
        "padding:0 3px;border-radius:3px;white-space:pre;"
    )
    return (
        f"<span style='color:{_PREFIX_COLOR}'>{prefix}</span>"
        f"<span style='color:{_LOOKAHEAD_COLOR}'>{middle}</span>"
        f"<span style='{final_box}' title='token id {final_id}'>{final}</span>"
        f"<span style='color:#888'>{tail} …</span>"
    )


def _label_vector_html(label: list[int]) -> str:
    cells = " ".join(
        f"<span style='color:{_POS_COLOR if v > 0 else _NEG_COLOR}'>{int(v):+d}</span>"
        for v in label
    )
    return f"<b>label ({len(label)}d):</b> {cells}"


def _inspect_corpus_samples(*, tokenizer: Any, samples: list[dict[str, Any]]):
    """+/- index Solara browser over pre-collected samples."""
    import solara

    @solara.component
    def _panel():
        idx, set_idx = solara.use_state(0)
        n = len(samples)
        sample = samples[idx]

        button_style = (
            "min-width:24px;width:24px;height:24px;padding:0;"
            "font-size:14px;line-height:1;"
        )
        with solara.Column(gap="8px", style={"max-width": "920px"}):
            with solara.Row(gap="4px", style={"align-items": "center"}):
                solara.Button(
                    "−",
                    on_click=lambda: set_idx(max(0, idx - 1)),
                    disabled=(idx == 0),
                    style=button_style,
                )
                solara.Button(
                    "+",
                    on_click=lambda: set_idx(min(n - 1, idx + 1)),
                    disabled=(idx == n - 1),
                    style=button_style,
                )
                solara.HTML(
                    tag="span",
                    unsafe_innerHTML=f"sample <b>{idx + 1}</b> / {n}",
                    style="font-family:monospace;font-size:13px;margin-left:4px;",
                )
            solara.HTML(
                tag="div",
                unsafe_innerHTML=_sample_body_html(tokenizer=tokenizer, sample=sample),
                style=(
                    "font-family:Georgia,serif;font-size:14px;line-height:1.5;"
                    "white-space:pre-wrap;"
                ),
            )
            solara.HTML(
                tag="hr",
                unsafe_innerHTML="",
                style="border:none;border-top:1px solid #ddd;margin:4px 0;",
            )
            solara.HTML(
                tag="div",
                unsafe_innerHTML=_label_vector_html(sample["label"]),
                style="font-family:monospace;font-size:12px;",
            )

    container, render_context = solara.render(_panel())
    setattr(container, "_solara_render_context", render_context)
    return container
