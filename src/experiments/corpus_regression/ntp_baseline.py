"""
NTP baseline for corpus-regression: intrinsic variance proxy.

Uses the pretrained SmolLM2-135M model's next-token distribution to compute
the Bayes-optimal prediction E[label | prefix] = Σ_t P(t|prefix) × Rademacher[t].
The MSE between this expected label and the true label is an upper bound on the
intrinsic variance (irreducible error) of the task.

Requires NO training — single inference pass over the dataset.
Scope: lookforward=1 only (exact via softmax @ Rademacher matmul).
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import polars as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoModelForCausalLM

from src.config.base import BaseConfig
from src.data.corpus_regression import (
    CorpusRegressionDataloadingConfig,
    CorpusRegressionDataset,
    CorpusRegressionDatasetConfig,
)
from src.experiments.corpus_regression.config import data_dir, artifacts_dir
from src.experiments.corpus_regression.state import (
    CorpusRegressionValidationOutput,
    _per_dim_stats,
)
from src.metrics import RegressionStatCounter


class NTPBaselineConfig(BaseConfig):
    """Configuration for the NTP intrinsic-variance baseline.

    No optimizer or training-loop config — this is inference-only.
    """

    data: CorpusRegressionDatasetConfig
    dataset_folder: Path
    dataloading: CorpusRegressionDataloadingConfig
    pretrained_model: str
    study_folder: Path

    @classmethod
    def get_canonical(
        cls,
        *,
        dataset_base_folder: Path,
        study_base_folder: Path,
        num_lookforward_tokens: int = 1,
        embedding_dim: int = 32,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
        normalize_labels: bool = False,
        label_range: tuple[float, float] = (0.0, 1.0),
        prefix_length: int = 128,
        num_samples: int = 100_000,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        eval_batch_size: int = 128,
    ) -> "NTPBaselineConfig":
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
        # Ensure dataset exists on disk.
        data_config.init_or_load_from(folder=dataset_folder, **data_config.model_dump())

        dataloading = CorpusRegressionDataloadingConfig(
            train_batch_size=eval_batch_size,
            eval_batch_size=eval_batch_size,
            drop_last=False,
        )

        study_folder = study_base_folder / dataset_folder.name

        config = cls(
            data=data_config,
            dataset_folder=dataset_folder,
            dataloading=dataloading,
            pretrained_model=model_name,
            study_folder=study_folder,
        )
        config.study_folder.mkdir(parents=True, exist_ok=True)
        config_path = config.study_folder / "config.json"
        config_path.write_text(config.model_dump_json(indent=2))
        return config


def run_ntp_baseline(
    config: NTPBaselineConfig,
    *,
    device: torch.device,
) -> None:
    """Run the NTP intrinsic-variance baseline (inference-only).

    For each prefix, computes:
        E[label | prefix] = softmax(logits) @ label_projector
    then evaluates MSE against the true label.

    For Rademacher labels: label_projector = Rademacher matrix (V, D).
    For token_id labels:   label_projector = arange(V).unsqueeze(1) (V, 1).
    """
    torch.set_float32_matmul_precision("medium")

    # Load dataset.
    dataset = CorpusRegressionDataset.load_from(config.dataset_folder)
    train_dl = config.dataloading.get_train_dataloader(dataset)
    val_dl = config.dataloading.get_val_dataloader(dataset)

    # Load pretrained model (no linear head).
    device_context = (
        torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    )
    with device_context:
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
    model.eval()

    # Build label projector on device.
    vocab_size = model.config.vocab_size
    if config.data.label_type == "rademacher":
        label_projector: Float[Tensor, "vocab D"] = (
            config.data._rademacher_matrix(vocab_size).float().to(device=device)
        )
    elif config.data.label_type == "token_id":
        label_projector: Float[Tensor, "vocab 1"] = (
            torch.arange(vocab_size, device=device, dtype=torch.float32).unsqueeze(1)
        )
        if config.data.normalize_labels:
            lo, hi = config.data.label_range
            label_projector = lo + (hi - lo) * (label_projector / vocab_size)
    else:
        raise ValueError(f"Unknown label_type: {config.data.label_type!r}")

    # Evaluate on both splits.
    train_counter = _evaluate_split(
        model=model,
        dataloader=train_dl,
        label_projector=label_projector,
        device=device,
        desc="ntp_baseline train",
    )
    val_output = _evaluate_split_full(
        model=model,
        dataloader=val_dl,
        label_projector=label_projector,
        device=device,
        desc="ntp_baseline val",
    )
    val_counter = val_output.compute_counter()

    # Serialize results at step=0 (NTP baseline is inference-only, single
    # pass — there is no real training step counter, but the analysis layer
    # ingests `val_metrics.parquet` keyed by `step`, so we write step=0).
    val_output.save_to(config.study_folder / "0")

    sufficient_stats: dict[str, object] = {}
    sufficient_stats.update(_per_dim_stats(train_counter, prefix="train_target"))
    sufficient_stats.update(_per_dim_stats(val_counter, prefix="val_target"))

    metrics_path = config.study_folder / "val_metrics.parquet"
    pl.DataFrame({
        "step": [0],
        **{key: [value] for key, value in sufficient_stats.items()},
    }).write_parquet(metrics_path)

    # Print summary.
    train_stats = train_counter.get_stats()
    val_stats = val_counter.get_stats()
    D = label_projector.shape[1]
    print(f"\n{'='*60}")
    print(f"NTP Baseline — Intrinsic Variance Proxy (label_type={config.data.label_type}, D={D})")
    print(f"{'='*60}")
    print(f"  Train MSE (mean over dims): {float(train_stats.mse.mean()):.6f}")
    print(f"  Val   MSE (mean over dims): {float(val_stats.mse.mean()):.6f}")
    print(f"  Train corr (mean over dims): {float(train_stats.corr.mean()):.6f}")
    print(f"  Val   corr (mean over dims): {float(val_stats.corr.mean()):.6f}")
    print(f"  Artifacts: {config.study_folder}")
    print(f"{'='*60}\n")


@torch.no_grad()
def _evaluate_split(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_projector: Float[Tensor, "vocab D"],
    device: torch.device,
    desc: str,
) -> RegressionStatCounter:
    """Evaluate NTP baseline on a split, returning sufficient statistics."""
    D = label_projector.shape[1]
    counter = RegressionStatCounter.initialize(dim=D)

    device_context = (
        torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    )
    for tokens, target, _lookahead_token_ids in tqdm(dataloader, desc=desc):
        tokens: Int[Tensor, "batch seq"] = tokens.to(device=device, dtype=torch.long)
        target: Float[Tensor, "batch D"] = target.float()

        with device_context, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(input_ids=tokens, use_cache=False)
            logits: Float[Tensor, "batch vocab"] = output.logits[:, -1, :]

        # Compute E[label | prefix] = softmax(logits) @ label_projector.
        probs: Float[Tensor, "batch vocab"] = F.softmax(logits.float(), dim=-1)
        expected_label: Float[Tensor, "batch D"] = probs @ label_projector

        counter.tick(x=expected_label.cpu(), y=target)

    return counter


@torch.no_grad()
def _evaluate_split_full(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_projector: Float[Tensor, "vocab D"],
    device: torch.device,
    desc: str,
) -> CorpusRegressionValidationOutput:
    """Evaluate NTP baseline on a split, returning full per-sample predictions."""
    predictions: list[Float[Tensor, "batch D"]] = []
    targets: list[Float[Tensor, "batch D"]] = []

    device_context = (
        torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    )
    for tokens, target, _lookahead_token_ids in tqdm(dataloader, desc=desc):
        tokens: Int[Tensor, "batch seq"] = tokens.to(device=device, dtype=torch.long)
        target: Float[Tensor, "batch D"] = target.float()

        with device_context, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(input_ids=tokens, use_cache=False)
            logits: Float[Tensor, "batch vocab"] = output.logits[:, -1, :]

        probs: Float[Tensor, "batch vocab"] = F.softmax(logits.float(), dim=-1)
        expected_label: Float[Tensor, "batch D"] = probs @ label_projector

        predictions.append(expected_label.cpu())
        targets.append(target)

    return CorpusRegressionValidationOutput(
        model_preds=torch.cat(predictions),
        target=torch.cat(targets),
    )
