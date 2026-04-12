from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import torch
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from src.data.parquet import TokenizedParquetDataset
from src.metrics import CorrelationCounter

if TYPE_CHECKING:
    from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class ValidationOutput:
    preds: Float[Tensor, "n"]
    ground_truths: Float[Tensor, "n"]

    def compute_corr(self) -> Float[Tensor, ""]:
        counter = CorrelationCounter.initialize(dim=1, device=self.preds.device)
        counter.tick(x=self.preds[:, None], y=self.ground_truths[:, None])
        return counter.get_stats().squeeze(0)

    def save_to(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "preds": self.preds.float().cpu().tolist(),
                "ground_truths": self.ground_truths.float().cpu().tolist(),
            }
        ).write_parquet(folder / "validation.parquet")


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class BagOfWordsStudyBaseState(ABC):
    config: BagOfWordsStudyBaseConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataset: TokenizedParquetDataset
    train_dl: DataLoader
    val_dl: DataLoader
    device: torch.device

    current_epoch: int = 0
    train_corr_counter: CorrelationCounter = field(
        default_factory=lambda: CorrelationCounter.initialize(dim=1)
    )

    @abstractmethod
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch"]: ...

    def compute_validation(self) -> ValidationOutput:
        model_was_training = self.model.training
        self.model.eval()

        predictions: list[Float[Tensor, "batch"]] = []
        ground_truths: list[Float[Tensor, "batch"]] = []
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with (
            torch.no_grad(),
            device_context,
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16),
        ):
            for tokens, _target, ground_truth in tqdm(
                self.val_dl,
                desc=f"validation epoch {self.current_epoch}",
            ):
                tokens: Int[Tensor, "batch seq"] = tokens.to(
                    device=self.device,
                    dtype=torch.long,
                )
                ground_truth: Float[Tensor, "batch"] = ground_truth.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                prediction = self.compute_last_step_projections(context=tokens)
                predictions.append(prediction.float().cpu())
                ground_truths.append(ground_truth.float().cpu())

        if model_was_training:
            self.model.train()

        return ValidationOutput(
            preds=torch.cat(predictions),
            ground_truths=torch.cat(ground_truths),
        )

    def serialize_at_end_of_epoch(self, *, validation: ValidationOutput) -> None:
        validation.save_to(self.config.study_folder / str(self.current_epoch))

        train_corr = float(self.train_corr_counter.get_stats().squeeze(0).cpu())
        val_corr = float(validation.compute_corr().cpu())
        metrics_path = self.config.study_folder / "metrics.parquet"
        new_row = pl.DataFrame(
            {
                "epoch": [self.current_epoch],
                "train_corr": [train_corr],
                "val_corr": [val_corr],
            }
        )
        if metrics_path.exists():
            metrics = pl.read_parquet(metrics_path).filter(
                pl.col("epoch") != self.current_epoch
            )
            new_row = pl.concat([metrics, new_row]).sort("epoch")
        new_row.write_parquet(metrics_path)

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.clip_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


__all__ = [
    "BagOfWordsStudyBaseState",
    "ValidationOutput",
]
