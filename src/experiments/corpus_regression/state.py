from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

import polars as pl
import torch
from jaxtyping import Float, Int
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from src.data.corpus_regression import CorpusRegressionDataset
from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.metrics import RegressionStatCounter


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionValidationOutput:
    model_preds: Float[Tensor, "n D"]
    target: Float[Tensor, "n D"]

    def compute_counter(self) -> RegressionStatCounter:
        D = self.model_preds.shape[1]
        counter = RegressionStatCounter.initialize(
            dim=D, device=self.model_preds.device
        )
        counter.tick(x=self.model_preds, y=self.target)
        return counter

    def save_to(self, folder: Path) -> None:
        """Per-row predictions and targets as list-columns of length D."""
        folder.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "model_preds": self.model_preds.float().cpu().tolist(),
            "target": self.target.float().cpu().tolist(),
        }).write_parquet(folder / "validation.parquet")


def _per_dim_stats(
    counter: RegressionStatCounter, *, prefix: str
) -> dict[str, object]:
    """Per-dim sufficient stats serialized as polars list-columns (length D),
    plus a scalar `n` (shared across dims). Downstream analysis recovers
    per-dim corr / R² / MSE via list-elementwise expressions and averages
    across dims with `.list.mean()`."""
    return {
        f"{prefix}_xx": counter.xx.float().cpu().tolist(),
        f"{prefix}_xy": counter.xy.float().cpu().tolist(),
        f"{prefix}_yy": counter.yy.float().cpu().tolist(),
        f"{prefix}_n": float(counter.n.cpu()),
    }


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionStudyBaseState(ABC):
    config: CorpusRegressionStudyBaseConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataset: CorpusRegressionDataset
    train_dl: DataLoader
    val_dl: DataLoader
    device: torch.device

    train_target_counter: RegressionStatCounter
    current_epoch: int = 0

    @abstractmethod
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]: ...

    def compute_validation(self) -> CorpusRegressionValidationOutput:
        model_was_training = self.model.training
        self.model.eval()

        predictions: list[Float[Tensor, "batch D"]] = []
        targets: list[Float[Tensor, "batch D"]] = []
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
            for tokens, target in tqdm(
                self.val_dl,
                desc=f"validation epoch {self.current_epoch}",
            ):
                tokens: Int[Tensor, "batch seq"] = tokens.to(
                    device=self.device,
                    dtype=torch.long,
                )
                target: Float[Tensor, "batch D"] = target.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                prediction = self.compute_last_step_projections(context=tokens)
                predictions.append(prediction.float().cpu())
                targets.append(target.float().cpu())

        if model_was_training:
            self.model.train()

        return CorpusRegressionValidationOutput(
            model_preds=torch.cat(predictions),
            target=torch.cat(targets),
        )

    def serialize_at_end_of_epoch(
        self,
        *,
        validation: CorpusRegressionValidationOutput,
    ) -> None:
        """Per-epoch sufficient stats per output dimension, written as polars
        list-columns. Downstream analysis derives per-dim corr / R² / MSE and
        averages across dims via `.list.mean()`. If this schema changes, update
        analysis.py."""
        validation.save_to(self.config.study_folder / str(self.current_epoch))

        val_counter = validation.compute_counter()
        sufficient_stats: dict[str, object] = {}
        sufficient_stats.update(
            _per_dim_stats(self.train_target_counter, prefix="train_target")
        )
        sufficient_stats.update(_per_dim_stats(val_counter, prefix="val_target"))

        metrics_path = self.config.study_folder / "metrics.parquet"
        new_row = pl.DataFrame({
            "epoch": [self.current_epoch],
            **{key: [value] for key, value in sufficient_stats.items()},
        })
        if metrics_path.exists():
            metrics = pl.read_parquet(metrics_path)
            if metrics.columns == new_row.columns:
                metrics = metrics.filter(pl.col("epoch") != self.current_epoch)
                new_row = pl.concat([metrics, new_row]).sort("epoch")
        new_row.write_parquet(metrics_path)

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.clip_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
