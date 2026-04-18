from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import field
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

from src.data.parquet import TokenizedParquetDataset
from src.experiments.bag_of_words.config import BagOfWordsStudyBaseConfig
from src.metrics import RegressionStatCounter


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class ValidationOutput:
    model_preds: Float[Tensor, "n"]
    ground_truth: Float[Tensor, "n"]
    target: Float[Tensor, "n"]

    def _compute_counter(self, *, y: Float[Tensor, "n"]) -> RegressionStatCounter:
        counter = RegressionStatCounter.initialize(
            dim=1, device=self.model_preds.device
        )
        counter.tick(x=self.model_preds[:, None], y=y[:, None])
        return counter

    def compute_counters(self) -> dict[str, RegressionStatCounter]:
        return {
            "target": self._compute_counter(y=self.target),
            "ground_truth": self._compute_counter(y=self.ground_truth),
        }

    def save_to(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "model_preds": self.model_preds.float().cpu().tolist(),
            "ground_truth": self.ground_truth.float().cpu().tolist(),
            "target": self.target.float().cpu().tolist(),
        }).write_parquet(folder / "validation.parquet")


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
    train_corr_target_counter: RegressionStatCounter = field(
        default_factory=lambda: RegressionStatCounter.initialize(dim=1)
    )
    train_corr_ground_truth_counter: RegressionStatCounter = field(
        default_factory=lambda: RegressionStatCounter.initialize(dim=1)
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
        targets: list[Float[Tensor, "batch"]] = []
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
            for tokens, target, ground_truth in tqdm(
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
                target: Float[Tensor, "batch"] = target.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                prediction = self.compute_last_step_projections(context=tokens)
                predictions.append(prediction.float().cpu())
                ground_truths.append(ground_truth.float().cpu())
                targets.append(target.float().cpu())

        if model_was_training:
            self.model.train()

        return ValidationOutput(
            model_preds=torch.cat(predictions),
            ground_truth=torch.cat(ground_truths),
            target=torch.cat(targets),
        )

    @staticmethod
    def _counter_sufficient_stats(
        counter: RegressionStatCounter, *, prefix: str
    ) -> dict[str, float]:
        return {
            f"{prefix}_xx": float(counter.xx.squeeze(0).cpu()),
            f"{prefix}_xy": float(counter.xy.squeeze(0).cpu()),
            f"{prefix}_yy": float(counter.yy.squeeze(0).cpu()),
            f"{prefix}_n": float(counter.n.cpu()),
        }

    def serialize_at_end_of_epoch(
        self,
        *,
        validation: ValidationOutput,
    ) -> None:
        """
        Writes per-epoch sufficient statistics (xx, xy, yy, n) for the four
        (split * y) combinations. Derived quantities (corr, mse, rsq, beta, ...)
        are computed downstream from these columns. If this schema changes,
        update analyze notebooks that consume `metrics.parquet` and their helpers.
        """
        validation.save_to(self.config.study_folder / str(self.current_epoch))

        val_counters = validation.compute_counters()
        counters: dict[str, RegressionStatCounter] = {
            "train_target": self.train_corr_target_counter,
            "train_ground_truth": self.train_corr_ground_truth_counter,
            "val_target": val_counters["target"],
            "val_ground_truth": val_counters["ground_truth"],
        }
        sufficient_stats: dict[str, float] = {}
        for prefix, counter in counters.items():
            sufficient_stats.update(
                self._counter_sufficient_stats(counter, prefix=prefix)
            )
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


__all__ = [
    "BagOfWordsStudyBaseState",
    "ValidationOutput",
]
