from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
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


@dataclass(kw_only=True)
class TrainStepOutput:
    """Per-step metrics returned by each algorithm's train_step."""

    loss: float  # algorithm-native loss (MSE for SL, policy-gradient for RL)
    mse: float  # MSE(f_theta(x), target) dim-averaged — deterministic mean, no rollout noise


def _cycle_dataloader(dl: DataLoader) -> Iterator:
    """Re-iterates the dataloader indefinitely (re-shuffles each pass)."""
    while True:
        yield from dl


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

    @abstractmethod
    def compute_last_step_projections(
        self,
        *,
        context: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch D"]: ...

    @abstractmethod
    def train_step(
        self,
        *,
        batch: tuple[Int[Tensor, "batch seq"], Float[Tensor, "batch D"]],
    ) -> TrainStepOutput: ...

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
                desc="validation",
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

    def _flush_train_metrics(self, buffer: list[dict[str, float]]) -> None:
        """Append buffered per-step train metrics to train_metrics.parquet."""
        if not buffer:
            return
        new_rows = pl.DataFrame(buffer, schema={"step": pl.Int64, "loss": pl.Float64, "mse": pl.Float64})
        metrics_path = self.config.study_folder / "train_metrics.parquet"
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            min_new_step = new_rows["step"].min()
            existing = existing.filter(pl.col("step") < min_new_step)
            new_rows = pl.concat([existing, new_rows]).sort("step")
        new_rows.write_parquet(metrics_path)

    def _serialize_val(
        self,
        *,
        step: int,
        validation: CorpusRegressionValidationOutput,
    ) -> None:
        """Append one validation row to val_metrics.parquet and save predictions."""
        validation.save_to(self.config.study_folder / f"val-step-{step}")

        val_counter = validation.compute_counter()
        sufficient_stats: dict[str, object] = {}
        sufficient_stats.update(_per_dim_stats(val_counter, prefix="val_target"))

        metrics_path = self.config.study_folder / "val_metrics.parquet"
        new_row = pl.DataFrame({
            "step": [step],
            **{key: [value] for key, value in sufficient_stats.items()},
        })
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            if existing.columns == new_row.columns:
                existing = existing.filter(pl.col("step") != step)
                new_row = pl.concat([existing, new_row]).sort("step")
        new_row.write_parquet(metrics_path)

    def run_training(self) -> None:
        """Flat step-based training loop. Validates every val_every_n_steps."""
        self.model.train()
        batch_iter = _cycle_dataloader(self.train_dl)
        train_buffer: list[dict[str, float]] = []

        pbar = tqdm(range(self.config.train_steps), desc="training")
        for step in pbar:
            output = self.train_step(batch=next(batch_iter))
            train_buffer.append({"step": step, "loss": output.loss, "mse": output.mse})
            pbar.set_postfix(loss=f"{output.loss:.4f}", mse=f"{output.mse:.4f}")

            is_val_step = (
                (step + 1) % self.config.val_every_n_steps == 0
                or step == self.config.train_steps - 1
            )
            if is_val_step:
                validation = self.compute_validation()
                self._flush_train_metrics(train_buffer)
                self._serialize_val(step=step, validation=validation)
                train_buffer = []
                self.model.train()

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.clip_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
