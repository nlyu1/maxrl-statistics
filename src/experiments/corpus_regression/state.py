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
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from src.data.corpus_regression import CorpusRegressionDataset
from src.experiments.corpus_regression.config import CorpusRegressionStudyBaseConfig
from src.metrics import RegressionStatCounter


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class TrainStepOutput:
    """Per-step metrics returned by each algorithm's train_step.

    `xx`, `xy`, `yy`, `pred_sum`, `target_sum`, `n` are per-batch sufficient
    stats over the deterministic mean prediction f_theta(x) vs. target y, in
    float32 (matching the precision used to compute `mse`). Downstream
    `run_training` (i) derives a scalar per-step `corr`, `pred_var`, and
    `target_var` for `train_metrics.parquet`, and (ii) sums them across the
    val window into `train_target_{xx,xy,yy,pred_sum,target_sum,n}` written
    alongside `val_target_*` in `val_metrics.parquet`. The latter is
    therefore a moving-window aggregate over training batches drawn from
    changing model snapshots — *not* a held-out train-set evaluation."""

    loss: float  # algorithm-native loss (MSE for SL, policy-gradient for RL)
    mse: float  # MSE(f_theta(x), target) dim-averaged — deterministic mean, no rollout noise
    xx: Float[Tensor, "D"]  # sum_b f_theta(x_b) ** 2 (per-dim, float32)
    xy: Float[Tensor, "D"]  # sum_b f_theta(x_b) * y_b
    yy: Float[Tensor, "D"]  # sum_b y_b ** 2
    pred_sum: Float[Tensor, "D"]  # sum_b f_theta(x_b)
    target_sum: Float[Tensor, "D"]  # sum_b y_b
    n: int  # batch size B (number of (x, y) pairs summed)


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
    per-dim corr / R² / MSE / Var via list-elementwise expressions and averages
    across dims with `.list.mean()`."""
    return {
        f"{prefix}_xx": counter.xx.float().cpu().tolist(),
        f"{prefix}_xy": counter.xy.float().cpu().tolist(),
        f"{prefix}_yy": counter.yy.float().cpu().tolist(),
        f"{prefix}_pred_sum": counter.x_sum.float().cpu().tolist(),
        f"{prefix}_target_sum": counter.y_sum.float().cpu().tolist(),
        f"{prefix}_n": float(counter.n.cpu()),
    }


@torch.no_grad()
def _batch_sufficient_stats(
    *,
    prediction: Float[Tensor, "B D"],
    target: Float[Tensor, "B D"],
) -> tuple[
    Float[Tensor, "D"],
    Float[Tensor, "D"],
    Float[Tensor, "D"],
    Float[Tensor, "D"],
    Float[Tensor, "D"],
    int,
]:
    """Per-dim sufficient stats for one training batch, in float32. Mirrors
    `RegressionStatCounter.tick` but stays inline so we can return them via
    `TrainStepOutput` without allocating a counter per step. Returns
    `(xx, xy, yy, pred_sum, target_sum, n)` so downstream code can recover
    Var[pred] and Var[target] from per-step or per-window aggregates."""
    x = prediction.float()
    y = target.float()
    xx = (x * x).sum(dim=0)
    xy = (x * y).sum(dim=0)
    yy = (y * y).sum(dim=0)
    pred_sum = x.sum(dim=0)
    target_sum = y.sum(dim=0)
    return xx, xy, yy, pred_sum, target_sum, x.shape[0]


def _scalar_corr_from_stats(
    *,
    xx: Float[Tensor, "D"],
    xy: Float[Tensor, "D"],
    yy: Float[Tensor, "D"],
) -> float:
    """Dim-averaged Pearson corr from sufficient stats. Same convention as
    `analysis._decode_dim_averaged` (per-dim corr then mean over D)."""
    eps = torch.finfo(xx.dtype).eps
    per_dim_corr = xy / (xx * yy).sqrt().clamp_min(eps)
    return float(per_dim_corr.mean().item())


def _scalar_var_from_stats(
    *,
    sq_sum: Float[Tensor, "D"],
    val_sum: Float[Tensor, "D"],
    n: int,
) -> float:
    """Dim-averaged variance from sufficient stats: `sq_sum / n − (val_sum/n)²`,
    averaged across the D dims. Used to populate per-step / per-window
    `pred_var` and `target_var` columns in `train_metrics.parquet`."""
    if n <= 0:
        return float("nan")
    n_t = torch.as_tensor(float(n), dtype=sq_sum.dtype, device=sq_sum.device)
    mean = val_sum / n_t
    per_dim_var = sq_sum / n_t - mean * mean
    return float(per_dim_var.mean().item())


def _per_dim_stats_from_tensors(
    *,
    xx: Float[Tensor, "D"],
    xy: Float[Tensor, "D"],
    yy: Float[Tensor, "D"],
    pred_sum: Float[Tensor, "D"],
    target_sum: Float[Tensor, "D"],
    n: int,
    prefix: str,
) -> dict[str, object]:
    """Same shape as `_per_dim_stats` but accepts raw aggregated tensors
    instead of a `RegressionStatCounter`. Used for the train-window aggregate
    written into `val_metrics.parquet` next to `val_target_*`."""
    return {
        f"{prefix}_xx": xx.float().cpu().tolist(),
        f"{prefix}_xy": xy.float().cpu().tolist(),
        f"{prefix}_yy": yy.float().cpu().tolist(),
        f"{prefix}_pred_sum": pred_sum.float().cpu().tolist(),
        f"{prefix}_target_sum": target_sum.float().cpu().tolist(),
        f"{prefix}_n": float(n),
    }


def _align_val_schema(
    existing: pl.DataFrame, *, target_columns: list[str],
) -> pl.DataFrame:
    """Backfill any missing columns from `target_columns` onto `existing`
    with null values of the right dtype, then re-order to match. Used when a
    resumed run encounters a `val_metrics.parquet` written before the
    `train_target_*`, `*_pred_sum`, or `*_target_sum` columns were added."""
    if existing.columns == target_columns:
        return existing
    # List dtype borrowed from val_target_xx so the polars cast doesn't trip
    # on dtype mismatch when the file is later read.
    list_dtype = existing.schema.get("val_target_xx", pl.List(pl.Float64))
    additions: list[pl.Expr] = []
    for col in target_columns:
        if col in existing.columns:
            continue
        if col.endswith("_n"):
            additions.append(pl.lit(None, dtype=pl.Float64).alias(col))
        else:
            additions.append(pl.lit(None, dtype=list_dtype).alias(col))
    return existing.with_columns(additions).select(target_columns)


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class CorpusRegressionStudyBaseState(ABC):
    config: CorpusRegressionStudyBaseConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    dataset: CorpusRegressionDataset
    train_dl: DataLoader
    val_dl: DataLoader
    device: torch.device
    # `None` for `lr_schedule == "flat"` — no scheduler is created and
    # `step_and_zero_grad` skips the scheduler.step() call. Otherwise this
    # is a `LambdaLR` driving the lockstep warmup+cosine factor across all
    # optimizer param groups.
    scheduler: LRScheduler | None = None

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
        batch: tuple[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch D"],
            Int[Tensor, "batch"],
        ],
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
            for tokens, target, _lookahead_token_ids in tqdm(
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
        """Append buffered per-step train metrics to train_metrics.parquet.

        Buffer entries have keys `step, loss, mse, corr, pred_var, target_var,
        lr`. The `lr` column captures the realized backbone-Muon group LR at
        the time of each step (param_groups[0]); under the lockstep schedule
        every group's LR is `base_lr * f(t)` for the same `f(t)`, so a single
        column suffices to verify schedule shape. Resumed runs whose existing
        parquet was written by an older code version (missing some columns)
        get backfilled with nulls so concat schemas line up."""
        if not buffer:
            return
        target_schema = {
            "step": pl.Int64,
            "loss": pl.Float64,
            "mse": pl.Float64,
            "corr": pl.Float64,
            "pred_var": pl.Float64,
            "target_var": pl.Float64,
            "lr": pl.Float64,
        }
        new_rows = pl.DataFrame(buffer, schema=target_schema)
        metrics_path = self.config.study_folder / "train_metrics.parquet"
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            for col, dtype in target_schema.items():
                if col not in existing.columns:
                    existing = existing.with_columns(
                        pl.lit(None, dtype=dtype).alias(col)
                    )
            existing = existing.select(list(target_schema.keys()))
            min_new_step = new_rows["step"].min()
            existing = existing.filter(pl.col("step") < min_new_step)
            new_rows = pl.concat([existing, new_rows]).sort("step")
        new_rows.write_parquet(metrics_path)

    def _serialize_val(
        self,
        *,
        step: int,
        validation: CorpusRegressionValidationOutput,
        train_window_xx: Float[Tensor, "D"],
        train_window_xy: Float[Tensor, "D"],
        train_window_yy: Float[Tensor, "D"],
        train_window_pred_sum: Float[Tensor, "D"],
        train_window_target_sum: Float[Tensor, "D"],
        train_window_n: int,
    ) -> None:
        """Append one row to val_metrics.parquet (and save predictions).

        Writes both the held-out validation sufficient stats (`val_target_*`)
        and a *training-window aggregate* (`train_target_*`) summed over all
        training batches since the last val step. The latter is **not** a
        held-out train-set evaluation — it's a moving aggregate over batches
        from changing model snapshots within the most recent val window. Used
        downstream as a cheap proxy for train-set fit; expect noisier and
        slightly more pessimistic numbers than a held-out pass would give.

        Resumed runs whose existing parquet predates the new schema (no
        `train_target_*`, `*_pred_sum`, or `*_target_sum` columns) get
        backfilled with nulls of the right list dtype so concat schemas line
        up."""
        validation.save_to(self.config.study_folder / f"val-step-{step}")

        val_counter = validation.compute_counter()
        sufficient_stats: dict[str, object] = {}
        sufficient_stats.update(_per_dim_stats(val_counter, prefix="val_target"))
        sufficient_stats.update(
            _per_dim_stats_from_tensors(
                xx=train_window_xx,
                xy=train_window_xy,
                yy=train_window_yy,
                pred_sum=train_window_pred_sum,
                target_sum=train_window_target_sum,
                n=train_window_n,
                prefix="train_target",
            )
        )

        metrics_path = self.config.study_folder / "val_metrics.parquet"
        new_row = pl.DataFrame({
            "step": [step],
            **{key: [value] for key, value in sufficient_stats.items()},
        })
        if metrics_path.exists():
            existing = pl.read_parquet(metrics_path)
            existing = _align_val_schema(existing, target_columns=new_row.columns)
            existing = existing.filter(pl.col("step") != step)
            new_row = pl.concat([existing, new_row]).sort("step")
        new_row.write_parquet(metrics_path)

    def run_training(self) -> None:
        """Flat step-based training loop. Validates every val_every_n_steps.

        Per gradient step we record `(loss, mse, corr, pred_var, target_var)`
        into `train_buffer` for `train_metrics.parquet`, and add per-batch
        sufficient stats into `train_window_*` for `val_metrics.parquet`.
        Both buffers are reset after each val flush so each val row's
        `train_target_*` summarises only the most recent val window."""
        self.model.train()
        batch_iter = _cycle_dataloader(self.train_dl)
        train_buffer: list[dict[str, float]] = []

        # Initialised lazily on the first batch so we can match D from the
        # actual prediction shape without poking the model config.
        train_window_xx: Tensor | None = None
        train_window_xy: Tensor | None = None
        train_window_yy: Tensor | None = None
        train_window_pred_sum: Tensor | None = None
        train_window_target_sum: Tensor | None = None
        train_window_n: int = 0

        pbar = tqdm(range(self.config.train_steps), desc="training")
        for step in pbar:
            output = self.train_step(batch=next(batch_iter))
            corr = _scalar_corr_from_stats(xx=output.xx, xy=output.xy, yy=output.yy)
            pred_var = _scalar_var_from_stats(
                sq_sum=output.xx, val_sum=output.pred_sum, n=output.n,
            )
            target_var = _scalar_var_from_stats(
                sq_sum=output.yy, val_sum=output.target_sum, n=output.n,
            )
            # Realized backbone-Muon LR at the step we just took.
            # Lockstep schedule ⇒ a single column captures schedule shape.
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            train_buffer.append({
                "step": step,
                "loss": output.loss,
                "mse": output.mse,
                "corr": corr,
                "pred_var": pred_var,
                "target_var": target_var,
                "lr": current_lr,
            })
            pbar.set_postfix(
                loss=f"{output.loss:.4f}",
                mse=f"{output.mse:.4f}",
                corr=f"{corr:.4f}",
                pred_var=f"{pred_var:.4f}",
                lr=f"{current_lr:.2e}",
            )

            if train_window_xx is None:
                train_window_xx = torch.zeros_like(output.xx)
                train_window_xy = torch.zeros_like(output.xy)
                train_window_yy = torch.zeros_like(output.yy)
                train_window_pred_sum = torch.zeros_like(output.pred_sum)
                train_window_target_sum = torch.zeros_like(output.target_sum)
            train_window_xx += output.xx
            train_window_xy += output.xy
            train_window_yy += output.yy
            train_window_pred_sum += output.pred_sum
            train_window_target_sum += output.target_sum
            train_window_n += output.n

            is_val_step = (
                (step + 1) % self.config.val_every_n_steps == 0
                or step == self.config.train_steps - 1
            )
            if is_val_step:
                validation = self.compute_validation()
                self._flush_train_metrics(train_buffer)
                assert train_window_xx is not None
                assert train_window_pred_sum is not None
                assert train_window_target_sum is not None
                self._serialize_val(
                    step=step,
                    validation=validation,
                    train_window_xx=train_window_xx,
                    train_window_xy=train_window_xy,
                    train_window_yy=train_window_yy,
                    train_window_pred_sum=train_window_pred_sum,
                    train_window_target_sum=train_window_target_sum,
                    train_window_n=train_window_n,
                )
                train_buffer = []
                train_window_xx.zero_()
                train_window_xy.zero_()
                train_window_yy.zero_()
                train_window_pred_sum.zero_()
                train_window_target_sum.zero_()
                train_window_n = 0
                self.model.train()

    def step_and_zero_grad(self) -> None:
        clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optimizer.clip_grad_norm,
        )
        self.optimizer.step()
        # Scheduler advances after the optimizer step so the LR used by the
        # step we just took matches `lr_lambda(last_epoch)` at the moment of
        # the step. With `lr_schedule == "flat"` the scheduler is `None` and
        # this is a no-op — preserves byte-for-byte today's behaviour.
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
