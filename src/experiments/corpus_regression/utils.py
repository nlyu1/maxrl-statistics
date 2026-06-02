from __future__ import annotations

import numpy as np
import polars as pl


def dim_averaged_metrics_from_parquet(metrics: pl.DataFrame, *, split: str) -> pl.DataFrame:
    """Per-dim correlation and R^2 averaged across the embedding dimensions.

    Reads the per-dim sufficient-stat list-columns written by
    `CorpusRegressionStudyBaseState._serialize_val` (`{split}_target_xx`,
    `_xy`, `_yy`, each `List[Float, length=D]`) and returns a frame with
    a step/epoch column, `{split}_avg_corr`, `{split}_avg_rsq`. Each scalar
    column is the mean across the D output dimensions of the per-dim derived
    metric.

    Supports both new-format (step column) and old-format (epoch column).
    """
    xx = np.array(metrics[f"{split}_target_xx"].to_list())
    xy = np.array(metrics[f"{split}_target_xy"].to_list())
    yy = np.array(metrics[f"{split}_target_yy"].to_list())
    x_col = "step" if "step" in metrics.columns else "epoch"
    return pl.DataFrame({
        x_col: metrics[x_col],
        f"{split}_avg_corr": (xy / np.sqrt(xx * yy)).mean(axis=1),
        f"{split}_avg_rsq": (1 - (xx - 2 * xy + yy) / yy).mean(axis=1),
    })
