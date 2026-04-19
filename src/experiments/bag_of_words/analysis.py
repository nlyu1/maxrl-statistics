import json
from pathlib import Path
from typing import Self

import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from src.config.base import BaseConfig


class BagOfWordsAnalysisConfig(BaseConfig):
    """
    Dictionary of legend names to study artifact paths. Construct via
    from_studies(); only studies with config + metrics on disk are retained.
    """

    studies: dict[str, Path]

    @staticmethod
    def has_study_started(path: Path) -> bool:
        return (path / "metrics.parquet").exists() and (path / "config.json").exists()

    @staticmethod
    def is_study_complete(path: Path) -> bool:
        """True iff the study at *path* has completed all its training epochs."""
        if not BagOfWordsAnalysisConfig.has_study_started(path):
            return False
        train_epochs = json.loads((path / "config.json").read_text())["train_epochs"]
        max_epoch = (
            pl
            .read_parquet(path / "metrics.parquet")
            .select(pl.col("epoch").max())
            .item()
        )
        return max_epoch >= train_epochs - 1

    @classmethod
    def from_studies(cls, studies: dict[str, Path]) -> Self:
        started = {k: v for k, v in studies.items() if cls.has_study_started(v)}
        if not started:
            raise ValueError("No studies with metrics found")
        return cls(studies=started)

    def get_metric_dataframe(self) -> pl.DataFrame:
        """Vertical concat of per-study metrics.parquet with a `study` key column."""
        frames = [
            pl.read_parquet(path / "metrics.parquet").with_columns(
                pl.lit(key).alias("study")
            )
            for key, path in self.studies.items()
        ]
        stat_cols = [c for c in frames[0].columns if c != "study"]
        return pl.concat(frames).select(["study", *stat_cols])

    def _add_per_epoch_traces(
        self,
        *,
        fig: go.Figure,
        df: pl.DataFrame,
        study_corrs: dict[str, float],
        study_colors: dict[str, str],
        x_name: str,
        y_name: str,
        col: int,
    ) -> None:
        for study in self.studies:
            sub = df.filter(pl.col("study") == study).sort("epoch")
            ds_corr = study_corrs[study]
            customdata = [[ds_corr, e] for e in sub["epoch"].to_list()]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_name].to_list(),
                    y=sub[y_name].to_list(),
                    mode="lines+markers",
                    name=study,
                    legendgroup=study,
                    showlegend=(col == 1),
                    line=dict(color=study_colors[study]),
                    marker=dict(color=study_colors[study]),
                    customdata=customdata,
                    hovertemplate=(
                        f"{x_name}: %{{x}}<br>"
                        f"{y_name}: %{{y}}<br>"
                        "dataset corr: %{customdata[0]}<br>"
                        "epoch: %{customdata[1]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    def _add_agg_trace(
        self,
        *,
        fig: go.Figure,
        df: pl.DataFrame,
        study_corrs: dict[str, float],
        x_name: str,
        y_name: str,
        col: int,
        argmax: bool,
    ) -> None:
        rows = []
        for study in self.studies:
            sub = df.filter(pl.col("study") == study)
            best = sub.sort(y_name, descending=argmax).head(1)
            rows.append((
                best[x_name].item(),
                best[y_name].item(),
                study_corrs[study],
                best["epoch"].item(),
                study,
            ))
        rows.sort(key=lambda r: r[2])
        xs, ys, corrs, epochs, names = map(list, zip(*rows))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{x_name} vs {y_name}",
                showlegend=False,
                customdata=list(zip(corrs, epochs, names)),
                hovertemplate=(
                    f"{x_name}: %{{x}}<br>"
                    f"{y_name}: %{{y}}<br>"
                    "dataset corr: %{customdata[0]}<br>"
                    "epoch: %{customdata[1]}"
                    "<extra>%{customdata[2]}</extra>"
                ),
            ),
            row=1,
            col=col,
        )

    def xy_plots(
        self,
        axes: list[tuple[pl.Expr, pl.Expr, bool | None]],
        *,
        title: str | None = None,
    ) -> go.Figure:
        """
        Horizontally laid-out xy subplots, one per (x_expr, y_expr, agg) triple.

        Each expression must carry its display name via `.alias(...)` — the
        helpers in this module (`corr_expr`, `mse_expr`, `rsq_expr`, `beta_expr`)
        already do this. The per-panel `agg` flag controls epoch aggregation:

        agg=None:  per-epoch traces, one line per study.
        agg=True:  one point per study at the epoch maximizing y.
        agg=False: one point per study at the epoch minimizing y.
        """
        df = self.get_metric_dataframe()
        study_corrs = {
            s: float(json.loads((path / "config.json").read_text())["data"]["corr"])
            for s, path in self.studies.items()
        }
        palette = qualitative.Plotly
        study_colors = {
            study: palette[i % len(palette)] for i, study in enumerate(self.studies)
        }

        fig = make_subplots(rows=1, cols=len(axes))
        for col, (x_expr, y_expr, agg) in enumerate(axes, start=1):
            x_name = x_expr.meta.output_name()
            y_name = y_expr.meta.output_name()
            evaluated = df.with_columns(x_expr, y_expr)
            if agg is None:
                self._add_per_epoch_traces(
                    fig=fig,
                    df=evaluated,
                    study_corrs=study_corrs,
                    study_colors=study_colors,
                    x_name=x_name,
                    y_name=y_name,
                    col=col,
                )
            else:
                self._add_agg_trace(
                    fig=fig,
                    df=evaluated,
                    study_corrs=study_corrs,
                    x_name=x_name,
                    y_name=y_name,
                    col=col,
                    argmax=agg,
                )
            fig.update_xaxes(title_text=x_name, row=1, col=col)
            fig.update_yaxes(title_text=y_name, row=1, col=col)
        if title is not None:
            fig.update_layout(title=title)
        return fig


# ── Expression helpers ────────────────────────────────────────────────────────
# All four quantities are *uncentered* (no mean subtraction) because
# CorrelationCounter accumulates raw sums Σx², Σxy, Σy², n. Here x = model
# prediction and y ∈ {target, ground_truth}.


def corr_expr(*, split: str, y: str) -> pl.Expr:
    """Uncentered correlation (cosine similarity): Σxy / √(Σx² · Σy²)."""
    xy = pl.col(f"{split}_{y}_xy")
    xx = pl.col(f"{split}_{y}_xx")
    yy = pl.col(f"{split}_{y}_yy")
    return (xy / (xx * yy).sqrt()).alias(f"{split}_corr_{y}")


def mse_expr(*, split: str, y: str) -> pl.Expr:
    """Mean squared error: Σ(x − y)² / n."""
    xx = pl.col(f"{split}_{y}_xx")
    xy = pl.col(f"{split}_{y}_xy")
    yy = pl.col(f"{split}_{y}_yy")
    n = pl.col(f"{split}_{y}_n")
    return ((xx - 2 * xy + yy) / n).alias(f"{split}_mse_{y}")


def rsq_expr(*, split: str, y: str) -> pl.Expr:
    """Uncentered R²: 1 − Σ(x − y)² / Σy²."""
    xx = pl.col(f"{split}_{y}_xx")
    xy = pl.col(f"{split}_{y}_xy")
    yy = pl.col(f"{split}_{y}_yy")
    return (1.0 - (xx - 2 * xy + yy) / yy).alias(f"{split}_rsq_{y}")


def beta_expr(*, split: str, y: str) -> pl.Expr:
    """OLS slope through the origin (y on x): Σxy / Σx²."""
    xx = pl.col(f"{split}_{y}_xx")
    xy = pl.col(f"{split}_{y}_xy")
    return (xy / xx).alias(f"{split}_beta_{y}")
