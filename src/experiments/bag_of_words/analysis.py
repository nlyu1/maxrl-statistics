import json
from pathlib import Path
from typing import Self

import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from src.config.base import BaseConfig
from src.data.bag_of_words import (
    candidate_corrs,
    candidate_rollout_steps,
    candidate_seeds,
)


# Canonical dataset-folder defaults shared by all single_run scripts.
# Must match BagOfWordsStudyBaseConfig.canonical_kwargs.
_CANONICAL_NUM_WORDS = 7
_CANONICAL_PROMPT_LENGTH = 128
_CANONICAL_WORD_DECAY_POWER = 1.0
_CANONICAL_AUX_WORDS_RATIO = 0.5


def _dataset_folder_name(*, corr: float) -> str:
    return (
        f"{_CANONICAL_NUM_WORDS}-words_corr-{corr}"
        f"_len-{_CANONICAL_PROMPT_LENGTH}"
        f"_pow-{_CANONICAL_WORD_DECAY_POWER}"
        f"_ar-{_CANONICAL_AUX_WORDS_RATIO}"
    )


def _seed_folder_name(seed: int) -> str:
    return f"seed-{seed}"


def _maxrl_baseline_folder(*, subtract_baseline: bool) -> str:
    return "subtract-baseline" if subtract_baseline else "no-subtract-baseline"


class BagOfWordsAnalysisConfig(BaseConfig):
    """
    Grouped view over completed/started study folders. Each `studies` key is a
    group name (e.g. `"corr=0.220 r=64"`); the value is the list of per-seed
    study folders backing it. Construct via the `from_*_sweep` factories, which
    consume the canonical grids in `src.data.bag_of_words`.
    """

    studies: dict[str, list[Path]]
    study_seeds: dict[str, list[int]]
    study_corrs: dict[str, float]

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
    def from_grouped(
        cls, grouped: dict[str, list[tuple[int, Path]]]
    ) -> Self | None:
        """
        Build an analysis config from a group-name -> [(seed, path), ...] mapping.
        Drops paths without `has_study_started`; drops groups that end up empty.
        Returns None if nothing survives (so notebook loops can skip cleanly).
        """
        studies: dict[str, list[Path]] = {}
        study_seeds: dict[str, list[int]] = {}
        study_corrs: dict[str, float] = {}
        for name, pairs in grouped.items():
            kept = [(s, p) for (s, p) in pairs if cls.has_study_started(p)]
            if not kept:
                continue
            studies[name] = [p for (_, p) in kept]
            study_seeds[name] = [s for (s, _) in kept]
            first_path = kept[0][1]
            study_corrs[name] = float(
                json.loads((first_path / "config.json").read_text())["data"]["corr"]
            )
        if not studies:
            return None
        return cls(
            studies=studies,
            study_seeds=study_seeds,
            study_corrs=study_corrs,
        )

    @classmethod
    def from_sl_sweep(cls, *, study_base: Path) -> Self | None:
        """Discover SL runs under `study_base/seed-{S}/{dataset_name}/`."""
        if not study_base.exists():
            return None
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for corr in candidate_corrs:
            name = f"corr={corr:.3f}"
            pairs: list[tuple[int, Path]] = []
            for seed in candidate_seeds:
                path = (
                    study_base
                    / _seed_folder_name(seed)
                    / _dataset_folder_name(corr=corr)
                )
                pairs.append((seed, path))
            grouped[name] = pairs
        return cls.from_grouped(grouped)

    @classmethod
    def from_grpo_sweep(cls, *, study_base: Path) -> Self | None:
        """Discover GRPO runs under `study_base/seed-{S}/rollouts-{N}/{dataset_name}/`."""
        if not study_base.exists():
            return None
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for corr in candidate_corrs:
            for r in candidate_rollout_steps:
                name = f"corr={corr:.3f} r={r}"
                pairs: list[tuple[int, Path]] = []
                for seed in candidate_seeds:
                    path = (
                        study_base
                        / _seed_folder_name(seed)
                        / f"rollouts-{r}"
                        / _dataset_folder_name(corr=corr)
                    )
                    pairs.append((seed, path))
                grouped[name] = pairs
        return cls.from_grouped(grouped)

    @classmethod
    def from_maxrl_sweep(
        cls, *, study_base: Path, subtract_baseline: bool
    ) -> Self | None:
        """Discover MaxRL runs under
        `study_base/seed-{S}/rollouts-{N}/{baseline_mode}/{dataset_name}/`.
        The baseline mode is fixed per factory call (and hence absent from
        group keys — surface it in the figure title instead)."""
        if not study_base.exists():
            return None
        baseline_folder = _maxrl_baseline_folder(
            subtract_baseline=subtract_baseline
        )
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for corr in candidate_corrs:
            for r in candidate_rollout_steps:
                name = f"corr={corr:.3f} r={r}"
                pairs: list[tuple[int, Path]] = []
                for seed in candidate_seeds:
                    path = (
                        study_base
                        / _seed_folder_name(seed)
                        / f"rollouts-{r}"
                        / baseline_folder
                        / _dataset_folder_name(corr=corr)
                    )
                    pairs.append((seed, path))
                grouped[name] = pairs
        return cls.from_grouped(grouped)

    def get_metric_dataframe(self) -> pl.DataFrame:
        """Vertical concat of per-(study, seed) metrics.parquet files."""
        frames: list[pl.DataFrame] = []
        for name, paths in self.studies.items():
            seeds = self.study_seeds[name]
            for seed, path in zip(seeds, paths, strict=True):
                frames.append(
                    pl.read_parquet(path / "metrics.parquet").with_columns(
                        pl.lit(name).alias("study"),
                        pl.lit(seed).alias("seed"),
                    )
                )
        stat_cols = [c for c in frames[0].columns if c not in ("study", "seed")]
        return pl.concat(frames).select(["study", "seed", *stat_cols])

    def _aggregate_by_epoch(
        self, *, df: pl.DataFrame, x_name: str, y_name: str
    ) -> pl.DataFrame:
        """Collapse across seed: (study, epoch) -> mean_x, mean_y, min_y, max_y, n_seeds."""
        return (
            df
            .group_by(["study", "epoch"])
            .agg(
                pl.col(x_name).mean().alias("mean_x"),
                pl.col(y_name).mean().alias("mean_y"),
                pl.col(y_name).min().alias("min_y"),
                pl.col(y_name).max().alias("max_y"),
                pl.len().alias("n_seeds"),
            )
            .sort(["study", "epoch"])
        )

    def _add_per_epoch_traces(
        self,
        *,
        fig: go.Figure,
        agg: pl.DataFrame,
        study_colors: dict[str, str],
        x_name: str,
        y_name: str,
        col: int,
        show_seed_bar: bool,
    ) -> None:
        for study in self.studies:
            sub = agg.filter(pl.col("study") == study).sort("epoch")
            if sub.is_empty():
                continue
            ds_corr = self.study_corrs[study]
            xs = sub["mean_x"].to_list()
            ys = sub["mean_y"].to_list()
            mins = sub["min_y"].to_list()
            maxs = sub["max_y"].to_list()
            n_seeds = sub["n_seeds"].to_list()
            epochs = sub["epoch"].to_list()
            customdata = [
                [ds_corr, e, n] for (e, n) in zip(epochs, n_seeds, strict=True)
            ]
            error_kwargs = {}
            if show_seed_bar:
                error_kwargs = dict(
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[mx - my for mx, my in zip(maxs, ys, strict=True)],
                        arrayminus=[my - mn for my, mn in zip(ys, mins, strict=True)],
                        thickness=1,
                        width=3,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
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
                        "epoch: %{customdata[1]}<br>"
                        "n_seeds: %{customdata[2]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                    **error_kwargs,
                ),
                row=1,
                col=col,
            )

    def _add_agg_trace(
        self,
        *,
        fig: go.Figure,
        agg: pl.DataFrame,
        x_name: str,
        y_name: str,
        col: int,
        argmax: bool,
        show_seed_bar: bool,
    ) -> None:
        # Per study: pick the epoch that argmax/argmin mean_y on the seed-averaged curve.
        rows: list[tuple[float, float, float, float, float, int, int, str]] = []
        for study in self.studies:
            sub = agg.filter(pl.col("study") == study)
            if sub.is_empty():
                continue
            best = sub.sort("mean_y", descending=argmax).head(1)
            rows.append((
                best["mean_x"].item(),
                best["mean_y"].item(),
                best["min_y"].item(),
                best["max_y"].item(),
                self.study_corrs[study],
                best["epoch"].item(),
                best["n_seeds"].item(),
                study,
            ))
        rows.sort(key=lambda r: r[4])  # sort by dataset corr for monotone legend
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        mins = [r[2] for r in rows]
        maxs = [r[3] for r in rows]
        customdata = [(r[4], r[5], r[6], r[7]) for r in rows]
        error_kwargs = {}
        if show_seed_bar:
            error_kwargs = dict(
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[mx - my for mx, my in zip(maxs, ys, strict=True)],
                    arrayminus=[my - mn for my, mn in zip(ys, mins, strict=True)],
                    thickness=1,
                    width=3,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{x_name} vs {y_name}",
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    f"{x_name}: %{{x}}<br>"
                    f"{y_name}: %{{y}}<br>"
                    "dataset corr: %{customdata[0]}<br>"
                    "epoch: %{customdata[1]}<br>"
                    "n_seeds: %{customdata[2]}"
                    "<extra>%{customdata[3]}</extra>"
                ),
                **error_kwargs,
            ),
            row=1,
            col=col,
        )

    def xy_plots(
        self,
        axes: list[tuple[pl.Expr, pl.Expr, bool | None]],
        *,
        title: str | None = None,
        show_seed_bar: bool = False,
    ) -> go.Figure:
        """
        Horizontally laid-out xy subplots, one per (x_expr, y_expr, agg) triple.

        Each expression must carry its display name via `.alias(...)` — the
        helpers in this module (`corr_expr`, `mse_expr`, `rsq_expr`, `beta_expr`)
        already do this. Metrics are first evaluated per (study, seed, epoch)
        and then averaged across seeds per (study, epoch). Hover reports the
        number of seeds averaged at that point. `agg` controls epoch aggregation:

        agg=None:  per-epoch traces, one line per group.
        agg=True:  one point per group at the epoch maximizing the seed-averaged y.
        agg=False: one point per group at the epoch minimizing the seed-averaged y.

        When `show_seed_bar=True`, each point carries an asymmetric min/max-of-seeds
        error bar.
        """
        df = self.get_metric_dataframe()
        palette = qualitative.Plotly
        study_colors = {
            study: palette[i % len(palette)] for i, study in enumerate(self.studies)
        }

        fig = make_subplots(rows=1, cols=len(axes))
        for col, (x_expr, y_expr, agg) in enumerate(axes, start=1):
            x_name = x_expr.meta.output_name()
            y_name = y_expr.meta.output_name()
            evaluated = df.with_columns(x_expr, y_expr)
            agg_df = self._aggregate_by_epoch(
                df=evaluated, x_name=x_name, y_name=y_name
            )
            if agg is None:
                self._add_per_epoch_traces(
                    fig=fig,
                    agg=agg_df,
                    study_colors=study_colors,
                    x_name=x_name,
                    y_name=y_name,
                    col=col,
                    show_seed_bar=show_seed_bar,
                )
            else:
                self._add_agg_trace(
                    fig=fig,
                    agg=agg_df,
                    x_name=x_name,
                    y_name=y_name,
                    col=col,
                    argmax=agg,
                    show_seed_bar=show_seed_bar,
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
