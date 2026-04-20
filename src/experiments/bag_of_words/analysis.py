import json
import re
from pathlib import Path
from typing import Literal, Self

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

_ROLLOUTS_RE = re.compile(r" r=(\d+)")


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
    def from_grouped(cls, grouped: dict[str, list[tuple[int, Path]]]) -> Self | None:
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
        baseline_folder = _maxrl_baseline_folder(subtract_baseline=subtract_baseline)
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

    def describe(self, label: str) -> None:
        """Print a one-line summary: run count, group count, max seeds/group."""
        n_runs = sum(len(v) for v in self.studies.values())
        n_groups = len(self.studies)
        max_seeds = max(len(v) for v in self.studies.values())
        print(
            f"{label:<32s}  {n_runs:>4d} runs   {n_groups:>3d} groups   "
            f"up to {max_seeds} seeds/group"
        )

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

    def _rollouts_groups(self) -> dict[int, list[str]] | None:
        """Split studies by the ` r=(\\d+)` suffix. Returns None if any study
        lacks the suffix (e.g. SL sweeps)."""
        groups: dict[int, list[str]] = {}
        for name in self.studies:
            m = _ROLLOUTS_RE.search(name)
            if m is None:
                return None
            groups.setdefault(int(m.group(1)), []).append(name)
        return dict(sorted(groups.items()))

    def _aggregate_by_epoch(self, *, df: pl.DataFrame, y_name: str) -> pl.DataFrame:
        """Collapse across seed: (study, epoch) -> mean_y, min_y, max_y, n_seeds."""
        return (
            df
            .group_by(["study", "epoch"])
            .agg(
                pl.col(y_name).mean().alias("mean_y"),
                pl.col(y_name).min().alias("min_y"),
                pl.col(y_name).max().alias("max_y"),
                pl.len().alias("n_seeds"),
            )
            .sort(["study", "epoch"])
        )

    @staticmethod
    def _seed_bar_error_kwargs(
        *, ys: list[float], mins: list[float], maxs: list[float]
    ) -> dict:
        return dict(
            error_y=dict(
                type="data",
                symmetric=False,
                array=[mx - my for mx, my in zip(maxs, ys, strict=True)],
                arrayminus=[my - mn for my, mn in zip(ys, mins, strict=True)],
                thickness=1,
                width=3,
            )
        )

    # Rollouts kept visible by default in per-epoch plots on rollout sweeps
    # (others start as 'legendonly'). Intersected with the max-corr default,
    # this gives a small but representative set of curves on first render.
    _DEFAULT_VISIBLE_ROLLOUTS: frozenset[int] = frozenset({4, 128, 1024})

    _DASH_STYLES: tuple[str, ...] = ("solid", "dash", "dashdot", "dot")

    @staticmethod
    def _apply_compact_layout(fig: go.Figure, *, has_title: bool) -> None:
        """Tighten margins/fonts so figures embed well in the writeup. No
        explicit width is set so the saved HTML renders responsively and fills
        the embedding iframe; ``margin.autoexpand`` (plotly default) still
        pushes the plot area aside to fit the right-hand legend."""
        fig.update_layout(
            height=320,
            margin=dict(l=50, r=15, t=40 if has_title else 15, b=40),
            title=dict(font=dict(size=13), x=0.02, xanchor="left", y=0.98, yanchor="top")
            if has_title
            else None,
            font=dict(size=11),
            legend=dict(font=dict(size=10)),
        )
        fig.update_xaxes(title_font=dict(size=11), tickfont=dict(size=10))
        fig.update_yaxes(title_font=dict(size=11), tickfont=dict(size=10))

    @staticmethod
    def _save_html(fig: go.Figure, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True},
        )

    def plot_vs_epoch(
        self,
        y_exprs: list[pl.Expr | list[pl.Expr]],
        *,
        title: str | None = None,
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Per-epoch traces, one per study.

        Each entry in `y_exprs` is one panel. An entry may be either a single
        polars expression (one line per study) or a list of expressions (one
        line per (study, expression) — expressions are distinguished by line
        dash: solid, dash, dashdot, dot).

        For GRPO/MaxRL sweeps (study names carry a ` r=<N>` suffix), traces are
        *colored* by rollouts value and *legend-grouped* by dataset correlation.
        `legend.groupclick='togglegroup'` is set so clicking any entry in a
        `corr=...` group toggles every rollout for that corr on/off at once. To
        keep the initial view uncluttered, only traces in the highest-corr
        group whose rollouts value is in `_DEFAULT_VISIBLE_ROLLOUTS` are drawn;
        the rest start as 'legendonly'. For SL sweeps, each study gets its own
        color and every trace is visible by default.

        Expressions must carry their display name via `.alias(...)` — the
        helpers `corr_expr`, `rsq_expr`, `mse_expr`, `beta_expr` already do
        this. Seeds are averaged per (study, epoch); with `show_seed_bar=True`,
        asymmetric min/max-of-seeds error bars are drawn.
        """
        panels: list[list[pl.Expr]] = [
            [e] if isinstance(e, pl.Expr) else list(e) for e in y_exprs
        ]

        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        if rollouts is None:
            study_colors = {
                s: palette[i % len(palette)] for i, s in enumerate(self.studies)
            }
            study_legendgroups = {s: s for s in self.studies}
            study_legend_names = {s: s for s in self.studies}
            study_rollouts: dict[str, int | None] = {s: None for s in self.studies}
            show_group_title = False
            default_visible_group: str | None = None
        else:
            rollouts_colors = {
                r: palette[i % len(palette)] for i, r in enumerate(rollouts.keys())
            }
            study_colors: dict[str, str] = {}
            study_legendgroups: dict[str, str] = {}
            study_legend_names: dict[str, str] = {}
            study_rollouts = {}
            for r, names in rollouts.items():
                for n in names:
                    study_colors[n] = rollouts_colors[r]
                    study_legendgroups[n] = f"corr={self.study_corrs[n]:.3f}"
                    study_legend_names[n] = f"r={r}"
                    study_rollouts[n] = r
            show_group_title = True
            max_corr = max(self.study_corrs.values())
            default_visible_group = f"corr={max_corr:.3f}"

        def is_default_visible(study: str) -> bool:
            if default_visible_group is None:
                return True
            if study_legendgroups[study] != default_visible_group:
                return False
            return study_rollouts[study] in self._DEFAULT_VISIBLE_ROLLOUTS

        fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.08)
        for col, panel_y_exprs in enumerate(panels, start=1):
            multi_y = len(panel_y_exprs) > 1
            y_names = [y.meta.output_name() for y in panel_y_exprs]
            evaluated = df.with_columns(*panel_y_exprs)
            for y_idx, (y_expr, y_name) in enumerate(
                zip(panel_y_exprs, y_names, strict=True)
            ):
                dash = self._DASH_STYLES[y_idx % len(self._DASH_STYLES)]
                agg_df = self._aggregate_by_epoch(df=evaluated, y_name=y_name)
                for study in self.studies:
                    sub = agg_df.filter(pl.col("study") == study).sort("epoch")
                    if sub.is_empty():
                        continue
                    ds_corr = self.study_corrs[study]
                    epochs = sub["epoch"].to_list()
                    ys = sub["mean_y"].to_list()
                    mins = sub["min_y"].to_list()
                    maxs = sub["max_y"].to_list()
                    n_seeds = sub["n_seeds"].to_list()
                    customdata = [[ds_corr, n] for n in n_seeds]
                    error_kwargs = (
                        self._seed_bar_error_kwargs(ys=ys, mins=mins, maxs=maxs)
                        if show_seed_bar
                        else {}
                    )
                    group_title_kwargs = (
                        dict(legendgrouptitle_text=study_legendgroups[study])
                        if show_group_title
                        else {}
                    )
                    visible_kwargs = (
                        {} if is_default_visible(study) else dict(visible="legendonly")
                    )
                    trace_name = (
                        f"{study_legend_names[study]} · {y_name}"
                        if multi_y
                        else study_legend_names[study]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=ys,
                            mode="lines+markers",
                            name=trace_name,
                            legendgroup=study_legendgroups[study],
                            showlegend=(col == 1),
                            **group_title_kwargs,
                            **visible_kwargs,
                            line=dict(color=study_colors[study], dash=dash),
                            marker=dict(color=study_colors[study]),
                            customdata=customdata,
                            hovertemplate=(
                                "epoch: %{x}<br>"
                                f"{y_name}: %{{y}}<br>"
                                "dataset corr: %{customdata[0]}<br>"
                                "n_seeds: %{customdata[1]}"
                                f"<extra>{study}</extra>"
                            ),
                            **error_kwargs,
                        ),
                        row=1,
                        col=col,
                    )
            fig.update_xaxes(title_text="epoch", row=1, col=col)
            fig.update_yaxes(
                title_text=" / ".join(y_names) if multi_y else y_names[0],
                row=1,
                col=col,
            )
        if rollouts is not None:
            fig.update_layout(legend=dict(groupclick="togglegroup"))
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig

    def plot_vs_corr(
        self,
        y_exprs: list[pl.Expr],
        *,
        title: str | None = None,
        x_scale: Literal["log", "uniform"] = "log",
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Best-epoch (argmax seed-averaged y) vs dataset correlation.

        For GRPO/MaxRL sweeps, one curve per rollouts value — by default only
        rollouts in `_DEFAULT_VISIBLE_ROLLOUTS` are drawn; the rest start as
        'legendonly' (click the legend entry to reveal). For SL, a single
        curve is drawn. `x_scale="log"` uses a plotly log x-axis on the real
        corr values. `x_scale="uniform"` places each study at its rank in
        `src.data.bag_of_words.candidate_corrs` and labels ticks with the real
        corr values — giving equal visual spacing across the canonical grid.
        """
        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        curve_groups: list[tuple[str, int | None, list[str]]]
        if rollouts is None:
            curve_groups = [("all", None, list(self.studies.keys()))]
        else:
            curve_groups = [(f"r={r}", r, names) for r, names in rollouts.items()]

        n_panels = len(y_exprs)
        fig = make_subplots(rows=1, cols=n_panels, horizontal_spacing=0.08)
        for panel_idx, y_expr in enumerate(y_exprs):
            row, col = 1, panel_idx + 1
            is_first_panel = panel_idx == 0
            y_name = y_expr.meta.output_name()
            evaluated = df.with_columns(y_expr)
            agg_df = self._aggregate_by_epoch(df=evaluated, y_name=y_name)
            for i, (curve_name, r_value, names) in enumerate(curve_groups):
                color = palette[i % len(palette)]
                visible_kwargs = (
                    dict(visible="legendonly")
                    if (
                        r_value is not None
                        and r_value not in self._DEFAULT_VISIBLE_ROLLOUTS
                    )
                    else {}
                )
                rows: list[tuple[float, float, float, float, int, int, str]] = []
                for study in names:
                    sub = agg_df.filter(pl.col("study") == study)
                    if sub.is_empty():
                        continue
                    best = sub.sort("mean_y", descending=True).head(1)
                    rows.append((
                        self.study_corrs[study],
                        best["mean_y"].item(),
                        best["min_y"].item(),
                        best["max_y"].item(),
                        best["epoch"].item(),
                        best["n_seeds"].item(),
                        study,
                    ))
                if not rows:
                    continue
                rows.sort(key=lambda r: r[0])
                ds_corrs = [r[0] for r in rows]
                if x_scale == "uniform":
                    xs = [candidate_corrs.index(c) for c in ds_corrs]
                else:
                    xs = ds_corrs
                ys = [r[1] for r in rows]
                mins = [r[2] for r in rows]
                maxs = [r[3] for r in rows]
                customdata = [(r[0], r[4], r[5], r[6]) for r in rows]
                error_kwargs = (
                    self._seed_bar_error_kwargs(ys=ys, mins=mins, maxs=maxs)
                    if show_seed_bar
                    else {}
                )
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=curve_name,
                        legendgroup=curve_name,
                        showlegend=(rollouts is not None) and is_first_panel,
                        **visible_kwargs,
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=customdata,
                        hovertemplate=(
                            "dataset_corr: %{customdata[0]}<br>"
                            f"{y_name}: %{{y}}<br>"
                            "epoch: %{customdata[1]}<br>"
                            "n_seeds: %{customdata[2]}"
                            "<extra>%{customdata[3]}</extra>"
                        ),
                        **error_kwargs,
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="dataset_corr", row=row, col=col)
            fig.update_yaxes(title_text=y_name, row=row, col=col)
            if x_scale == "log":
                fig.update_xaxes(type="log", row=row, col=col)
            else:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(range(len(candidate_corrs))),
                    ticktext=[f"{c:.2f}" for c in candidate_corrs],
                    tickangle=-90,
                    row=row,
                    col=col,
                )
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
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
    """Mean squared error: Σ(x - y)² / n."""
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
