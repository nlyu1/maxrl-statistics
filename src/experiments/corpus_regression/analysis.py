import json
import re
from pathlib import Path
from typing import Literal, Self

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from src.config.base import BaseConfig
from src.data.corpus_regression import (
    CorpusRegressionDatasetConfig,
    candidate_lookforward_tokens,
    candidate_rollout_steps,
    candidate_seeds,
)

_ROLLOUTS_RE = re.compile(r" r=(\d+)")
_SWEEP_ROOT_BY_METHOD: dict[str, str] = {
    "sl": "cr-sl-sweep",
    "grpo": "cr-grpo-sweep",
    "maxrl": "cr-maxrl-sweep",
}


def _seed_folder_name(seed: int) -> str:
    return f"seed-{seed}"


def baseline_mode_folder(*, subtract_baseline: bool) -> str:
    return "subtract-baseline" if subtract_baseline else "no-subtract-baseline"


def sweep_root_name(*, method: str) -> str:
    return _SWEEP_ROOT_BY_METHOD[method]


def canonical_dataset_folder_name(*, num_lookforward_tokens: int) -> str:
    """Single source of truth for the per-lookforward dataset slug. Reuses
    `CorpusRegressionDatasetConfig.get_canonical_folder` so the name stays in
    sync with run-time artifacts."""
    cfg = CorpusRegressionDatasetConfig(
        **{
            **CorpusRegressionDatasetConfig.canonical_kwargs(),
            "num_lookforward_tokens": num_lookforward_tokens,
        }
    )
    return cfg.get_canonical_folder(Path("/")).name


def _decode_dim_averaged(df: pl.DataFrame) -> pl.DataFrame:
    """Append scalar `{split}_corr` and `{split}_mse` columns. Sufficient stats
    are stored as length-D list-columns; per-dim metrics are averaged across D
    output dimensions (same recipe as `utils.dim_averaged_metrics_from_parquet`)."""
    out = df
    for split in ("train", "val"):
        xx = np.asarray(df[f"{split}_target_xx"].to_list(), dtype=np.float64)
        xy = np.asarray(df[f"{split}_target_xy"].to_list(), dtype=np.float64)
        yy = np.asarray(df[f"{split}_target_yy"].to_list(), dtype=np.float64)
        n = df[f"{split}_target_n"].to_numpy().astype(np.float64)
        per_dim_corr = xy / np.sqrt(xx * yy)
        per_dim_mse = (xx - 2.0 * xy + yy) / n[:, None]
        out = out.with_columns(
            pl.Series(f"{split}_corr", per_dim_corr.mean(axis=1)),
            pl.Series(f"{split}_mse", per_dim_mse.mean(axis=1)),
        )
    return out


class CorpusRegressionAnalysisConfig(BaseConfig):
    """
    Grouped view over completed/started study folders. Each `studies` key is a
    group name (e.g. `"look=4"` for SL, `"look=4 r=128"` for GRPO/MaxRL); the
    value is the list of per-seed study folders backing it. Construct via the
    `from_*_sweep` factories, which consume the canonical grids in
    `src.data.corpus_regression`.
    """

    studies: dict[str, list[Path]]
    study_seeds: dict[str, list[int]]
    study_lookforwards: dict[str, int]

    @classmethod
    def from_grouped(cls, grouped: dict[str, list[tuple[int, Path]]]) -> Self | None:
        """Build from group-name -> [(seed, path), ...]. Drops paths missing
        `metrics.parquet`/`config.json`; drops empty groups. Returns None if
        nothing survives so notebook loops can skip cleanly."""
        studies: dict[str, list[Path]] = {}
        study_seeds: dict[str, list[int]] = {}
        study_lookforwards: dict[str, int] = {}
        for name, pairs in grouped.items():
            kept = [
                (s, p)
                for (s, p) in pairs
                if (p / "metrics.parquet").exists() and (p / "config.json").exists()
            ]
            if not kept:
                continue
            studies[name] = [p for (_, p) in kept]
            study_seeds[name] = [s for (s, _) in kept]
            first_path = kept[0][1]
            study_lookforwards[name] = int(
                json.loads((first_path / "config.json").read_text())[
                    "data"
                ]["num_lookforward_tokens"]
            )
        if not studies:
            return None
        return cls(
            studies=studies,
            study_seeds=study_seeds,
            study_lookforwards=study_lookforwards,
        )

    @classmethod
    def from_sl_sweep(cls, *, artifacts_root: Path) -> Self | None:
        """Discover SL runs under
        `<artifacts_root>/cr-sl-sweep/seed-{S}/{dataset_folder}/`."""
        study_base = artifacts_root / sweep_root_name(method="sl")
        if not study_base.exists():
            return None
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            name = f"look={n}"
            grouped[name] = [
                (
                    s,
                    study_base
                    / _seed_folder_name(s)
                    / canonical_dataset_folder_name(num_lookforward_tokens=n),
                )
                for s in candidate_seeds
            ]
        return cls.from_grouped(grouped)

    @classmethod
    def from_grpo_sweep(cls, *, artifacts_root: Path) -> Self | None:
        """Discover GRPO runs under
        `<artifacts_root>/cr-grpo-sweep/seed-{S}/rollouts-{N}/{dataset_folder}/`."""
        study_base = artifacts_root / sweep_root_name(method="grpo")
        if not study_base.exists():
            return None
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            for r in candidate_rollout_steps:
                name = f"look={n} r={r}"
                grouped[name] = [
                    (
                        s,
                        study_base
                        / _seed_folder_name(s)
                        / f"rollouts-{r}"
                        / canonical_dataset_folder_name(num_lookforward_tokens=n),
                    )
                    for s in candidate_seeds
                ]
        return cls.from_grouped(grouped)

    @classmethod
    def from_maxrl_sweep(
        cls,
        *,
        artifacts_root: Path,
        subtract_baseline: bool,
    ) -> Self | None:
        """Discover MaxRL runs under
        `<artifacts_root>/cr-maxrl-sweep/seed-{S}/rollouts-{N}/{baseline_mode}/{dataset_folder}/`.
        The baseline mode is fixed per factory call; surface it in the figure
        title rather than the group key."""
        study_base = artifacts_root / sweep_root_name(method="maxrl")
        if not study_base.exists():
            return None
        baseline_folder = baseline_mode_folder(subtract_baseline=subtract_baseline)
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            for r in candidate_rollout_steps:
                name = f"look={n} r={r}"
                grouped[name] = [
                    (
                        s,
                        study_base
                        / _seed_folder_name(s)
                        / f"rollouts-{r}"
                        / baseline_folder
                        / canonical_dataset_folder_name(num_lookforward_tokens=n),
                    )
                    for s in candidate_seeds
                ]
        return cls.from_grouped(grouped)

    def describe(self, label: str) -> None:
        """One-line summary: run count, group count, max seeds/group."""
        n_runs = sum(len(v) for v in self.studies.values())
        n_groups = len(self.studies)
        max_seeds = max(len(v) for v in self.studies.values())
        print(
            f"{label:<32s}  {n_runs:>4d} runs   {n_groups:>3d} groups   "
            f"up to {max_seeds} seeds/group"
        )

    def get_metric_dataframe(self) -> pl.DataFrame:
        """Vertical concat of per-(study, seed) metrics.parquet files, with
        scalar `{split}_corr` and `{split}_mse` columns appended."""
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
        return _decode_dim_averaged(pl.concat(frames))

    def _rollouts_groups(self) -> dict[int, list[str]] | None:
        """Split studies by ` r=(\\d+)` suffix. Returns None if any study lacks
        the suffix (i.e. SL sweeps)."""
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

    # Default-visible rollouts on first render of rollout sweeps. Intersected
    # with the max-lookforward default group, this gives a small but
    # representative initial view.
    _DEFAULT_VISIBLE_ROLLOUTS: frozenset[int] = frozenset({4, 128, 1024})

    @staticmethod
    def _apply_compact_layout(fig: go.Figure, *, has_title: bool) -> None:
        fig.update_layout(
            height=320,
            margin=dict(l=50, r=105, t=40 if has_title else 15, b=40),
            title=dict(font=dict(size=13), x=0.02, xanchor="left", y=0.98, yanchor="top")
            if has_title
            else None,
            font=dict(size=11),
            legend=dict(
                font=dict(size=10),
                x=1.01,
                xanchor="left",
                y=1.0,
                yanchor="top",
            ),
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

    def _study_styling(self) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, str],
        bool,
        str | None,
    ]:
        """Resolve per-study color, legendgroup, legend display name, whether
        to show legend group titles, and the default-visible group key.

        For SL-shaped sweeps (no `r=` suffix), each study gets its own color
        and is visible by default. For rollout-shaped sweeps, color comes from
        rollouts, legendgroup from `look=<N>`, and the default-visible group is
        the maximum lookforward."""
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        if rollouts is None:
            study_colors = {
                s: palette[i % len(palette)] for i, s in enumerate(self.studies)
            }
            study_legendgroups = {s: s for s in self.studies}
            study_legend_names = {s: s for s in self.studies}
            return study_colors, study_legendgroups, study_legend_names, False, None
        rollouts_colors = {
            r: palette[i % len(palette)] for i, r in enumerate(rollouts.keys())
        }
        study_colors: dict[str, str] = {}
        study_legendgroups: dict[str, str] = {}
        study_legend_names: dict[str, str] = {}
        for r, names in rollouts.items():
            for n in names:
                study_colors[n] = rollouts_colors[r]
                study_legendgroups[n] = f"look={self.study_lookforwards[n]}"
                study_legend_names[n] = f"r={r}"
        max_look = max(self.study_lookforwards.values())
        return (
            study_colors,
            study_legendgroups,
            study_legend_names,
            True,
            f"look={max_look}",
        )

    def plot_vs_epoch(
        self,
        metric: Literal["corr", "mse"],
        *,
        title: str | None = None,
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Per-epoch traces, two panels (train left, val right) of
        `{train,val}_{metric}`. One line per study (seed-mean).

        For rollout sweeps (study names carry ` r=<N>`), traces are colored by
        rollouts and legend-grouped by lookforward; only the max-lookforward
        group with rollouts in `_DEFAULT_VISIBLE_ROLLOUTS` is visible at first
        render. For SL sweeps, every study is colored uniquely and visible by
        default."""
        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        (
            study_colors,
            study_legendgroups,
            study_legend_names,
            show_group_title,
            default_visible_group,
        ) = self._study_styling()

        def is_default_visible(study: str) -> bool:
            if default_visible_group is None:
                return True
            if study_legendgroups[study] != default_visible_group:
                return False
            r = int(_ROLLOUTS_RE.search(study).group(1))
            return r in self._DEFAULT_VISIBLE_ROLLOUTS

        splits = ("train", "val")
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
        for col, split in enumerate(splits, start=1):
            y_name = f"{split}_{metric}"
            agg_df = self._aggregate_by_epoch(df=df, y_name=y_name)
            for study in self.studies:
                sub = agg_df.filter(pl.col("study") == study).sort("epoch")
                if sub.is_empty():
                    continue
                look = self.study_lookforwards[study]
                epochs = sub["epoch"].to_list()
                ys = sub["mean_y"].to_list()
                mins = sub["min_y"].to_list()
                maxs = sub["max_y"].to_list()
                n_seeds = sub["n_seeds"].to_list()
                customdata = [[look, n] for n in n_seeds]
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
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=ys,
                        mode="lines+markers",
                        name=study_legend_names[study],
                        legendgroup=study_legendgroups[study],
                        showlegend=(col == 1),
                        **group_title_kwargs,
                        **visible_kwargs,
                        line=dict(color=study_colors[study]),
                        marker=dict(color=study_colors[study]),
                        customdata=customdata,
                        hovertemplate=(
                            "epoch: %{x}<br>"
                            f"{y_name}: %{{y}}<br>"
                            "num_lookforward_tokens: %{customdata[0]}<br>"
                            "n_seeds: %{customdata[1]}"
                            f"<extra>{study}</extra>"
                        ),
                        **error_kwargs,
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="epoch", row=1, col=col)
            fig.update_yaxes(title_text=y_name, row=1, col=col)
        if rollouts is not None:
            fig.update_layout(legend=dict(groupclick="togglegroup"))
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig

    def plot_vs_lookforward(
        self,
        *,
        title: str | None = None,
        x_scale: Literal["log", "uniform"] = "uniform",
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Best-epoch (argmax seed-averaged value) of `train_corr` / `val_corr`
        vs `num_lookforward_tokens`. Two panels (train left, val right).

        For rollout sweeps, one curve per rollouts value — only those in
        `_DEFAULT_VISIBLE_ROLLOUTS` start visible. For SL, a single curve.
        `x_scale="log"` plots on the real lookforward axis; `x_scale="uniform"`
        ranks values within `candidate_lookforward_tokens` for equal spacing
        and labels ticks with the real values."""
        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        if rollouts is None:
            curve_groups: list[tuple[str, int | None, list[str]]] = [
                ("all", None, list(self.studies.keys()))
            ]
        else:
            curve_groups = [(f"r={r}", r, names) for r, names in rollouts.items()]

        splits = ("train", "val")
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
        for col, split in enumerate(splits, start=1):
            y_name = f"{split}_corr"
            agg_df = self._aggregate_by_epoch(df=df, y_name=y_name)
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
                rows: list[tuple[int, float, float, float, int, int, str]] = []
                for study in names:
                    sub = agg_df.filter(pl.col("study") == study)
                    if sub.is_empty():
                        continue
                    best = sub.sort("mean_y", descending=True).head(1)
                    rows.append((
                        self.study_lookforwards[study],
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
                looks = [r[0] for r in rows]
                if x_scale == "uniform":
                    xs = [candidate_lookforward_tokens.index(v) for v in looks]
                else:
                    xs = looks
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
                        showlegend=(rollouts is not None) and col == 1,
                        **visible_kwargs,
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=customdata,
                        hovertemplate=(
                            "num_lookforward_tokens: %{customdata[0]}<br>"
                            f"{y_name}: %{{y}}<br>"
                            "epoch: %{customdata[1]}<br>"
                            "n_seeds: %{customdata[2]}"
                            "<extra>%{customdata[3]}</extra>"
                        ),
                        **error_kwargs,
                    ),
                    row=1,
                    col=col,
                )
            fig.update_xaxes(title_text="num_lookforward_tokens", row=1, col=col)
            fig.update_yaxes(title_text=y_name, row=1, col=col)
            if x_scale == "log":
                fig.update_xaxes(type="log", row=1, col=col)
            else:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(range(len(candidate_lookforward_tokens))),
                    ticktext=[str(v) for v in candidate_lookforward_tokens],
                    row=1,
                    col=col,
                )
        if title is not None:
            fig.update_layout(title=title)
        self._apply_compact_layout(fig, has_title=title is not None)
        if save_path is not None:
            self._save_html(fig, save_path)
        return fig
