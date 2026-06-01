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
from src.experiments.corpus_regression.config import (
    baseline_mode_folder,
    factorized_mode_folder,
    likelihood_mode_folder,
    sigma_folder,
)

_ROLLOUTS_RE = re.compile(r" r=(\d+)")


def _seed_folder_name(seed: int) -> str:
    return f"seed-{seed}"


def canonical_dataset_folder_name(
    *,
    num_lookforward_tokens: int,
    num_samples: int = 100_000,
    label_type: Literal["rademacher", "token_id"] = "rademacher",
) -> str:
    """Slug used by single_run scripts as the leaf study folder. Computed from
    `CorpusRegressionDatasetConfig.get_canonical_folder` so it stays in sync
    with run-time artifacts."""
    cfg = CorpusRegressionDatasetConfig(
        **{
            **CorpusRegressionDatasetConfig.canonical_kwargs(label_type=label_type),
            "num_lookforward_tokens": num_lookforward_tokens,
            "num_samples": num_samples,
        }
    )
    return cfg.get_canonical_folder(Path("/")).name


def _decode_dim_averaged(df: pl.DataFrame) -> pl.DataFrame:
    """Append scalar `{split}_corr` and `{split}_mse` columns by averaging
    per-dim metrics across the D output dimensions of the list-column
    sufficient stats."""
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
    """Grouped view over completed/started study folders. Each `studies` key
    is a group name (`"look=4"` for SL, `"look=4 r=128"` for GRPO/MaxRL); the
    value is the list of per-seed study folders backing it."""

    studies: dict[str, list[Path]]
    study_seeds: dict[str, list[int]]
    study_lookforwards: dict[str, int]

    @classmethod
    def from_grouped(cls, grouped: dict[str, list[tuple[int, Path]]]) -> Self | None:
        """Drop paths missing `metrics.parquet`/`config.json`; drop empty
        groups; return None if nothing survives."""
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
    def from_ntp_baseline(
        cls,
        *,
        artifacts_root: Path,
        num_lookforward_tokens: int = 1,
        num_samples: int = 100_000,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
    ) -> Self | None:
        """`<artifacts_root>/ntp_baseline/{dataset_folder}/`.

        NTP baseline has no seeds (deterministic given dataset). We wrap
        the single result as seed=0 for compatibility with the grouped API.
        """
        study_base = artifacts_root / "ntp_baseline"
        if not study_base.exists():
            return None
        folder_name = canonical_dataset_folder_name(
            num_lookforward_tokens=num_lookforward_tokens,
            num_samples=num_samples,
            label_type=label_type,
        )
        grouped: dict[str, list[tuple[int, Path]]] = {
            f"look={num_lookforward_tokens}": [
                (0, study_base / folder_name),
            ]
        }
        return cls.from_grouped(grouped)

    @classmethod
    def from_sl_sweep(
        cls,
        *,
        artifacts_root: Path,
        num_samples: int = 100_000,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
    ) -> Self | None:
        """`<artifacts_root>/sl/seed-{S}/{dataset_folder}/`."""
        study_base = artifacts_root / "sl"
        if not study_base.exists():
            return None
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            grouped[f"look={n}"] = [
                (
                    s,
                    study_base
                    / _seed_folder_name(s)
                    / canonical_dataset_folder_name(
                        num_lookforward_tokens=n, num_samples=num_samples,
                        label_type=label_type,
                    ),
                )
                for s in candidate_seeds
            ]
        return cls.from_grouped(grouped)

    @classmethod
    def from_grpo_sweep(
        cls,
        *,
        artifacts_root: Path,
        num_samples: int = 100_000,
        gaussian_stdev: float = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
    ) -> Self | None:
        """`<artifacts_root>/grpo/seed-{S}/rollouts-{N}/sigma-{σ}/{dataset_folder}/`."""
        study_base = artifacts_root / "grpo"
        if not study_base.exists():
            return None
        sigma = sigma_folder(gaussian_stdev=gaussian_stdev)
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            for r in candidate_rollout_steps:
                grouped[f"look={n} r={r}"] = [
                    (
                        s,
                        study_base
                        / _seed_folder_name(s)
                        / f"rollouts-{r}"
                        / sigma
                        / canonical_dataset_folder_name(
                            num_lookforward_tokens=n, num_samples=num_samples,
                            label_type=label_type,
                        ),
                    )
                    for s in candidate_seeds
                ]
        return cls.from_grouped(grouped)

    @classmethod
    def from_rloo_sweep(
        cls,
        *,
        artifacts_root: Path,
        factorized: bool,
        num_samples: int = 100_000,
        gaussian_stdev: float = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
    ) -> Self | None:
        """`<artifacts_root>/rloo/seed-{S}/rollouts-{N}/sigma-{σ}/{factorized_mode}/{dataset_folder}/`.
        `factorized` is fixed per call — surface it in the figure title."""
        study_base = artifacts_root / "rloo"
        if not study_base.exists():
            return None
        factorized_mode = factorized_mode_folder(factorized=factorized)
        sigma = sigma_folder(gaussian_stdev=gaussian_stdev)
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            for r in candidate_rollout_steps:
                grouped[f"look={n} r={r}"] = [
                    (
                        s,
                        study_base
                        / _seed_folder_name(s)
                        / f"rollouts-{r}"
                        / sigma
                        / factorized_mode
                        / canonical_dataset_folder_name(
                            num_lookforward_tokens=n, num_samples=num_samples,
                            label_type=label_type,
                        ),
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
        use_factorized_likelihoods: bool,
        num_samples: int = 100_000,
        gaussian_stdev: float = 1.0,
        label_type: Literal["rademacher", "token_id"] = "rademacher",
    ) -> Self | None:
        """`<artifacts_root>/maxrl/seed-{S}/rollouts-{N}/sigma-{σ}/{baseline_mode}/{likelihood_mode}/{dataset_folder}/`.
        Both flags are fixed per call — surface them in the figure title."""
        study_base = artifacts_root / "maxrl"
        if not study_base.exists():
            return None
        baseline = baseline_mode_folder(subtract_baseline=subtract_baseline)
        likelihood = likelihood_mode_folder(
            use_factorized_likelihoods=use_factorized_likelihoods,
        )
        sigma = sigma_folder(gaussian_stdev=gaussian_stdev)
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for n in candidate_lookforward_tokens:
            for r in candidate_rollout_steps:
                grouped[f"look={n} r={r}"] = [
                    (
                        s,
                        study_base
                        / _seed_folder_name(s)
                        / f"rollouts-{r}"
                        / sigma
                        / baseline
                        / likelihood
                        / canonical_dataset_folder_name(
                            num_lookforward_tokens=n, num_samples=num_samples,
                            label_type=label_type,
                        ),
                    )
                    for s in candidate_seeds
                ]
        return cls.from_grouped(grouped)

    def describe(self, label: str) -> None:
        n_runs = sum(len(v) for v in self.studies.values())
        n_groups = len(self.studies)
        max_seeds = max(len(v) for v in self.studies.values())
        print(
            f"{label:<32s}  {n_runs:>4d} runs   {n_groups:>3d} groups   "
            f"up to {max_seeds} seeds/group"
        )

    def get_metric_dataframe(self) -> pl.DataFrame:
        """Concat per-(study, seed) metrics.parquet, then append scalar
        `{split}_corr` / `{split}_mse` columns."""
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
        """Split studies by ` r=(\\d+)` suffix; None if any study lacks it
        (i.e. SL sweeps)."""
        groups: dict[int, list[str]] = {}
        for name in self.studies:
            m = _ROLLOUTS_RE.search(name)
            if m is None:
                return None
            groups.setdefault(int(m.group(1)), []).append(name)
        return dict(sorted(groups.items()))

    def _aggregate_by_epoch(self, *, df: pl.DataFrame, y_name: str) -> pl.DataFrame:
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

    _DEFAULT_VISIBLE_ROLLOUTS: frozenset[int] = frozenset({4, 128, 1024})

    @staticmethod
    def _apply_compact_layout(fig: go.Figure, *, has_title: bool) -> None:
        # Right margin is explicit so the legend doesn't get clipped when the
        # figure is first rendered hidden inside a collapsed callout.
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
        dict[str, str], dict[str, str], dict[str, str], bool, str | None,
    ]:
        """For SL: each study gets a unique color, no group title. For rollout
        sweeps: color by rollouts, legend-group by lookforward, default-visible
        is the min-look group available (so partial sweeps still show
        something on first render)."""
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
        min_look = min(self.study_lookforwards.values())
        return (
            study_colors,
            study_legendgroups,
            study_legend_names,
            True,
            f"look={min_look}",
        )

    def plot_vs_epoch(
        self,
        metric: Literal["corr", "mse"],
        *,
        title: str | None = None,
        show_seed_bar: bool = False,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Two panels (train left, val right) of `{train,val}_{metric}` per
        epoch, one line per study (seed-mean). Rollout sweeps default-show
        the min-look group with rollouts in `_DEFAULT_VISIBLE_ROLLOUTS`."""
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

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
        for col, split in enumerate(("train", "val"), start=1):
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
        """Best-epoch (argmax seed-mean) `train_corr` / `val_corr` vs
        `num_lookforward_tokens`, two panels. Rollout sweeps draw one curve
        per rollouts value; only `_DEFAULT_VISIBLE_ROLLOUTS` start visible."""
        df = self.get_metric_dataframe()
        rollouts = self._rollouts_groups()
        palette = qualitative.Plotly
        if rollouts is None:
            curve_groups: list[tuple[str, int | None, list[str]]] = [
                ("all", None, list(self.studies.keys()))
            ]
        else:
            curve_groups = [(f"r={r}", r, names) for r, names in rollouts.items()]

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
        for col, split in enumerate(("train", "val"), start=1):
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
                rows = _best_epoch_rows_for_studies(
                    cfg=self, agg_df=agg_df, studies=names,
                )
                if not rows:
                    continue
                looks = [r[0] for r in rows]
                xs = (
                    [candidate_lookforward_tokens.index(v) for v in looks]
                    if x_scale == "uniform"
                    else looks
                )
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


# Cross-method comparison helpers.

_METHOD_DASH: dict[str, str] = {
    "sl": "solid",
    "grpo": "dash",
    "maxrl": "dot",
    "rloo": "longdash",
    "ntp_baseline": "dashdot",
}


def _best_epoch_rows_for_studies(
    *,
    cfg: CorpusRegressionAnalysisConfig,
    agg_df: pl.DataFrame,
    studies: list[str],
    higher_is_better: bool = True,
) -> list[tuple[int, float, float, float, int, int, str]]:
    """Per-study best-epoch row, sorted by lookforward. Uses argmax when
    `higher_is_better=True` (corr) and argmin when False (mse).
    Skips studies with no rows so partial sweeps render cleanly."""
    out: list[tuple[int, float, float, float, int, int, str]] = []
    for study in studies:
        sub = agg_df.filter(pl.col("study") == study)
        if sub.is_empty():
            continue
        best = sub.sort("mean_y", descending=higher_is_better).head(1)
        out.append((
            cfg.study_lookforwards[study],
            best["mean_y"].item(),
            best["min_y"].item(),
            best["max_y"].item(),
            best["epoch"].item(),
            best["n_seeds"].item(),
            study,
        ))
    out.sort(key=lambda r: r[0])
    return out


def plot_methods_vs_lookforward(
    *,
    sl: CorpusRegressionAnalysisConfig | None = None,
    grpo: CorpusRegressionAnalysisConfig | None = None,
    maxrl: CorpusRegressionAnalysisConfig | None = None,
    rloo: CorpusRegressionAnalysisConfig | None = None,
    ntp_baseline: CorpusRegressionAnalysisConfig | None = None,
    metric: Literal["corr", "mse"] = "corr",
    title: str | None = None,
    x_scale: Literal["log", "uniform"] = "uniform",
    save_path: Path | None = None,
) -> go.Figure:
    """Cross-method best-epoch metric vs `num_lookforward_tokens`. Two panels
    (train, val). SL (when present) is a single standalone curve; GRPO, MaxRL
    and RLOO traces are grouped by rollouts — one legend group per `r=N`,
    containing one curve per RL method that ran that rollouts value. Color =
    rollouts; dash = method. Default-visible: SL plus the highest-rollouts
    group.

    `ntp_baseline` (when present) is drawn as a standalone curve representing
    the intrinsic variance floor."""
    methods_rl: list[tuple[str, CorpusRegressionAnalysisConfig]] = [
        (name, cfg)
        for name, cfg in (("grpo", grpo), ("maxrl", maxrl), ("rloo", rloo))
        if cfg is not None
    ]
    if sl is None and not methods_rl and ntp_baseline is None:
        raise ValueError("at least one of sl/grpo/maxrl/rloo/ntp_baseline must be provided")

    palette = qualitative.Plotly
    rollouts_seen: list[int] = []
    for _, cfg in methods_rl:
        rg = cfg._rollouts_groups()
        if rg is None:
            continue
        for r in rg:
            if r not in rollouts_seen:
                rollouts_seen.append(r)
    rollouts_seen.sort()
    color_by_rollouts = {
        r: palette[i % len(palette)] for i, r in enumerate(rollouts_seen)
    }
    sl_color = palette[len(rollouts_seen) % len(palette)]
    ntp_color = palette[(len(rollouts_seen) + 1) % len(palette)]
    max_rollouts = rollouts_seen[-1] if rollouts_seen else None

    sl_df = sl.get_metric_dataframe() if sl is not None else None
    ntp_df = ntp_baseline.get_metric_dataframe() if ntp_baseline is not None else None
    rl_dfs: dict[str, pl.DataFrame] = {
        name: cfg.get_metric_dataframe() for name, cfg in methods_rl
    }
    rl_groups: dict[str, dict[int, list[str]]] = {
        name: cfg._rollouts_groups() or {} for name, cfg in methods_rl
    }
    rl_cfgs: dict[str, CorpusRegressionAnalysisConfig] = dict(methods_rl)

    higher_is_better = metric == "corr"

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
    for col, split in enumerate(("train", "val"), start=1):
        y_name = f"{split}_{metric}"

        if sl is not None:
            sl_agg = sl._aggregate_by_epoch(df=sl_df, y_name=y_name)
            sl_rows = _best_epoch_rows_for_studies(
                cfg=sl, agg_df=sl_agg, studies=list(sl.studies.keys()),
                higher_is_better=higher_is_better,
            )
            if sl_rows:
                _add_methods_curve(
                    fig=fig,
                    rows=sl_rows,
                    col=col,
                    trace_name="sl",
                    legendgroup="sl",
                    legendgrouptitle_text=None,
                    color=sl_color,
                    dash=_METHOD_DASH["sl"],
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(col == 1),
                    visible_default=True,
                )

        if ntp_baseline is not None:
            ntp_agg = ntp_baseline._aggregate_by_epoch(df=ntp_df, y_name=y_name)
            ntp_rows = _best_epoch_rows_for_studies(
                cfg=ntp_baseline, agg_df=ntp_agg, studies=list(ntp_baseline.studies.keys()),
                higher_is_better=higher_is_better,
            )
            if ntp_rows:
                _add_methods_curve(
                    fig=fig,
                    rows=ntp_rows,
                    col=col,
                    trace_name="ntp_baseline",
                    legendgroup="ntp_baseline",
                    legendgrouptitle_text=None,
                    color=ntp_color,
                    dash=_METHOD_DASH["ntp_baseline"],
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(col == 1),
                    visible_default=True,
                )

        for r in rollouts_seen:
            for method_name, _ in methods_rl:
                names_for_r = rl_groups[method_name].get(r, [])
                if not names_for_r:
                    continue
                cfg = rl_cfgs[method_name]
                agg_df = cfg._aggregate_by_epoch(
                    df=rl_dfs[method_name], y_name=y_name,
                )
                rows = _best_epoch_rows_for_studies(
                    cfg=cfg, agg_df=agg_df, studies=names_for_r,
                    higher_is_better=higher_is_better,
                )
                if not rows:
                    continue
                _add_methods_curve(
                    fig=fig,
                    rows=rows,
                    col=col,
                    trace_name=method_name,
                    legendgroup=f"r={r}",
                    legendgrouptitle_text=f"r={r}",
                    color=color_by_rollouts[r],
                    dash=_METHOD_DASH[method_name],
                    y_name=y_name,
                    x_scale=x_scale,
                    show_legend=(col == 1),
                    visible_default=(r == max_rollouts),
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
    fig.update_layout(legend=dict(groupclick="togglegroup"))
    if title is not None:
        fig.update_layout(title=title)
    CorpusRegressionAnalysisConfig._apply_compact_layout(
        fig, has_title=title is not None
    )
    if save_path is not None:
        CorpusRegressionAnalysisConfig._save_html(fig, save_path)
    return fig


def _add_methods_curve(
    *,
    fig: go.Figure,
    rows: list[tuple[int, float, float, float, int, int, str]],
    col: int,
    trace_name: str,
    legendgroup: str,
    legendgrouptitle_text: str | None,
    color: str,
    dash: str,
    y_name: str,
    x_scale: Literal["log", "uniform"],
    show_legend: bool,
    visible_default: bool,
) -> None:
    looks = [r[0] for r in rows]
    xs = (
        [candidate_lookforward_tokens.index(v) for v in looks]
        if x_scale == "uniform"
        else looks
    )
    ys = [r[1] for r in rows]
    mins = [r[2] for r in rows]
    maxs = [r[3] for r in rows]
    customdata = [(r[0], r[4], r[5], r[6]) for r in rows]
    error_kwargs = CorpusRegressionAnalysisConfig._seed_bar_error_kwargs(
        ys=ys, mins=mins, maxs=maxs,
    )
    group_title_kwargs = (
        dict(legendgrouptitle_text=legendgrouptitle_text)
        if legendgrouptitle_text is not None
        else {}
    )
    visible_kwargs = {} if visible_default else dict(visible="legendonly")
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=trace_name,
            legendgroup=legendgroup,
            **group_title_kwargs,
            showlegend=show_legend,
            line=dict(color=color, dash=dash),
            marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "num_lookforward_tokens: %{customdata[0]}<br>"
                f"{y_name}: %{{y}}<br>"
                "epoch: %{customdata[1]}<br>"
                "n_seeds: %{customdata[2]}"
                f"<extra>{trace_name} %{{customdata[3]}}</extra>"
            ),
            **visible_kwargs,
            **error_kwargs,
        ),
        row=1,
        col=col,
    )


def _add_epoch_curve(
    *,
    fig: go.Figure,
    agg_df: pl.DataFrame,
    col: int,
    trace_name: str,
    legendgroup: str,
    legendgrouptitle_text: str | None,
    color: str,
    dash: str,
    y_name: str,
    show_legend: bool,
    visible_default: bool,
    show_seed_bar: bool,
) -> None:
    """Add a single per-epoch Scatter trace to `fig`. `agg_df` must already be
    filtered to one study and sorted by epoch."""
    epochs = agg_df["epoch"].to_list()
    ys = agg_df["mean_y"].to_list()
    mins = agg_df["min_y"].to_list()
    maxs = agg_df["max_y"].to_list()
    n_seeds = agg_df["n_seeds"].to_list()
    error_kwargs = (
        CorpusRegressionAnalysisConfig._seed_bar_error_kwargs(
            ys=ys, mins=mins, maxs=maxs
        )
        if show_seed_bar
        else {}
    )
    group_title_kwargs = (
        dict(legendgrouptitle_text=legendgrouptitle_text)
        if legendgrouptitle_text is not None
        else {}
    )
    visible_kwargs = {} if visible_default else dict(visible="legendonly")
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=ys,
            mode="lines+markers",
            name=trace_name,
            legendgroup=legendgroup,
            **group_title_kwargs,
            showlegend=show_legend,
            line=dict(color=color, dash=dash),
            marker=dict(color=color),
            customdata=[[n] for n in n_seeds],
            hovertemplate=(
                "epoch: %{x}<br>"
                f"{y_name}: %{{y}}<br>"
                "n_seeds: %{customdata[0]}"
                f"<extra>{trace_name}</extra>"
            ),
            **visible_kwargs,
            **error_kwargs,
        ),
        row=1,
        col=col,
    )


def plot_methods_vs_epoch(
    *,
    sl: CorpusRegressionAnalysisConfig | None = None,
    grpo: CorpusRegressionAnalysisConfig | None = None,
    maxrl: CorpusRegressionAnalysisConfig | None = None,
    rloo: CorpusRegressionAnalysisConfig | None = None,
    ntp_baseline: CorpusRegressionAnalysisConfig | None = None,
    num_lookforward_tokens: int = 1,
    metric: Literal["corr", "mse"] = "corr",
    show_seed_bar: bool = False,
    title: str | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """Cross-method per-epoch metric curves for a single `num_lookforward_tokens`
    value. Two panels (train, val). Color = rollouts; dash = method."""
    methods_rl: list[tuple[str, CorpusRegressionAnalysisConfig]] = [
        (name, cfg)
        for name, cfg in (("grpo", grpo), ("maxrl", maxrl), ("rloo", rloo))
        if cfg is not None
    ]
    if sl is None and not methods_rl and ntp_baseline is None:
        raise ValueError("at least one of sl/grpo/maxrl/rloo/ntp_baseline must be provided")

    palette = qualitative.Plotly

    # Collect rollout values across RL methods.
    rollouts_seen: list[int] = []
    for _, cfg in methods_rl:
        rg = cfg._rollouts_groups()
        if rg is None:
            continue
        for r in rg:
            if r not in rollouts_seen:
                rollouts_seen.append(r)
    rollouts_seen.sort()
    color_by_rollouts = {
        r: palette[i % len(palette)] for i, r in enumerate(rollouts_seen)
    }
    sl_color = palette[len(rollouts_seen) % len(palette)]
    ntp_color = palette[(len(rollouts_seen) + 1) % len(palette)]
    max_rollouts = rollouts_seen[-1] if rollouts_seen else None

    # Pre-load metric DataFrames.
    sl_df = sl.get_metric_dataframe() if sl is not None else None
    ntp_df = ntp_baseline.get_metric_dataframe() if ntp_baseline is not None else None
    rl_dfs: dict[str, pl.DataFrame] = {
        name: cfg.get_metric_dataframe() for name, cfg in methods_rl
    }

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)
    for col, split in enumerate(("train", "val"), start=1):
        y_name = f"{split}_{metric}"

        # SL (no rollouts): single study "look=N".
        if sl is not None and sl_df is not None:
            study_name = f"look={num_lookforward_tokens}"
            if study_name in sl.studies:
                agg = sl._aggregate_by_epoch(
                    df=sl_df.filter(pl.col("study") == study_name), y_name=y_name
                )
                if not agg.is_empty():
                    _add_epoch_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name="sl",
                        legendgroup="sl",
                        legendgrouptitle_text=None,
                        color=sl_color,
                        dash=_METHOD_DASH["sl"],
                        y_name=y_name,
                        show_legend=(col == 1),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        # NTP baseline: single study "look=N".
        if ntp_baseline is not None and ntp_df is not None:
            study_name = f"look={num_lookforward_tokens}"
            if study_name in ntp_baseline.studies:
                agg = ntp_baseline._aggregate_by_epoch(
                    df=ntp_df.filter(pl.col("study") == study_name), y_name=y_name
                )
                if not agg.is_empty():
                    _add_epoch_curve(
                        fig=fig,
                        agg_df=agg,
                        col=col,
                        trace_name="ntp_baseline",
                        legendgroup="ntp_baseline",
                        legendgrouptitle_text=None,
                        color=ntp_color,
                        dash=_METHOD_DASH["ntp_baseline"],
                        y_name=y_name,
                        show_legend=(col == 1),
                        visible_default=True,
                        show_seed_bar=show_seed_bar,
                    )

        # RL methods: one trace per (rollouts, method) pair.
        for r in rollouts_seen:
            for method_name, cfg in methods_rl:
                study_name = f"look={num_lookforward_tokens} r={r}"
                if study_name not in cfg.studies:
                    continue
                agg = cfg._aggregate_by_epoch(
                    df=rl_dfs[method_name].filter(pl.col("study") == study_name),
                    y_name=y_name,
                )
                if agg.is_empty():
                    continue
                _add_epoch_curve(
                    fig=fig,
                    agg_df=agg,
                    col=col,
                    trace_name=method_name,
                    legendgroup=f"r={r}",
                    legendgrouptitle_text=f"r={r}",
                    color=color_by_rollouts[r],
                    dash=_METHOD_DASH[method_name],
                    y_name=y_name,
                    show_legend=(col == 1),
                    visible_default=(r == max_rollouts),
                    show_seed_bar=show_seed_bar,
                )

        fig.update_xaxes(title_text="epoch", row=1, col=col)
        fig.update_yaxes(title_text=y_name, row=1, col=col)

    fig.update_layout(legend=dict(groupclick="togglegroup"))
    if title is not None:
        fig.update_layout(title=title)
    CorpusRegressionAnalysisConfig._apply_compact_layout(
        fig, has_title=title is not None
    )
    if save_path is not None:
        CorpusRegressionAnalysisConfig._save_html(fig, save_path)
    return fig
