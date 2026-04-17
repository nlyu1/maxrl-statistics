import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from src.config.base import BaseConfig
from src.experiments.bag_of_words.sl import BagOfWordsSLConfig


class BagOfWordsAnalysisConfig(BaseConfig):
    """
    Dictionary of legend names and study artifact paths.
    Construct via from_studies(); only fully-completed experiments
    (those that finished all training epochs) are retained.
    """

    studies: dict[str, Path]
    metrics: list[str]

    @staticmethod
    def is_study_complete(path: Path) -> bool:
        """True iff the study at *path* has completed all its training epochs."""
        if not BagOfWordsAnalysisConfig.has_study_started(path):
            return False
        train_epochs = json.loads((path / "config.json").read_text())["train_epochs"]
        max_epoch = (
            pl.read_parquet(path / "metrics.parquet")
            .select(pl.col("epoch").max())
            .item()
        )
        return max_epoch >= train_epochs - 1

    @staticmethod
    def has_study_started(path: Path) -> bool:
        """True iff *path* has a config and at least one epoch of metrics written."""
        return (path / "metrics.parquet").exists() and (path / "config.json").exists()

    @classmethod
    def from_studies(cls, studies: dict[str, Path]) -> "BagOfWordsAnalysisConfig":
        started = {k: v for k, v in studies.items() if cls.has_study_started(v)}
        if not started:
            raise ValueError("No studies with metrics found")
        sample = pl.read_parquet(next(iter(started.values())) / "metrics.parquet")
        metrics = [c for c in sample.columns if c != "epoch"]
        return cls(studies=started, metrics=metrics)

    def get_config(self, study_name: str) -> BagOfWordsSLConfig:
        return BagOfWordsSLConfig.model_validate_json(
            (self.studies[study_name] / "config.json").read_text()
        )

    def get_metric_dataframe(self) -> pl.DataFrame:
        """
        Vertically joins all metrics parquets with columns:
        study, epoch, <metric columns>
        """
        frames = [
            pl.read_parquet(path / "metrics.parquet").with_columns(
                pl.lit(key).alias("study")
            )
            for key, path in self.studies.items()
        ]
        return pl.concat(frames).select(["study", "epoch", *self.metrics])

    def _add_per_epoch_traces(
        self,
        *,
        fig: go.Figure,
        df: pl.DataFrame,
        study_corrs: dict[str, float],
        x_axis: str,
        y_axis: str,
        col: int,
    ) -> None:
        for study in self.studies:
            sub = df.filter(pl.col("study") == study).sort("epoch")
            ds_corr = study_corrs[study]
            customdata = [[ds_corr, e] for e in sub["epoch"].to_list()]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_axis].to_list(),
                    y=sub[y_axis].to_list(),
                    mode="lines+markers",
                    name=study,
                    legendgroup=study,
                    showlegend=(col == 1),
                    customdata=customdata,
                    hovertemplate=(
                        f"{x_axis}: %{{x}}<br>"
                        f"{y_axis}: %{{y}}<br>"
                        "dataset corr: %{customdata[0]}<br>"
                        "epoch: %{customdata[1]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    def _add_best_epoch_trace(
        self,
        *,
        fig: go.Figure,
        df: pl.DataFrame,
        study_corrs: dict[str, float],
        x_axis: str,
        y_axis: str,
        col: int,
    ) -> None:
        # "best" is ambiguous without more info; use the mse→min, else→max convention.
        minimize = "mse" in y_axis.lower()
        rows = []
        for study in self.studies:
            sub = df.filter(pl.col("study") == study)
            best = sub.sort(y_axis, descending=not minimize).head(1)
            rows.append(
                (
                    best[x_axis].item(),
                    best[y_axis].item(),
                    study_corrs[study],
                    best["epoch"].item(),
                    study,
                )
            )
        rows.sort(key=lambda r: r[0])
        xs, ys, corrs, epochs, names = map(list, zip(*rows))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"{x_axis} vs {y_axis}",
                showlegend=False,
                customdata=list(zip(corrs, epochs, names)),
                hovertemplate=(
                    f"{x_axis}: %{{x}}<br>"
                    f"{y_axis}: %{{y}}<br>"
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
        axes: list[tuple[str, str]],
        *,
        best_epoch_only: bool,
        title: str | None = None,
    ) -> go.Figure:
        """
        Horizontally laid-out xy plots, one subplot per (x, y) pair.

        best_epoch_only=False: one line per study, parameterized by epoch; legend shared.
        best_epoch_only=True:  one line per subplot; each point is a study's best-epoch
                               value (argmin if "mse" in y_axis else argmax), sorted by x.
        """
        from plotly.subplots import make_subplots

        df = self.get_metric_dataframe()
        study_corrs = {s: self.get_config(s).data.corr for s in self.studies}
        add_trace = (
            self._add_best_epoch_trace
            if best_epoch_only
            else self._add_per_epoch_traces
        )

        fig = make_subplots(rows=1, cols=len(axes))
        for col, (x_axis, y_axis) in enumerate(axes, start=1):
            add_trace(
                fig=fig,
                df=df,
                study_corrs=study_corrs,
                x_axis=x_axis,
                y_axis=y_axis,
                col=col,
            )
            fig.update_xaxes(title_text=x_axis, row=1, col=col)
            fig.update_yaxes(title_text=y_axis, row=1, col=col)
        if title is not None:
            fig.update_layout(title=title)
        return fig
