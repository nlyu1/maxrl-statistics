import json

import plotly.graph_objects as go
import polars as pl
from pathlib import Path

from src.config.base import BaseConfig
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig


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
        metrics_path = path / "metrics.parquet"
        config_path = path / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            return False
        train_epochs = json.loads(config_path.read_text())["train_epochs"]
        max_epoch = pl.read_parquet(metrics_path).select(pl.col("epoch").max()).item()
        return max_epoch >= train_epochs - 1

    @classmethod
    def from_studies(cls, studies: dict[str, Path]) -> "BagOfWordsAnalysisConfig":
        completed = {k: v for k, v in studies.items() if cls.is_study_complete(v)}
        if not completed:
            raise ValueError("No completed studies found")
        sample = pl.read_parquet(next(iter(completed.values())) / "metrics.parquet")
        metrics = [c for c in sample.columns if c != "epoch"]
        return cls(studies=completed, metrics=metrics)

    def get_config(self, study_name: str) -> BagOfWordsSFTConfig:
        return BagOfWordsSFTConfig.model_validate_json(
            (self.studies[study_name] / "config.json").read_text()
        )

    def get_metric_dataframe(self) -> pl.DataFrame:
        """
        Vertically joins all metrics parquets with columns:
        study, epoch, <metric columns>
        """
        frames = [
            pl.read_parquet(path / "metrics.parquet").with_columns(pl.lit(key).alias("study"))
            for key, path in self.studies.items()
        ]
        return pl.concat(frames).select(["study", "epoch", *self.metrics])

    def xy_plots(self, axes: list[tuple[str, str]]) -> go.Figure:
        """
        Horizontally laid-out epoch-parameterized xy line plots with point markers.
        One subplot per (x, y) pair; legend is shared across all subplots.
        """
        from plotly.subplots import make_subplots

        df = self.get_metric_dataframe()
        n = len(axes)
        fig = make_subplots(rows=1, cols=n)
        for col, (x_axis, y_axis) in enumerate(axes, start=1):
            for i, study in enumerate(self.studies):
                sub = df.filter(pl.col("study") == study).sort("epoch")
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_axis].to_list(),
                        y=sub[y_axis].to_list(),
                        mode="lines+markers",
                        name=study,
                        legendgroup=study,
                        showlegend=(col == 1),
                        text=sub["epoch"].to_list(),
                        hovertemplate=f"{x_axis}: %{{x}}<br>{y_axis}: %{{y}}<br>epoch: %{{text}}<extra>%{{fullData.name}}</extra>",
                    ),
                    row=1, col=col,
                )
            fig.update_xaxes(title_text=x_axis, row=1, col=col)
            fig.update_yaxes(title_text=y_axis, row=1, col=col)
        return fig
