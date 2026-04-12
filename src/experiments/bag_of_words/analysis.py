import plotly.graph_objects as go
import polars as pl
from pathlib import Path

from src.config.base import BaseConfig
from src.experiments.bag_of_words.sft import BagOfWordsSFTConfig


class BagOfWordsAnalysisConfig(BaseConfig):
    """
    Dictionary of legend names and study artifact paths.
    Construct via from_studies(); only completed experiments (those with
    a metrics.parquet) are retained.
    """

    studies: dict[str, Path]
    metrics: list[str]

    @classmethod
    def from_studies(cls, studies: dict[str, Path]) -> "BagOfWordsAnalysisConfig":
        completed = {k: v for k, v in studies.items() if (v / "metrics.parquet").exists()}
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

    def xy_plot(self, x_axis: str, y_axis: str) -> go.Figure:
        """
        Epoch-parameterized xy line plot with point markers.
        One trace per study; legend keys are study names.
        """
        df = self.get_metric_dataframe()
        fig = go.Figure()
        for study in self.studies:
            sub = df.filter(pl.col("study") == study).sort("epoch")
            fig.add_trace(go.Scatter(
                x=sub[x_axis].to_list(),
                y=sub[y_axis].to_list(),
                mode="lines+markers",
                name=study,
            ))
        fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        return fig
