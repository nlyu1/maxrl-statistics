import polars as pl
from pathlib import Path

from src.config.base import BaseConfig


class BagOfWordsAnalysisConfig(BaseConfig):
    """
    Dictionary of legend names and study artifact path
    """
    studies: dict[str, Path]
    metrics: list[str]

    @classmethod
    def from_studies(
        cls, studies: dict[str, Path]
    ) -> "BagOfWordsAnalysisConfig":
        # Parse, read parquets etc

    def get_config(
        self, study_name: str
    ) -> "config...":
        # Returns the config corresponding to the study

    def get_metric_dataframe(
        self
    ) -> pl.DataFrame:
        """
        Vertically joins all the metrics parquets, following columns:
        - study (study key above)
        - epoch
        - {all metric columns}
        """

    def xy_plot(
        self,
        x_axis: str,
        y_axis: str,
    ) -> go.Figure:
        """
        Makes xy-line-plot with point marker emphasis
        """