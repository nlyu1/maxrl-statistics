import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor


def token_length_distribution_plot(
    train_lengths: Tensor,
    val_lengths: Tensor,
    filter_threshold: int | None = None,
) -> go.Figure:
    def value_counts(t: Tensor) -> tuple:
        vals, cnts = torch.unique(t, return_counts=True)
        return vals.numpy(), cnts.numpy()

    tx, ty = value_counts(train_lengths)
    vx, vy = value_counts(val_lengths)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=tx, y=ty, mode="lines+markers", name="train"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=vx, y=vy, mode="lines+markers", name="val"),
        secondary_y=True,
    )
    if filter_threshold is not None:
        fig.add_vline(x=filter_threshold, line_dash="dash", annotation_text="threshold")
    return fig
