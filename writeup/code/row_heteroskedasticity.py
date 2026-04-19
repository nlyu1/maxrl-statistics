"""Row-level per-sample correlation histogram with two sliders.

Slider axes:
- `snr_halflife_in_quantile` (linear, 9 steps)
- `corr` (log-scaled labels, 16 steps over [0.01, 1.0])

`a_i = tilde_a(u_i) / sqrt(E[tilde_a^2])` depends only on halflife, so we
precompute it per halflife and recombine across corrs analytically:
    rho_i = corr / sqrt(corr^2 + (1 - corr^2) * a_i^2).

Visibility is coordinated by a JS post-script that reads both sliders; each
trace is tagged with `meta = {kind: 'hist', h, c}`.

Writes `writeup/assets/bow/row_heteroskedasticity.html`.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.data.heterogeneous_bag_of_words import RowHeterogeneousBagOfWordsDatasetConfig
from writeup.code._common import (
    CORRS,
    HALFLIVES,
    NUM_SAMPLES,
    coordinated_slider_js,
    default_indices,
)

OUTPUT = (
    Path(__file__).resolve().parents[1] / "assets" / "bow" / "row_heteroskedasticity.html"
)

SOURCE_WIDTH = 900
SOURCE_HEIGHT = 620


def a_per_halflife(*, halflife: float) -> np.ndarray:
    u = (np.arange(NUM_SAMPLES) + 0.5) / NUM_SAMPLES
    tilde_a = RowHeterogeneousBagOfWordsDatasetConfig.raw_noise_schedule(
        quantiles=u, snr_halflife_in_quantile=halflife,
    )
    return tilde_a / np.sqrt(np.mean(tilde_a**2))


def build_figure() -> go.Figure:
    h_default, c_default = default_indices()
    edges = np.linspace(0.0, 1.0, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()
    for i, h in enumerate(HALFLIVES):
        a = a_per_halflife(halflife=h)
        a_sq = a**2
        for j, rho0 in enumerate(CORRS):
            rho = rho0 / np.sqrt(rho0**2 + (1 - rho0**2) * a_sq)
            counts, _ = np.histogram(rho, bins=edges)
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=edges[1] - edges[0],
                    visible=(i == h_default and j == c_default),
                    marker=dict(color="#2e6fb7"),
                    hovertemplate="rho=%{x:.2f}<br>count=%{y}<extra></extra>",
                    showlegend=False,
                    meta=dict(kind="hist", h=i, c=j),
                )
            )

    halflife_steps = [
        dict(method="skip", label=f"{h:g}") for h in HALFLIVES
    ]
    corr_steps = [
        dict(method="skip", label=f"{c:g}") for c in CORRS
    ]

    fig.update_layout(
        title=dict(
            text="per-row corr(signal, target)",
            font=dict(size=14),
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(title="count"),
        bargap=0.02,
        sliders=[
            dict(
                active=h_default,
                currentvalue=dict(
                    prefix="snr_halflife_in_quantile = ", font=dict(size=12)
                ),
                pad=dict(t=10, b=0),
                len=0.9,
                y=-0.05,
                steps=halflife_steps,
            ),
            dict(
                active=c_default,
                currentvalue=dict(prefix="corr = ", font=dict(size=12)),
                pad=dict(t=10, b=0),
                len=0.9,
                y=-0.22,
                steps=corr_steps,
            ),
        ],
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        margin=dict(l=60, r=30, t=35, b=95),
    )
    return fig


def main() -> None:
    fig = build_figure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=coordinated_slider_js(),
    )
    print(f"wrote {OUTPUT} (source {SOURCE_WIDTH}x{SOURCE_HEIGHT})")


if __name__ == "__main__":
    main()
