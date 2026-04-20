"""Row-level per-sample correlation histogram with two sliders.

Slider axes:
- `row_hardness_eta` (9-step canonical visualization grid)
- `corr` (log-scaled labels, 16 steps over [0.01, 1.0])

`a_i²` is drawn per (eta, corr) cell from the harmonic-beta law and empirically
renormalized so mean(a_i²) = 1 exactly on the cell. The beta parameters depend
on both eta and m = corr², so draws cannot be shared across corrs.

Titles carry empirical mean(ρ_j) and max(|target|); visibility and title text
are coordinated by a JS post-script that reads both sliders, with each trace
tagged as `meta = {kind: 'hist', h, c, title}`.

Writes `writeup/assets/bow/row_heteroskedasticity.html`.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.data.heterogeneous_bag_of_words import RowHeterogeneousBagOfWordsDatasetConfig
from writeup.code._common import (
    CORRS,
    DEFAULT_CORR,
    NUM_SAMPLES,
    RNG_SEED,
    coordinated_slider_js,
)

ETAS: tuple[float, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_ETA: float = 16.0

OUTPUT = (
    Path(__file__).resolve().parents[1] / "assets" / "bow" / "row_heteroskedasticity.html"
)

SOURCE_WIDTH = 900
SOURCE_HEIGHT = 620


def default_indices() -> tuple[int, int]:
    e_idx = ETAS.index(DEFAULT_ETA)
    c_idx = min(range(len(CORRS)), key=lambda i: abs(CORRS[i] - DEFAULT_CORR))
    return e_idx, c_idx


def a_sq_per_cell(
    *, corr: float, eta: float, rng: np.random.Generator, n: int
) -> np.ndarray:
    raw = RowHeterogeneousBagOfWordsDatasetConfig.raw_noise_schedule(
        corr=corr, row_hardness_eta=eta, rng=rng, n=n,
    )
    return raw / np.mean(raw)


def build_figure() -> go.Figure:
    e_default, c_default = default_indices()
    edges = np.linspace(0.0, 1.0, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(RNG_SEED)

    fig = go.Figure()
    for i, eta in enumerate(ETAS):
        for j, rho0 in enumerate(CORRS):
            a_sq = a_sq_per_cell(corr=rho0, eta=eta, rng=rng, n=NUM_SAMPLES)
            if rho0 >= 1.0:
                rho = np.ones(NUM_SAMPLES)
            else:
                rho = rho0 / np.sqrt(rho0**2 + (1 - rho0**2) * a_sq)
            signal = rng.normal(0.0, rho0, size=NUM_SAMPLES)
            noise = (
                np.sqrt(max(1 - rho0**2, 0.0))
                * np.sqrt(a_sq)
                * rng.standard_normal(NUM_SAMPLES)
            )
            target = signal + noise
            mean_rho = float(np.mean(rho))
            max_abs_target = float(np.max(np.abs(target)))
            counts, _ = np.histogram(rho, bins=edges)
            title = (
                "per-row corr(signal, target) — "
                f"mean(ρ_j)={mean_rho:.3f}, max|target|={max_abs_target:.2f}"
            )
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=edges[1] - edges[0],
                    visible=(i == e_default and j == c_default),
                    marker=dict(color="#2e6fb7"),
                    hovertemplate="rho=%{x:.2f}<br>count=%{y}<extra></extra>",
                    showlegend=False,
                    meta=dict(kind="hist", h=i, c=j, title=title),
                )
            )

    eta_steps = [
        dict(method="skip", label=f"{e:g}") for e in ETAS
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
                active=e_default,
                currentvalue=dict(
                    prefix="row_hardness_eta = ", font=dict(size=12)
                ),
                pad=dict(t=10, b=0),
                len=0.9,
                y=-0.05,
                steps=eta_steps,
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
