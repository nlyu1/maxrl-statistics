"""Word-level heteroskedasticity figure: per-prompt correlation histogram plus
the density / hardness profile that induced it.

Two sliders:
- `snr_halflife_in_word_quantile` (9 steps) — drives both the histogram and
  the hardness bars.
- `corr` (16 log-spaced steps) — drives only the histogram.

The per-prompt prompt-variance multiplier A(x) is sampled once per halflife;
rho(x) at each corr is recombined analytically without re-sampling.

Writes `writeup/assets/bow/word_heteroskedasticity.html`.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.bag_of_words import _derived_fields, canonical_bags
from src.data.heterogeneous_bag_of_words import (
    SignalHeterogeneousBagOfWordsDatasetConfig,
)
from writeup.code._common import (
    AUX_WORDS_RATIO,
    CORRS,
    DEFAULT_CORR,
    DEFAULT_HALFLIFE,
    HALFLIVES,
    NUM_SAMPLES,
    NUM_SEMANTIC_WORDS,
    PROMPT_LENGTH,
    RNG_SEED,
    WORD_DECAY_POWER,
    coordinated_slider_js,
)

# Drop h=0.05 locally; word-level hardness at that halflife explodes visually.
WORD_HALFLIVES: tuple[float, ...] = tuple(h for h in HALFLIVES if h != 0.05)

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "bow"
    / "word_heteroskedasticity.html"
)

SOURCE_WIDTH = 1000
SOURCE_HEIGHT = 900


def _prepare_vocab() -> tuple[
    tuple[str, ...], list[str], np.ndarray, dict[str, int], dict[str, float]
]:
    semantic = tuple(canonical_bags[NUM_SEMANTIC_WORDS])
    density, values, _ = _derived_fields(
        word_assignments=semantic,
        aux_words_ratio=AUX_WORDS_RATIO,
        word_decay_power=WORD_DECAY_POWER,
    )
    words = list(density)
    probs = np.array([density[w] for w in words], dtype=float)
    return semantic, words, probs, values, density


def per_halflife_stats(
    *,
    halflife: float,
    semantic: tuple[str, ...],
    values: dict[str, int],
    density: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(A_sq, semantic_a_norm)` for one halflife.

    `A_sq[n]` is the RMS-normalized prompt variance multiplier (50k prompts);
    `semantic_a_norm[w]` is the per-semantic-word normalized hardness.
    """
    A_sq, normalized_mults = (
        SignalHeterogeneousBagOfWordsDatasetConfig.sample_prompt_variance_multiplier(
            word_assignments=semantic,
            word_values=values,
            word_density=density,
            snr_halflife_in_word_quantile=halflife,
            prompt_length=PROMPT_LENGTH,
            n=NUM_SAMPLES,
            rng=rng,
        )
    )
    semantic_a_norm = np.array([normalized_mults[w] for w in semantic])
    return A_sq, semantic_a_norm


def build_figure() -> go.Figure:
    semantic, words, probs, values, density = _prepare_vocab()
    semantic_density = np.array([probs[words.index(w)] for w in semantic])
    h_default = min(
        range(len(WORD_HALFLIVES)),
        key=lambda i: abs(WORD_HALFLIVES[i] - DEFAULT_HALFLIFE),
    )
    c_default = min(range(len(CORRS)), key=lambda i: abs(CORRS[i] - DEFAULT_CORR))
    rng = np.random.default_rng(RNG_SEED)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.10,
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=(
            "per-prompt corr(signal, target)",
            "Semantic-word density vs normalized hardness",
        ),
    )

    edges = np.linspace(0.0, 1.0, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hardness_by_halflife: list[np.ndarray] = []
    for i, h in enumerate(WORD_HALFLIVES):
        A_sq, semantic_a_norm = per_halflife_stats(
            halflife=h,
            semantic=semantic,
            values=values,
            density=density,
            rng=rng,
        )
        hardness_by_halflife.append(semantic_a_norm)
        for j, rho0 in enumerate(CORRS):
            if rho0 >= 1.0:
                rho = np.ones(NUM_SAMPLES)
            else:
                rho = rho0 / np.sqrt(rho0**2 + (1 - rho0**2) * A_sq)
            signal = rng.normal(0.0, rho0, size=NUM_SAMPLES)
            noise = (
                np.sqrt(max(1 - rho0**2, 0.0))
                * np.sqrt(A_sq)
                * rng.standard_normal(NUM_SAMPLES)
            )
            target = signal + noise
            mean_rho = float(np.mean(rho))
            max_abs_target = float(np.max(np.abs(target)))
            counts, _ = np.histogram(rho, bins=edges)
            title = (
                "per-prompt corr(signal, target) — "
                f"mean(ρ_j)={mean_rho:.3f}, max|target|={max_abs_target:.2f}"
            )
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=edges[1] - edges[0],
                    visible=(i == h_default and j == c_default),
                    marker=dict(color="#2e6fb7"),
                    showlegend=False,
                    hovertemplate="rho=%{x:.2f}<br>count=%{y}<extra></extra>",
                    meta=dict(kind="hist", h=i, c=j, title=title),
                ),
                row=1,
                col=1,
            )

    # Density bar — invariant across both sliders.
    x_positions = np.arange(len(semantic))
    bar_width = 0.4
    fig.add_trace(
        go.Bar(
            x=x_positions - bar_width / 2,
            y=semantic_density,
            width=bar_width,
            name="density p(w)",
            marker=dict(color="#8c8c8c"),
            opacity=0.9,
            showlegend=False,
            customdata=list(semantic),
            hovertemplate="%{customdata}<br>p=%{y:.4f}<extra></extra>",
            meta=dict(kind="density"),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Hardness bars — one per halflife (independent of corr).
    for i, (h, hardness) in enumerate(zip(WORD_HALFLIVES, hardness_by_halflife)):
        fig.add_trace(
            go.Bar(
                x=x_positions + bar_width / 2,
                y=hardness,
                width=bar_width,
                name=f"hardness a(w) | halflife={h}",
                marker=dict(color="#d9534f"),
                opacity=0.9,
                visible=(i == h_default),
                customdata=list(semantic),
                hovertemplate="%{customdata}<br>a(w)=%{y:.3f}<extra></extra>",
                showlegend=False,
                meta=dict(kind="hardness", h=i),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

    halflife_steps = [dict(method="skip", label=f"{h:g}") for h in WORD_HALFLIVES]
    corr_steps = [dict(method="skip", label=f"{c:g}") for c in CORRS]

    fig.update_xaxes(range=[0.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions),
        ticktext=list(semantic),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="density p(w)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="hardness a(w)",
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        barmode="overlay",
        bargap=0.02,
        sliders=[
            dict(
                active=h_default,
                currentvalue=dict(
                    prefix="snr_halflife_in_word_quantile = ", font=dict(size=12)
                ),
                pad=dict(t=10, b=0),
                len=0.9,
                y=-0.06,
                steps=halflife_steps,
            ),
            dict(
                active=c_default,
                currentvalue=dict(prefix="corr = ", font=dict(size=12)),
                pad=dict(t=10, b=0),
                len=0.9,
                y=-0.18,
                steps=corr_steps,
            ),
        ],
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        margin=dict(l=70, r=70, t=40, b=140),
    )
    return fig


def main() -> None:
    fig = build_figure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        OUTPUT,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=coordinated_slider_js(title_path="annotations[0].text"),
    )
    print(f"wrote {OUTPUT} (source {SOURCE_WIDTH}x{SOURCE_HEIGHT})")


if __name__ == "__main__":
    main()
