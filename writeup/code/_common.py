"""Shared constants and helpers for heteroskedasticity visualizations."""

HALFLIVES: tuple[float, ...] = (
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0,
)

# Log-spaced correlations on [0.01, 1.0]; mirror the 16-point canonical grid
# but with a clean geometric progression so the slider itself reads as log.
CORRS: tuple[float, ...] = (
    0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0,
)

NUM_SAMPLES: int = 50_000
PROMPT_LENGTH: int = 128
AUX_WORDS_RATIO: float = 0.5
WORD_DECAY_POWER: float = 1.0
NUM_SEMANTIC_WORDS: int = 15
RNG_SEED: int = 0

DEFAULT_HALFLIFE: float = 0.5
DEFAULT_CORR: float = 0.3


def default_indices() -> tuple[int, int]:
    h_idx = HALFLIVES.index(DEFAULT_HALFLIFE)
    c_idx = min(
        range(len(CORRS)),
        key=lambda i: abs(CORRS[i] - DEFAULT_CORR),
    )
    return h_idx, c_idx


def coordinated_slider_js() -> str:
    """Post-script JS that keeps trace visibility in sync with two sliders.

    Each trace must carry `meta = {"kind": "hist"|"hardness"|"density",
    "h": h_idx, "c": c_idx (hist only)}`. Density traces are always visible;
    hardness traces respond to the halflife slider; histogram traces respond
    to both. Uses plotly's `{plot_id}` substitution.
    """
    return """
var gd = document.getElementById('{plot_id}');

function syncVisibility() {
    var h = gd.layout.sliders[0].active;
    var c = gd.layout.sliders[1].active;
    var vis = gd.data.map(function(trace) {
        var m = trace.meta || {};
        if (m.kind === 'hist') return (m.h === h && m.c === c);
        if (m.kind === 'hardness') return (m.h === h);
        return true;
    });
    Plotly.restyle(gd, {visible: vis});
}

gd.on('plotly_sliderchange', syncVisibility);
syncVisibility();
"""
