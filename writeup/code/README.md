# writeup figure-generation scripts

Thin drivers that render interactive plotly HTMLs into `writeup/assets/bow/`.
They reuse library code from `src/` without modifying it.

Shared defaults live in `_common.py`. Both figures expose two sliders
(`snr_halflife_in_*_quantile` and `corr`); because pure-plotly sliders operate
independently, trace visibility is coordinated by the `coordinated_slider_js`
post-script, which keys off each trace's `meta` tag.

## Regenerating the assets

From the repo root:

```bash
uv run python -m writeup.code.row_heteroskedasticity
uv run python -m writeup.code.word_heteroskedasticity
```

Figures scale to container width in-page via the `.scaled-plotly-frame` CSS +
JS helper in `writeup/styles.css` and `writeup/bag-of-words.qmd`.
