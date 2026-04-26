"""Render a self-contained interactive demo of the corpus-regression dataset.

Streams `NUM_SAMPLES` fresh fineweb-edu prefixes through the canonical config,
formats each one with the same per-sample / label HTML helpers used by the
Solara browser in `src/data/corpus_regression.py`, then bakes them into a
single static HTML page with a vanilla-JS +/- carousel. The resulting file is
iframe-embeddable from `writeup/corpus-regression.qmd` (no Python kernel
needed at view time).

Run from the repo root:

    uv run python -m writeup.code.corpus_regression_demo
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from src.data.corpus_regression import (
    CorpusRegressionDatasetConfig,
    _label_vector_html,
    _sample_body_html,
    _stream_token_prefixes,
)

NUM_SAMPLES: int = 50

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "corpus-regression"
    / "demonstrate.html"
)

SOURCE_WIDTH = 920
SOURCE_HEIGHT = 360


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Corpus regression — sample browser</title>
<style>
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: Georgia, serif;
    font-size: 14px;
    line-height: 1.5;
    color: #111;
    padding: 12px 16px;
    box-sizing: border-box;
  }}
  .controls {{
    display: flex;
    gap: 4px;
    align-items: center;
    margin-bottom: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
  }}
  .controls button {{
    width: 26px; height: 26px; padding: 0;
    font-size: 15px; line-height: 1;
    border: 1px solid #bbb; border-radius: 4px;
    background: #f5f5f5; cursor: pointer;
  }}
  .controls button:disabled {{ opacity: 0.4; cursor: default; }}
  .body {{
    white-space: pre-wrap;
    word-break: break-word;
  }}
  hr.sep {{ border: none; border-top: 1px solid #ddd; margin: 8px 0; }}
  .label {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
</style>
</head>
<body>
  <div class="controls">
    <button id="prev">−</button>
    <button id="next">+</button>
    <span>sample <b id="cur">1</b> / <span id="total"></span></span>
  </div>
  <div id="body" class="body"></div>
  <hr class="sep">
  <div id="label" class="label"></div>
<script>
  const SAMPLES = __SAMPLES_JSON__;
  let idx = 0;
  const $body = document.getElementById('body');
  const $label = document.getElementById('label');
  const $cur = document.getElementById('cur');
  const $prev = document.getElementById('prev');
  const $next = document.getElementById('next');
  document.getElementById('total').textContent = SAMPLES.length;
  function render() {{
    const s = SAMPLES[idx];
    $body.innerHTML = s.body;
    $label.innerHTML = s.label;
    $cur.textContent = idx + 1;
    $prev.disabled = (idx === 0);
    $next.disabled = (idx === SAMPLES.length - 1);
  }}
  $prev.addEventListener('click', () => {{ if (idx > 0) {{ idx--; render(); }} }});
  $next.addEventListener('click', () => {{ if (idx < SAMPLES.length - 1) {{ idx++; render(); }} }});
  render();
</script>
</body>
</html>
"""


def build_html(*, config: CorpusRegressionDatasetConfig, num_samples: int) -> str:
    tok = AutoTokenizer.from_pretrained(config.pretrained_tokenizer_model_name)
    rademacher = config._rademacher_matrix(len(tok))

    min_length = config.prefix_length + config.num_lookforward_tokens
    collected = _stream_token_prefixes(
        tokenizer=tok,
        min_length=min_length,
        target_count=num_samples,
        desc="streaming demo samples",
        overshoot=16,
    )
    last = config.prefix_length + config.num_lookforward_tokens - 1
    payload = []
    for ids in collected:
        sample = {
            "prefix_ids": ids[: config.prefix_length],
            "middle_ids": ids[config.prefix_length : last],
            "final_id": ids[last],
            "tail_ids": ids[last + 1 :],
            "label": rademacher[ids[last]].tolist(),
        }
        payload.append(
            {
                "body": _sample_body_html(tokenizer=tok, sample=sample),
                "label": _label_vector_html(sample["label"]),
            }
        )

    samples_json = json.dumps(payload)
    return _PAGE_TEMPLATE.replace("__SAMPLES_JSON__", samples_json)


def main() -> None:
    config = CorpusRegressionDatasetConfig.get_canonical()
    html = build_html(config=config, num_samples=NUM_SAMPLES)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT} ({len(html)} bytes; {NUM_SAMPLES} samples)")


if __name__ == "__main__":
    main()
