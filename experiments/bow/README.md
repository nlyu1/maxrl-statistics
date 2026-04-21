# Bag-of-words experiments

Each flavor (`sl-corr/`, `grpo-corr/`, `maxrl-corr/`) has two scripts:

- `single_run.py` — one training run. Resumable: loudly skips if the study folder already has `metrics.parquet` for all epochs; wipes and retries otherwise. Manually delete the study folder to overwrite.
- `orchestrate.py` — hard-codes `cuda:0` and `cuda:1`, partitions work across both GPUs, spawns one `single_run.py` subprocess per job. Seeds are outermost with a cross-device barrier.

Canonical grids live in [`src/data/bag_of_words.py`](../../src/data/bag_of_words.py).

## Run the full sweeps

```bash
uv run python experiments/bow/sl-corr/orchestrate.py
uv run python experiments/bow/grpo-corr/orchestrate.py
...
```

Add `--dry-run` to print the per-device job lists and exit without training.
For MaxRL, choose exactly one of `--subtract-baseline` or `--no-subtract-baseline`.

## Run a single experiment

```bash
# SL: (corr, seed)
uv run python experiments/bow/sl-corr/single_run.py \
    --corr 0.22 --seed 51 --device cuda:0

# GRPO: (corr, num_rollouts, seed)
uv run python experiments/bow/grpo-corr/single_run.py \
    --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0

# MaxRL: (corr, num_rollouts, seed)
uv run python experiments/bow/maxrl-corr/single_run.py \
    --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0 \
    --subtract-baseline True
```

## Analysis

Run `analyze.ipynb` top-to-bottom to visualize and save the experiment artifacts.