# Bag-of-words experiments

Each flavor (`sl-corr/`, `grpo-corr/`, `maxrl-corr/`) has a `single_run.py` (one training run) and an `orchestrate.py` (one `(dataset, seed)` fanned across `cuda:0`/`cuda:1`). Seed iteration lives in the bash drivers. Dataset choices: `homoskedastic`, `row_heteroskedastic`, `word_heteroskedastic`.

## Sweeps

```bash
experiments/bow/sweep_homoskedastic.bash
experiments/bow/sweep_row_heteroskedastic.bash
experiments/bow/sweep_word_heteroskedastic.bash
```

## Single orchestrator call

```bash
uv run python experiments/bow/sl-corr/orchestrate.py \
    --dataset homoskedastic --seed 51
uv run python experiments/bow/grpo-corr/orchestrate.py \
    --dataset word_heteroskedastic --seed 51
uv run python experiments/bow/maxrl-corr/orchestrate.py \
    --dataset row_heteroskedastic --seed 51 --subtract-baseline True
```

Add `--dry-run` to print per-device job lists.

## Single experiment

```bash
uv run python experiments/bow/sl-corr/single_run.py \
    --dataset homoskedastic --corr 0.22 --seed 51 --device cuda:0
uv run python experiments/bow/grpo-corr/single_run.py \
    --dataset homoskedastic --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0
uv run python experiments/bow/maxrl-corr/single_run.py \
    --dataset homoskedastic --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0 \
    --subtract-baseline True
```

## Analysis

Run `analyze.ipynb` top-to-bottom to visualize and save the experiment artifacts.
