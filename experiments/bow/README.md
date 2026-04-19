# Bag-of-words experiments

Each flavor (`sl-corr/`, `grpo-corr/`, `maxrl-corr/`) has two scripts:

- `single_run.py` — one training run. Resumable: loudly skips if the study folder already has `metrics.parquet` for all epochs; wipes and retries otherwise. Manually delete the study folder to overwrite.
- `orchestrate.py` — hard-codes `cuda:0` and `cuda:1`, partitions work across both GPUs, spawns one `single_run.py` subprocess per job. Seeds are outermost with a cross-device barrier.

Canonical grids live in [`src/data/bag_of_words.py`](../../src/data/bag_of_words.py).

## Run the full sweeps

```bash
uv run python experiments/bow/sl-corr/orchestrate.py
uv run python experiments/bow/grpo-corr/orchestrate.py
uv run python experiments/bow/maxrl-corr/orchestrate.py
```

Add `--dry-run` to print the per-device job lists and exit without training.

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
    --corr 0.22 --num-rollouts 64 --seed 51 --device cuda:0
```

Both accept `--train-epochs INT` (default 20).

## Artifact layout

```
artifacts/
├── bow-data/{dataset_name}/                       # parquets, shared across seeds
├── bow-sl-sweep/seed-{S}/{dataset_name}/
└── bow-{grpo/maxrl}-sweep/seed-{S}/rollouts-{N}/{dataset_name}/
```

`{dataset_name} = {W}-words_corr-{c}_len-{L}_pow-{p}_ar-{a}`. Each study folder contains `config.json`, `metrics.parquet`, and `{epoch}/validation.parquet`.

## Sharding strategy

Both orchestrators enforce: finish seed *k* on both devices before starting seed *k+1*.

- **GRPO / MaxRL** — both devices traverse all 16 corrs (disjoint rollout halves, no collision). GPU0 walks stride-4 top-down × `rollouts[:3]` natural; GPU1 walks stride-4 bottom-up × `rollouts[3:]` reversed. Prioritizes rollout-axis coverage first, then corr.
- **SL** — corrs are partitioned: GPU0 gets `corrs[:8]` stride-4 top-down, GPU1 gets `corrs[8:]` stride-4 bottom-up.

Stride-4 ordering means if a run is interrupted early, both ends of the corr range are still sampled — good coverage-under-time.
