# Corpus-regression experiments

Each method (`sl/`, `grpo/`, `maxrl/`) has a `single_run.py` (one training run) and an `orchestrate.py` (sweeps `num_lookforward_tokens` × `num_rollouts` for one seed across `cuda:0`/`cuda:1`). Seed iteration lives in `sweep.bash`. `dataset.ipynb` exemplifies the dataset.

All methods write to a shared artifacts root:

```
artifacts/corpus-regression-sweep/{sl,grpo,maxrl}/
```

## Sweep

```bash
experiments/corpus-regression/sweep.bash
```

## Single orchestrator call

```bash
uv run python experiments/corpus-regression/sl/orchestrate.py --seed 51
uv run python experiments/corpus-regression/grpo/orchestrate.py --seed 51
uv run python experiments/corpus-regression/maxrl/orchestrate.py \
    --seed 51 --subtract-baseline True --use-factorized-likelihoods True
```

Add `--dry-run` to print per-device job lists.

## Single experiment

```bash
uv run python experiments/corpus-regression/sl/single_run.py \
    --num-lookforward-tokens 4 --seed 51 --device cuda:0
uv run python experiments/corpus-regression/grpo/single_run.py \
    --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0
uv run python experiments/corpus-regression/maxrl/single_run.py \
    --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0 \
    --subtract-baseline True --use-factorized-likelihoods True
```

## Analysis

Run `analyze.ipynb` top-to-bottom to visualize and save the experiment artifacts.
