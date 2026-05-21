# Corpus-regression experiments

Each method (`sl/`, `grpo/`, `maxrl/`, `rloo/`) has a `single_run.py` (one training run). The unified `orchestrate.py` sweeps the full grid (`seeds × lookforward_tokens × rollouts`) dynamically across all available GPUs.

All methods write to a shared artifacts root:

```
artifacts/corpus-regression/artifacts/{sl,grpo,maxrl,rloo}/
```

## Full Sweep

```bash
experiments/corpus-regression/sweep.bash
```

## Unified Orchestrator

```bash
# All defaults (all 8 lookforward tokens, all rollout steps, all GPUs)
uv run python experiments/corpus-regression/orchestrate.py \
    --method sl --seeds 51,61,121

# Custom grid, specific GPUs, custom sample count
uv run python experiments/corpus-regression/orchestrate.py \
    --method grpo --seeds 51 \
    --lookforward-tokens 1,4,8 \
    --rollout-steps 16,1024 \
    --num-samples 50000 \
    --train-epochs 5 \
    --gpu-ids 0,1,2,3

# MaxRL with method-specific flags
uv run python experiments/corpus-regression/orchestrate.py \
    --method maxrl --seeds 51,61 \
    --subtract-baseline --use-factorized-likelihoods

# RLOO with factorized advantages
uv run python experiments/corpus-regression/orchestrate.py \
    --method rloo --seeds 51 --factorized

# Dry run (print job list without executing)
uv run python experiments/corpus-regression/orchestrate.py \
    --method grpo --seeds 51 --dry-run

# Fail-fast mode (abort on first failure)
uv run python experiments/corpus-regression/orchestrate.py \
    --method sl --seeds 51 --fail-fast
```

## Single Experiment

```bash
uv run python experiments/corpus-regression/sl/single_run.py \
    --num-lookforward-tokens 4 --seed 51 --device cuda:0 --num-samples 50000
uv run python experiments/corpus-regression/grpo/single_run.py \
    --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0
uv run python experiments/corpus-regression/maxrl/single_run.py \
    --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0 \
    --subtract-baseline True --use-factorized-likelihoods True
uv run python experiments/corpus-regression/rloo/single_run.py \
    --num-lookforward-tokens 4 --num-rollouts 64 --seed 51 --device cuda:0 \
    --factorized True
```

## Orchestrator Arguments

| Argument | Type | Default | Notes |
|----------|------|---------|-------|
| `--method` | choice | required | `sl`, `grpo`, `rloo`, `maxrl` |
| `--seeds` | int list | required | e.g. `51,61,121` |
| `--lookforward-tokens` | int list | `1,2,3,4,5,6,7,8` | Sweep grid |
| `--rollout-steps` | int list | `4,16,128,1024` | Ignored for SL |
| `--num-samples` | int | `100000` | Dataset size |
| `--train-epochs` | int | `5` | Epochs per run |
| `--gpu-ids` | int list | all visible | e.g. `0,1,2,3` |
| `--subtract-baseline` | flag | True | MaxRL only |
| `--use-factorized-likelihoods` | flag | True | MaxRL only |
| `--factorized` | flag | True | RLOO only |
| `--dry-run` | flag | off | Print jobs, don't run |
| `--fail-fast` | flag | off | Stop on first failure |

## Analysis

Run `analyze.ipynb` top-to-bottom to visualize and save the experiment artifacts.
