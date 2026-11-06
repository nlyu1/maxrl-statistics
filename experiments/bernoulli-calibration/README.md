# Bernoulli bag-of-words calibration

Each length-16 context samples uniformly from 32 words. Four words each have
teacher values -2, -1, +1, +2; sixteen fillers have value zero. The latent
probability is `sigmoid(sum(word_values) / 4)`. Labels are fresh Bernoulli draws.
Training, validation and test word-count vectors are disjoint. Training prefixes
of 8192, 16384 and 32768 bags share fixed 8192-bag validation and test sets.

The model is trained from scratch: 32-by-128 embeddings, normalized sum pooling,
Linear(128,32), GELU, Linear(32,1). There are 8257 parameters. All methods use the
same per-seed initialization, minibatch order, and separate label/action random
generators. CUDA execution uses device contexts and bfloat16 autocast; loss
arithmetic and model probabilities use float32. AdamW has zero weight decay;
there is no entropy bonus, KL penalty, dropout, gradient clipping or LR schedule.

## Methods and grouping

- CE, scalar MSE and scalar MAE apply losses to individual binary labels before
  averaging. Scalar predictions represent probabilities, not sampled actions.
- RLOO, GRPO and binary MaxRL sample Bernoulli actions and reward label equality.
- Shared: one observed label for all N actions.
- Split: N independent labels, grouped by label for advantage calculation. The
  loss averages all N trajectories, retaining subgroup-size weights. Empty
  groups contribute zero. GRPO and centered MaxRL give zero singleton updates;
  singleton RLOO uses an unbaselined REINFORCE score update.

Rollouts are 2, 4, 8, 16, 64, 256. Supervised methods receive the same labels as
RL within each protocol. Shared-label supervised runs are reused across N;
split-label supervised baselines retain matching label counts. Independent label
draws and policy rollouts are recorded separately. Randomized rounding is not
part of the initial matrix.

## Selection and artifacts

The initial LR grid is 1e-4, 3e-4, 1e-3, 3e-3, 1e-2. Pilots use seed 42,
256 updates and batches of 256 contexts. Select by the final validation loss
native to each method: Bernoulli NLL for CE, expected binary-label MSE/MAE for
scalar regression, expected mismatch for RLOO/GRPO, and negative finite-compute
log success for MaxRL. GRPO is ranked by its underlying reward, not its detached
normalized policy-gradient surrogate. Calibration RMSE never selects rates.

For the centered, zero-success-skipping MaxRL estimator, define
`J_D(s) = -sum((1-s)**k / k, k=1..D)` with `D=N-1`. Shared validation loss is
`-p*J_D(q) - (1-p)*J_D(1-q)`. Independent split-label validation loss is
`-p*J_D(p*q) - (1-p)*J_D((1-p)*(1-q))`. These are the exact expected finite-N
objectives, including random subgroup sizes. Exact-enumeration tests verify
their gradients. The split objective tends to CE plus a teacher-dependent
constant as N increases. At small N, probability matching is not guaranteed.

Main runs use seeds 42,43,44 and 2048 updates. Evaluate full training/validation
sets every 32 updates, including step zero. Retain final and best-native-loss
checkpoints, and evaluate both on test only after training. Save native loss,
probability MAE/RMSE, Pearson with a defined flag, label NLL, excess NLL, entropy,
true entropy, expected reward, and sampled subgroup/update diagnostics.

The automatic training-size check uses shared-label CE/MSE only. Increase to
the next prefix if at least two seeds have validation RMSE more than 0.02 above
training RMSE and over 1.5 times training RMSE, or if validation RMSE seed SD
exceeds 0.02. This diagnostic never selects learning rates or checkpoints, and
does not treat the expected sharpening of MAE/RL as overfitting. Each increased
size reruns the whole comparison and LR sweep; validation/test sets stay fixed.

Artifacts live outside the repository. Each run retains config, JSON/Parquet
metric histories, result metadata and, for main runs, checkpoints and test
probabilities. Reports include all seed results, full curves, calibration
scatters and rollout comparisons. Jobs per GPU are explicitly configurable.

```bash
uv run python -m src.experiments.bernoulli_calibration.study \
  --root ~/.local/state/maxrl-statistics/bernoulli-calibration-20261107 \
  --gpu 1 --jobs-per-gpu 4
uv run python -m src.experiments.bernoulli_calibration.report \
  --root ~/.local/state/maxrl-statistics/bernoulli-calibration-20261107
```

Run Python, tests and experiments on the server as specified in `server.md`.
