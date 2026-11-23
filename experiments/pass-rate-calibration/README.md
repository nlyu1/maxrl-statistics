# MaxRL pass-rate calibration

Predict, from the problem text alone, the per-problem pass rate p(x) of the
official MaxRL Qwen3-1.7B checkpoint. This scales the
[Bernoulli calibration study](../bernoulli-calibration/README.md) from a
bag-of-words model to pretrained and reasoning LLM predictors.

Status: roadmap. The first implementation covers Phases 0-2. Phase 3 starts
only after review.

## Target

The solver is `ftajwar/qwen3_1.7B_Base_MaxRL_Polaris_1000_steps`, trained from
Qwen3-1.7B-Base on POLARIS (arXiv:2602.02710). Rollout settings follow the MaxRL
repository, `tajwarfahim/maxrl` at commit 7197bbb:

- Prompt: the checkpoint's chat template, one user turn, no system turn, no think
  block. The problem is followed by the literal
  `"\nPlease reason step by step, and put your final answer within \\boxed{{}}."`
- Sampling: temperature 0.6, top-p 0.95, top-k off, 4096 new tokens. Stop on
  token ids 151643 and 151645.
- Grading: a copy of `verl/utils/reward_score/math_verify.py`. The gold answer is
  wrapped in `\boxed{}` and checked with `math-verify` 0.9.0, with no extra
  tolerance. Truncated responses count as wrong.

Each problem gets K = 256 rollouts, and the ground truth is p̂(x) = mean@256.
Training labels are fresh Bernoulli(p̂) draws at every problem visit.

| Split | Source | Problems |
|---|---|---|
| train | `math-ai/minervamath` test, random | 240 |
| val | same, random, fixed seed | 32 |
| OOD test | `daman1209arora/jeebench_numeric` test (math, phy, chem) | 137 |

JEEBench is reported with subjects mixed, plus a per-subject breakdown as a
diagnostic.

## Predictor

Every method family starts from `Qwen/Qwen3-8B`. Qwen3.5-4B is the Phase 1
alternative. The predictor sees only the problem, in a single user turn:

````
An independent small 1.7B base model will try to solve the following problem. You should predict whether that model will get a correct answer. Do not solve the question yourself. Respond with exactly one word: Yes or No.

Problem to be solved by the other model:
```
{problem}
```
````

The prompt pilot in the status section chose this prompt. The solver is
RL-trained from a base model, so "base model" is a deliberate simplification.
Phases 1 and 2 used the original prompt:

```
A specific AI model will attempt the math problem below once, reasoning step by step before giving a final answer. Predict whether its final answer will be correct. Respond with exactly one word: Yes or No.

Problem:
{problem}
```

Reasoning mode:
- Thinking is on, with at most 4096 completion tokens. Sampling temperature is
  1.0 for both training and evaluation.
- The answer is the text after `</think>` and must be exactly `Yes` or `No`.
  Overlong and malformed responses get zero reward.
- q(x) is the fraction of Yes among valid rollouts. Abort and format-failure
  rates are reported separately.
- Each rollout logs P(answer | reasoning) at the answer position. Values near 1
  mean the reasoning decides the answer.

The non-reasoning readout used by the linear head takes the same prompt with the
thinking-disabled template. It reads the hidden state after the final norm, at
the first answer position.

## Method families

| Family | Reasoning | Output | Training | Phase |
|---|---|---|---|---|
| Linear head, frozen backbone | no | σ(w·h + b) | {CE, MSE} × {binary labels, soft p̂} | 2 |
| Linear head, full fine-tune | no | σ(w·h + b) | same four | 2 |
| Binary RL | yes | Yes/No after reasoning | GRPO, RLOO, MaxRL-shared, MaxRL-split | 3 |
| Scalar RL | yes | verbalized pass rate | GRPO, RLOO, TailRL | 4 |

- The head is initialized with w = lm_head[Yes] - lm_head[No] and b = 0. Its
  initial q is therefore the base model's two-token Yes probability.
- Binary-label supervised runs draw one fresh label per problem visit.
- Shared: one label per problem visit, scored against all N rollouts.
- Split: N independent labels, with advantages computed within label subgroups,
  as in the toy study.
- GRPO and RLOO use one shared label.

## Phase 0: ground truth (server L40s)

1. Run vLLM rollouts for 409 problems × 256 at the settings above, with one engine
   per idle GPU. Store per-rollout text, token count, finish reason and
   correctness. Store per-problem p̂ and truncation rate.
2. Gate: pass@1 should be close to the paper's plotted values, about 0.29-0.30 on
   Minerva and 0.25 on JEEBench (approximate, read off figures). A clear miss
   stops work until the settings are reconciled.
3. Draw the 240/32 split, then plot:
   - p̂ histograms for train, val, and JEEBench (mixed and per subject);
   - solver response-length histograms by dataset, correct vs. incorrect, with
     4096-token truncations marked.
4. Choose the Phase 3 N grid. Run
   `bernoulli_calibration.distribution_shift.pointwise_maxrl_probability` on the
   training p̂ for N in {8, 16, 32, 64, 128}, shared and split. Tabulate the
   optimum's RMSE and MAE, and the fraction of problems the optimum sends to 0 or 1.

## Phase 1: predictor pilot (Tinker)

Compare Qwen3-8B (`qwen3` renderer) with Qwen3.5-4B (`qwen3_5`), thinking on.
Run 32 training problems × 8 samples per model, about $2 in total. Report:
- reasoning-length histograms;
- abort and format-failure rates;
- the untrained Yes rate and P(answer | reasoning);
- time per request and cost per rollout.

Outputs: the model choice, confirmation of the 4096-token cap, and the
learning-rate protocol for Phases 2-3.

## Phase 2: linear-head baselines (server L40s)

The four losses are {CE, MSE} on q × {fresh binary labels, soft p̂}. Each run
uses 16 problems per step for 300 steps and logs to W&B.

- Frozen backbone: cache hidden states for all 409 problems once, then train
  only the head.
- Full fine-tune: queued after the frozen runs; overfitting is intentional.
  Qwen3-8B with fp32 master weights and AdamW states needs about 131 GB. That
  means FSDP on 4 L40s, or on 2 with optimizer CPU offload, under bf16 autocast.

## Phase 3: binary RL with reasoning (Tinker, after review)

Configs:
- GRPO and RLOO at N = 16.
- MaxRL-shared and MaxRL-split at N in {16, 64, 128}, revised after Phase 0.

Training setup:
- 16 problems per step for 300 steps (20 epochs), one seed, no KL penalty.
- Advantages are computed client-side with the toy estimators and passed to
  Tinker's `importance_sampling` loss.
- Tinker trains LoRA only, with rank 32 by default. Thinking Machines report
  that LoRA matches full fine-tuning for policy-gradient RL even at low rank;
  external replications are mixed.

Evaluation: val and JEEBench every 30 steps with 16 samples per problem. The
full training set at steps 0, 150 and 300.

Implementation (`rl.py`):
- One synchronous on-policy step:
  - sample N completions for each of 16 problems from the current LoRA weights;
  - reward is 1 when a valid Yes/No matches a fresh Bernoulli(p̂) label;
  - advantages come from `bernoulli_calibration.objectives.grouped_advantages`;
  - one `importance_sampling` forward-backward on the rollouts with nonzero
    advantage, then one Adam step (Tinker defaults: β = (0.9, 0.95), no weight
    decay, no clipping).
- The loss is the toy objective: the mean over rollouts of A · log π(completion).
  Token log-probabilities are summed within a completion, with no length
  normalization. Adam makes the global scale irrelevant.
- Problem order and labels are a function of (seed, step) alone. Every config
  sees the same batches and the same shared labels, and a resumed run continues
  exactly.
- Evaluation uses 16 samples per problem.
  - Training, train and val evaluations use the 4096-token cap. JEEBench
    evaluation uses 8192, because 35% of base-model JEEBench traces hit 4096,
    mostly while solving the problem themselves.
  - Every 15 steps (one epoch) the run evaluates the 32 val problems and a fixed
    48-problem JEEBench subset (`test_subset/*`). The subset is stratified by
    subject (21 math, 12 phy, 15 chem) and by p̂ tercile within each subject.
  - The full train, val and JEEBench sets (`train_eval/*`, `test/*`) are
    evaluated at step 0, which is the shared base model, and at the last step.
  - Each epoch covers the training set once, so the on-policy rollouts also give
    a free train-set estimate (`train_epoch/*`).
- Every 75 steps, and at the last step, the run saves sampler weights and a full
  Tinker state with optimizer moments. Evaluations at other steps use temporary
  sampler snapshots, which don't count against checkpoint storage. Rerunning a
  sweep with a longer horizon resumes from the last checkpoint, so a 30-step
  pilot extends to 300 steps without restarting.
- Tinker's free checkpoint storage appears to be shared across the organization.
  Saving every 15 steps filled it at step 150 of the 300-step runs (102
  checkpoints, 75 GB). The superseded checkpoints were deleted with the user's
  approval; the manifest is `rl/deleted-checkpoints-*.json`.
- W&B logs, per run (group `phase3-<sweep>`):
  - rollout diagnostics: reward, Yes rate, abort and malformed rates, length,
    nonzero-advantage and no-success group fractions, batch bias;
  - trainer-vs-sampler log-probability gap, cost and time per phase;
  - evaluation metrics, plus metrics without the e-notation golds, distance to
    the objective's own pointwise optimum, and dead-zone collapse.

Estimated cost on Qwen3-8B, at $0.195 prefill, $0.60 sampling and $0.44 training
per million tokens. Each rollout is assumed to use 350 prompt and 3000
completion tokens, about $0.0033; replace this with the Phase 1 measurement.

| N | Per step | Per config, 300 steps | With evaluation |
|---|---|---|---|
| 16 | $0.84 | $251 | $326 |
| 32 | $1.68 | $503 | $578 |
| 64 | $3.35 | $1,005 | $1,080 |
| 128 | $6.70 | $2,011 | $2,086 |

The proposed 8-config matrix costs about $7.6k, or $3.5k without N = 128. Cost
scales linearly with mean completion length: about 1.35× if responses run to the
4096 cap, and 1.68× on Qwen3.5-4B at equal length.

## Phase 4 (later)

Scalar RL that predicts a verbalized pass rate (GRPO, RLOO, TailRL). This
includes a proper-score reward on P(Yes | reasoning).

## Metrics

Primary metrics on train, val and JEEBench:
- MAE, RMSE, Pearson and Spearman of q against p̂.
- Expected reward, q·p̂ + (1 - q)(1 - p̂). A 0/1 threshold maximizes it, so it
  is reported but never used for selection.

Reference predictors: the constant at the training-set mean p̂, and the
pointwise native optimum q*_N(p̂).

Per-step diagnostics, logged to W&B:
- sampled reward;
- fractions of groups with nonzero advantage and with no success;
- empty and singleton subgroup fractions (split only);
- completion-length mean, p90 and histogram;
- abort and format-failure rates;
- mean P(answer | reasoning);
- fraction of problems with q in {0, 1};
- dead-zone collapse: among problems with 0 < p̂ < 1 whose pointwise optimum is
  0 or 1, the fraction already at that boundary;
- bias, mean q - mean p̂, to track the out-of-distribution base-rate shift;
- time per step, and tokens and dollars per step.

Per-problem q is kept at every evaluation.

## Code and artifacts

Code lives in `src/experiments/pass_rate_calibration/` and reuses:
- `bernoulli_calibration.objectives`
- `bernoulli_calibration.distribution_shift.pointwise_maxrl_probability`
- `utkface.advantages`
- `gpu_pool`

New dependencies: `math-verify==0.9.0`, `tinker`, `tinker-cookbook` (for
renderers), and `vllm`. vLLM may need its own environment if its `transformers`
pin conflicts with `transformers>=5.5.3`.

Artifacts go under `~/.local/state/maxrl-statistics/pass-rate-calibration-<date>/`.

Secrets (`TINKER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`) come from the gitignored
`.env` at the repository root. Code runs on the server (see `server.md`), so the
`.env` must be copied there out of band, never through Git.

## Status (2026-11-20)

**Phase 0 is done.**
- The gate passed: pass@1 is 0.312 on Minerva (paper about 0.295) and 0.248 on
  JEEBench (paper about 0.25).
- The training prior is 0.300. Of the 240 training problems, 88 have p̂ = 0
  and 10 have p̂ = 1.
- The random validation set is easier, with mean 0.403.
- JEEBench has mean 0.248: math 0.316, phy 0.205, chem 0.185.
- Truncation at 4096 tokens is 2.0% on Minerva and 8.3% on JEEBench.
- Grader artifacts are inherited by design. Examples are rounded gold values
  and the gold answer `np.arcsin(10/13)`.
- The largest artifact: `math_verify` parses e-notation golds such as `3.83e35`
  as 3.83·e·35, with e as Euler's number. Only the literal string then scores.
  - 53 of 240 training golds and 5 of 32 validation golds use e-notation, and
    52 of the 53 training ones have p̂ = 0. JEEBench has none.
  - Regrading them against `a \times 10^{b}` would raise the training prior to
    0.343 and cut the training problems with p̂ = 0 from 88 to 51.
  - The grader stays unchanged for now.

Pointwise optimum RMSE on the training p̂:

| Method | N = 16 | N = 64 | N = 128 |
|---|---|---|---|
| MaxRL-shared | 0.019 | 0.003 | 0.002 |
| MaxRL-split | 0.078 | 0.035 | 0.022 |
| RLOO/GRPO | 0.167 | 0.167 | 0.167 |

The constant training prior gives RMSE 0.376 on train.

**Phase 1 is done, and Qwen3-8B is the predictor.**
- Thinking traces average 2292 tokens. At the 4096 cap, 26% of rollouts abort
  and 3% are malformed.
- The untrained predictor answers Yes 98.9% of the time: mean q is 0.99 against
  mean p̂ 0.36, with Pearson -0.06.
- P(answer | reasoning) averages 0.999.
- Qwen3.5-4B averages 3720 tokens, and 71% of its rollouts abort.
- Projected Phase 3 cost per 300-step config is $190, $758 and $1,515 at
  N = 16, 64 and 128.
- A shared Tinker seed with `num_samples > 1` returns identical samples, so
  never pass one.

**Phase 2 is done.**
- Frozen heads: 24 runs.
- Full fine-tunes: 8 runs, FSDP2 on 2 L40s with CPU offload, about 7.8 s per
  step. Every full run selected lr 3e-6.

Each row below is the run with the lowest final validation native loss:

| Run | Val MAE | Val Pearson | JEEBench MAE | JEEBench Pearson | JEEBench bias |
|---|---|---|---|---|---|
| Constant prior | 0.396 | - | 0.256 | - | +0.05 |
| Frozen CE, binary labels | 0.223 | 0.73 | 0.233 | 0.14 | -0.05 |
| Frozen CE, soft p̂ | 0.218 | 0.74 | 0.234 | 0.16 | -0.03 |
| Frozen MSE, binary labels | 0.291 | 0.45 | 0.368 | 0.29 | +0.23 |
| Frozen MSE, soft p̂ | 0.299 | 0.43 | 0.285 | 0.31 | +0.09 |
| Full CE, binary labels | 0.211 | 0.74 | 0.243 | 0.28 | +0.04 |
| Full CE, soft p̂ | 0.163 | 0.81 | 0.303 | 0.31 | +0.16 |
| Full MSE, binary labels | 0.216 | 0.70 | 0.223 | 0.29 | -0.01 |
| Full MSE, soft p̂ | 0.212 | 0.70 | 0.242 | 0.31 | +0.04 |

- **Frozen MSE saturates.** MSE through the sigmoid cannot leave the saturated
  Yes/No initialization: the median |logit| is 4.25, and 42% exceed 5. Frozen
  MSE collapses at lr ≥ 1e-3, and full MSE collapses to q = 0 at lr 1e-5.
- **Full fine-tuning improves JEEBench ranking but not absolute error.**
  Pearson rises from about 0.15 to 0.3. JEEBench MAE barely moves relative to
  the constant prior.
- **Validation and JEEBench disagree.** The validation set has mean 0.40 and
  JEEBench 0.25, so the best validation fit over-predicts JEEBench (full CE on
  soft p̂, bias +0.16). The 32-problem selection does not transfer.

**The prompt pilot replaced the predictor prompt.**
- Setup: Qwen3-8B with thinking on, a 4096-token cap and temperature 1.0, on the
  Phase 1 training problems.
- Why: under the original prompt, the traces solve the problem and check their
  own derivation. Nos appear only when the problem itself looks defective.
- Outputs are in `$ROOT/pilot-variants/`.

| Prompt | Problems × samples | Yes rate | Spearman vs p̂ | Median tokens | Capped |
|---|---|---|---|---|---|
| Original | 32 × 8 | 0.99 | 0.03 | 1,901 | 26% |
| Original, answer in `\boxed{}` | 32 × 8 | 0.99 | 0.00 | 2,672 | 33% |
| "Independent 1.7B model", "without solving" | 16 × 4 | 0.93 | 0.10 | 1,298 | 11% |
| Same, plus "solves about 30%", exact match | 16 × 4 | 0.22 | 0.17 | 2,540 | 20% |
| "Independent small 1.7B model", "Do not solve" | 32 × 4 | 0.92 | 0.09 | 1,073 | 6% |
| "Independent small 1.7B base model", "Do not solve" (current) | 32 × 8 | 0.83 | 0.50 | 1,165 | 6% |

- **The current prompt ranks problems.**
  - The permutation p is 0.004, and the bootstrap 95% interval is 0.21-0.72.
  - Mean q is 0.67 for p̂ < 0.01, 0.81 for 0.01 ≤ p̂ < 0.5, and 1.0 for
    p̂ ≥ 0.5.
  - Its Nos cite missing domain knowledge, while mechanical calculations get Yes.
- **It is still biased upward.** Mean q is 0.81 against a mean p̂ of 0.31. MAE
  is 0.50, against 0.34 for the constant prior.
- **The ranking needs replication.** Six prompts were compared on the same
  problems, and the result has not yet been tested on fresh problems.
- **RL signal is dense from the start.** At N = 16, about 94% of groups whose
  label is No contain at least one No (12% under the original prompt).
- **It is cheaper.** The projected cost is about $130 per 300-step N = 16
  config, down from $190.

**Phase 3 learning-rate pilot (2026-11-21).**
- Setup: RLOO, MaxRL-shared and MaxRL-split at lr ∈ {5e-6, 1e-5, 3e-5, 1e-4}.
  All runs use N = 16 and 30 steps, with shared batches; shared-label runs also
  share labels. Cost: $235 for the twelve runs plus $12.7 for the shared base
  evaluations.
- Results: the outputs are in `$ROOT/rl/lr-pilot.md` and the W&B group
  `phase3-lr-pilot`. At 8192 tokens the base model hits the cap on 7% of
  JEEBench traces (35% at 4096), and JEEBench Spearman barely moves (0.45 → 0.47).

| Run (step 30) | Yes rate | Val MAE | Val ρ | JEE MAE | JEE ρ | JEE bias |
|---|---|---|---|---|---|---|
| Base model (step 0) | 0.79 | 0.497 | 0.39 | 0.466 | 0.47 | +0.45 |
| RLOO 5e-6 | 0.81 | 0.462 | 0.48 | 0.426 | 0.46 | +0.40 |
| RLOO 1e-5 | 0.70 | 0.358 | 0.48 | 0.250 | 0.48 | +0.14 |
| RLOO 3e-5 | 0.01 (collapsed) | 0.403 | – | 0.248 | – | −0.25 |
| RLOO 1e-4 | 0.00 (collapsed by step 11) | 0.403 | – | 0.248 | – | −0.25 |
| MaxRL 5e-6 | 0.80 | 0.484 | 0.48 | 0.433 | 0.45 | +0.40 |
| MaxRL 1e-5 | 0.66 | 0.328 | 0.52 | 0.236 | 0.45 | +0.10 |
| MaxRL 3e-5 | 0.05 (collapsing) | 0.407 | 0.23 | 0.246 | 0.02 | −0.24 |
| MaxRL 1e-4 | 0.30 (dipped to 0.09, recovered) | 0.276 | 0.49 | 0.209 | 0.41 | −0.17 |
| MaxRL-split 5e-6 | 0.80 | 0.450 | 0.37 | 0.418 | 0.53 | +0.40 |
| MaxRL-split 1e-5 | 0.56 | 0.277 | 0.61 | 0.219 | 0.41 | +0.04 |
| MaxRL-split 3e-5 | 0.21 (dipped to 0.07, recovered) | 0.289 | 0.52 | 0.228 | 0.25 | −0.21 |
| MaxRL-split 1e-4 | 0.38 (dipped to 0.09, recovered) | 0.254 | 0.48 | 0.197 | 0.42 | −0.11 |

- **5e-6 barely moves in 30 steps.** The Yes rate stays at 0.72-0.81 and the
  val bias at +0.42 to +0.48, for every method.
- **lr 1e-5 is stable for all three methods.** MaxRL-split corrects the bias
  fastest at that rate (val bias +0.18, against +0.24 for shared and +0.30 for
  RLOO). The split runs draw different labels from the shared runs, and each
  configuration is a single seed.
- **Split labels resist collapse.**
  - No MaxRL-split run collapsed. At 3e-5, the Yes rate bottomed at 0.07 and
    the nonzero-advantage fraction stayed at or above 0.38. MaxRL-shared at 3e-5
    fell to Yes 0.01-0.05.
  - MaxRL-split at 1e-4 has the best val MAE (0.254, bias −0.02) and JEEBench
    MAE (0.197) of all twelve runs.
  - So far no run shows the toy study's split dead zone.
  - Each configuration is a single seed.
- **Ranking:** on the full training set, Spearman rises from 0.35 to about
  0.45-0.52 at lr ≥ 1e-5. JEEBench Spearman stays at 0.41-0.53 for every run,
  around the base model's 0.47.

- **Collapse to all-No is absorbing.** Once every rollout in a group gives the
  same answer, every advantage is zero, so nothing pushes back. The completions
  also shrink to about 500 tokens of quick Nos.
- **MaxRL can recover.** It gives a rare correct Yes a weight of up to N − 1 = 15,
  against about 1 under RLOO. This is a single seed.
- **JEEBench MAE rewards collapse.** An all-No predictor scores 0.248 against
  0.256 for the constant prior, so rank by Spearman and bias.
- **The fixed 48-problem subset reads JEEBench ranking low.** Its step-0
  Spearman is 0.28 against 0.47 for the full set, about 2 sd. The offset is
  constant, but subset ρ is not JEEBench ρ.

**Phase 3, 300 steps at lr 1e-5 (2026-11-21).**
- Setup: RLOO, MaxRL-shared and MaxRL-split, extended from the pilot's step-30
  checkpoints, one seed each.
- Cost: $119, $144 and $142. The outputs are in `$ROOT/rl/long.md` and the W&B
  group `phase3-long`.
- Step 300 is scored on the full sets. JEEBench uses the 8192-token cap. The
  constant predictor at the training prior scores val MAE 0.396 (RMSE 0.434)
  and JEEBench MAE 0.256 (RMSE 0.294).

| Full set, step 300 | Base | RLOO | MaxRL-shared | MaxRL-split |
|---|---|---|---|---|
| Train MAE / RMSE | 0.501 / 0.613 | 0.204 / 0.328 | 0.225 / 0.319 | 0.216 / 0.317 |
| Train Spearman / Pearson | 0.35 / 0.33 | 0.51 / 0.58 | 0.53 / 0.57 | 0.52 / 0.58 |
| Train bias | +0.47 | −0.09 | 0.00 | −0.05 |
| Train fraction with q ∈ {0, 1} | 0.42 | 0.51 | 0.26 | 0.31 |
| Val MAE / RMSE / Spearman | 0.497 / 0.621 / 0.39 | 0.253 / 0.381 / 0.56 | 0.279 / 0.375 / 0.45 | 0.252 / 0.355 / 0.49 |
| Val bias | +0.49 | −0.12 | −0.05 | −0.07 |
| JEEBench MAE / RMSE | 0.466 / 0.546 | 0.228 / 0.346 | 0.193 / 0.282 | 0.205 / 0.304 |
| JEEBench Spearman / Pearson | 0.47 / 0.42 | 0.42 / 0.40 | 0.48 / 0.49 | 0.48 / 0.43 |
| JEEBench mean q (mean p̂ 0.25) | 0.69 | 0.03 | 0.12 | 0.09 |
| JEEBench answer entropy (bits) | – | 0.13 | 0.41 | 0.31 |

- **All three methods remove the base-rate bias and improve in-distribution
  ranking.** Train Spearman rises from 0.35 to about 0.52. Most of the gain is
  in place by step 60-120.
- **The methods differ in sharpness, in the order the toy study predicts.**
  - RLOO pushes the most problems to q ∈ {0, 1} and has the lowest answer
    entropy. On JEEBench it says No to almost everything.
  - MaxRL-shared stays the most probabilistic and is unbiased on train.
  - MaxRL-split lies in between.
  - All three remain far from their own pointwise optima (optimum MAE about
    0.23), so model capacity dominates the error, not objective bias.
- **Dynamics.**
  - RLOO fell into a low-Yes regime (Yes about 0.07, steps 61-150) and then
    partly recovered to about 0.2.
  - Over the last 30 steps, the Yes rate on Yes-labelled versus No-labelled
    rollouts is 0.41 vs 0.10 for RLOO, 0.53 vs 0.21 for MaxRL-shared, and
    0.45 vs 0.17 for MaxRL-split.
  - Token entropy drifts up for RLOO (0.40 → 0.44) and down for MaxRL-split
    (0.39 → 0.34). Traces shorten from about 1,560 to 965-1,120 tokens, and the
    abort rate falls to about 1%.
- **JEEBench ranking does not improve.** Spearman stays at 0.42-0.48 against
  0.47 for the base model. Every method under-predicts JEEBench, with bias
  −0.13 to −0.22. Only MaxRL-shared beats the constant prior on JEEBench RMSE
  (0.282 vs 0.294).
- **The e-notation artifacts lower ranking rather than help it.** Without them,
  train Spearman is 0.58-0.61. There is no sign of a shortcut that predicts No
  from scientific-notation answers.
- **Comparison with the frozen CE head (new prompt):** the head scores val
  MAE 0.227 and val Spearman 0.69, but JEEBench MAE 0.291 and Spearman 0.26.
  Reasoning RL transfers better to JEEBench, while the head (selected on val)
  fits Minerva better.
- **Caveats.** These are single seeds, and val (32 problems) and the 48-problem
  JEEBench subset are noisy from one evaluation to the next. RLOO repeated steps
  166-179 after a storage-limit crash; W&B keeps the first copy of those steps.

**Phase 3, round 2 (2026-11-23).**
- Setup:
  - The three lr 1e-5 runs continued from step 300 to 450.
  - Fresh runs (`-r1`, LoRA init seed + 1, same batches and labels): MaxRL-shared
    and MaxRL-split at 3e-5, and RLOO and GRPO at 1e-5, all 300 steps.
  - Each run kept only its latest checkpoint.
- Cost: $617. The outputs are in `$ROOT/rl/round2.md` and the W&B group
  `phase3-round2`.
- Constant predictor at the training prior: train RMSE 0.376, val MAE 0.396 /
  RMSE 0.434, JEEBench MAE 0.256 / RMSE 0.294.

| Full set | Train MAE / RMSE / ρ / bias | Val MAE / ρ | JEEBench MAE / RMSE / ρ / Pearson / bias |
|---|---|---|---|
| Base | 0.501 / 0.613 / 0.35 / +0.47 | 0.497 / 0.39 | 0.466 / 0.546 / 0.47 / 0.42 / +0.45 |
| RLOO r0, 300 | 0.204 / 0.328 / 0.51 / −0.09 | 0.253 / 0.56 | 0.228 / 0.346 / 0.42 / 0.40 / −0.22 |
| RLOO r1, 300 | 0.210 / 0.335 / 0.51 / −0.12 | 0.295 / 0.45 | 0.229 / 0.353 / 0.36 / 0.40 / −0.23 |
| GRPO r1, 300 | 0.206 / 0.317 / 0.52 / −0.07 | 0.231 / 0.51 | 0.220 / 0.332 / 0.37 / 0.41 / −0.20 |
| MaxRL-shared 1e-5, 300 | 0.225 / 0.319 / 0.53 / 0.00 | 0.279 / 0.45 | 0.193 / 0.282 / 0.48 / 0.49 / −0.13 |
| MaxRL-split 1e-5, 300 | 0.216 / 0.317 / 0.52 / −0.05 | 0.252 / 0.49 | 0.205 / 0.304 / 0.48 / 0.43 / −0.16 |
| **MaxRL-shared 3e-5 r1, 300** | 0.189 / 0.277 / 0.64 / +0.05 | 0.237 / 0.60 | 0.180 / 0.262 / 0.55 / 0.53 / −0.05 |
| **MaxRL-split 3e-5 r1, 300** | 0.187 / 0.274 / 0.65 / +0.03 | 0.212 / 0.61 | 0.167 / 0.246 / 0.56 / 0.65 / −0.11 |
| RLOO r0, 450 | 0.184 / 0.315 / 0.58 / −0.10 | 0.273 / 0.45 | 0.235 / 0.358 / 0.27 / 0.28 / −0.23 |
| MaxRL-shared 1e-5, 450 | 0.206 / 0.304 / 0.56 / −0.05 | 0.235 / 0.54 | 0.203 / 0.300 / 0.40 / 0.44 / −0.15 |
| MaxRL-split 1e-5, 450 | 0.203 / 0.295 / 0.56 / −0.03 | 0.255 / 0.52 | 0.209 / 0.312 / 0.43 / 0.43 / −0.17 |

- **Seed noise (RLOO r0 vs r1).**
  - Train metrics and JEEBench MAE/RMSE reproduce within about 0.01.
  - Val metrics (32 problems) and Spearman vary by up to about 0.1.
  - So compare runs on the full train set and on JEEBench RMSE or Pearson.
- **MaxRL at 3e-5 is the best setting so far.**
  - Against 1e-5, train Spearman rises by +0.11 to +0.13 and JEEBench RMSE
    falls by 0.02 to 0.06. These gaps are several times the seed noise.
  - JEEBench Spearman improves over the base model (0.55-0.56 vs 0.47) for the
    first time.
  - Predictions are less compressed: problems with p̂ ≈ 0.83 get q 0.65-0.68,
    against about 0.5 at 1e-5.
  - Neither fresh run collapsed. The pilot's MaxRL-shared at 3e-5 (another
    seed) collapsed by step 25, so collapse at 3e-5 depends on the seed.
- **GRPO behaves like RLOO.** It dipped deeper mid-run (Yes 0.04-0.06 over
  steps 61-150, against 0.13-0.15 for RLOO r1) and ends slightly less sharp.
  Its edge over RLOO r1 (JEEBench RMSE 0.332 vs 0.353) is about 3× the seed
  gap. Both stay sharper and more biased than MaxRL at the same learning rate.
- **Training longer at 1e-5 overfits Minerva.**
  - From step 300 to 450, train Spearman rises (+0.03 to +0.08) while
    JEEBench Spearman falls: RLOO 0.42 → 0.27, MaxRL-shared 0.48 → 0.40,
    MaxRL-split 0.48 → 0.43.
  - RLOO keeps sharpening toward its 0/1 target: 65% of training problems sit
    at q ∈ {0, 1}, and JEEBench answer entropy is 0.09 bits.

## Commands

Run these on the server from a checkout of this branch. The solver worker runs
in an isolated `uv` environment with vLLM 0.32.0, which keeps it separate from
the project's torch and transformers pins. GPUs are claimed through
`~/.local/state/maxrl-statistics/gpu-claims`.

```bash
ROOT=~/.local/state/maxrl-statistics/pass-rate-calibration-20261120
M=src.experiments.pass_rate_calibration
uv run python -m $M.solver launch --root $ROOT --gpus 6 7
uv run python -m $M.solver grade --root $ROOT
uv run python -m $M.ground_truth --root $ROOT
uv run python -m $M.pilot run --root $ROOT --models qwen3-8b qwen3.5-4b
uv run python -m $M.pilot report --root $ROOT --models qwen3-8b qwen3.5-4b --with-labels
uv run python -m $M.head features --root $ROOT
uv run python -m $M.head train-frozen --root $ROOT
uv run python -m $M.head queue-full --root $ROOT --gpus 2 --cpu-offload \
  --identifiers full-ce-soft-lr3e-06
uv run python -m $M.head report --root $ROOT
uv run python -m $M.head compare --root $ROOT --reference $OLD_ROOT
uv run python -m $M.rl launch --root $ROOT --sweep lr-pilot
uv run python -m $M.rl report --root $ROOT --sweep lr-pilot
```

`features` and `queue-full` wait for free GPUs. `queue-full` defaults to all
eight full fine-tunes. `rl launch` runs the shared base evaluation if it is
missing, then starts one process per unfinished run and writes
`$ROOT/rl/<sweep>.md`. Secrets come from `set -a; . ./.env; set +a`.

Reports are written to `$ROOT/ground-truth.md`, `$ROOT/pilot/pilot.md` and
`$ROOT/heads/phase2.md`. Training curves are logged to the W&B project
`maxrl-pass-rate-calibration`.

Head features, head results and pilot outputs record the predictor prompt. The
head commands refuse artifacts made with a different prompt. To rerun Phases 1-2
under a new prompt, start a fresh `ROOT` and copy `problems.parquet` and
`labels.parquet` into it from the Phase 0 root.

## Open decisions

- Learning-rate protocol: after Phase 1.
- N grid: after Phase 0.
- Whether GRPO and RLOO run at N = 16 only.
- Predictor model: after Phase 1.
- Phase 2 under the current prompt (root `pass-rate-calibration-20261120-small-base`):
  all 24 frozen heads were rerun. Train and val barely change. JEEBench MAE is
  slightly worse (0.29 vs 0.23) and JEEBench Pearson about 0.20. The full
  fine-tune reruns were dropped.
