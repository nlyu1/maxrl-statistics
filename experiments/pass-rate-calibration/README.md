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
alternative. The predictor sees only the problem, in a single user turn
(draft prompt):

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

**Phase 2 frozen heads are done; the full fine-tunes are queued.** The queue
runs on 2 L40s with CPU offload, at about 7.7 s per step. Selected frozen runs:

| Run | Val MAE | Val Pearson | JEEBench MAE | JEEBench Pearson |
|---|---|---|---|---|
| Constant prior | 0.396 | - | 0.256 | - |
| CE, binary labels, lr 1e-3 | 0.223 | 0.73 | 0.233 | 0.14 |
| CE, soft p̂, lr 1e-3 | 0.218 | 0.74 | 0.234 | 0.16 |
| MSE, binary labels, lr 3e-4 | 0.291 | 0.45 | 0.368 | 0.29 |
| MSE, soft p̂, lr 3e-4 | 0.299 | 0.43 | 0.285 | 0.31 |

MSE through the sigmoid cannot leave the saturated Yes/No initialization. The
median |logit| is 4.25, and 42% of logits exceed 5. At lr ≥ 1e-3 every
prediction collapses to 0 or 1.

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
uv run python -m $M.head features --root $ROOT --gpu 7
uv run python -m $M.head train-frozen --root $ROOT
uv run python -m $M.head queue-full --root $ROOT --gpus 2 --cpu-offload
uv run python -m $M.head report --root $ROOT
```

Reports are written to `$ROOT/ground-truth.md`, `$ROOT/pilot/pilot.md` and
`$ROOT/heads/phase2.md`. Training curves are logged to the W&B project
`maxrl-pass-rate-calibration`.

## Open decisions

- Learning-rate protocol: after Phase 1.
- N grid: after Phase 0.
- Whether GRPO and RLOO run at N = 16 only.
- Predictor model: after Phase 1.
