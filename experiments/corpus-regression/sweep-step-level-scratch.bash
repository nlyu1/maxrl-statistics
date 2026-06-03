#!/usr/bin/env bash
set -euo pipefail

# From-scratch sweep using the new per-step logging regime.
# Mirror of sweep-step-level.bash with two differences:
#   - --train-from-scratch appended (random-init backbone, lr_divisor forced
#     to 1.0, artifacts land under <method>_scratch/).
#   - ntp_baseline excluded (inference-only on pretrained weights, has no
#     meaningful from-scratch counterpart).
# Trains SL, GRPO, RLOO, MaxRL with canonical step-level defaults.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

# uv run python "${ORCHESTRATE}" \
#     --method sl \
#     --method grpo \
#     --method rloo \
#     --method maxrl \
#     --seeds "${SEEDS}" \
#     --train-steps 10000 \
#     --val-every-n-steps 1000 \
#     --lookforward-tokens 1 \
#     --rollout-steps 64,256,1024 \
#     --gaussian-stdev 1.0,0.5,0.1 \
#     --num-samples 100000,500000 \
#     --subtract-baseline \
#     --use-factorized-likelihoods \
#     --factorized \
#     --train-from-scratch

uv run python "${ORCHESTRATE}" \
    --method sl \
    --method grpo \
    --method rloo \
    --method maxrl \
    --seeds "${SEEDS}" \
    --label-type token_id \
    --normalize-labels \
    --label-range 0.0 1.0 \
    --train-steps 10000 \
    --val-every-n-steps 1000 \
    --lookforward-tokens 1 \
    --rollout-steps 1024 \
    --gaussian-stdev 1.0,0.5 \
    --num-samples 100000,500000 \
    --subtract-baseline \
    --use-factorized-likelihoods \
    --factorized \
    --train-from-scratch
