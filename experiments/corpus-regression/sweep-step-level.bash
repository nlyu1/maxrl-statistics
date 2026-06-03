#!/usr/bin/env bash
set -euo pipefail

# Production sweep using the new per-step logging regime.
# Runs all methods (SL, GRPO, RLOO, MaxRL, NTP baseline) with canonical
# step-level defaults: 10K train steps, validation every 2K steps.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

# uv run python "${ORCHESTRATE}" \
#     --method sl \
#     --method grpo \
#     --method rloo \
#     --method maxrl \
#     --method ntp_baseline \
#     --seeds "${SEEDS}" \
#     --train-steps 10000 \
#     --val-every-n-steps 1000 \
#     --lookforward-tokens 1 \
#     --rollout-steps 64,256,1024 \
#     --gaussian-stdev 1.0,0.5,0.1 \
#     --num-samples 100000,500000 \
#     --subtract-baseline \
#     --use-factorized-likelihoods \
#     --factorized

uv run python "${ORCHESTRATE}" \
    --method sl \
    --method grpo \
    --method rloo \
    --method maxrl \
    --method ntp_baseline \
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
    --factorized
