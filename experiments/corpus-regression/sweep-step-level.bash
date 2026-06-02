#!/usr/bin/env bash
set -euo pipefail

# Production sweep using the new per-step logging regime.
# Runs all methods (SL, GRPO, RLOO, MaxRL, NTP baseline) with canonical
# step-level defaults: 10K train steps, validation every 2K steps.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121,153,228"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

uv run python "${ORCHESTRATE}" \
    --method sl \
    --method grpo \
    --method rloo \
    --method maxrl \
    --method ntp_baseline \
    --seeds "${SEEDS}" \
    --train-steps 10000 \
    --val-every-n-steps 2000 \
    --lookforward-tokens 1,2,4,8 \
    --rollout-steps 64,256,1024 \
    --gaussian-stdev 1.0 \
    --num-samples 100000 \
    --subtract-baseline \
    --use-factorized-likelihoods \
    --factorized
