#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

# SL sweep (token_id)
uv run python "${ORCHESTRATE}" \
    --method sl_mse \
    --method grpo \
    --method rloo \
    --method maxrl \
    --method pretrained_baseline \
    --seeds "${SEEDS}" \
    --label-type token_id \
    --lookforward-tokens 1 \
    --rollout-steps 1024 \
    --gaussian-stdev 1.0,9.0,100.0,900.0,10000.0 \
    --num-samples 50000,100000,500000 \
    --subtract-baseline \
    --use-factorized-likelihoods
