#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

# SL sweep (token_id)
uv run python "${ORCHESTRATE}" \
    --method sl \
    --method grpo \
    --method rloo \
    --method maxrl \
    --seeds "${SEEDS}" \
    --label-type token_id \
    --lookforward-tokens 1 \
    --rollout-steps 16,128,1024 \
    --gaussian-stdev 1.0,4.0,9.0 \
    --num-samples 50000,100000,500000 \
    --subtract-baseline \
    --use-factorized-likelihoods

# NTP baseline (token_id) — single inference run
uv run python experiments/corpus-regression/ntp_baseline/single_run.py \
    --num-lookforward-tokens 1 \
    --device cuda:0 \
    --label-type token_id
