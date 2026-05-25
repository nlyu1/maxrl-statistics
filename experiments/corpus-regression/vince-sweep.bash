#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

uv run python "${ORCHESTRATE}" \
    --method maxrl \
    --method sl \
    --method grpo \
    --method rloo \
    --seeds 51,61,121 \
    --lookforward-tokens 1,2,4 \
    --rollout-steps 128 \
    --num-samples 100000,500000,1000000 \
    --gaussian-stdev 1.0,0.5 \
    --subtract-baseline \
    --use-factorized-likelihoods \
    --factorized
