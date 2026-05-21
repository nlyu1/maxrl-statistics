#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121,153,228,19,143,151,181,210"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

uv run python "${ORCHESTRATE}" \
    --method maxrl \
    --seeds "${SEEDS}" \
    --subtract-baseline \
    --use-factorized-likelihoods

uv run python "${ORCHESTRATE}" \
    --method sl \
    --seeds "${SEEDS}"

uv run python "${ORCHESTRATE}" \
    --method grpo \
    --seeds "${SEEDS}"

uv run python "${ORCHESTRATE}" \
    --method rloo \
    --seeds "${SEEDS}" \
    --factorized
