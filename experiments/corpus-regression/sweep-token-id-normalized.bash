#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

# Same sweep as sweep-token-id.bash but with labels normalized to [0, 1].
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
    --lookforward-tokens 1 \
    --rollout-steps 1024 \
    --gaussian-stdev 1.0,0.5 \
    --num-samples 500000 \
    --subtract-baseline \
    --use-factorized-likelihoods
