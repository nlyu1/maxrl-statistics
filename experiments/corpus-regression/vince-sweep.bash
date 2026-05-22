#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
LOOKFORWARD_TOKENS="1,2,4"
ROLLOUT_STEPS="128"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

NUM_SAMPLES_LIST=(100000 500000 1000000)
GAUSSIAN_STDEVS=(1.0 0.5 0.1)

for NUM_SAMPLES in "${NUM_SAMPLES_LIST[@]}"; do
    for STDEV in "${GAUSSIAN_STDEVS[@]}"; do
        echo "=== num_samples=${NUM_SAMPLES}  gaussian_stdev=${STDEV} ==="

        uv run python "${ORCHESTRATE}" \
            --method maxrl \
            --seeds "${SEEDS}" \
            --subtract-baseline \
            --use-factorized-likelihoods \
            --rollout-steps "${ROLLOUT_STEPS}" \
            --lookforward-tokens "${LOOKFORWARD_TOKENS}" \
            --num-samples "${NUM_SAMPLES}" \
            --gaussian-stdev "${STDEV}"

        uv run python "${ORCHESTRATE}" \
            --method sl \
            --seeds "${SEEDS}" \
            --lookforward-tokens "${LOOKFORWARD_TOKENS}" \
            --num-samples "${NUM_SAMPLES}" \
            --gaussian-stdev "${STDEV}"

        uv run python "${ORCHESTRATE}" \
            --method grpo \
            --seeds "${SEEDS}" \
            --rollout-steps "${ROLLOUT_STEPS}" \
            --lookforward-tokens "${LOOKFORWARD_TOKENS}" \
            --num-samples "${NUM_SAMPLES}" \
            --gaussian-stdev "${STDEV}"

        uv run python "${ORCHESTRATE}" \
            --method rloo \
            --seeds "${SEEDS}" \
            --factorized \
            --rollout-steps "${ROLLOUT_STEPS}" \
            --lookforward-tokens "${LOOKFORWARD_TOKENS}" \
            --num-samples "${NUM_SAMPLES}" \
            --gaussian-stdev "${STDEV}"
    done
done
