#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(51 61 121 153 228 19 143 151 181 210)

for seed in "${SEEDS[@]}"; do
    uv run python experiments/corpus-regression/maxrl/orchestrate.py \
        --seed "${seed}" \
        --subtract-baseline True \
        --use-factorized-likelihoods True
    uv run python experiments/corpus-regression/sl/orchestrate.py \
        --seed "${seed}"
    uv run python experiments/corpus-regression/grpo/orchestrate.py \
        --seed "${seed}"
    uv run python experiments/corpus-regression/rloo/orchestrate.py \
        --seed "${seed}" \
        --factorized True
done
