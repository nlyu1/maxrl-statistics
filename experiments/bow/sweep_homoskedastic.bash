#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="homoskedastic"
SEEDS=(51 61 121 153 228)

for seed in "${SEEDS[@]}"; do
    uv run python experiments/bow/maxrl-corr/orchestrate.py \
        --dataset "${DATASET}" --seed "${seed}" --subtract-baseline True
    uv run python experiments/bow/grpo-corr/orchestrate.py \
        --dataset "${DATASET}" --seed "${seed}"
    uv run python experiments/bow/sl-corr/orchestrate.py \
        --dataset "${DATASET}" --seed "${seed}"
done
