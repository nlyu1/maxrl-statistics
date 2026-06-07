#!/usr/bin/env bash
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ORCHESTRATE="experiments/corpus-regression/orchestrate.py"
SEEDS="51,61,121"
LABEL="token_id"
ROLLOUT_STEPS=16,64,256
NUM_SAMPLES=100000,500000
GAUSSIAN_STDEV=1.0
FROM_PRETRAIN=True
FROM_SCRATCH=True
LEARNING_RATE=1e-4

METHOD_ARGS=(
    --method maxrl \
    --method sl \
    --method grpo \
    --method rloo \
    --method sl_ce \
)

SEED_ARGS=(
    --seeds "${SEEDS}"
)

if [ "${LABEL}" = "token_id" ]; then
    LABEL_ARGS=(
        --label-type token_id \
        --normalize-labels \
        --label-range 0.0 1.0 \
    )
fi

if [ "${LABEL}" = "rademacher" ]; then
    LABEL_ARGS=(
        --label-type rademacher \
    )
fi

TRAIN_ARGS=(
    --train-steps 10000 \
    --val-every-n-steps 500 \
    --lookforward-tokens 1 \
    --rollout-steps ${ROLLOUT_STEPS} \
    --gaussian-stdev ${GAUSSIAN_STDEV} \
    --num-samples ${NUM_SAMPLES} \
    --lr-per-sample ${LEARNING_RATE} \
    --subtract-baseline \
    --use-factorized-likelihoods \
    --factorized \
)

ARGS=(
    ${METHOD_ARGS[@]}
    ${SEED_ARGS[@]}
    ${LABEL_ARGS[@]}
    ${TRAIN_ARGS[@]}
)

if [ "${FROM_PRETRAIN}" = "True" ] && [ "${FROM_SCRATCH}" = "False" ]; then
    uv run python "${ORCHESTRATE}" \
        --method ntp_baseline \
        "${ARGS[@]}" 
fi

if [ "${FROM_SCRATCH}" = "True" ] && [ "${FROM_PRETRAIN}" = "False" ]; then
    uv run python "${ORCHESTRATE}" \
        "${ARGS[@]}" \
        --train-from-scratch
fi

if [ "${FROM_SCRATCH}" = "True" ] && [ "${FROM_PRETRAIN}" = "True" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python "${ORCHESTRATE}" \
        --method ntp_baseline \
        "${ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python "${ORCHESTRATE}" \
        "${ARGS[@]}" \
        --train-from-scratch &
    wait
fi