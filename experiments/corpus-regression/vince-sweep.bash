#!/usr/bin/env bash
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ORCHESTRATE="experiments/corpus-regression/orchestrate.py"
SEEDS="51,61,121"
LABEL="token_id"
ROLLOUT_STEPS=16,256
NUM_SAMPLES=500000
GAUSSIAN_STDEV=1.0
# FROM_PRETRAIN=False
# FROM_SCRATCH=True
FLAT=True
COSINE=True
LEARNING_RATES=2e-5,1e-5
LR_MIN_RATIO=0.1
BATCH_SIZE=256

METHOD_ARGS=(
    --method maxrl \
    --method sl_mse \
    --method sl_ce \
    --method grpo \
    --method rloo \
)

SEED_ARGS=(
    --seeds "${SEEDS}"
)

SCHEDULER_ARGS=(
    --lr-schedule cosine \
    --warmup-ratio 0.05 \
    --lr-min-ratio ${LR_MIN_RATIO} \
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
    --train-steps 1500 \
    --val-every-n-steps 50 \
    --batch-size ${BATCH_SIZE} \
    --lookforward-tokens 1 \
    --rollout-steps ${ROLLOUT_STEPS} \
    --gaussian-stdev ${GAUSSIAN_STDEV} \
    --lr-per-sample ${LEARNING_RATES} \
    --num-samples ${NUM_SAMPLES} \
    --subtract-baseline \
    --use-factorized-likelihoods \
    --factorized \
    --train-from-scratch \
)

ARGS=(
    ${METHOD_ARGS[@]}
    ${SEED_ARGS[@]}
    ${LABEL_ARGS[@]}
    ${TRAIN_ARGS[@]}
)

ARGS_WITH_SCHEDULER=(
    ${METHOD_ARGS[@]}
    ${SEED_ARGS[@]}
    ${LABEL_ARGS[@]}
    ${TRAIN_ARGS[@]}
    ${SCHEDULER_ARGS[@]}
)

if [ "${FLAT}" = "True" ] && [ "${COSINE}" = "False" ]; then
    uv run python "${ORCHESTRATE}" \
        "${ARGS[@]}" 
fi

if [ "${COSINE}" = "True" ] && [ "${FLAT}" = "False" ]; then
    uv run python "${ORCHESTRATE}" \
        "${ARGS_WITH_SCHEDULER[@]}" 
fi

if [ "${COSINE}" = "True" ] && [ "${FLAT}" = "True" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python "${ORCHESTRATE}" \
        "${ARGS[@]}" &
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python "${ORCHESTRATE}" \
        "${ARGS_WITH_SCHEDULER[@]}" &
    wait
fi