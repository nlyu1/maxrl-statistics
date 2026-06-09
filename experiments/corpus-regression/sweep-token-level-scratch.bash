#!/usr/bin/env bash
set -euo pipefail

# Token-level (NTP cross-entropy) sweep. Counterpart to
# sweep-step-level-scratch.bash:
#   - Uses the new sl_ce trainer (CE on the lookahead token id; full-param
#     fine-tune of an AutoModelForCausalLM that retains its native lm_head).
#   - --train-from-scratch (random-init backbone; artifacts land under
#     sl_ce/from_scratch/ alongside the from-scratch sl_mse/grpo/rloo/maxrl trees).
#   - K=1 only — token-level CE only makes sense at the next-token horizon.
# Validation reuses the projection from pretrained_baseline.py:
#     softmax(logits) @ label_projector  →  MSE
# so the projected MSE is directly comparable to SL_MSE/GRPO/RLOO/MaxRL.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="51,61,121"
ORCHESTRATE="experiments/corpus-regression/orchestrate.py"

uv run python "${ORCHESTRATE}" \
    --method sl_ce \
    --seeds "${SEEDS}" \
    --label-type token_id \
    --normalize-labels \
    --label-range 0.0 1.0 \
    --train-steps 10000 \
    --val-every-n-steps 1000 \
    --lookforward-tokens 1 \
    --num-samples 100000,500000 \
    --train-from-scratch
