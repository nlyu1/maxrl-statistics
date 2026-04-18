#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STUDY_BASE="${REPO_ROOT}/artifacts/bow-grpo-coarse-sweep"

mkdir -p "${STUDY_BASE}/logs"

cd "${REPO_ROOT}"

# coarse_sweep.py tees its own stdout/stderr to ${STUDY_BASE}/logs/device_<id>.log
uv run python experiments/bow/grpo-corr/coarse_sweep.py 0 &
PID0=$!
uv run python experiments/bow/grpo-corr/coarse_sweep.py 1 &
PID1=$!

wait "${PID0}" "${PID1}"
