#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

uv run python experiments/bow/sl-corr/orchestrate.py
uv run python experiments/bow/grpo-corr/orchestrate.py
