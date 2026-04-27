#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOTEBOOK="${REPO_ROOT}/experiments/corpus-regression/analyze.ipynb"
ASSETS_DIR="${REPO_ROOT}/writeup/assets/corpus-regression"

export PATH="/home/nlyu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HOME="${HOME:-/home/nlyu}"

cd "${REPO_ROOT}"

echo "[$(date -Is)] executing notebook"
uv run jupyter nbconvert --to notebook --execute --inplace "${NOTEBOOK}"

git add "${NOTEBOOK}" "${ASSETS_DIR}"

if git diff --staged --quiet; then
    echo "[$(date -Is)] no changes to commit"
    exit 0
fi

echo "[$(date -Is)] committing"
git commit -m "updates"

echo "[$(date -Is)] pushing"
git push origin main

echo "[$(date -Is)] done"
