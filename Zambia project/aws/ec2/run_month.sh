#!/usr/bin/env bash
set -euo pipefail

START_DATE="${1:?Need START_DATE like 2025-12-01}"
END_DATE="${2:?Need END_DATE like 2025-12-31}"
REPO_DIR="${3:-$HOME/Zambia project}"

cd "$REPO_DIR"
source .venv/bin/activate

python pipeline/pipeline.py \
  --start-date "$START_DATE" \
  --end-date "$END_DATE"