#!/usr/bin/env bash
set -euo pipefail

BUCKET="${1:?Need S3 bucket name}"
REPO_DIR="${2:-$HOME/news_pipelines}"

cd "$REPO_DIR"

aws s3 sync data/processed/pipeline_runs "s3://$BUCKET/pipeline_runs"