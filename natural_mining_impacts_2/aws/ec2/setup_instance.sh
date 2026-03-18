#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:?Need GitHub repo URL}"
REPO_DIR="${2:-$(basename "$REPO_URL" .git)}"

if command -v dnf >/dev/null 2>&1; then
  sudo dnf update -y
  sudo dnf install -y git python3 python3-pip
else
  sudo apt update -y
  sudo apt install -y git python3 python3-pip python3-venv
fi

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
echo "Now create a .env file with OPENAI_API_KEY."