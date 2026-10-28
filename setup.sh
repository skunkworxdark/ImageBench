#!/usr/bin/env bash
# ImageBench - one-time environment setup (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "[ERROR] Python 3.11+ not found"; exit 1; }

[ -x .venv/bin/python ] || { echo "[setup] Creating .venv with $PY"; "$PY" -m venv .venv; }
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "[setup] Done. Start the dashboard with: ./run.sh"
