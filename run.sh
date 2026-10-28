#!/usr/bin/env bash
# ImageBench - launch the dashboard (runs setup automatically on first use)
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python -m streamlit run app.py "$@"
