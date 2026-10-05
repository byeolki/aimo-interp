#!/usr/bin/env bash
# Full experiment 01 on a GPU host: labels -> features for all four models -> CV analysis.
# Long-running; launch inside tmux and read runs/01-layer-probe/log.txt.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
mkdir -p runs/01-layer-probe

uv run scripts/fetch_labels.py
uv run experiments/01-layer-probe/extract.py
uv run experiments/01-layer-probe/analyze.py
