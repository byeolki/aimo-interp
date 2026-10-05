#!/usr/bin/env bash
# One-time setup on a fresh GPU host (vast.ai PyTorch template), run from the repo root.
# Installs uv and the pinned evaluation-runtime packages into .venv.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# The host's own PyTorch install must not leak into the pinned environment.
unset VIRTUAL_ENV
uv sync
uv run python -c "import torch, transformers, sklearn; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, sklearn.__version__)"
uv run python -m unittest discover -s tests
