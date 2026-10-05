#!/usr/bin/env bash
# vLLM lives in its own venv: it pins its own torch/transformers, which must not touch the
# evaluation-pinned .venv used for hidden-state extraction.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv "$root/.venv-vllm" --python 3.12
uv pip install --python "$root/.venv-vllm/bin/python" vllm
"$root/.venv-vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
