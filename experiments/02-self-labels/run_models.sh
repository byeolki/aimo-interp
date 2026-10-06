#!/usr/bin/env bash
# Generate self-labels for the given models one after another, then extract their hidden
# states for the problem pool. Resumable: rerun after an interruptible instance is outbid.
# Usage: run_models.sh Qwen/Qwen3.5-4B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B ...
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export HF_HOME=/workspace/hf PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
mkdir -p runs/02-self-labels

for model in "$@"; do
  echo "=== generate $model $(date -u +%H:%M)"
  kv_dtype=fp8
  [[ "$model" == Skywork/* ]] && kv_dtype=auto
  .venv-vllm/bin/python experiments/02-self-labels/generate.py --model "$model" --problems data/problems_small.jsonl --kv-cache-dtype "$kv_dtype"
  echo "=== extract $model $(date -u +%H:%M)"
  .venv/bin/python experiments/01-layer-probe/extract.py --models "$model" \
    --problems data/problems_small.jsonl --output-dir runs/features-self
done
echo "=== all done $(date -u +%H:%M)"
