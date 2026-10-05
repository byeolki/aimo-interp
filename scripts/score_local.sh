#!/usr/bin/env bash
# Score a built ZIP on the public val-sample with the official ingestion/scoring code.
# Usage: scripts/score_local.sh dist/always-true-small.zip
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
kit="$root/third_party/getting-started"
data="$root/data/val-sample"
zip_path="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

# The harness is stdlib-only, but solutions that import torch need the project venv
# (run `uv sync` first); the fallback only covers constant baselines.
python="$root/.venv/bin/python"
[[ -x "$python" ]] || python="python3"

# Data goes to the repo's ignored data/ so the pinned submodule stays clean.
if [[ ! -f "$data/input/cases.jsonl" ]]; then
  "$python" "$kit/scripts/import_hf_dataset.py" --output-dir "$data"
fi

track_flag=()
if unzip -Z1 "$zip_path" | grep -qix 'small.txt'; then
  track_flag=(--small)
fi

"$python" "$kit/scripts/run_local.py" "$zip_path" "${track_flag[@]}" \
  --input-dir "$data/input" --reference-dir "$data/reference"
