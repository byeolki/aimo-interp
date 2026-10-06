#!/usr/bin/env bash
# Run a submission through the official ingestion on the 34 organizer-labeled small-track rows,
# offline, and keep per-case predictions. The counterfactual method never sees labels, so this
# is a held-out check against the official labeling.
# Usage: benchmark.sh <submission-dir-or-zip> <out-name> [time-limit-seconds]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AIMO_TIME_LIMIT_S="${3:-3600}"
out="runs/03-counterfactual/$2"
rm -rf "$out" && mkdir -p "$out"
submission="$1"
if [[ "$submission" == *.zip ]]; then
  mkdir -p "$out/submission" && unzip -q "$submission" -d "$out/submission" && submission="$out/submission"
fi
started=$(date +%s)
.venv/bin/python third_party/getting-started/components/ingestion_program/ingestion.py \
  runs/contract-test/input "$out/predictions" "$submission" --track small 2> "$out/stderr.txt"
echo "wall $(( $(date +%s) - started ))s"
grep -a -E "budgets|failed" "$out/stderr.txt" | cut -c1-200 || true
.venv/bin/python experiments/03-counterfactual/score.py "$out/predictions/predictions.jsonl"
