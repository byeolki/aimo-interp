#!/usr/bin/env bash
# Score the counterfactual submission on the 34 organizer-labeled small-track rows, offline.
# The method never sees labels, so this is a held-out check against the official labeling.
# Usage: benchmark.sh <zip> [time-limit-seconds]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AIMO_TIME_LIMIT_S="${2:-3600}"
mkdir -p runs/03-counterfactual
started=$(date +%s)
.venv/bin/python third_party/getting-started/scripts/run_local.py "$1" --small \
  --input-dir runs/contract-test/input --reference-dir runs/contract-test/reference \
  2> runs/03-counterfactual/stderr.txt | tr -d '\n '
echo " wall $(( $(date +%s) - started ))s"
grep -a -E "budgets|failed|fallback" runs/03-counterfactual/stderr.txt || true
