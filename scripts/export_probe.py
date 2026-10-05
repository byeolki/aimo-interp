#!/usr/bin/env python3
"""Fit a shared-encoder probe on all labeled rows and write submissions/<name>/artifacts.

Usage:
  uv run scripts/export_probe.py --encoder deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --pooling last --layer 20 --name layer-probe

Choose encoder/pooling/layer from experiments/01-layer-probe CV results, never from the
leaderboard.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "01-layer-probe"))

from analyze import load_encoder_features, load_table  # noqa: E402

from aimo_interp.models import SMALL_TRACK_MODELS  # noqa: E402
from aimo_interp.probe import fit_probe  # noqa: E402
from aimo_interp.spec import WILDCARD, Head, Spec  # noqa: E402

PROBE_FILE = "shared_probe.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--pooling", choices=["last", "mean"], required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--name", default="layer-probe")
    args = parser.parse_args()

    table = load_table()
    rows = table.rows
    hidden = load_encoder_features(args.encoder, args.pooling, rows["problem_id"])[args.layer]
    features = np.hstack([hidden, table.model_onehot])
    probe = fit_probe(features, rows["is_robust"].to_numpy())

    artifact_dir = ROOT / "submissions" / args.name / "artifacts"
    probe.save(artifact_dir / PROBE_FILE)
    head = Head(args.encoder, args.pooling, args.layer, PROBE_FILE, uses_model_onehot=True)
    spec = Spec(
        model_order=list(SMALL_TRACK_MODELS),
        heads={WILDCARD: head},
        fallback_prediction=True,
        notes={"trained_rows": str(len(rows)), "train_acc": f"{np.mean(np.array(probe.predict(features)) == rows['is_robust'].to_numpy()):.3f}"},
    )
    spec.save(artifact_dir)
    print(f"wrote {artifact_dir.relative_to(ROOT)}: {spec.notes}")


if __name__ == "__main__":
    main()
