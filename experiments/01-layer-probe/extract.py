#!/usr/bin/env python3
"""Extract all-layer hidden states (last token + mean pool) for every labeled problem.

Every problem goes through every small-track model, so the analysis can compare a
model's own representation against a shared encoder. Writes one npz per model to
runs/features/. Run on a GPU host after `uv run scripts/fetch_labels.py`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import read_rows  # noqa: E402
from aimo_interp.features import extract_hidden_states, load_model, release  # noqa: E402
from aimo_interp.models import SMALL_TRACK_MODELS, safe_model_id  # noqa: E402

LABELS = ROOT / "data" / "labels.jsonl"
OUTPUT_DIR = ROOT / "runs" / "features"


def unique_problems(labels_path: Path) -> tuple[list[str], list[str]]:
    by_id = {row.problem_id: row.problem for row in read_rows(labels_path)}
    problem_ids = sorted(by_id)
    return problem_ids, [by_id[problem_id] for problem_id in problem_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(SMALL_TRACK_MODELS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    problem_ids, problems = unique_problems(LABELS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        output = OUTPUT_DIR / f"{safe_model_id(model_id)}.npz"
        if output.exists():
            print(f"skip {model_id}: {output.name} exists")
            continue
        started = time.time()
        loaded = load_model(model_id, local_files_only=False)
        loaded_at = time.time()
        pooled = extract_hidden_states(loaded, problems, batch_size=args.batch_size)
        release(loaded)
        # float16 halves disk use; probes are trained on standardized features anyway.
        np.savez(
            output,
            problem_ids=np.array(problem_ids),
            **{name: values.astype(np.float16) for name, values in pooled.items()},
        )
        print(
            f"{model_id}: {pooled['last'].shape} load {loaded_at - started:.0f}s "
            f"forward {time.time() - loaded_at:.1f}s for {len(problems)} problems"
        )


if __name__ == "__main__":
    main()
