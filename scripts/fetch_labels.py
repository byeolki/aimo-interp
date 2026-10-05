#!/usr/bin/env python3
"""Write data/labels.jsonl: every public labeled row for the four small-track models."""

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import fetch_labeled_rows, write_rows  # noqa: E402

OUTPUT = ROOT / "data" / "labels.jsonl"


def main() -> None:
    rows = fetch_labeled_rows()
    write_rows(rows, OUTPUT)
    counts = collections.Counter((row.source, row.model_id, row.is_robust) for row in rows)
    print(f"{OUTPUT.relative_to(ROOT)}: {len(rows)} rows, {len({r.problem_id for r in rows})} problems")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
