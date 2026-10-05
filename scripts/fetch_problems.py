#!/usr/bin/env python3
"""Write data/problems.jsonl: integer-answer competition problems not in the official data."""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import read_rows  # noqa: E402
from aimo_interp.problems import collect_problems, write_problems  # noqa: E402

LABELS = ROOT / "data" / "labels.jsonl"
VAL_CASES = ROOT / "data" / "val-sample" / "input" / "cases.jsonl"
OUTPUT = ROOT / "data" / "problems.jsonl"


def official_texts() -> set[str]:
    texts = {row.problem for row in read_rows(LABELS)}
    if VAL_CASES.exists():
        with VAL_CASES.open(encoding="utf-8") as handle:
            texts |= {json.loads(line)["problem"] for line in handle if line.strip()}
    return texts


def main() -> None:
    problems = collect_problems(official_texts())
    write_problems(problems, OUTPUT)
    print(f"{OUTPUT.relative_to(ROOT)}: {len(problems)} problems")
    for source, count in collections.Counter(problem.source for problem in problems).most_common():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
