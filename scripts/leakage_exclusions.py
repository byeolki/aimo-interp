#!/usr/bin/env python3
"""Write data/excluded_problems.txt: pool problems the organizers advise against using.

On 2026-10-09 the organizers advised excluding AIME 2024, 2025 and 2026 (plus AIMO 2/3 public
references, JMO and problems-public, none of which are in our pool) to avoid val/test leakage.
AIME 2024 problems come from AI-MO/aimo-validation-aime and are identified by their source URL.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import PAGE_SIZE, ROWS_API, problem_key  # noqa: E402

POOL = ROOT / "data" / "problems_small.jsonl"
OUTPUT = ROOT / "data" / "excluded_problems.txt"
EXCLUDED_SOURCES = {"aime_2025", "aime_2026"}
EXCLUDED_AIME_URL_YEARS = ("2024",)


def aime_years() -> dict[str, str]:
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": "AI-MO/aimo-validation-aime", "config": "default", "split": "train", "offset": offset, "length": PAGE_SIZE}
        )
        with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
            page = [item["row"] for item in json.load(response)["rows"]]
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return {problem_key(row["problem"]): row["url"] for row in rows}


def main() -> None:
    urls = aime_years()
    excluded = []
    with POOL.open(encoding="utf-8") as handle:
        for line in handle:
            problem = json.loads(line)
            from_aime_2024 = problem["source"] == "aimo-validation-aime" and any(
                year in urls.get(problem["problem_id"], "") for year in EXCLUDED_AIME_URL_YEARS
            )
            if problem["source"] in EXCLUDED_SOURCES or from_aime_2024:
                excluded.append(problem["problem_id"])
    OUTPUT.write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(excluded)} problems")


if __name__ == "__main__":
    main()
