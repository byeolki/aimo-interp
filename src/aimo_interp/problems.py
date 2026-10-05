"""Public competition problems with integer answers, used to generate our own labels.

Integer answers keep answer checking exact. Problems that also appear in the official
train data are dropped so self-generated labels never duplicate official rows.
"""

import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from aimo_interp.data import ROWS_API, PAGE_SIZE, problem_key

SOURCES: tuple[tuple[str, str, str, str], ...] = (
    # (dataset, split, problem field, answer field)
    ("AI-MO/aimo-validation-aime", "train", "problem", "answer"),
    ("AI-MO/aimo-validation-amc", "train", "problem", "answer"),
    ("MathArena/aime_2025", "train", "problem", "answer"),
    ("MathArena/aime_2026", "train", "problem", "answer"),
    ("MathArena/hmmt_feb_2025", "train", "problem", "answer"),
    ("MathArena/hmmt_feb_2026", "train", "problem", "answer"),
    ("MathArena/brumo_2025", "train", "problem", "answer"),
    ("MathArena/smt_2025", "train", "problem", "answer"),
    ("MathArena/cmimc_2025", "train", "problem", "answer"),
)
INTEGER = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    source: str
    problem: str
    answer: int


def _fetch(dataset: str, split: str) -> list[dict]:
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": PAGE_SIZE}
        )
        with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
            page = [item["row"] for item in json.load(response)["rows"]]
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def _integer_answer(raw: object) -> int | None:
    text = str(raw).strip().strip("$").replace(",", "")
    # AMC answers in this dataset are floats such as "142.0".
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return int(text) if INTEGER.match(text) else None


def collect_problems(excluded_texts: set[str]) -> list[Problem]:
    excluded = {problem_key(text) for text in excluded_texts}
    problems: dict[str, Problem] = {}
    for dataset, split, problem_field, answer_field in SOURCES:
        for row in _fetch(dataset, split):
            answer = _integer_answer(row[answer_field])
            text = str(row[problem_field]).strip()
            key = problem_key(text)
            if answer is None or key in excluded or key in problems:
                continue
            problems[key] = Problem(key, dataset.split("/")[-1], text, answer)
    return list(problems.values())


def write_problems(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps(asdict(problem), ensure_ascii=False) + "\n")


def read_problems(path: Path) -> list[Problem]:
    with path.open(encoding="utf-8") as handle:
        return [Problem(**json.loads(line)) for line in handle if line.strip()]
