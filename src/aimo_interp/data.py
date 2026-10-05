"""Collect labeled (model, problem) rows from the public aimo-interp datasets.

Standard library only, so the label table can be built on any machine.
"""

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from aimo_interp.models import SMALL_TRACK_MODELS, resolve_model_id

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


@dataclass(frozen=True)
class LabelSource:
    dataset: str
    split: str
    model_field: str
    problem_field: str
    label_field: str


# train-main-v2 holds the competition models; augmented-sample-math-agg adds MATH-level
# problems for qwen3-8b:low (= DeepSeek-R1-0528-Qwen3-8B), which is easier than the
# olympiad test distribution, so it is tagged and evaluated separately.
LABEL_SOURCES: tuple[LabelSource, ...] = (
    LabelSource("aimo-interp/train-main-v2", "train", "model_id", "problem", "is_robust"),
    LabelSource(
        "aimo-interp/augmented-sample-math-agg",
        "validation",
        "model_id",
        "original_problem",
        "model_is_robust",
    ),
)


@dataclass(frozen=True)
class LabeledRow:
    source: str
    model_id: str
    problem_id: str
    problem: str
    is_robust: bool


def problem_key(problem: str) -> str:
    """Stable id for a problem text; used for grouping folds and caching features."""
    return hashlib.sha256(problem.strip().encode()).hexdigest()[:16]


def _fetch_rows(dataset: str, split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
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


def _to_labeled_rows(source: LabelSource, raw_rows: list[dict]) -> list[LabeledRow]:
    labeled = []
    for raw in raw_rows:
        model_id = resolve_model_id(raw[source.model_field])
        label = raw[source.label_field]
        # Rows between the robust/spurious thresholds carry no label.
        if model_id not in SMALL_TRACK_MODELS or label is None:
            continue
        problem = raw[source.problem_field]
        labeled.append(
            LabeledRow(
                source=source.dataset.split("/")[-1],
                model_id=model_id,
                problem_id=problem_key(problem),
                problem=problem,
                is_robust=bool(label),
            )
        )
    return labeled


def fetch_labeled_rows() -> list[LabeledRow]:
    """Download every labeled small-track row, deduplicated on (model, problem)."""
    seen: set[tuple[str, str]] = set()
    rows: list[LabeledRow] = []
    for source in LABEL_SOURCES:
        for row in _to_labeled_rows(source, _fetch_rows(source.dataset, source.split)):
            key = (row.model_id, row.problem_id)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def write_rows(rows: list[LabeledRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[LabeledRow]:
    with path.open(encoding="utf-8") as handle:
        return [LabeledRow(**json.loads(line)) for line in handle if line.strip()]
