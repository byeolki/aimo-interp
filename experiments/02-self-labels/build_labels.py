#!/usr/bin/env python3
"""Turn sampled answers into per-(model, problem) robustness labels.

base     = accuracy on the original over K_ORIGINAL samples
drop_f   = base - accuracy over all samples of perturbation family f
max_drop = max over families; robust if <= 0.10, spurious if >= 0.25, unlabeled between.
A problem the model never solves gets label robust with ``base_zero = True``: accuracy cannot
drop below zero, and the public challenge sample labels all such cases robust (20 of 20).

Writes data/self_labels.jsonl.
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.models import SMALL_TRACK_MODELS, safe_model_id  # noqa: E402
from aimo_interp.perturb import FAMILIES  # noqa: E402
from aimo_interp.problems import read_problems  # noqa: E402

RUNS = ROOT / "runs" / "02-self-labels"
OUTPUT = ROOT / "data" / "self_labels.jsonl"
ROBUST_MAX_DROP = 0.10
SPURIOUS_MIN_DROP = 0.25


def mean(values: list[bool]) -> float:
    return sum(values) / len(values)


def label_model(model_id: str, problem_text: dict[str, str]) -> list[dict]:
    path = RUNS / f"{safe_model_id(model_id)}.jsonl"
    if not path.exists():
        return []
    by_problem: dict[str, dict[str, list[bool]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    truncation: dict[str, list[bool]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_problem[record["problem_id"]][record["family"]].extend(record["correct"])
            if record["family"] == "original":
                truncation[record["problem_id"]].extend(record["truncated"])

    rows = []
    for problem_id, families in by_problem.items():
        if "original" not in families:
            continue
        base = mean(families["original"])
        row = {"model_id": model_id, "problem_id": problem_id, "problem": problem_text[problem_id],
               "base": base, "truncated_rate": mean(truncation[problem_id]), "base_zero": base == 0.0}
        if base == 0.0:
            rows.append({**row, "max_drop": 0.0, "is_robust": True})
            continue
        if not all(family in families for family in FAMILIES):
            continue
        drops = {family: base - mean(families[family]) for family in FAMILIES}
        max_drop = max(drops.values())
        label = True if max_drop <= ROBUST_MAX_DROP else False if max_drop >= SPURIOUS_MIN_DROP else None
        rows.append({**row, "drops": drops, "max_drop": max_drop, "is_robust": label})
    return rows


def main() -> None:
    problem_text = {p.problem_id: p.problem for p in read_problems(ROOT / "data" / "problems.jsonl")}
    rows = [row for model in SMALL_TRACK_MODELS for row in label_model(model, problem_text)]
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    labeled = [row for row in rows if row["is_robust"] is not None]
    print(f"{OUTPUT.relative_to(ROOT)}: {len(rows)} rows, {len(labeled)} labeled")
    for model in SMALL_TRACK_MODELS:
        mine = [row for row in labeled if row["model_id"] == model]
        if mine:
            print(f"  {model}: n={len(mine)} robust={sum(r['is_robust'] for r in mine)} base0={sum(r['base_zero'] for r in mine)}")
    per_problem = collections.defaultdict(set)
    for row in labeled:
        per_problem[row["problem_id"]].add(row["is_robust"])
    multi = [labels for labels in per_problem.values()]
    print(f"  problems with mixed labels across models: {sum(len(s) == 2 for s in multi)} / {len(multi)}")


if __name__ == "__main__":
    main()
