#!/usr/bin/env python3
"""Per-model accuracy of a predictions.jsonl on the official contract-test rows."""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "runs" / "contract-test" / "input" / "cases.jsonl"
LABELS = ROOT / "runs" / "contract-test" / "reference" / "labels.jsonl"


def main() -> None:
    predictions = {r["id"]: r for r in map(json.loads, Path(sys.argv[1]).read_text().splitlines())}
    labels = {r["id"]: r["is_robust"] for r in map(json.loads, LABELS.read_text().splitlines())}
    models = {r["id"]: r["model_id"].split("/")[-1] for r in map(json.loads, CASES.read_text().splitlines())}
    per_model = collections.defaultdict(lambda: [0, 0])
    by_label = collections.defaultdict(lambda: [0, 0])
    for case_id, label in labels.items():
        correct = predictions[case_id]["is_robust"] == label
        per_model[models[case_id]][0] += correct
        per_model[models[case_id]][1] += 1
        by_label[label][0] += correct
        by_label[label][1] += 1
    total = sum(c for c, _ in per_model.values()), sum(n for _, n in per_model.values())
    balanced = sum(c / n for c, n in by_label.values()) / len(by_label)
    print(f"accuracy {total[0]}/{total[1]} = {total[0] / total[1]:.3f}, balanced accuracy {balanced:.3f}, "
          f"predicted robust {sum(p['is_robust'] for p in predictions.values())}/{len(predictions)}")
    for model, (c, n) in sorted(per_model.items()):
        print(f"  {model}: {c}/{n}")


if __name__ == "__main__":
    main()
