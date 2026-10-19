#!/usr/bin/env python3
"""White-box ablation: the same probes with the hidden state replaced by black-box inputs.

Black-box inputs are what a method sees without model internals: surface text features of the
problem (length, digit count, LaTeX density, largest number) plus the target-model one-hot.
Everything else matches analyze.py / analyze_v2.py: leakage-free rows, problem-grouped folds,
3 seeds, logistic C=0.05 balanced, ridge alpha 10000, mean pooling over relative depth 0.5-0.7.

Reports (1) out-of-fold correlation of predicted vs sampled solve rate and (2) robustness
accuracy on all / mixed / official rows, for white-box and black-box inputs.
"""

import os
import sys

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from analyze import MODELS, FeatureStore, folds, load_rows, score  # noqa: E402
from analyze_v2 import band_features, solve_rate_rows, standardize  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("layer_probe_analyze", ROOT / "experiments" / "01-layer-probe" / "analyze.py")
_layer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layer)
text_features = _layer.text_features

BAND, POOLING, ALPHA, C = (0.5, 0.7), "mean", 10000.0, 0.05
SEEDS = (0, 1, 2)


def problem_texts() -> dict[str, str]:
    import json

    texts = {}
    for path in (ROOT / "data" / "problems_small.jsonl", ROOT / "data" / "labels.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                texts[row["problem_id"]] = row["problem"]
    return texts


def inputs(kind: str, model: str, problem_ids, stores, texts) -> np.ndarray:
    if kind == "white-box":
        return band_features(stores[model], problem_ids, BAND)
    return text_features(pd.Series([texts[p] for p in problem_ids]))


def solve_rate_r(kind: str, frame, solve, stores, texts) -> float:
    preds, truth = [], []
    for seed in SEEDS:
        for train, test in folds(frame, seed):
            held = set(frame.loc[test, "problem_id"])
            for m in MODELS:
                fit = solve[(solve.model_id == m) & ~solve.problem_id.isin(held)]
                ev = solve[(solve.model_id == m) & solve.problem_id.isin(held)]
                if len(ev) == 0:
                    continue
                z_fit, z_ev = standardize(inputs(kind, m, fit.problem_id, stores, texts), inputs(kind, m, ev.problem_id, stores, texts))
                preds.extend(Ridge(alpha=ALPHA if kind == "white-box" else 1.0).fit(z_fit, fit.base).predict(z_ev))
                truth.extend(ev.base)
    return float(np.corrcoef(preds, truth)[0, 1])


def robustness(kind: str, frame, stores, texts) -> dict:
    labels = frame["is_robust"].to_numpy()
    results = []
    for seed in SEEDS:
        predictions = np.zeros(len(frame), dtype=bool)
        for train, test in folds(frame, seed):
            for m in MODELS:
                tr = train[(frame.loc[train, "model_id"] == m).to_numpy()]
                te = test[(frame.loc[test, "model_id"] == m).to_numpy()]
                if len(te) == 0:
                    continue
                if len(set(labels[tr])) < 2:
                    predictions[te] = labels[tr][0]
                    continue
                z_tr, z_te = standardize(inputs(kind, m, frame.loc[tr, "problem_id"], stores, texts), inputs(kind, m, frame.loc[te, "problem_id"], stores, texts))
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(z_tr, labels[tr])
                predictions[te] = clf.predict(z_te)
        results.append(score(frame, predictions))
    return pd.DataFrame(results).mean().to_dict()


def main() -> None:
    frame = load_rows()
    solve = solve_rate_rows()
    stores = {m: FeatureStore(m, POOLING) for m in MODELS}
    texts = problem_texts()
    rows = []
    for kind in ("white-box", "black-box"):
        rows.append({"inputs": kind, "solve_rate_r": solve_rate_r(kind, frame, solve, stores, texts), **robustness(kind, frame, stores, texts)})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "noleak" / "ablation_whitebox.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
