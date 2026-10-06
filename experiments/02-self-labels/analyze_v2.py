#!/usr/bin/env python3
"""Two-step self-probe: predict the model's own solve rate, then robustness from it.

Motivation (analyze.py, 2026-10-06): on problems whose labels differ across models, the
sampled base accuracy alone reaches ~0.82 while a direct robustness probe reaches ~0.62. The
solve rate is also available for every (model, problem) pair, including rows whose robustness
label falls in the unlabeled band, so it is a denser training target than the label itself.

Per model, out-of-fold with problem-grouped folds:
1. ridge regression from hidden states (mean of a band of layers) to base accuracy
2. logistic regression from [predicted base, its square, model one-hot] to is_robust
Plus a direct probe on the same band-averaged features, and the blend of both.
Official rows have no base accuracy; they are used only in step 2 and in scoring.
"""

import os
import sys

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import MODELS, FeatureStore, folds, load_rows, score  # noqa: E402

OUTPUT = ROOT / "runs" / "02-self-labels" / "cv_v2.csv"
SEEDS = (0, 1, 2)
BANDS = ((0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0), (0.6, 1.0))
RIDGE_ALPHAS = (100.0, 1000.0, 10000.0)


def solve_rate_rows() -> pd.DataFrame:
    """Every generated (model, problem) pair with a base accuracy, labeled or not."""
    with (ROOT / "data" / "self_labels.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frame = pd.DataFrame(rows)
    return frame[frame["model_id"].isin(MODELS)][["model_id", "problem_id", "base"]]


def band_features(store: FeatureStore, problem_ids, band: tuple[float, float]) -> np.ndarray:
    count = store.layer_count()
    layers = [l for l in range(1, count) if band[0] <= l / (count - 1) <= band[1]]
    stacked = np.stack([store.get(problem_ids, l) for l in layers])
    return stacked.mean(axis=0)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train - mean) / scale, (test - mean) / scale


def run(frame: pd.DataFrame, solve: pd.DataFrame, band, pooling: str, alpha: float, seed: int) -> dict[str, np.ndarray]:
    stores = {m: FeatureStore(m, pooling) for m in MODELS}
    labels = frame["is_robust"].to_numpy()
    onehot = np.stack([(frame["model_id"] == m).to_numpy(float) for m in MODELS], axis=1)
    predicted_base = np.zeros(len(frame))
    direct_logit = np.zeros(len(frame))
    fold_list = folds(frame, seed)

    for train, test in fold_list:
        test_problems = set(frame.loc[test, "problem_id"])
        for model in MODELS:
            store = stores[model]
            fit_rows = solve[(solve.model_id == model) & ~solve.problem_id.isin(test_problems)]
            x_fit = band_features(store, fit_rows["problem_id"], band)
            mask_te = (frame.loc[test, "model_id"] == model).to_numpy()
            te = test[mask_te]
            if len(te) == 0:
                continue
            x_te = band_features(store, frame.loc[te, "problem_id"], band)
            z_fit, z_te = standardize(x_fit, x_te)
            regressor = Ridge(alpha=alpha).fit(z_fit, fit_rows["base"].to_numpy())
            predicted_base[te] = np.clip(regressor.predict(z_te), 0, 1)

            tr = train[(frame.loc[train, "model_id"] == model).to_numpy()]
            x_tr = band_features(store, frame.loc[tr, "problem_id"], band)
            z_tr, z_te2 = standardize(x_tr, x_te)
            if len(set(labels[tr])) == 2:
                classifier = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000).fit(z_tr, labels[tr])
                direct_logit[te] = classifier.decision_function(z_te2)

    # Step 2 needs out-of-fold predicted base on the training rows too, so it is fitted on the
    # out-of-fold predictions of a second pass over the same folds.
    two_step_logit = np.zeros(len(frame))
    stacked = np.column_stack([predicted_base, predicted_base**2, onehot])
    for train, test in fold_list:
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(stacked[train], labels[train])
        two_step_logit[test] = classifier.decision_function(stacked[test])

    blend = two_step_logit / (np.std(two_step_logit) + 1e-9) + direct_logit / (np.std(direct_logit) + 1e-9)
    return {"two_step": two_step_logit >= 0, "direct_band": direct_logit >= 0, "blend": blend >= 0,
            "_base_corr": np.array([np.corrcoef(predicted_base[frame.base.notna()], frame.base[frame.base.notna()])[0, 1]])}


def evaluate(frame, solve, band, pooling, alpha) -> list[dict]:
    per_seed = [run(frame, solve, band, pooling, alpha, seed) for seed in SEEDS]
    records = []
    for name in ("two_step", "direct_band", "blend"):
        scores = pd.DataFrame([score(frame, result[name]) for result in per_seed]).mean().to_dict()
        records.append({"config": name, "band": f"{band[0]}-{band[1]}", "pooling": pooling, "alpha": alpha,
                        "base_corr": float(np.mean([r["_base_corr"][0] for r in per_seed])), **scores})
    return records


def main() -> None:
    frame = load_rows()
    solve = solve_rate_rows()
    print(f"models {MODELS}; labeled rows {len(frame)}, mixed {int(frame.mixed.sum())}; solve-rate rows {len(solve)}")
    tasks = [delayed(evaluate)(frame, solve, band, pooling, alpha)
             for band in BANDS for pooling in ("last", "mean") for alpha in RIDGE_ALPHAS]
    results = pd.DataFrame([r for rs in Parallel(n_jobs=min(len(tasks), os.cpu_count() or 1), verbose=2)(tasks) for r in rs])
    results.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        for config, group in results.groupby("config"):
            print(f"\n== {config}")
            print(group.sort_values("acc_balanced", ascending=False).head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
