#!/usr/bin/env python3
"""Fit the chosen blend heads on all rows and write submissions/self-probe/artifacts.

Chosen in RESULTS.md: mean pooling, layers in relative depth 0.5-0.7 averaged, ridge
alpha 10000, logistic C=0.05 (direct) and C=1.0 (step 2), class_weight balanced.
Step 2 is fitted on out-of-fold solve-rate predictions so it sees the noise it will see at
test time. Models without self labels use the shared v1 head copied from layer-probe.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import MODELS, FeatureStore, load_rows  # noqa: E402
from analyze_v2 import solve_rate_rows  # noqa: E402

from aimo_interp.blend import BlendHead  # noqa: E402
from aimo_interp.models import safe_model_id  # noqa: E402

BAND = (0.5, 0.7)
POOLING = "mean"
RIDGE_ALPHA = 10000.0
DIRECT_C = 0.05
STEP2_C = 1.0
N_FOLDS = 5
OUT = ROOT / "submissions" / "self-probe" / "artifacts"
FALLBACK_SOURCE = ROOT / "submissions" / "layer-probe" / "artifacts"


def band_layers(store: FeatureStore) -> list[int]:
    count = store.layer_count()
    return [l for l in range(1, count) if BAND[0] <= l / (count - 1) <= BAND[1]]


def features_for(store: FeatureStore, layers: list[int], problem_ids) -> np.ndarray:
    return np.stack([store.get(problem_ids, l) for l in layers]).mean(axis=0)


def standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    return mean, np.where(scale < 1e-6, 1.0, scale)


def fit_ridge(x: np.ndarray, y: np.ndarray):
    mean, scale = standardizer(x)
    model = Ridge(alpha=RIDGE_ALPHA).fit((x - mean) / scale, y)
    return mean, scale, model


def main() -> None:
    frame = load_rows()
    solve = solve_rate_rows()
    onehot_all = np.stack([(frame["model_id"] == m).to_numpy(float) for m in MODELS], axis=1)
    labels = frame["is_robust"].to_numpy()

    stores = {m: FeatureStore(m, POOLING) for m in MODELS}
    layers = {m: band_layers(stores[m]) for m in MODELS}

    # Out-of-fold solve-rate predictions for every labeled row, grouped by problem.
    oof_rate = np.zeros(len(frame))
    problem_codes = frame["problem_id"].astype("category").cat.codes.to_numpy()
    for train, test in GroupKFold(n_splits=N_FOLDS).split(frame, groups=problem_codes):
        held_out = set(frame.loc[test, "problem_id"])
        for m in MODELS:
            fit_rows = solve[(solve.model_id == m) & ~solve.problem_id.isin(held_out)]
            mean, scale, ridge = fit_ridge(features_for(stores[m], layers[m], fit_rows["problem_id"]), fit_rows["base"].to_numpy())
            te = test[(frame.loc[test, "model_id"] == m).to_numpy()]
            if len(te):
                x = features_for(stores[m], layers[m], frame.loc[te, "problem_id"])
                oof_rate[te] = np.clip(ridge.predict((x - mean) / scale), 0, 1)

    step2_inputs = np.column_stack([oof_rate, oof_rate**2, onehot_all])
    step2 = LogisticRegression(C=STEP2_C, class_weight="balanced", max_iter=5000).fit(step2_inputs, labels)
    two_step_spread = float(np.std(step2.decision_function(step2_inputs)))

    OUT.mkdir(parents=True, exist_ok=True)
    heads = {}
    for index, m in enumerate(MODELS):
        rows = frame.index[frame["model_id"] == m].to_numpy()
        fit_rows = solve[solve.model_id == m]
        r_mean, r_scale, ridge = fit_ridge(features_for(stores[m], layers[m], fit_rows["problem_id"]), fit_rows["base"].to_numpy())
        x = features_for(stores[m], layers[m], frame.loc[rows, "problem_id"])
        d_mean, d_scale = standardizer(x)
        direct = LogisticRegression(C=DIRECT_C, class_weight="balanced", max_iter=5000).fit((x - d_mean) / d_scale, labels[rows])

        # Spread of out-of-fold direct logits for this model, to put both routes on one scale.
        oof_direct = np.zeros(len(rows))
        codes = problem_codes[rows]
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(x, groups=codes):
            mean, scale = standardizer(x[tr])
            if len(set(labels[rows][tr])) == 2:
                clf = LogisticRegression(C=DIRECT_C, class_weight="balanced", max_iter=5000).fit((x[tr] - mean) / scale, labels[rows][tr])
                oof_direct[te] = clf.decision_function((x[te] - mean) / scale)
        onehot = np.eye(len(MODELS))[index]
        head = BlendHead(
            layers=np.array(layers[m]), ridge_mean=r_mean, ridge_scale=r_scale, ridge_coef=ridge.coef_,
            ridge_intercept=float(ridge.intercept_), step2_coef=step2.coef_[0], step2_intercept=float(step2.intercept_[0]),
            model_onehot=onehot, direct_mean=d_mean, direct_scale=d_scale, direct_coef=direct.coef_[0],
            direct_intercept=float(direct.intercept_[0]), two_step_spread=two_step_spread,
            direct_spread=float(np.std(oof_direct)) or 1.0,
        )
        file_name = f"{safe_model_id(m)}.npz"
        head.save(OUT / file_name)
        train_acc = float(np.mean(np.array(head.predict(x)) == labels[rows]))
        heads[m] = {"file": file_name, "pooling": POOLING, "layers": layers[m], "train_acc": round(train_acc, 3), "rows": int(len(rows))}

    fallback_dir = OUT / "fallback"
    if fallback_dir.exists():
        shutil.rmtree(fallback_dir)
    shutil.copytree(FALLBACK_SOURCE, fallback_dir)
    config = {"heads": heads, "fallback_dir": "fallback", "band": list(BAND), "ridge_alpha": RIDGE_ALPHA}
    (OUT / "self_heads.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
