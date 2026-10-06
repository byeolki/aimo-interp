#!/usr/bin/env python3
"""Does matching the test design improve accuracy for free?

The organizers state the val/test labels are balanced per problem and per model (always-true
scored exactly 0.50 on val). Our classifiers are fitted on data that is 75-85% robust, so their
threshold is miscalibrated for a balanced test. Two label-free post-processing rules use only
what are_robust receives:

- per-model median: inside one call (one model, many problems) predict robust for the top
  half of scores. Pure calibration; the ranking is unchanged.
- per-problem rank: score every problem with every model's head (other checkpoints are
  cached on the worker) and predict robust for the models ranked in the top half on that
  problem, using per-model standardized scores.

Evaluated on balanced subsamples of our mixed problems: for each mixed problem keep one
robust and one spurious row (different models), repeated over random draws. Scores are the
out-of-fold blend logits of the chosen config.
"""

import os
import sys

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import MODELS, FeatureStore, folds, load_rows  # noqa: E402
from analyze_v2 import band_features, solve_rate_rows, standardize  # noqa: E402

BAND, POOLING, ALPHA = (0.5, 0.7), "mean", 10000.0
DRAWS = 2000


def oof_scores(frame: pd.DataFrame, solve: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Out-of-fold blend logit of every model's head on every problem in the frame."""
    stores = {m: FeatureStore(m, POOLING) for m in MODELS}
    labels = frame["is_robust"].to_numpy()
    problems = frame["problem_id"].unique()
    scores = {m: pd.Series(np.nan, index=problems) for m in MODELS}
    for train, test in folds(frame, seed):
        held = frame.loc[test, "problem_id"].unique()
        rate = {}
        direct = {}
        for m in MODELS:
            fit = solve[(solve.model_id == m) & ~solve.problem_id.isin(held)]
            z_fit, z_held = standardize(band_features(stores[m], fit.problem_id, BAND), band_features(stores[m], held, BAND))
            rate[m] = np.clip(Ridge(alpha=ALPHA).fit(z_fit, fit.base).predict(z_held), 0, 1)
            tr = train[(frame.loc[train, "model_id"] == m).to_numpy()]
            z_tr, z_h = standardize(band_features(stores[m], frame.loc[tr, "problem_id"], BAND), band_features(stores[m], held, BAND))
            direct[m] = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000).fit(z_tr, labels[tr]).decision_function(z_h)
        # Step 2 on training rows' in-sample rates is a simplification; it only sets the scale.
        onehot = lambda m, n: np.tile(np.eye(len(MODELS))[MODELS.index(m)], (n, 1))  # noqa: E731
        tr_rate = np.zeros(len(train))
        for m in MODELS:
            idx = np.where((frame.loc[train, "model_id"] == m).to_numpy())[0]
            fit = solve[(solve.model_id == m) & ~solve.problem_id.isin(held)]
            z_fit, z_tr = standardize(band_features(stores[m], fit.problem_id, BAND), band_features(stores[m], frame.loc[train[idx], "problem_id"], BAND))
            tr_rate[idx] = np.clip(Ridge(alpha=ALPHA).fit(z_fit, fit.base).predict(z_tr), 0, 1)
        tr_onehot = np.stack([(frame.loc[train, "model_id"] == m).to_numpy(float) for m in MODELS], axis=1)
        step2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(
            np.column_stack([tr_rate, tr_rate**2, tr_onehot]), labels[train])
        for m in MODELS:
            two = step2.decision_function(np.column_stack([rate[m], rate[m] ** 2, onehot(m, len(held))]))
            scores[m].loc[held] = two / (two.std() + 1e-9) + direct[m] / (direct[m].std() + 1e-9)
    return pd.DataFrame(scores)


def evaluate(frame: pd.DataFrame, scores: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    mixed = frame[frame.mixed]
    # Per-model standardization over all problems, as the solution could do with training stats.
    z = (scores - scores.mean()) / scores.std()
    results = {"raw threshold": [], "per-model median": [], "per-problem rank": [], "per-problem rank raw": [], "model prior only": []}
    prior = frame.groupby("model_id").is_robust.mean()
    groups = {p: g for p, g in mixed.groupby("problem_id")}
    for _ in range(DRAWS):
        rows = []
        for p, g in groups.items():
            pos, neg = g[g.is_robust], g[~g.is_robust]
            if len(pos) and len(neg):
                rows.append(pos.sample(1, random_state=int(rng.integers(1 << 31))))
                rows.append(neg.sample(1, random_state=int(rng.integers(1 << 31))))
        sample = pd.concat(rows)
        s = np.array([scores.loc[p, m] for p, m in zip(sample.problem_id, sample.model_id)])
        zs = np.array([z.loc[p, m] for p, m in zip(sample.problem_id, sample.model_id)])
        y = sample.is_robust.to_numpy()
        results["raw threshold"].append(np.mean((s >= 0) == y))
        median_pred = np.zeros(len(sample), dtype=bool)
        for m in MODELS:
            mask = (sample.model_id == m).to_numpy()
            if mask.any():
                median_pred[mask] = s[mask] >= np.median(s[mask])
        results["per-model median"].append(np.mean(median_pred == y))
        # Within each problem the two sampled models compete; higher standardized score = robust.
        rank_pred = np.zeros(len(sample), dtype=bool)
        for start in range(0, len(sample), 2):
            a, b = start, start + 1
            winner = a if zs[a] >= zs[b] else b
            rank_pred[winner] = True
        results["per-problem rank"].append(np.mean(rank_pred == y))
        raw_pred = np.zeros(len(sample), dtype=bool)
        prior_pred = np.zeros(len(sample), dtype=bool)
        models = sample.model_id.to_numpy()
        for start in range(0, len(sample), 2):
            a, b = start, start + 1
            raw_pred[a if s[a] >= s[b] else b] = True
            prior_pred[a if prior[models[a]] >= prior[models[b]] else b] = True
        results["per-problem rank raw"].append(np.mean(raw_pred == y))
        results["model prior only"].append(np.mean(prior_pred == y))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


def main() -> None:
    frame = load_rows()
    frame = frame[frame.source == "generated"].reset_index(drop=True)
    frame["mixed"] = frame.problem_id.map(frame.groupby("problem_id").is_robust.nunique()) == 2
    solve = solve_rate_rows()
    rng = np.random.default_rng(0)
    for seed in (0, 1):
        scores = oof_scores(frame, solve, seed)
        print(f"seed {seed}:", {k: f"{m:.3f} (sd {s:.3f})" for k, (m, s) in evaluate(frame, scores, rng).items()}, flush=True)


if __name__ == "__main__":
    main()
