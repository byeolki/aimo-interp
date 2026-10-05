#!/usr/bin/env python3
"""Self-probe vs shared-encoder probe on self-generated labels, scored where it matters.

Rows: data/self_labels.jsonl (labeled only) plus the 34 official rows.
Features: runs/features-self/<model>.npz (hidden states of the 316 pool problems) and
runs/features/<model>.npz (the official problems), all layers, last token and mean pool.

Configurations, out-of-fold with folds grouped by problem across all models:
- self:   per-model probe on the target model's own hidden state
- shared: one probe on DeepSeek-8B hidden state + model one-hot (the v1 design)
- base:   oracle-ish behavioural baseline, logistic regression on the sampled base accuracy
- majority per model

Reported subsets:
- all:     every labeled row
- mixed:   only problems whose labels differ across models (what the val/test set rewards)
- balanced: accuracy averaged per problem, then over problems, on mixed problems
- official: the 34 organizer-labeled rows
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
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import read_rows  # noqa: E402
from aimo_interp.models import SMALL_TRACK_MODELS, safe_model_id  # noqa: E402
from aimo_interp.probe import fit_probe  # noqa: E402

FEATURE_DIRS = (ROOT / "runs" / "features-self", ROOT / "runs" / "features")
OUTPUT = ROOT / "runs" / "02-self-labels" / "cv.csv"
SHARED_ENCODER = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
N_FOLDS = 5
SEEDS = (0, 1, 2)
INCLUDE_BASE_ZERO = os.environ.get("INCLUDE_BASE_ZERO", "1") == "1"


def load_rows() -> pd.DataFrame:
    with (ROOT / "data" / "self_labels.jsonl").open(encoding="utf-8") as handle:
        generated = [json.loads(line) for line in handle if line.strip()]
    generated = [r for r in generated if r["is_robust"] is not None and (INCLUDE_BASE_ZERO or not r["base_zero"])]
    official = [
        {"model_id": r.model_id, "problem_id": r.problem_id, "problem": r.problem, "is_robust": r.is_robust, "base": np.nan, "base_zero": False}
        for r in read_rows(ROOT / "data" / "labels.jsonl") if r.source == "train-main-v2"
    ]
    frame = pd.DataFrame(
        [{**r, "source": "generated"} for r in generated] + [{**r, "source": "official"} for r in official]
    )
    frame["is_robust"] = frame["is_robust"].astype(bool)
    per_problem = frame.groupby("problem_id")["is_robust"].nunique()
    frame["mixed"] = frame["problem_id"].map(per_problem) == 2
    return frame.reset_index(drop=True)


class FeatureStore:
    """All-layer hidden states of one encoder, looked up by problem id across feature dirs."""

    def __init__(self, encoder_id: str, pooling: str):
        self.blocks, self.index = [], {}
        for directory in FEATURE_DIRS:
            path = directory / f"{safe_model_id(encoder_id)}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as arrays:
                block = arrays[pooling].astype(np.float32)
                for position, problem_id in enumerate(arrays["problem_ids"]):
                    self.index.setdefault(str(problem_id), (len(self.blocks), position))
                self.blocks.append(block)

    def layer_count(self) -> int:
        return self.blocks[0].shape[0]

    def get(self, problem_ids, layer: int) -> np.ndarray:
        return np.stack([self.blocks[b][layer, p] for b, p in (self.index[pid] for pid in problem_ids)])


def folds(frame: pd.DataFrame, seed: int):
    problems = frame["problem_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = dict(zip(problems, rng.permutation(len(problems))))
    groups = frame["problem_id"].map(shuffled).to_numpy()
    return list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(frame)), groups=groups))


def fit_predict(features: np.ndarray, labels: np.ndarray, train, test) -> np.ndarray:
    if len(set(labels[train])) < 2:
        return np.full(len(test), labels[train][0])
    return np.array(fit_probe(features[train], labels[train]).predict(features[test]))


def self_predictions(frame: pd.DataFrame, layer_by_model: dict[str, int], pooling: str, seed: int) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=bool)
    stores = {model: FeatureStore(model, pooling) for model in SMALL_TRACK_MODELS}
    for train, test in folds(frame, seed):
        for model in SMALL_TRACK_MODELS:
            rows = frame.index[frame["model_id"] == model].to_numpy()
            tr, te = np.intersect1d(train, rows), np.intersect1d(test, rows)
            if len(te) == 0:
                continue
            store = stores[model]
            layer = layer_by_model[model]
            x_tr = store.get(frame.loc[tr, "problem_id"], layer)
            x_te = store.get(frame.loc[te, "problem_id"], layer)
            features = np.vstack([x_tr, x_te])
            labels = np.concatenate([frame.loc[tr, "is_robust"].to_numpy(), frame.loc[te, "is_robust"].to_numpy()])
            predictions[te] = fit_predict(features, labels, np.arange(len(tr)), np.arange(len(tr), len(labels)))
    return predictions


def shared_predictions(frame: pd.DataFrame, layer: int, pooling: str, seed: int) -> np.ndarray:
    store = FeatureStore(SHARED_ENCODER, pooling)
    onehot = np.stack([(frame["model_id"] == m).to_numpy(float) for m in SMALL_TRACK_MODELS], axis=1)
    features = np.hstack([store.get(frame["problem_id"], layer), onehot])
    labels = frame["is_robust"].to_numpy()
    predictions = np.zeros(len(frame), dtype=bool)
    for train, test in folds(frame, seed):
        predictions[test] = fit_predict(features, labels, train, test)
    return predictions


def base_predictions(frame: pd.DataFrame, seed: int) -> np.ndarray:
    has_base = frame["base"].notna().to_numpy()
    base = frame["base"].fillna(0.5).to_numpy()
    features = np.stack([base, base**2, (base == 0).astype(float), has_base.astype(float)], axis=1)
    labels = frame["is_robust"].to_numpy()
    predictions = np.zeros(len(frame), dtype=bool)
    for train, test in folds(frame, seed):
        predictions[test] = fit_predict(features, labels, train, test)
    return predictions


def majority_predictions(frame: pd.DataFrame, seed: int) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=bool)
    for train, test in folds(frame, seed):
        rates = frame.iloc[train].groupby("model_id")["is_robust"].mean()
        predictions[test] = [rates.get(m, 0.5) >= 0.5 for m in frame.iloc[test]["model_id"]]
    return predictions


def score(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    correct = predictions == frame["is_robust"].to_numpy()
    mixed = frame["mixed"].to_numpy()
    official = (frame["source"] == "official").to_numpy()
    per_problem = pd.Series(correct[mixed]).groupby(frame.loc[mixed, "problem_id"].to_numpy()).mean()
    return {
        "acc_all": correct.mean(),
        "acc_mixed": correct[mixed].mean() if mixed.any() else np.nan,
        "acc_balanced": per_problem.mean() if len(per_problem) else np.nan,
        "acc_official": correct[official].mean() if official.any() else np.nan,
    }


def averaged(frame: pd.DataFrame, predict, config: dict) -> dict:
    scores = pd.DataFrame([score(frame, predict(seed)) for seed in SEEDS]).mean().to_dict()
    return {**config, **scores}


def main() -> None:
    frame = load_rows()
    print(f"rows {len(frame)} (generated {sum(frame.source == 'generated')}, official {sum(frame.source == 'official')}), "
          f"mixed rows {int(frame.mixed.sum())} over {frame[frame.mixed].problem_id.nunique()} problems")
    print(frame.groupby(["model_id", "source"])["is_robust"].agg(["count", "mean"]).round(2).to_string())

    tasks = [delayed(averaged)(frame, lambda s: majority_predictions(frame, s), {"config": "majority"}),
             delayed(averaged)(frame, lambda s: base_predictions(frame, s), {"config": "base-accuracy"})]
    for pooling in ("last", "mean"):
        layers = FeatureStore(SHARED_ENCODER, pooling).layer_count()
        for layer in range(1, layers):
            tasks.append(delayed(averaged)(frame, lambda s, l=layer, p=pooling: shared_predictions(frame, l, p, s),
                                           {"config": "shared", "pooling": pooling, "rel_depth": layer / (layers - 1)}))
        # Self probes use the same relative depth in every model so one knob is swept.
        counts = {m: FeatureStore(m, pooling).layer_count() for m in SMALL_TRACK_MODELS}
        for depth in np.linspace(0.1, 1.0, 19):
            layer_by_model = {m: int(round(depth * (counts[m] - 1))) for m in SMALL_TRACK_MODELS}
            tasks.append(delayed(averaged)(frame, lambda s, lm=layer_by_model, p=pooling: self_predictions(frame, lm, p, s),
                                           {"config": "self", "pooling": pooling, "rel_depth": round(float(depth), 3)}))
    results = pd.DataFrame(Parallel(n_jobs=os.cpu_count() or 1, verbose=2)(tasks))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for config, group in results.groupby("config"):
            top = group.sort_values("acc_balanced", ascending=False).head(6)
            print(f"\n== {config}")
            print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
