#!/usr/bin/env python3
"""Layer-wise probe accuracy under problem-grouped cross-validation, against baselines.

Configurations, all scored out-of-fold:
- self:     probe on the target model's own hidden states, trained on that model's rows
- shared:   one probe across all four targets on one encoder's hidden states + model one-hot
- text:     surface difficulty features (length, digits, LaTeX density) + model one-hot
- majority: per-model majority label from the training fold

The primary subset is the official train-main-v2 rows (34). The augmented MATH rows are
trained on but reported separately because they come from an easier distribution.
Writes runs/01-layer-probe/cv.csv and prints the best layer per configuration. Picking
the best layer on this same CV is optimistic; treat those numbers as upper bounds.
"""

import os
import re
import sys

# Each fit is a tiny matrix; multithreaded BLAS thrashes. Parallelize across configs instead.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.data import read_rows  # noqa: E402
from aimo_interp.models import SMALL_TRACK_MODELS, safe_model_id  # noqa: E402
from aimo_interp.probe import fit_probe  # noqa: E402

LABELS = ROOT / "data" / "labels.jsonl"
FEATURES_DIR = ROOT / "runs" / "features"
OUTPUT = ROOT / "runs" / "01-layer-probe" / "cv.csv"
OFFICIAL_SOURCE = "train-main-v2"
N_FOLDS = 5
SEEDS = (0, 1, 2)
MIN_ROWS_FOR_SELF_PROBE = 8


@dataclass(frozen=True)
class Table:
    rows: pd.DataFrame
    model_onehot: np.ndarray


def load_table() -> Table:
    rows = pd.DataFrame([row.__dict__ for row in read_rows(LABELS)]).reset_index(drop=True)
    onehot = np.stack([(rows["model_id"] == model).to_numpy(float) for model in SMALL_TRACK_MODELS], axis=1)
    return Table(rows=rows, model_onehot=onehot)


def load_encoder_features(encoder_id: str, pooling: str, problem_ids: pd.Series) -> np.ndarray:
    """Return array[n_layers, n_rows, hidden] aligned with ``problem_ids``."""
    with np.load(FEATURES_DIR / f"{safe_model_id(encoder_id)}.npz", allow_pickle=False) as arrays:
        index = {problem_id: position for position, problem_id in enumerate(arrays["problem_ids"])}
        values = arrays[pooling].astype(np.float32)
    return values[:, [index[problem_id] for problem_id in problem_ids], :]


def text_features(problems: pd.Series) -> np.ndarray:
    def describe(text: str) -> list[float]:
        numbers = [int(match) for match in re.findall(r"\d+", text)[:50]]
        return [
            np.log1p(len(text)),
            np.log1p(sum(ch.isdigit() for ch in text)),
            np.log1p(text.count("$") + text.count("\\")),
            np.log1p(max(numbers, default=0)) if numbers else 0.0,
        ]

    return np.array([describe(text) for text in problems], dtype=np.float64)


def folds(rows: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = rows["is_robust"].astype(int).to_numpy()
    n_splits = min(N_FOLDS, int(np.bincount(labels).min()), rows["problem_id"].nunique())
    splitter = StratifiedGroupKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(rows)), labels, groups=rows["problem_id"]))


def oof_probe_predictions(features: np.ndarray, rows: pd.DataFrame, seed: int) -> np.ndarray:
    labels = rows["is_robust"].to_numpy()
    predictions = np.zeros(len(rows), dtype=bool)
    for train, test in folds(rows, seed):
        if len(set(labels[train])) < 2:
            predictions[test] = labels[train][0]
            continue
        probe = fit_probe(features[train], labels[train])
        predictions[test] = probe.predict(features[test])
    return predictions


def oof_majority_predictions(rows: pd.DataFrame, seed: int) -> np.ndarray:
    predictions = np.zeros(len(rows), dtype=bool)
    for train, test in folds(rows, seed):
        rates = rows.iloc[train].groupby("model_id")["is_robust"].mean()
        global_rate = rows.iloc[train]["is_robust"].mean()
        test_models = rows.iloc[test]["model_id"]
        predictions[test] = [rates.get(model, global_rate) >= 0.5 for model in test_models]
    return predictions


def score(rows: pd.DataFrame, predictions: np.ndarray, config: dict) -> list[dict]:
    """One record per reporting subset: official rows overall, per model, augmented rows."""
    subsets = {"official": rows["source"] == OFFICIAL_SOURCE, "augmented": rows["source"] != OFFICIAL_SOURCE}
    for model in SMALL_TRACK_MODELS:
        subsets[f"official:{model.split('/')[-1]}"] = (rows["source"] == OFFICIAL_SOURCE) & (rows["model_id"] == model)
    records = []
    labels = rows["is_robust"].to_numpy()
    for name, mask in subsets.items():
        mask = mask.to_numpy()
        if not mask.any():
            continue
        has_both = len(set(labels[mask])) == 2
        records.append(
            {
                **config,
                "subset": name,
                "n": int(mask.sum()),
                "acc": float((predictions[mask] == labels[mask]).mean()),
                "bal_acc": float(balanced_accuracy_score(labels[mask], predictions[mask])) if has_both else np.nan,
            }
        )
    return records


def averaged_over_seeds(predict, rows: pd.DataFrame, config: dict) -> list[dict]:
    records = [record for seed in SEEDS for record in score(rows, predict(seed), {**config, "seed": seed})]
    frame = pd.DataFrame(records)
    keys = [column for column in frame.columns if column not in {"seed", "acc", "bal_acc", "n"}]
    return frame.groupby(keys, dropna=False).agg(n=("n", "first"), acc=("acc", "mean"), bal_acc=("bal_acc", "mean")).reset_index().to_dict("records")


def run_baselines(table: Table) -> list[dict]:
    rows = table.rows
    text = np.hstack([text_features(rows["problem"]), table.model_onehot])
    records = averaged_over_seeds(lambda seed: oof_majority_predictions(rows, seed), rows, {"config": "majority"})
    records += averaged_over_seeds(lambda seed: oof_probe_predictions(text, rows, seed), rows, {"config": "text"})
    return records


def run_shared(table: Table, encoder_id: str, pooling: str) -> list[dict]:
    rows = table.rows
    stacked = load_encoder_features(encoder_id, pooling, rows["problem_id"])
    records = []
    for layer in range(stacked.shape[0]):
        features = np.hstack([stacked[layer], table.model_onehot])
        config = {"config": "shared", "encoder": encoder_id, "pooling": pooling, "layer": layer, "rel_depth": layer / (stacked.shape[0] - 1)}
        records += averaged_over_seeds(lambda seed: oof_probe_predictions(features, rows, seed), rows, config)
    return records


def run_self(table: Table, target_id: str, pooling: str) -> list[dict]:
    rows = table.rows[table.rows["model_id"] == target_id].reset_index(drop=True)
    if len(rows) < MIN_ROWS_FOR_SELF_PROBE or rows["is_robust"].nunique() < 2:
        return []
    stacked = load_encoder_features(target_id, pooling, rows["problem_id"])
    records = []
    for layer in range(stacked.shape[0]):
        config = {"config": "self", "encoder": target_id, "pooling": pooling, "layer": layer, "rel_depth": layer / (stacked.shape[0] - 1)}
        records += averaged_over_seeds(lambda seed: oof_probe_predictions(stacked[layer], rows, seed), rows, config)
    return records


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    official = frame[frame["subset"] == "official"]
    keys = ["config", "encoder", "pooling"]
    best = official.sort_values("bal_acc", ascending=False).groupby(keys, dropna=False).head(1)
    return best[keys + ["layer", "rel_depth", "n", "acc", "bal_acc"]].sort_values("bal_acc", ascending=False)


def main() -> None:
    table = load_table()
    available = [model for model in SMALL_TRACK_MODELS if (FEATURES_DIR / f"{safe_model_id(model)}.npz").exists()]
    tasks = [delayed(run_baselines)(table)]
    for encoder_id in available:
        for pooling in ("last", "mean"):
            tasks.append(delayed(run_shared)(table, encoder_id, pooling))
            tasks.append(delayed(run_self)(table, encoder_id, pooling))
    results = Parallel(n_jobs=min(len(tasks), os.cpu_count() or 1), verbose=5)(tasks)
    records = [record for result in results for record in result]

    frame = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame[frame["config"].isin(["majority", "text"])][["config", "subset", "n", "acc", "bal_acc"]].to_string(index=False))
        print()
        print(summarize(frame).to_string(index=False))


if __name__ == "__main__":
    main()
