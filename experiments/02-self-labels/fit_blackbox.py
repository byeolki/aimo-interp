#!/usr/bin/env python3
"""Fit the black-box ablation heads in submissions/self-probe-blackbox.

Mirrors export_blend.py (the submitted self-probe) with the hidden state replaced by four
surface text features: same training pool (all 150 sampled problems plus the official
train-main-v2 rows, i.e. EXCLUDE_LEAKAGE=0 as for the submitted v1), same routes (ridge
solve rate -> step-2 logistic on [rate, rate^2, model one-hot], plus a direct logistic),
same C values and class weighting, and the same out-of-fold spread scaling. Ridge alpha
is 1.0 instead of 10000 because there are 4 inputs instead of thousands.

Standard library only so the result can be checked without numpy/sklearn: logistic
regression is solved exactly with Newton's method on sklearn's objective
(0.5*|w|^2 + C * sum(weight * logloss), unpenalized intercept), and folds follow
sklearn's GroupKFold assignment (largest groups first into the lightest fold).
Requires data/labels.jsonl from scripts/fetch_labels.py.
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submissions" / "self-probe-blackbox" / "artifacts" / "blackbox_heads.json"
MODELS = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
RIDGE_ALPHA, DIRECT_C, STEP2_C, N_FOLDS = 1.0, 0.05, 1.0, 5


def text_features(text: str) -> list[float]:
    numbers = [int(match) for match in re.findall(r"\d+", text)[:50]]
    return [
        math.log1p(len(text)),
        math.log1p(sum(ch.isdigit() for ch in text)),
        math.log1p(text.count("$") + text.count("\\")),
        math.log1p(max(numbers)) if numbers else 0.0,
    ]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def standardizer(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    n, d = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(d)]
    scale = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in rows) / n) for j in range(d)]
    return mean, [s if s >= 1e-6 else 1.0 for s in scale]


def apply_z(rows, stats):
    mean, scale = stats
    return [[(v - m) / s for v, m, s in zip(r, mean, scale)] for r in rows]


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def ridge_fit(x, y, alpha):
    n, d = len(x), len(x[0])
    xm = [sum(r[j] for r in x) / n for j in range(d)]
    ym = sum(y) / n
    a = [[sum((r[j] - xm[j]) * (r[k] - xm[k]) for r in x) + (alpha if j == k else 0.0) for k in range(d)] for j in range(d)]
    b = [sum((r[j] - xm[j]) * (t - ym) for r, t in zip(x, y)) for j in range(d)]
    w = solve(a, b)
    return w, ym - sum(wj * mj for wj, mj in zip(w, xm))


def logistic_fit(x, y, c):
    n, d = len(x), len(x[0])
    positives = sum(y)
    weights = [n / (2 * positives) if t else n / (2 * (n - positives)) for t in y]
    w = [0.0] * (d + 1)
    for _ in range(100):
        grad = [w[j] if j < d else 0.0 for j in range(d + 1)]
        hess = [[1.0 if (j == k and j < d) else 0.0 for k in range(d + 1)] for j in range(d + 1)]
        for row, target, weight in zip(x, y, weights):
            xi = row + [1.0]
            p = 1.0 / (1.0 + math.exp(-sum(a * b for a, b in zip(xi, w))))
            e, h = c * weight * (p - target), c * weight * p * (1 - p)
            for j in range(d + 1):
                grad[j] += e * xi[j]
                for k in range(d + 1):
                    hess[j][k] += h * xi[j] * xi[k]
        step = solve(hess, grad)
        w = [a - b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return w[:d], w[d]


def decision(model, rows):
    coef, intercept = model
    return [sum(a * b for a, b in zip(r, coef)) + intercept for r in rows]


def group_k_fold(groups: list[str], k: int):
    counts: dict[str, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    ordered = sorted(enumerate(sorted(counts)), key=lambda item: (-counts[item[1]], item[0]))
    load, fold = [0] * k, {}
    for _, g in ordered:
        f = load.index(min(load))
        fold[g] = f
        load[f] += counts[g]
    return [([i for i, g in enumerate(groups) if fold[g] != f], [i for i, g in enumerate(groups) if fold[g] == f]) for f in range(k)]


def stdev(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def fit_ridge(rows):
    x = [text_features(r["problem"]) for r in rows]
    stats = standardizer(x)
    return stats, ridge_fit(apply_z(x, stats), [r["base"] for r in rows], RIDGE_ALPHA)


def main() -> None:
    generated = read_jsonl(ROOT / "data" / "self_labels.jsonl")
    official = [r for r in read_jsonl(ROOT / "data" / "labels.jsonl") if r["source"] == "train-main-v2"]
    frame = [r for r in generated if r["is_robust"] is not None and r["model_id"] in MODELS]
    frame += [r for r in official if r["model_id"] in MODELS]
    solve_rows = [r for r in generated if r["model_id"] in MODELS]
    labels = [bool(r["is_robust"]) for r in frame]
    onehot = lambda m: [1.0 if m == other else 0.0 for other in MODELS]

    oof_rate = [0.0] * len(frame)
    for _, test in group_k_fold([r["problem_id"] for r in frame], N_FOLDS):
        held = {frame[i]["problem_id"] for i in test}
        for m in MODELS:
            stats, ridge = fit_ridge([r for r in solve_rows if r["model_id"] == m and r["problem_id"] not in held])
            idx = [i for i in test if frame[i]["model_id"] == m]
            if idx:
                preds = decision(ridge, apply_z([text_features(frame[i]["problem"]) for i in idx], stats))
                for i, p in zip(idx, preds):
                    oof_rate[i] = min(1.0, max(0.0, p))
    step2_x = [[rate, rate * rate, *onehot(r["model_id"])] for rate, r in zip(oof_rate, frame)]
    step2 = logistic_fit(step2_x, labels, STEP2_C)
    two_step_spread = stdev(decision(step2, step2_x))

    heads = {}
    for m in MODELS:
        rows = [i for i, r in enumerate(frame) if r["model_id"] == m]
        (r_mean, r_scale), ridge = fit_ridge([r for r in solve_rows if r["model_id"] == m])
        x = [text_features(frame[i]["problem"]) for i in rows]
        y = [labels[i] for i in rows]
        d_stats = standardizer(x)
        direct = logistic_fit(apply_z(x, d_stats), y, DIRECT_C)
        oof_direct = [0.0] * len(rows)
        for train, test in group_k_fold([frame[i]["problem_id"] for i in rows], N_FOLDS):
            if len({y[i] for i in train}) < 2:
                continue
            stats = standardizer([x[i] for i in train])
            clf = logistic_fit(apply_z([x[i] for i in train], stats), [y[i] for i in train], DIRECT_C)
            for i, p in zip(test, decision(clf, apply_z([x[i] for i in test], stats))):
                oof_direct[i] = p
        heads[m] = {
            "ridge_mean": r_mean, "ridge_scale": r_scale, "ridge_coef": ridge[0], "ridge_intercept": ridge[1],
            "step2_coef": step2[0], "step2_intercept": step2[1], "model_onehot": onehot(m),
            "direct_mean": d_stats[0], "direct_scale": d_stats[1], "direct_coef": direct[0], "direct_intercept": direct[1],
            "two_step_spread": two_step_spread, "direct_spread": stdev(oof_direct) or 1.0, "rows": len(rows),
        }
    config = {"heads": heads, "inputs": "four surface text features; model one-hot in step 2",
              "ridge_alpha": RIDGE_ALPHA, "direct_C": DIRECT_C, "step2_C": STEP2_C, "folds": N_FOLDS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print({m: h["rows"] for m, h in heads.items()})


if __name__ == "__main__":
    main()
