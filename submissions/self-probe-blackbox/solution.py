"""White-box ablation entry: the self-probe blend head with the hidden state removed.

Same per-model recipe as submissions/self-probe (ridge solve-rate route plus direct logistic
route, each scaled by its out-of-fold spread), but the inputs are four surface text features
of the problem instead of the model's hidden state. No model is loaded. Models without a
head (Olmo) get the constant fallback, since the self-probe used a hidden-state probe there.
"""

import json
import math
import re
from pathlib import Path

HEADS = json.loads((Path(__file__).resolve().parent / "artifacts" / "blackbox_heads.json").read_text(encoding="utf-8"))
MODEL_ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
FALLBACK_PREDICTION = True


def text_features(text: str) -> list[float]:
    numbers = [int(match) for match in re.findall(r"\d+", text)[:50]]
    return [
        math.log1p(len(text)),
        math.log1p(sum(ch.isdigit() for ch in text)),
        math.log1p(text.count("$") + text.count("\\")),
        math.log1p(max(numbers)) if numbers else 0.0,
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _z(x: list[float], mean: list[float], scale: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(x, mean, scale)]


def decision(head: dict, x: list[float]) -> float:
    rate = _dot(_z(x, head["ridge_mean"], head["ridge_scale"]), head["ridge_coef"]) + head["ridge_intercept"]
    rate = min(1.0, max(0.0, rate))
    two_step = _dot([rate, rate * rate, *head["model_onehot"]], head["step2_coef"]) + head["step2_intercept"]
    direct = _dot(_z(x, head["direct_mean"], head["direct_scale"]), head["direct_coef"]) + head["direct_intercept"]
    return two_step / head["two_step_spread"] + direct / head["direct_spread"]


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for each problem, preserving input order."""
    del reasoning_effort
    head = HEADS["heads"].get(MODEL_ALIASES.get(model_id, model_id))
    if head is None:
        return [FALLBACK_PREDICTION for _ in problems]
    return [decision(head, text_features(problem)) >= 0.0 for problem in problems]
