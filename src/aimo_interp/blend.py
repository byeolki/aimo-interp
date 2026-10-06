"""Per-model blend head: solve-rate route plus direct robustness route, numpy-only inference.

Route A predicts the model's own solve rate from its hidden state (ridge), then maps
[rate, rate^2, model one-hot] to a robustness logit. Route B is a logistic probe on the same
hidden state. The decision is the sum of both logits, each divided by its spread on
out-of-fold training predictions so neither route dominates by scale.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BlendHead:
    layers: np.ndarray
    ridge_mean: np.ndarray
    ridge_scale: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: float
    step2_coef: np.ndarray
    step2_intercept: float
    model_onehot: np.ndarray
    direct_mean: np.ndarray
    direct_scale: np.ndarray
    direct_coef: np.ndarray
    direct_intercept: float
    two_step_spread: float
    direct_spread: float

    def solve_rate(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.ridge_mean) / self.ridge_scale
        return np.clip(z @ self.ridge_coef + self.ridge_intercept, 0.0, 1.0)

    def decision(self, features: np.ndarray) -> np.ndarray:
        rate = self.solve_rate(features)
        onehot = np.tile(self.model_onehot, (len(features), 1))
        step2_inputs = np.column_stack([rate, rate**2, onehot])
        two_step = step2_inputs @ self.step2_coef + self.step2_intercept
        direct = ((features - self.direct_mean) / self.direct_scale) @ self.direct_coef + self.direct_intercept
        return two_step / self.two_step_spread + direct / self.direct_spread

    def predict(self, features: np.ndarray) -> list[bool]:
        return [bool(value >= 0.0) for value in self.decision(features)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{name: np.asarray(value, dtype=np.float32 if name != "layers" else np.int64)
                          for name, value in self.__dict__.items()})

    @classmethod
    def load(cls, path: Path) -> "BlendHead":
        with np.load(path, allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        scalars = {"ridge_intercept", "step2_intercept", "direct_intercept", "two_step_spread", "direct_spread"}
        return cls(**{name: float(value) if name in scalars else value for name, value in values.items()})
