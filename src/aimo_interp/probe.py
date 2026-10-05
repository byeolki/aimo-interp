"""Standardized logistic-regression probe stored as plain numpy arrays.

Storing raw arrays instead of a pickled sklearn estimator keeps the submission loadable
regardless of sklearn pickling details; inference needs numpy only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_C = 0.05


@dataclass(frozen=True)
class LinearProbe:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float

    def decision(self, features: np.ndarray) -> np.ndarray:
        """Logit of P(robust), shifted so that ``>= 0`` means robust."""
        standardized = (features - self.mean) / self.scale
        return standardized @ self.coef + self.intercept - self.threshold

    def predict(self, features: np.ndarray) -> list[bool]:
        return [bool(value >= 0.0) for value in self.decision(features)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean.astype(np.float32),
            scale=self.scale.astype(np.float32),
            coef=self.coef.astype(np.float32),
            intercept=np.float32(self.intercept),
            threshold=np.float32(self.threshold),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                mean=arrays["mean"],
                scale=arrays["scale"],
                coef=arrays["coef"],
                intercept=float(arrays["intercept"]),
                threshold=float(arrays["threshold"]),
            )


def fit_probe(features: np.ndarray, labels: np.ndarray, c: float = DEFAULT_C) -> LinearProbe:
    """Fit an L2 logistic regression on standardized features.

    ``class_weight="balanced"`` because the label mix differs a lot between sources
    (official rows ~70% robust, augmented MATH rows ~26%). The threshold is left at 0;
    tuning it on n < 200 rows would mostly fit noise.
    """
    from sklearn.linear_model import LogisticRegression

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    classifier.fit((features - mean) / scale, labels.astype(int))
    return LinearProbe(
        mean=mean,
        scale=scale,
        coef=classifier.coef_[0],
        intercept=float(classifier.intercept_[0]),
        threshold=0.0,
    )
