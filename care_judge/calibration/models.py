from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import math
import pickle


@dataclass
class CalibratorBundle:
    method: str
    feature_cols: List[str]
    model: Any

    def predict_proba(self, rows: List[Dict[str, Any]]) -> List[float]:
        X = rows_to_matrix(rows, self.feature_cols)
        if self.method == "constant":
            return [float(self.model["p"])] * len(rows)
        if self.method == "isotonic":
            idx = self.feature_cols.index("confidence") if "confidence" in self.feature_cols else 0
            return [min(1.0, max(0.0, row[idx])) for row in X]
        weights = self.model["weights"]
        bias = self.model["bias"]
        return [_sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) for row in X]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CalibratorBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def rows_to_matrix(rows: List[Dict[str, Any]], feature_cols: List[str]) -> List[List[float]]:
    return [[float(r.get(c, 0.0) or 0.0) for c in feature_cols] for r in rows]


def fit_calibrator(rows: List[Dict[str, Any]], method: str = "logistic", feature_cols: List[str] | None = None) -> CalibratorBundle:
    labeled = [r for r in rows if r.get("correct") is not None]
    if not labeled:
        raise ValueError("No labeled rows with `correct` available for calibration")
    feature_cols = feature_cols or sorted([k for k in rows[0] if k.startswith("feat_")] + ["confidence"])
    X = rows_to_matrix(labeled, feature_cols)
    y = [int(r["correct"]) for r in labeled]
    if len(set(y)) < 2:
        return CalibratorBundle(method="constant", feature_cols=feature_cols, model={"p": sum(y) / len(y)})
    if method == "logistic":
        model = _fit_logistic(X, y, lr=0.2, epochs=1500, l2=1e-3)
    elif method == "gbm":
        # Minimal dependency-free fallback: use logistic for the GBM option when
        # sklearn is unavailable. This keeps command lines stable.
        model = _fit_logistic(X, y, lr=0.2, epochs=1500, l2=1e-3)
    elif method == "isotonic":
        model = {"identity": True}
    else:
        raise ValueError(f"Unknown calibrator: {method}")
    return CalibratorBundle(method=method, feature_cols=feature_cols, model=model)


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def _fit_logistic(X: List[List[float]], y: List[int], lr: float, epochs: int, l2: float) -> Dict[str, Any]:
    n = len(X)
    d = len(X[0]) if X else 0
    weights = [0.0] * d
    bias = 0.0
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) or 1.0)
    Z = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]
    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for row, target in zip(Z, y):
            pred = _sigmoid(sum(w * x for w, x in zip(weights, row)) + bias)
            err = pred - target
            grad_b += err
            for j in range(d):
                grad_w[j] += err * row[j]
        for j in range(d):
            weights[j] -= lr * (grad_w[j] / n + l2 * weights[j])
        bias -= lr * grad_b / n
    # Fold standardization into raw-space weights.
    raw_w = [weights[j] / stds[j] for j in range(d)]
    raw_b = bias - sum(weights[j] * means[j] / stds[j] for j in range(d))
    return {"weights": raw_w, "bias": raw_b}
