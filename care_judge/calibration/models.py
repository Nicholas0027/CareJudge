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
            return _isotonic_predict(self.model, X, self.feature_cols)
        if self.method == "gbm":
            return _sklearn_predict(self.model, X)
        # logistic (dependency-free) and sklearn-logistic both stored as dict
        if isinstance(self.model, dict) and "weights" in self.model:
            weights = self.model["weights"]
            bias = self.model["bias"]
            return [_sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) for row in X]
        return _sklearn_predict(self.model, X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CalibratorBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def rows_to_matrix(rows: List[Dict[str, Any]], feature_cols: List[str]) -> List[List[float]]:
    return [[float(r.get(c, 0.0) or 0.0) for c in feature_cols] for r in rows]


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


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
        model = _fit_logistic_sklearn(X, y) if _has_sklearn() else _fit_logistic(X, y, lr=0.2, epochs=1500, l2=1e-3)
        return CalibratorBundle(method="logistic", feature_cols=feature_cols, model=model)

    if method == "gbm":
        if _has_sklearn():
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(random_state=0)
            clf.fit(X, y)
            return CalibratorBundle(method="gbm", feature_cols=feature_cols, model=clf)
        # Honest fallback: report that GBM is unavailable by using logistic but
        # tagging the method so downstream logs are truthful.
        model = _fit_logistic(X, y, lr=0.2, epochs=1500, l2=1e-3)
        return CalibratorBundle(method="logistic_gbm_fallback", feature_cols=feature_cols, model=model)

    if method == "isotonic":
        # Real isotonic regression on the primary confidence score.
        idx = feature_cols.index("confidence") if "confidence" in feature_cols else 0
        scores = [row[idx] for row in X]
        if _has_sklearn():
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores, y)
            return CalibratorBundle(method="isotonic", feature_cols=feature_cols, model={"sklearn": iso, "idx": idx})
        pav = _pav_isotonic(scores, y)
        return CalibratorBundle(method="isotonic", feature_cols=feature_cols, model={"pav": pav, "idx": idx})

    raise ValueError(f"Unknown calibrator: {method}")


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def _fit_logistic_sklearn(X: List[List[float]], y: List[int]) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return {"weights": list(map(float, clf.coef_[0])), "bias": float(clf.intercept_[0])}


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
    raw_w = [weights[j] / stds[j] for j in range(d)]
    raw_b = bias - sum(weights[j] * means[j] / stds[j] for j in range(d))
    return {"weights": raw_w, "bias": raw_b}


def _sklearn_predict(clf: Any, X: List[List[float]]) -> List[float]:
    return [float(p[1]) for p in clf.predict_proba(X)]


def _isotonic_predict(model: Dict[str, Any], X: List[List[float]], feature_cols: List[str]) -> List[float]:
    idx = model["idx"]
    scores = [row[idx] for row in X]
    if "sklearn" in model:
        return [float(v) for v in model["sklearn"].predict(scores)]
    return [_pav_apply(model["pav"], s) for s in scores]


def _pav_isotonic(scores: List[float], y: List[int]) -> Dict[str, List[float]]:
    """Pool-adjacent-violators isotonic regression (dependency-free)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(y[i]) for i in order]
    # blocks of (sum, count, value)
    blocks: List[List[float]] = []
    for val in ys:
        blocks.append([val, 1.0, val])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            s2, c2, _ = blocks.pop()
            s1, c1, _ = blocks.pop()
            s, c = s1 + s2, c1 + c2
            blocks.append([s, c, s / c])
    fitted = []
    for s, c, v in blocks:
        fitted.extend([v] * int(c))
    return {"x": xs, "y": fitted}


def _pav_apply(pav: Dict[str, List[float]], s: float) -> float:
    xs, ys = pav["x"], pav["y"]
    if not xs:
        return 0.5
    if s <= xs[0]:
        return ys[0]
    if s >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < s:
            lo = mid + 1
        else:
            hi = mid
    return ys[lo]
