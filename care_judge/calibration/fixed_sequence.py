from __future__ import annotations

from typing import Iterable, Tuple
import math


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    # Dependency-free conservative Hoeffding upper bound. If scipy is available,
    # users may replace this with exact Clopper-Pearson; this bound is valid but
    # slightly more conservative.
    empirical = errors / n
    return min(1.0, empirical + math.sqrt(math.log(1 / max(delta, 1e-12)) / (2 * n)))


def calibrate_threshold(conf: Iterable[float], correct: Iterable[int], alpha: float = 0.1, delta: float = 0.1, min_keep: int = 20) -> Tuple[float, dict]:
    pairs = sorted(zip([float(x) for x in conf], [int(x) for x in correct]), key=lambda x: -x[0])
    sorted_conf = [p[0] for p in pairs]
    sorted_correct = [p[1] for p in pairs]
    best = 1.01
    trace = []
    for i in range(len(sorted_conf)):
        n = i + 1
        errors = int(sum(1 - x for x in sorted_correct[:n]))
        risk_upper = clopper_pearson_upper(errors, n, delta)
        trace.append({"threshold": float(sorted_conf[i]), "n": n, "errors": errors, "risk_upper": risk_upper})
        if n >= min_keep and risk_upper <= alpha:
            best = float(sorted_conf[i])
        elif n >= min_keep and best <= 1.0:
            break
    return best, {"alpha": alpha, "delta": delta, "min_keep": min_keep, "trace": trace}
