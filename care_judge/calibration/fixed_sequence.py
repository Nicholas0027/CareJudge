from __future__ import annotations

from typing import Iterable, List, Tuple
import math


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(Bin(n,p) <= k) computed stably in log-space."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("-inf") if k < n else 0.0
    # Sum probabilities up to k using log-space accumulation.
    log_p = math.log(p)
    log_q = math.log1p(-p)
    # log C(n, i) via lgamma
    def log_comb(i: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)

    terms = [log_comb(i) + i * log_p + (n - i) * log_q for i in range(0, k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson (1-delta) upper confidence bound on the
    error rate, given `errors` observed among `n` trials.

    Defined as sup{ p : P(Bin(n, p) <= errors) >= delta }. Solved by bisection
    on the monotone binomial CDF. Uses only the standard library.
    """
    if n <= 0:
        return 1.0
    if errors >= n:
        return 1.0
    delta = min(max(delta, 1e-12), 1.0)
    lo, hi = errors / n, 1.0
    # Bisection: cdf(errors; n, p) is decreasing in p.
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        cdf = math.exp(_log_binom_cdf(errors, n, mid))
        if cdf >= delta:
            lo = mid
        else:
            hi = mid
    return hi


def hoeffding_upper(errors: int, n: int, delta: float) -> float:
    """Conservative Hoeffding upper bound (kept for ablation/comparison)."""
    if n <= 0:
        return 1.0
    empirical = errors / n
    return min(1.0, empirical + math.sqrt(math.log(1.0 / max(delta, 1e-12)) / (2 * n)))


def calibrate_threshold(
    conf: Iterable[float],
    correct: Iterable[int],
    alpha: float = 0.1,
    delta: float = 0.1,
    min_keep: int = 20,
    bound: str = "clopper_pearson",
) -> Tuple[float, dict]:
    """Fixed-sequence selective-risk threshold search.

    IMPORTANT: `conf`/`correct` here MUST come from a CALIBRATION split that is
    disjoint from the evaluation/test split. The returned threshold is then
    applied on unseen data to obtain a valid (1-delta) risk guarantee.

    We sweep thresholds from most-confident downward and keep the lowest
    threshold whose (1-delta) upper risk bound remains <= alpha.
    """
    upper = clopper_pearson_upper if bound == "clopper_pearson" else hoeffding_upper
    pairs = sorted(zip([float(x) for x in conf], [int(x) for x in correct]), key=lambda x: -x[0])
    sorted_conf = [p[0] for p in pairs]
    sorted_correct = [p[1] for p in pairs]
    best = 1.01  # by default accept nothing
    trace = []
    for i in range(len(sorted_conf)):
        n = i + 1
        errors = int(sum(1 - x for x in sorted_correct[:n]))
        risk_upper = upper(errors, n, delta)
        trace.append({"threshold": float(sorted_conf[i]), "n": n, "errors": errors, "risk_upper": risk_upper})
        if n >= min_keep and risk_upper <= alpha:
            best = float(sorted_conf[i])
        elif n >= min_keep and best <= 1.0:
            # Fixed-sequence testing: stop at first violation after acceptance.
            break
    return best, {"alpha": alpha, "delta": delta, "min_keep": min_keep, "bound": bound, "trace": trace}