from __future__ import annotations

from typing import Any, Dict, List

import random


def expected_calibration_error(y_true: List[int], p: List[float], n_bins: int = 10) -> float:
    bins = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, x in enumerate(p) if x >= lo and (x < hi if hi < 1 else x <= hi)]
        if not idx:
            continue
        ece += (len(idx) / len(p)) * abs(sum(y_true[i] for i in idx) / len(idx) - sum(p[i] for i in idx) / len(idx))
    return float(ece)


def calibration_report(rows: List[Dict[str, Any]], p_correct: List[float]) -> Dict[str, float]:
    y = [int(r["correct"]) for r in rows if r.get("correct") is not None]
    p = [float(prob) for r, prob in zip(rows, p_correct) if r.get("correct") is not None]
    out = {"n_labeled": float(len(y))}
    if len(y) == 0 or len(set(y)) < 2:
        return out
    out.update({
        "ece": expected_calibration_error(y, p),
        "brier": float(sum((yy - pp) ** 2 for yy, pp in zip(y, p)) / len(y)),
        "auroc": float(_auroc(y, p)),
        "auprc": float(_auprc(y, p)),
    })
    return out


def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    arr = [float(x) for x in values]
    if len(arr) == 0:
        return float("nan"), float("nan")
    samples = [sum(rng.choice(arr) for _ in arr) / len(arr) for _ in range(n_boot)]
    samples.sort()
    return samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot) - 1]


def _auroc(y: List[int], p: List[float]) -> float:
    pos = [pp for yy, pp in zip(y, p) if yy == 1]
    neg = [pp for yy, pp in zip(y, p) if yy == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def _auprc(y: List[int], p: List[float]) -> float:
    pairs = sorted(zip(p, y), key=lambda x: -x[0])
    total_pos = sum(y)
    if total_pos == 0:
        return float("nan")
    tp = 0
    area = 0.0
    prev_recall = 0.0
    for i, (_, yy) in enumerate(pairs, start=1):
        if yy == 1:
            tp += 1
        precision = tp / i
        recall = tp / total_pos
        area += precision * max(0.0, recall - prev_recall)
        prev_recall = recall
    return area
