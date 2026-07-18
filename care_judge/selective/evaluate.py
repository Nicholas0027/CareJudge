from __future__ import annotations

from typing import Any, Dict, List


def apply_threshold(rows: List[Dict[str, Any]], p_correct: List[float], threshold: float) -> List[Dict[str, Any]]:
    out = []
    for row, p in zip(rows, p_correct):
        r = dict(row)
        r["p_correct"] = float(p)
        r["accepted"] = bool(p >= threshold)
        r["final_pred"] = r["pred"] if r["accepted"] else None
        out.append(r)
    return out


def summarize_selective(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    accepted = [r for r in rows if r.get("accepted")]
    labeled = [r for r in accepted if r.get("correct") is not None]
    all_labeled = [r for r in rows if r.get("correct") is not None]
    coverage = len(accepted) / max(1, len(rows))
    acc = sum(int(r["correct"]) for r in labeled) / max(1, len(labeled)) if labeled else None
    raw_acc = sum(int(r["correct"]) for r in all_labeled) / max(1, len(all_labeled)) if all_labeled else None
    return {
        "n": float(len(rows)),
        "accepted": float(len(accepted)),
        "coverage": float(coverage),
        "accuracy_accepted": float(acc) if acc is not None else None,
        "risk_accepted": float(1 - acc) if acc is not None else None,
        "raw_accuracy": float(raw_acc) if raw_acc is not None else None,
        "mean_cost": float(sum(float(r.get("feat_cost", 0.0) or 0.0) for r in rows) / max(1, len(rows))),
    }
