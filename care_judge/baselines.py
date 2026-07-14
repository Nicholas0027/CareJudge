from __future__ import annotations

from typing import Any, Dict, List

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.selective.evaluate import summarize_selective


def _prob_from_feature(row: Dict[str, Any], feature: str, default: float = 0.5) -> float:
    if feature == "confidence":
        return float(row.get("confidence", default) or default)
    return float(row.get(f"feat_{feature}", row.get(feature, default)) or default)


def _accepted_rows(rows: List[Dict[str, Any]], scores: List[float], threshold: float) -> List[Dict[str, Any]]:
    out = []
    for row, score in zip(rows, scores):
        r = dict(row)
        r["p_correct"] = float(score)
        r["accepted"] = bool(score >= threshold)
        r["final_pred"] = r.get("pred") if r["accepted"] else None
        out.append(r)
    return out


def run_feature_threshold_baseline(
    cal_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    feature: str,
    alpha: float,
    delta: float,
    min_keep: int,
    bound: str = "clopper_pearson",
) -> Dict[str, Any]:
    """Select threshold on the calibration split, evaluate on the test split."""
    cal_scores = [_prob_from_feature(r, feature) for r in cal_rows]
    cal_labeled = [(s, int(r["correct"])) for s, r in zip(cal_scores, cal_rows) if r.get("correct") is not None]
    threshold, trace = calibrate_threshold(
        [x[0] for x in cal_labeled], [x[1] for x in cal_labeled],
        alpha=alpha, delta=delta, min_keep=min_keep, bound=bound,
    )
    test_scores = [_prob_from_feature(r, feature) for r in test_rows]
    selected = _accepted_rows(test_rows, test_scores, threshold)
    report = summarize_selective(selected)
    report.update({"baseline": feature, "type": "thresholded", "threshold": threshold})
    return report


def run_rule_baseline(test_rows: List[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    out = []
    for row in test_rows:
        r = dict(row)
        if rule == "raw":
            accept = True
        elif rule == "position_swap":
            accept = float(row.get("feat_swap_consistency", 0.0) or 0.0) >= 1.0
        elif rule == "rubric_stable":
            accept = float(row.get("feat_rubric_flip", 1.0) or 1.0) == 0.0
        elif rule == "simulated_annotators":
            accept = float(row.get("feat_sim_vote_share", 0.0) or 0.0) >= 0.8
        elif rule == "self_consistency":
            accept = float(row.get("feat_self_vote_share", 0.0) or 0.0) >= 0.8
        else:
            raise ValueError(rule)
        r["accepted"] = accept
        r["final_pred"] = r.get("pred") if accept else None
        r["p_correct"] = float(row.get("confidence", 0.5) or 0.5)
        out.append(r)
    report = summarize_selective(out)
    report.update({"baseline": rule, "type": "rule", "threshold": None})
    return report


def run_all_baselines(
    cal_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    alpha: float = 0.1,
    delta: float = 0.1,
    min_keep: int = 20,
    bound: str = "clopper_pearson",
) -> List[Dict[str, Any]]:
    reports = []
    for rule in ["raw", "position_swap", "rubric_stable", "self_consistency", "simulated_annotators"]:
        reports.append(run_rule_baseline(test_rows, rule))
    for feat in ["confidence", "base_conf", "mean_conf", "self_vote_share", "rubric_vote_share", "swap_consistency", "sim_vote_share", "ensemble_vote_share"]:
        if feat == "confidence" or any(f"feat_{feat}" in r for r in test_rows):
            reports.append(run_feature_threshold_baseline(cal_rows, test_rows, feat, alpha=alpha, delta=delta, min_keep=min_keep, bound=bound))
    return reports
