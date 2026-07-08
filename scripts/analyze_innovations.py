#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator
from care_judge.evaluation.metrics import calibration_report
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.utils import read_jsonl

FEATURE_GROUPS = {
    "base_conf_only": ["confidence", "feat_base_conf", "feat_mean_conf", "feat_std_conf"],
    "self_consistency": ["confidence", "feat_self_vote_share", "feat_self_entropy", "feat_adaptive_calls"],
    "position_swap": ["confidence", "feat_swap_consistency", "feat_swap_conf_gap"],
    "rubric_stability": ["confidence", "feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip"],
    "sim_annotators": ["confidence", "feat_sim_vote_share", "feat_sim_entropy", "feat_sim_flip"],
    "bias_proxy": ["confidence", "feat_length_gap_norm"],
    "full_care": [],
}

DROP_GROUPS = {
    "full_minus_self": ["feat_self_vote_share", "feat_self_entropy", "feat_adaptive_calls"],
    "full_minus_swap": ["feat_swap_consistency", "feat_swap_conf_gap"],
    "full_minus_rubric": ["feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip"],
    "full_minus_sim": ["feat_sim_vote_share", "feat_sim_entropy", "feat_sim_flip"],
    "full_minus_bias_proxy": ["feat_length_gap_norm"],
}


def all_feature_cols(rows: List[Dict[str, Any]]) -> List[str]:
    cols = sorted([k for k in rows[0] if k.startswith("feat_")])
    if "confidence" in rows[0]:
        cols = ["confidence"] + cols
    return cols


def existing(cols: List[str], rows: List[Dict[str, Any]]) -> List[str]:
    keys = set(rows[0].keys())
    return [c for c in cols if c in keys]


def split_rows(rows: List[Dict[str, Any]], seed: int, frac: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labeled = [r for r in rows if r.get("correct") is not None]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n = max(2, int(len(labeled) * frac))
    return labeled[:n], labeled[n:]


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not reports:
        return out
    keys = sorted({k for r in reports for k, v in r.items() if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v))})
    for k in keys:
        vals = [float(r[k]) for r in reports if isinstance(r.get(k), (int, float)) and r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
        if vals:
            mean = sum(vals) / len(vals)
            out[k] = mean
            out[f"{k}_std"] = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
    return out


def evaluate_feature_set(rows: List[Dict[str, Any]], cols: List[str], alpha: float, delta: float, min_keep: int, seeds: int) -> Dict[str, Any]:
    if not cols:
        cols = all_feature_cols(rows)
    cols = existing(cols, rows)
    fold_reports = []
    for seed in range(seeds):
        train, test = split_rows(rows, seed, 0.5)
        if len(test) < 2:
            continue
        bundle = fit_calibrator(train, method="logistic", feature_cols=cols)
        probs = bundle.predict_proba(test)
        cal = calibration_report(test, probs)
        labeled = [(p, int(r["correct"])) for r, p in zip(test, probs) if r.get("correct") is not None]
        threshold, _ = calibrate_threshold([x[0] for x in labeled], [x[1] for x in labeled], alpha=alpha, delta=delta, min_keep=min_keep)
        selected = apply_threshold(test, probs, threshold)
        sel = summarize_selective(selected)
        fold_reports.append({**cal, **{f"selective_{k}": v for k, v in sel.items()}, "threshold": threshold})
    return aggregate(fold_reports) | {"feature_cols": ";".join(cols), "folds": len(fold_reports)}


def signal_failure_rates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = [
        ("position_consistent", lambda r: float(r.get("feat_swap_consistency", 0)) >= 1.0),
        ("position_inconsistent", lambda r: float(r.get("feat_swap_consistency", 0)) < 1.0),
        ("rubric_stable", lambda r: float(r.get("feat_rubric_flip", 1)) == 0.0),
        ("rubric_unstable", lambda r: float(r.get("feat_rubric_flip", 0)) > 0.0),
        ("self_high_agree", lambda r: float(r.get("feat_self_vote_share", 0)) >= 0.8),
        ("self_low_agree", lambda r: float(r.get("feat_self_vote_share", 1)) < 0.8),
        ("sim_high_agree", lambda r: float(r.get("feat_sim_vote_share", 0)) >= 0.8),
        ("sim_low_agree", lambda r: float(r.get("feat_sim_vote_share", 1)) < 0.8),
    ]
    out = []
    for name, pred in specs:
        group = [r for r in rows if r.get("correct") is not None and pred(r)]
        if not group:
            out.append({"signal_group": name, "n": 0, "accuracy": None, "error_rate": None})
        else:
            acc = sum(int(r["correct"]) for r in group) / len(group)
            out.append({"signal_group": name, "n": len(group), "accuracy": acc, "error_rate": 1 - acc})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    rows = [r for r in read_jsonl(args.features) if r.get("correct") is not None]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cols = all_feature_cols(rows)
    reports = []
    for name, cols in FEATURE_GROUPS.items():
        report = evaluate_feature_set(rows, cols if cols else all_cols, args.alpha, args.delta, args.min_keep, args.seeds)
        report.update({"dataset": args.dataset, "variant": name, "type": "feature_group"})
        reports.append(report)
    for name, drop in DROP_GROUPS.items():
        cols = [c for c in all_cols if c not in drop]
        report = evaluate_feature_set(rows, cols, args.alpha, args.delta, args.min_keep, args.seeds)
        report.update({"dataset": args.dataset, "variant": name, "type": "ablation"})
        reports.append(report)
    signal_rows = signal_failure_rates(rows)
    for r in signal_rows:
        r["dataset"] = args.dataset
    with open(out_dir / f"{args.dataset}_innovation_ablation.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(out_dir / f"{args.dataset}_signal_failure_rates.json", "w", encoding="utf-8") as f:
        json.dump(signal_rows, f, indent=2)
    fieldnames = sorted({k for r in reports for k in r})
    with open(out_dir / f"{args.dataset}_innovation_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(reports)
    with open(out_dir / f"{args.dataset}_signal_failure_rates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in signal_rows for k in r})); w.writeheader(); w.writerows(signal_rows)
    print(json.dumps({"dataset": args.dataset, "n": len(rows), "reports": reports, "signal_failure_rates": signal_rows}, indent=2))


if __name__ == "__main__":
    main()
