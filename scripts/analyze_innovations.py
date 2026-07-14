#!/usr/bin/env python3
"""Innovation validation with methodologically valid held-out evaluation.

For each feature group (single-signal and full CARE) and each leave-one-signal-out
ablation, we:
  1. fit the calibrator on a TRAIN split,
  2. select the risk-controlled threshold on a disjoint CALIBRATION split,
  3. report discrimination (AUROC/AUPRC/ECE/Brier) and selective risk/coverage on
     a disjoint TEST split,
averaged over multiple seeds with bootstrap confidence intervals.
"""
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
from care_judge.evaluation.metrics import calibration_report, bootstrap_ci
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


def three_way(rows: List[Dict[str, Any]], seed: int, train_frac: float, cal_frac: float):
    labeled = [r for r in rows if r.get("correct") is not None]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n = len(labeled)
    n_train = max(2, int(n * train_frac))
    n_cal = max(2, int(n * cal_frac))
    train = labeled[:n_train]
    cal = labeled[n_train:n_train + n_cal]
    test = labeled[n_train + n_cal:]
    if not test:
        test = cal
    return train, cal, test


def aggregate(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not reports:
        return out
    numeric_keys = sorted({k for r in reports for k, v in r.items() if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v))})
    for k in numeric_keys:
        vals = [float(r[k]) for r in reports if isinstance(r.get(k), (int, float)) and r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
        if vals:
            mean = sum(vals) / len(vals)
            out[k] = mean
            out[f"{k}_std"] = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
            lo, hi = bootstrap_ci(vals)
            out[f"{k}_ci_low"], out[f"{k}_ci_high"] = lo, hi
    return out


def evaluate_feature_set(rows, cols, alpha, delta, min_keep, seeds, method, bound, train_frac, cal_frac):
    if not cols:
        cols = all_feature_cols(rows)
    cols = existing(cols, rows)
    fold_reports = []
    for seed in range(seeds):
        train, cal, test = three_way(rows, seed, train_frac, cal_frac)
        if len(test) < 2 or len(cal) < 2 or len(train) < 2:
            continue
        try:
            bundle = fit_calibrator(train, method=method, feature_cols=cols)
        except Exception:
            continue
        p_cal = bundle.predict_proba(cal)
        cal_pairs = [(pp, int(r["correct"])) for r, pp in zip(cal, p_cal)]
        threshold, _ = calibrate_threshold([x[0] for x in cal_pairs], [x[1] for x in cal_pairs], alpha=alpha, delta=delta, min_keep=min_keep, bound=bound)
        p_test = bundle.predict_proba(test)
        cal_rep = calibration_report(test, p_test)
        selected = apply_threshold(test, p_test, threshold)
        sel = summarize_selective(selected)
        fold_reports.append({**cal_rep, **{f"selective_{k}": v for k, v in sel.items() if isinstance(v, (int, float))}, "threshold": threshold})
    agg = aggregate(fold_reports)
    agg["feature_cols"] = ";".join(cols)
    agg["folds"] = len(fold_reports)
    return agg


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
    p.add_argument("--method", default="logistic", choices=["logistic", "isotonic", "gbm"])
    p.add_argument("--bound", default="clopper_pearson", choices=["clopper_pearson", "hoeffding"])
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--cal-frac", type=float, default=0.3)
    args = p.parse_args()

    rows = [r for r in read_jsonl(args.features) if r.get("correct") is not None]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cols = all_feature_cols(rows)

    reports = []
    for name, cols in FEATURE_GROUPS.items():
        rep = evaluate_feature_set(rows, cols or all_cols, args.alpha, args.delta, args.min_keep, args.seeds, args.method, args.bound, args.train_frac, args.cal_frac)
        rep.update({"dataset": args.dataset, "variant": name, "type": "feature_group"})
        reports.append(rep)
    for name, drop in DROP_GROUPS.items():
        cols = [c for c in all_cols if c not in drop]
        rep = evaluate_feature_set(rows, cols, args.alpha, args.delta, args.min_keep, args.seeds, args.method, args.bound, args.train_frac, args.cal_frac)
        rep.update({"dataset": args.dataset, "variant": name, "type": "ablation"})
        reports.append(rep)

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

    print(json.dumps({"dataset": args.dataset, "n": len(rows), "method": args.method, "bound": args.bound, "reports": reports, "signal_failure_rates": signal_rows}, indent=2))


if __name__ == "__main__":
    main()
