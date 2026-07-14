#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import CalibratorBundle, fit_calibrator
from care_judge.evaluation.metrics import calibration_report
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Risk-controlled selective evaluation with disjoint calibration/test")
    p.add_argument("--input", required=True, help="Feature JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--calibrator", default=None, help="Optional pre-fit calibrator; else fit on train split")
    p.add_argument("--method", default="logistic", choices=["logistic", "isotonic", "gbm"])
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--cal-frac", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--bound", default="clopper_pearson", choices=["clopper_pearson", "hoeffding"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [r for r in read_jsonl(args.input) if r.get("correct") is not None]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = max(2, int(n * args.train_frac))
    n_cal = max(2, int(n * args.cal_frac))
    train, cal, test = rows[:n_train], rows[n_train:n_train + n_cal], rows[n_train + n_cal:]
    if not test:
        test = cal

    if args.calibrator:
        bundle = CalibratorBundle.load(args.calibrator)
    else:
        bundle = fit_calibrator(train, method=args.method)

    p_cal = bundle.predict_proba(cal)
    cal_pairs = [(pp, int(r["correct"])) for r, pp in zip(cal, p_cal)]
    threshold, trace = calibrate_threshold(
        [x[0] for x in cal_pairs], [x[1] for x in cal_pairs],
        alpha=args.alpha, delta=args.delta, min_keep=args.min_keep, bound=args.bound,
    )

    p_test = bundle.predict_proba(test)
    selected = apply_threshold(test, p_test, threshold)
    write_jsonl(args.out, selected)
    report = {
        "method": bundle.method,
        "bound": args.bound,
        "n_train": len(train), "n_calibration": len(cal), "n_test": len(test),
        "threshold": threshold,
        "threshold_trace": trace,
        "calibration_test": calibration_report(test, p_test),
        "selective_test": summarize_selective(selected),
    }
    report_path = args.report or args.out.replace(".jsonl", ".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["selective_test"], indent=2))
    print(f"wrote selected rows to {args.out} and report to {report_path}")


if __name__ == "__main__":
    main()
