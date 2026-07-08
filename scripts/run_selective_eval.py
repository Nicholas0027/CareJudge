#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import CalibratorBundle
from care_judge.evaluation.metrics import calibration_report
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Run risk-controlled selective evaluation")
    p.add_argument("--input", required=True)
    p.add_argument("--calibrator", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    args = p.parse_args()

    rows = read_jsonl(args.input)
    bundle = CalibratorBundle.load(args.calibrator)
    p_correct = bundle.predict_proba(rows)
    labeled = [(p, int(r["correct"])) for r, p in zip(rows, p_correct) if r.get("correct") is not None]
    threshold, trace = calibrate_threshold([x[0] for x in labeled], [x[1] for x in labeled], alpha=args.alpha, delta=args.delta, min_keep=args.min_keep)
    selected = apply_threshold(rows, p_correct, threshold)
    write_jsonl(args.out, selected)
    report = {"threshold": threshold, "threshold_trace": trace, "calibration": calibration_report(rows, p_correct), "selective": summarize_selective(selected)}
    report_path = args.report or args.out.replace(".jsonl", ".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["selective"], indent=2))
    print(f"wrote selected rows to {args.out} and report to {report_path}")


if __name__ == "__main__":
    main()
