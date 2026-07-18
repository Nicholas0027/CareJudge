#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator
from care_judge.evaluation.metrics import calibration_report
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-dataset calibration transfer: fit on source features, evaluate on target features")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["logistic", "isotonic", "gbm"], default="logistic")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    args = p.parse_args()
    source = [r for r in read_jsonl(args.source) if r.get("correct") is not None]
    target = read_jsonl(args.target)
    bundle = fit_calibrator(source, method=args.method)
    p_target = bundle.predict_proba(target)
    labeled_target = [(p, int(r["correct"])) for r, p in zip(target, p_target) if r.get("correct") is not None]
    threshold, trace = calibrate_threshold([x[0] for x in labeled_target], [x[1] for x in labeled_target], alpha=args.alpha, delta=args.delta, min_keep=args.min_keep)
    selected = apply_threshold(target, p_target, threshold)
    write_jsonl(args.out, selected)
    report = {"source": args.source, "target": args.target, "threshold": threshold, "threshold_trace": trace, "calibration": calibration_report(target, p_target), "selective": summarize_selective(selected)}
    with open(args.out.replace(".jsonl", ".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["selective"], indent=2))


if __name__ == "__main__":
    main()
