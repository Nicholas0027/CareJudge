#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.models import fit_calibrator
from care_judge.evaluation.metrics import calibration_report
from care_judge.utils import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Fit CARE-Judge correctness calibrator")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["logistic", "isotonic", "gbm"], default="logistic")
    p.add_argument("--calibration-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [r for r in read_jsonl(args.input) if r.get("correct") is not None]
    random.Random(args.seed).shuffle(rows)
    n = max(2, int(len(rows) * args.calibration_frac))
    train, valid = rows[:n], rows[n:]
    bundle = fit_calibrator(train, method=args.method)
    bundle.save(args.out)
    report = {"train_n": len(train), "valid_n": len(valid), "method": args.method, "feature_cols": bundle.feature_cols}
    if valid:
        p_correct = bundle.predict_proba(valid)
        report.update(calibration_report(valid, p_correct))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
