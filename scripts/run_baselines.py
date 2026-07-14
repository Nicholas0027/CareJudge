#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.baselines import run_all_baselines
from care_judge.utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Baselines with disjoint calibration/test threshold selection")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cal-frac", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--bound", default="clopper_pearson", choices=["clopper_pearson", "hoeffding"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [r for r in read_jsonl(args.input) if r.get("correct") is not None]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_cal = max(2, int(len(rows) * args.cal_frac))
    cal, test = rows[:n_cal], rows[n_cal:]
    if not test:
        test = cal
    reports = run_all_baselines(cal, test, alpha=args.alpha, delta=args.delta, min_keep=args.min_keep, bound=args.bound)
    write_jsonl(args.out, reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
