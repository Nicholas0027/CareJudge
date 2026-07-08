#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.experiments import run_single_dataset_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Run full CARE-Judge main experiment on one dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judge", default="mock:0.72")
    p.add_argument("--method", choices=["logistic", "isotonic", "gbm"], default="logistic")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-frac", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--k-self", type=int, default=3)
    p.add_argument("--sim-annotators", type=int, default=0)
    p.add_argument("--sim-shots", type=int, default=3)
    args = p.parse_args()
    report = run_single_dataset_experiment(
        dataset_path=args.input,
        out_dir=args.out_dir,
        judge_spec=args.judge,
        method=args.method,
        limit=args.limit,
        seed=args.seed,
        calibration_frac=args.calibration_frac,
        alpha=args.alpha,
        delta=args.delta,
        min_keep=args.min_keep,
        k_self=args.k_self,
        sim_annotators=args.sim_annotators,
        sim_shots=args.sim_shots,
    )
    print(json.dumps({"care_selective": report["care_selective"], "care_calibration": report["care_calibration"]}, indent=2))


if __name__ == "__main__":
    main()
