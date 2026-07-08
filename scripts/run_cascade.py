#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from care_judge.calibration.models import CalibratorBundle
from care_judge.data.loaders import load_jsonl_pairs
from care_judge.judges.factory import make_judge
from care_judge.selective.cascade import run_cascade_item
from care_judge.utils import write_jsonl


def parse_thresholds(s: str) -> Dict[str, float]:
    out = {}
    if not s:
        return out
    for part in s.split(","):
        name, val = part.split("=", 1)
        out[name] = float(val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run CARE-Judge cascade")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judges", required=True, help="Comma specs, e.g. cheap=mock:0.65,strong=mock:0.85")
    p.add_argument("--calibrators", default="", help="Comma paths, e.g. cheap=cheap.pkl,strong=strong.pkl")
    p.add_argument("--thresholds", required=True, help="Comma thresholds, e.g. cheap=0.8,strong=0.9")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k-self", type=int, default=3)
    p.add_argument("--sim-annotators", type=int, default=0)
    p.add_argument("--sim-shots", type=int, default=3)
    args = p.parse_args()

    judges = []
    for part in args.judges.split(","):
        name, spec = part.split("=", 1)
        judges.append((name, make_judge(spec)))
    calibrators = {}
    if args.calibrators:
        for part in args.calibrators.split(","):
            name, path = part.split("=", 1)
            calibrators[name] = CalibratorBundle.load(path)
    thresholds = parse_thresholds(args.thresholds)
    items = load_jsonl_pairs(args.input, limit=args.limit)
    rows = [run_cascade_item(item, judges, calibrators, thresholds, k_self=args.k_self, sim_examples=items, sim_annotators=args.sim_annotators, sim_shots=args.sim_shots) for item in tqdm(items, desc="cascade")]
    write_jsonl(args.out, rows)
    accepted = [r for r in rows if not r["abstained"]]
    labeled = [r for r in accepted if r.get("correct") is not None]
    report = {
        "n": len(rows),
        "coverage": len(accepted) / max(1, len(rows)),
        "accuracy_accepted": sum(r["correct"] for r in labeled) / max(1, len(labeled)) if labeled else None,
        "accepted_by": {name: sum(r["accepted_by"] == name for r in rows) for name, _ in judges} | {"abstain": sum(r["abstained"] for r in rows)},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
