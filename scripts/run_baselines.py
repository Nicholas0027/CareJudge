#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.baselines import run_all_baselines
from care_judge.utils import read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Run offline baselines on CARE feature rows")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    args = p.parse_args()
    rows = read_jsonl(args.input)
    reports = run_all_baselines(rows, alpha=args.alpha, delta=args.delta, min_keep=args.min_keep)
    write_jsonl(args.out, reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
