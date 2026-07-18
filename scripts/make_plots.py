#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export reliability-plot data as CSV; draws PNG if matplotlib exists")
    p.add_argument("--selected", required=True, help="selected.jsonl with p_correct and correct")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()
    rows = []
    with open(args.selected, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("correct") is not None and r.get("p_correct") is not None:
                    rows.append(r)
    bins = []
    for i in range(args.bins):
        lo, hi = i / args.bins, (i + 1) / args.bins
        group = [r for r in rows if float(r["p_correct"]) >= lo and (float(r["p_correct"]) < hi if hi < 1 else float(r["p_correct"]) <= hi)]
        if group:
            bins.append({"lo": lo, "hi": hi, "n": len(group), "mean_conf": sum(float(r["p_correct"]) for r in group) / len(group), "accuracy": sum(int(r["correct"]) for r in group) / len(group)})
    csv_path = f"{args.out_prefix}_reliability.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("lo,hi,n,mean_conf,accuracy\n")
        for b in bins:
            f.write(f"{b['lo']},{b['hi']},{b['n']},{b['mean_conf']},{b['accuracy']}\n")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.scatter([b["mean_conf"] for b in bins], [b["accuracy"] for b in bins])
        plt.xlabel("Predicted correctness")
        plt.ylabel("Empirical accuracy")
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_reliability.png", dpi=200)
    except Exception:
        pass
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
