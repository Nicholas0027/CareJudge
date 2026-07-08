#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tqdm import tqdm
except ImportError:  # keep smoke tests runnable in minimal Python envs
    def tqdm(x, **kwargs):
        return x

from care_judge.data.loaders import load_jsonl_pairs
from care_judge.judges.factory import make_judge
from care_judge.uncertainty.feature_builder import collect_uncertainty_features, record_to_row
from care_judge.utils import set_seed, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Collect CARE-Judge uncertainty features")
    p.add_argument("--input", required=True, help="Pairwise JSONL dataset")
    p.add_argument("--out", required=True, help="Output JSONL feature file")
    p.add_argument("--judge", default="mock:0.72", help="mock[:acc] or litellm:<model>")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k-self", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-swap", action="store_true")
    p.add_argument("--no-rubrics", action="store_true")
    p.add_argument("--sim-annotators", type=int, default=0, help="Number of simulated annotator prompts")
    p.add_argument("--sim-shots", type=int, default=3, help="Few-shot examples per simulated annotator")
    p.add_argument("--adaptive-k", action="store_true", help="Stop self-consistency early when stable")
    p.add_argument("--adaptive-tau", type=float, default=0.85)
    p.add_argument("--min-self", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    items = load_jsonl_pairs(args.input, limit=args.limit)
    judge = make_judge(args.judge)
    rows = []
    for item in tqdm(items, desc="collect"):
        rec = collect_uncertainty_features(
            item,
            judge,
            k_self=args.k_self,
            temperature=args.temperature,
            use_swap=not args.no_swap,
            use_rubrics=not args.no_rubrics,
            sim_examples=items,
            sim_annotators=args.sim_annotators,
            sim_shots=args.sim_shots,
            adaptive_k=args.adaptive_k,
            adaptive_tau=args.adaptive_tau,
            min_self=args.min_self,
        )
        row = record_to_row(rec)
        row["domain"] = item.domain
        rows.append(row)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
