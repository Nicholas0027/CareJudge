#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(x, **kwargs):
        return x

from care_judge.data.loaders import load_jsonl_pairs
from care_judge.judges.factory import make_judge
from care_judge.judges.prompts import BASE_RUBRIC
from care_judge.utils import entropy_binary, majority, vote_share, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Collect multi-judge ensemble disagreement features")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judges", required=True, help="Comma-separated specs, e.g. mock:0.65,mock:0.8,litellm:gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    items = load_jsonl_pairs(args.input, limit=args.limit)
    judges = [make_judge(s) for s in args.judges.split(",")]
    rows = []
    for item in tqdm(items, desc="ensemble"):
        votes = []
        confs = []
        raw = []
        for judge in judges:
            j = judge.judge(item, rubric=BASE_RUBRIC, temperature=0.0)
            votes.append(j.normalized_winner())
            confs.append(j.confidence)
            raw.append({"judge": judge.name, "winner": j.winner, "confidence": j.confidence, "reason": j.reason})
        pred = majority(votes)
        if pred == "tie":
            pred = votes[0]
        p_a = votes.count("A") / max(1, len(votes))
        row = {
            "id": item.id,
            "pred": pred,
            "label": item.label,
            "correct": None if item.label is None else int(pred == item.label),
            "confidence": vote_share(votes),
            "feat_ensemble_vote_share": vote_share(votes),
            "feat_ensemble_entropy": entropy_binary(p_a),
            "feat_ensemble_mean_conf": sum(confs) / max(1, len(confs)),
            "feat_ensemble_size": len(judges),
            "raw": {"ensemble": raw},
            "domain": item.domain,
        }
        rows.append(row)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} ensemble rows to {args.out}")


if __name__ == "__main__":
    main()
