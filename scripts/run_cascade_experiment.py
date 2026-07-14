#!/usr/bin/env python3
"""Cost-aware cascade experiment.

Runs a cheap->strong judge cascade with per-tier calibrated thresholds and
reports coverage, accuracy, per-tier routing, and total cost. Costs are read
from per-judge `--costs` (USD per judged example) so the cost-quality tradeoff
is quantified for the paper.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import CalibratorBundle
from care_judge.utils import read_jsonl, write_jsonl


def parse_kv(s: str) -> Dict[str, str]:
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Cost-aware cascade over precomputed per-tier feature files")
    p.add_argument("--tier-features", required=True,
                   help="Comma list name=path.jsonl, ordered cheap->strong, e.g. qwen1.5b=f1.jsonl,qwen7b=f2.jsonl,gpt41=f3.jsonl")
    p.add_argument("--costs", required=True, help="Comma list name=usd_per_example, e.g. qwen1.5b=0,qwen7b=0,gpt41=0.02")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="logistic", choices=["logistic", "isotonic", "gbm"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--min-keep", type=int, default=20)
    p.add_argument("--bound", default="clopper_pearson", choices=["clopper_pearson", "hoeffding"])
    p.add_argument("--cal-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tier_paths = parse_kv(args.tier_features)
    costs = {k: float(v) for k, v in parse_kv(args.costs).items()}
    tiers = list(tier_paths.keys())

    # Load per-tier features and align by id.
    feats: Dict[str, Dict[str, dict]] = {}
    for name, path in tier_paths.items():
        feats[name] = {r["id"]: r for r in read_jsonl(path) if r.get("correct") is not None}
    common_ids = set.intersection(*[set(feats[t].keys()) for t in tiers])
    ids = sorted(common_ids)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_cal = max(2, int(len(ids) * args.cal_frac))
    cal_ids, test_ids = set(ids[:n_cal]), ids[n_cal:]

    from care_judge.calibration.models import fit_calibrator
    calibrators, thresholds = {}, {}
    for name in tiers:
        cal_rows = [feats[name][i] for i in cal_ids]
        bundle = fit_calibrator(cal_rows, method=args.method)
        calibrators[name] = bundle
        p_cal = bundle.predict_proba(cal_rows)
        pairs = [(pp, int(r["correct"])) for r, pp in zip(cal_rows, p_cal)]
        thr, _ = calibrate_threshold([x[0] for x in pairs], [x[1] for x in pairs],
                                     alpha=args.alpha, delta=args.delta, min_keep=args.min_keep, bound=args.bound)
        thresholds[name] = thr

    routed, results = {t: 0 for t in tiers}, []
    routed["abstain"] = 0
    total_cost = 0.0
    for i in test_ids:
        decided = False
        spent = 0.0
        for name in tiers:
            row = feats[name][i]
            spent += costs.get(name, 0.0)
            p = float(calibrators[name].predict_proba([row])[0])
            if p >= thresholds[name]:
                results.append({"id": i, "accepted_by": name, "correct": int(row["correct"]), "cost": spent})
                routed[name] += 1
                decided = True
                break
        if not decided:
            routed["abstain"] += 1
            results.append({"id": i, "accepted_by": "abstain", "correct": None, "cost": spent})
        total_cost += spent

    accepted = [r for r in results if r["accepted_by"] != "abstain"]
    acc = sum(r["correct"] for r in accepted) / len(accepted) if accepted else None
    strong = tiers[-1]
    strong_only_cost = len(test_ids) * sum(costs.get(t, 0.0) for t in tiers[:1]) + len(test_ids) * costs.get(strong, 0.0)
    report = {
        "tiers": tiers,
        "thresholds": thresholds,
        "n_test": len(test_ids),
        "coverage": len(accepted) / max(1, len(test_ids)),
        "accuracy_accepted": acc,
        "risk_accepted": (1 - acc) if acc is not None else None,
        "routing": routed,
        "total_cost": total_cost,
        "mean_cost_per_example": total_cost / max(1, len(test_ids)),
        "strong_only_cost_estimate": len(test_ids) * costs.get(strong, 0.0),
        "cost_savings_vs_strong_only": (len(test_ids) * costs.get(strong, 0.0) - total_cost),
    }
    write_jsonl(args.out, results)
    with open(args.out.replace(".jsonl", ".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
