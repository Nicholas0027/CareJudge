#!/usr/bin/env python3
"""
Rigorous cost-aware cascade + cross-dataset transfer.

Fixes over the exploratory version:
  (1) Unified STRICT 3-way split (40/30/30 train/cal/test), identical to the
      main single-model experiments -> comparable sample sizes.
  (2) CONDITIONAL per-tier calibration: tier t's threshold is selected on the
      subset of the calibration split that REACHES tier t (i.e. was rejected
      by tiers 1..t-1), matching the deployment routing distribution.
  (3) UNION-BOUND error budget: each tier is calibrated at delta/T so the whole
      cascade enjoys a simultaneous (alpha, delta) guarantee by a union bound.
  (4) Calibrators fit on TRAIN only; thresholds on CAL only; metrics on TEST.

Transfer uses the same strict 3-way split: calibrator+threshold fit on the
SOURCE dataset (train+cal), evaluated on the TARGET dataset test split.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import Any, Dict, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator

ALPHAS = [0.15, 0.20, 0.25, 0.30]


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get("correct") is not None]


def split3_ids(ids, seed, tr=0.4, ca=0.3):
    rng = random.Random(seed); ids = list(ids); rng.shuffle(ids)
    n = len(ids); nt, nc = int(n*tr), int(n*ca)
    return set(ids[:nt]), set(ids[nt:nt+nc]), set(ids[nt+nc:])


def cascade(tier_paths, costs, delta=0.10, seed=0, min_keep=20):
    """tier_paths: ordered list of (name, path) cheap->strong. Aligned by id."""
    feats = {name: {r["id"]: r for r in read_jsonl(p)} for name, p in tier_paths}
    names = [n for n, _ in tier_paths]
    common = set.intersection(*[set(feats[n]) for n in names])
    ids = sorted(common)
    T = len(names)
    delta_t = delta / T  # union bound
    tr_ids, ca_ids, te_ids = split3_ids(ids, seed)

    # Fit each tier's calibrator on TRAIN (marginal is fine for fitting the model).
    cals = {}
    for n in names:
        train_rows = [feats[n][i] for i in tr_ids]
        cals[n] = fit_calibrator(train_rows, method="logistic")

    out = {}
    for alpha in ALPHAS:
        # CONDITIONAL calibration: walk tiers, each threshold chosen on the
        # calibration items that REACH that tier.
        thresholds = {}
        remaining_cal = set(ca_ids)
        for n in names:
            rows = [feats[n][i] for i in remaining_cal]
            if len(rows) < min_keep:
                thresholds[n] = float("inf")  # cannot certify -> never accept here
                continue
            p = cals[n].predict_proba(rows)
            y = [int(r["correct"]) for r in rows]
            thr, _ = calibrate_threshold(p, y, alpha=alpha, delta=delta_t,
                                         min_keep=min_keep, bound="clopper_pearson")
            thresholds[n] = thr
            # items accepted at this tier leave the pool
            acc = {i for i, pp in zip(remaining_cal, p) if pp >= thr}
            remaining_cal = remaining_cal - acc

        # Route TEST items through the cascade.
        routed = {n: 0 for n in names}; routed["abstain"] = 0
        errs = 0; nacc = 0; total_cost = 0.0
        strong_only_cost = 0.0
        for i in sorted(te_ids):
            spent = 0.0; decided = False
            for n in names:
                spent += costs.get(n, 0.0)
                p = cals[n].predict_proba([feats[n][i]])[0]
                if p >= thresholds[n]:
                    routed[n] += 1; nacc += 1
                    errs += 1 - int(feats[n][i]["correct"])
                    decided = True; break
            if not decided:
                routed["abstain"] += 1
            total_cost += spent
            strong_only_cost += costs.get(names[-1], 0.0)
        cov = nacc / len(te_ids) if te_ids else 0.0
        acc = 1 - errs / nacc if nacc else None
        save = 100 * (1 - total_cost / strong_only_cost) if strong_only_cost > 0 else 0.0
        out[f"alpha{alpha}"] = {
            "alpha": alpha, "coverage": cov, "accuracy_accepted": acc,
            "routing": routed, "n_test": len(te_ids),
            "cost_savings_pct": save, "delta_per_tier": delta_t}
    return out


def transfer(src_path, tgt_path, delta=0.10, seed=0, min_keep=20):
    src = read_jsonl(src_path); tgt = read_jsonl(tgt_path)
    # source: fit calibrator on src train, threshold on src cal
    sids = list(range(len(src)))
    tr, ca, _ = split3_ids(sids, seed)
    train = [src[i] for i in tr]; cal = [src[i] for i in ca]
    bundle = fit_calibrator(train, method="logistic")
    p_cal = bundle.predict_proba(cal); y_cal = [int(r["correct"]) for r in cal]
    # target: evaluate on target test split (held-out on target)
    tids = list(range(len(tgt)))
    _, _, te = split3_ids(tids, seed)
    test = [tgt[i] for i in te]
    out = {}
    for alpha in ALPHAS:
        thr, _ = calibrate_threshold(p_cal, y_cal, alpha=alpha, delta=delta, min_keep=min_keep)
        p_test = bundle.predict_proba(test)
        acc_idx = [i for i in range(len(test)) if p_test[i] >= thr]
        cov = len(acc_idx)/len(test) if test else 0.0
        acc = (sum(int(test[i]["correct"]) for i in acc_idx)/len(acc_idx)) if acc_idx else None
        out[f"alpha{alpha}"] = {"coverage": cov, "accuracy_accepted": acc, "n_test": len(test)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cascade", "transfer"], required=True)
    ap.add_argument("--tiers", help="name=path,name=path (cheap->strong)")
    ap.add_argument("--costs", help="name=usd,...")
    ap.add_argument("--src"); ap.add_argument("--tgt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "cascade":
        tier_paths = [(kv.split("=")[0], kv.split("=")[1]) for kv in args.tiers.split(",")]
        costs = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.costs.split(",")}
        # average over seeds (pooled per alpha)
        agg = {}
        for seed in range(args.seeds):
            r = cascade(tier_paths, costs, seed=seed)
            for k, v in r.items():
                a = agg.setdefault(k, {"cov": [], "acc": [], "save": [], "route": []})
                a["cov"].append(v["coverage"])
                if v["accuracy_accepted"] is not None:
                    a["acc"].append(v["accuracy_accepted"])
                a["save"].append(v["cost_savings_pct"])
                a["route"].append(v["routing"])
        final = {}
        for k, v in agg.items():
            m = lambda xs: sum(xs)/len(xs) if xs else None
            route_avg = {}
            for rk in v["route"][0]:
                route_avg[rk] = sum(rr[rk] for rr in v["route"])/len(v["route"])
            final[k] = {"coverage": m(v["cov"]), "accuracy_accepted": m(v["acc"]),
                        "cost_savings_pct": m(v["save"]), "routing_avg": route_avg}
        json.dump(final, open(args.out, "w"), indent=2)
        print(json.dumps(final, indent=2))
    else:
        agg = {}
        for seed in range(args.seeds):
            r = transfer(args.src, args.tgt, seed=seed)
            for k, v in r.items():
                a = agg.setdefault(k, {"cov": [], "acc": []})
                a["cov"].append(v["coverage"])
                if v["accuracy_accepted"] is not None:
                    a["acc"].append(v["accuracy_accepted"])
        m = lambda xs: sum(xs)/len(xs) if xs else None
        final = {k: {"coverage": m(v["cov"]), "accuracy_accepted": m(v["acc"]),
                     "n_seeds": len(v["cov"])} for k, v in agg.items()}
        json.dump(final, open(args.out, "w"), indent=2)
        print(json.dumps(final, indent=2))
