#!/usr/bin/env python3
"""Three-way head-to-head comparison: CARE vs Trust-or-Escalate vs SCOPE.

All three use the SAME 6 judge calls per item (matched budget).
Each method computes its confidence score from the same per-call traces,
then enters the same rigorous fixed-sequence calibration + evaluation pipeline.
"""
import json, random, sys; sys.path.insert(0,'.')
import argparse
from typing import Dict, List, Any
from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def split3(rows, seed, tr=0.4, ca=0.3):
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    n = len(idx); nt, nc = int(n*tr), int(n*ca)
    return [rows[i] for i in idx[:nt]], [rows[i] for i in idx[nt:nt+nc]], [rows[i] for i in idx[nt+nc:]]

def auroc(y,s):
    p=[s[i] for i in range(len(y)) if y[i]==1]; n=[s[i] for i in range(len(y)) if y[i]==0]
    if not p or not n: return None
    import math
    c=sum(1 for a in p for b in n if a>b)+0.5*sum(1 for a in p for b in n if a==b)
    return c/(len(p)*len(n))

def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get("correct") is not None]

def score_care(rows):
    """CARE: logistic fusion over all feat_* columns."""
    bundle = fit_calibrator(rows, method="logistic")
    return bundle.predict_proba(rows)

def score_toe(rows):
    """Trust-or-Escalate proxy: simulated annotator agreement from multi-call agreement.
    Compute as: mean(rubric_consistency, self_consistency, swap_consistency).
    This uses the same 6 calls as CARE (via the stored feat_* aggregations)."""
    scores = []
    for r in rows:
        rubric = float(r.get("feat_rubric_vote_share", 0.5))
        self_v = float(r.get("feat_self_vote_share", 0.5))
        swap = float(r.get("feat_swap_consistency", 0.5))
        scores.append((rubric + self_v + swap) / 3.0)
    return scores

def score_scope(rows):
    """SCOPE-style: bidirectional consistency as confidence.
    Uses feat_swap_consistency (binary: 1.0 = consistent, 0.0 = flipped).
    For soft SCOPE, also use feat_swap_conf_gap and feat_base_conf."""
    scores = []
    for r in rows:
        swap = float(r.get("feat_swap_consistency", 0.5))
        conf_gap = float(r.get("feat_swap_conf_gap", 0))
        base = float(r.get("feat_base_conf", 0.5))
        # Aggregated: base_conf weighted by consistency
        scores.append(base * (1.0 - min(conf_gap, 1.0)) * swap + 0.5 * (1 - swap))
    return scores

METHODS = {"care": score_care, "toe": score_toe, "scope": score_scope}

def run(features_path, dataset, out_dir, seeds=10, delta=0.10):
    rows = read_jsonl(features_path)
    bests = {}
    for method_name, score_fn in METHODS.items():
        records = []
        for seed in range(min(seeds, 50)):
            tr, ca, te = split3(rows, seed)
            if len(te) < 10: continue
            yt = [int(r["correct"]) for r in te]
            # fit on train-only
            bundle = fit_calibrator(tr, method="logistic") if method_name == "care" else None
            if method_name == "care":
                p_cal = bundle.predict_proba(ca)
                p_test = bundle.predict_proba(te)
            else:
                p_cal = score_fn(ca)
                p_test = score_fn(te)
            # AUROC
            au = auroc(yt, p_test)
            yc = [int(r["correct"]) for r in ca]
            # Register scores for each alpha
            for alpha in ALPHAS:
                opt = {"alpha": alpha, "seed": seed, "method": method_name}
                try:
                    thr, info = calibrate_threshold(list(p_cal), yc, alpha=alpha, delta=delta, min_keep=20, bound="clopper_pearson")
                    accs = [i for i in range(len(te)) if p_test[i] >= thr]
                    cov = len(accs) / len(te) if len(te) else 0
                    acc = sum(int(te[i]["correct"]) for i in accs) / len(accs) if accs else None
                    opt.update({"coverage": cov, "accuracy_accepted": acc, "auroc": au, "n_test": len(te), "n_acc": len(accs)})
                except Exception as e:
                    opt.update({"coverage": 0, "accuracy_accepted": None, "auroc": au})
                records.append(opt)
        bests[method_name] = records
    # Aggregate
    agg = {}
    for method, recs in bests.items():
        for alpha in ALPHAS:
            ar = [r for r in recs if r.get("alpha") == alpha]
            covs = [r["coverage"] for r in ar]
            accs = [r["accuracy_accepted"] for r in ar if r["accuracy_accepted"] is not None]
            aucs = [r["auroc"] for r in ar if r.get("auroc") is not None]
            def mean(x): return sum(x)/len(x) if x else 0
            agg[f"{method}@{alpha}"] = {"alpha": alpha, "method": method,
                "cov_mean": mean(covs), "acc_mean": mean(accs),
                "auroc_mean": mean(aucs)}
    return agg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    import os; os.makedirs(args.out_dir, exist_ok=True)
    agg = run(args.features, args.dataset, args.out_dir, seeds=args.seeds)
    json.dump(agg, open(f"{args.out_dir}/{args.dataset}_threeway.json","w"), indent=2)
    # Print summary
    print(f"\n=== {args.dataset} (seeds={args.seeds}) ===")
    print(f"{'method':>8s} {'alpha':>5s}  AUC    cov    acc")
    for k, v in sorted(agg.items()):
        print(f"{v['method']:>8s} {v['alpha']:>5.2f}  {v['auroc_mean']:.3f}  {v['cov_mean']:.3f}  {v['acc_mean']:.3f}" if v['acc_mean'] else f"{v['method']:>8s} {v['alpha']:>5.2f}  {v['auroc_mean']:.3f}  {v['cov_mean']:.3f}  ----")
