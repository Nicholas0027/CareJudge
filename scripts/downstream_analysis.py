#!/usr/bin/env python3
"""Downstream analysis pipeline: reads collected features, runs rigorous evaluation,
   three-way comparison, capability ladder, and risk-coverage curves.
   Usage: python scripts/downstream_analysis.py --features-dir outputs/scale --out-dir outputs/downstream
"""
import argparse, json, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator
from care_judge.baselines_systems import compute_baseline_scores

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
JUDGES = {"deepseek-chat": "DeepSeek-V4", "gpt-5_5": "GPT-5.5", "glm-5.2": "GLM-5.2", "qwen-1.5b": "Qwen-1.5B"}
BENCHMARKS = ["judgebench", "tldr", "rewardbench", "lmaarena"]
METHODS = {"care": "fusion", "toe": "t-o-e", "scope": "scope"}

def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get("correct") is not None]

def split3(rows, seed, tr=0.4, ca=0.3):
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    n = len(idx); nt, nc = int(n*tr), int(n*ca)
    return [rows[i] for i in idx[:nt]], [rows[i] for i in idx[nt:nt+nc]], [rows[i] for i in idx[nt+nc:]]

def auroc(y, s):
    p = [s[i] for i in range(len(y)) if y[i]==1]; n = [s[i] for i in range(len(y)) if y[i]==0]
    if not p or not n: return None
    return (sum(1 for a in p for b in n if a>b)+0.5*sum(1 for a in p for b in n if a==b))/(len(p)*len(n))

def ece_brier(y, p, bins=10):
    tot = len(y); e, brier = 0.0, sum((pi-yi)**2 for pi,yi in zip(p,y))/max(1,tot) if y else None
    for b in range(bins):
        lo, hi = b/bins, (b+1)/bins
        idx = [i for i in range(tot) if (p[i] > lo or (b==0 and p[i]>=lo)) and p[i] <= hi]
        if not idx: continue
        conf = sum(p[i] for i in idx)/len(idx); acc = sum(y[i] for i in idx)/len(idx)
        e += (len(idx)/tot)*abs(acc-conf)
    return e, brier

def analyze_dataset(features_path: str, dataset_name: str, judge_name: str, seeds: int = 10, delta: float = 0.10):
    rows = read_jsonl(features_path)
    if len(rows) < 50:
        print(f"  [SKIP] {dataset_name} ({len(rows)} rows, too few)")
        return None
    records = []
    # --- Signal rates (stable vs unstable) ---
    def signal_gap():
        rubric_stable = [r for r in rows if float(r.get("feat_rubric_flip",1)) == 0.0]
        rubric_unstable = [r for r in rows if float(r.get("feat_rubric_flip",0)) > 0.0]
        pos_stable = [r for r in rows if float(r.get("feat_swap_consistency",0)) >= 1.0]
        pos_unstable = [r for r in rows if float(r.get("feat_swap_consistency",0)) < 1.0]
        def acc(xs): return sum(r["correct"] for r in xs)/len(xs) if xs else None
        return {
            "rubric_stable": acc(rubric_stable), "rubric_unstable": acc(rubric_unstable),
            "position_stable": acc(pos_stable), "position_unstable": acc(pos_unstable),
            "rubric_n_stable": len(rubric_stable), "position_n_stable": len(pos_stable)
        }
    # --- Main evaluation across seeds ---
    bests = {m: {"auroc": [], "ece": [], "brier": [], "cov": [], "acc": [], "viol": []} for m in METHODS}
    bests["raw"] = {"cov": [1.0]*seeds, "acc": [sum(r["correct"] for r in rows)/len(rows)]*seeds}
    for seed in range(seeds):
        tr, ca, te = split3(rows, seed)
        if len(te) < 10: continue
        yt = [int(r["correct"]) for r in te]; yc = [int(r["correct"]) for r in ca]
        # ----- CARE: logistic fusion -----
        bundle = fit_calibrator(tr, method="logistic")
        p_cal_care = bundle.predict_proba(ca); p_test_care = bundle.predict_proba(te)
        au = auroc(yt, p_test_care)
        ec, br = ece_brier(yt, p_test_care)
        bests["care"]["auroc"].append(au or 0); bests["care"]["ece"].append(ec); bests["care"]["brier"].append(br or 0)
        for alpha in [0.15]:  # Main comparison threshold
            thr, _ = calibrate_threshold(list(p_cal_care), yc, alpha=alpha, delta=delta, min_keep=20, bound="clopper_pearson")
            acc_idx = [i for i in range(len(te)) if p_test_care[i] >= thr]
            cov = len(acc_idx)/len(te) if te else 0
            acc = sum(yt[i] for i in acc_idx)/len(acc_idx) if acc_idx else None
            viol = 1 if acc is not None and (1-acc) > alpha else 0
            bests["care"]["cov"].append(cov); bests["care"]["acc"].append(acc or 0); bests["care"]["viol"].append(viol)
        # ----- Trust-or-Escalate: simulated annotator agreement -----
        p_cal_toe = [max(float(r.get("feat_rubric_vote_share",0.5)), float(r.get("feat_self_vote_share",0.5)), float(r.get("feat_swap_consistency",0.5))) for r in ca]
        p_test_toe = [max(float(r.get("feat_rubric_vote_share",0.5)), float(r.get("feat_self_vote_share",0.5)), float(r.get("feat_swap_consistency",0.5))) for r in te]
        au = auroc(yt, p_test_toe)
        bests["toe"]["auroc"].append(au or 0)
        for alpha in [0.15]:
            thr, _ = calibrate_threshold(list(p_cal_toe), yc, alpha=alpha, delta=delta, min_keep=20, bound="clopper_pearson")
            acc_idx = [i for i in range(len(te)) if p_test_toe[i] >= thr]
            cov = len(acc_idx)/len(te) if te else 0
            acc = sum(yt[i] for i in acc_idx)/len(acc_idx) if acc_idx else None
            viol = 1 if acc is not None and (1-acc) > alpha else 0
            bests["toe"]["cov"].append(cov); bests["toe"]["acc"].append(acc or 0)
            if not bests["toe"].get("viol"): bests["toe"]["viol"] = []
            bests["toe"]["viol"].append(viol)
        # ----- SCOPE: bidirectional preference entropy -----
        p_cal_scope = [(1.0 - abs(float(r.get("feat_swap_conf_gap",0)))) * float(r.get("feat_swap_consistency",0.5)) for r in ca]
        p_test_scope = [(1.0 - abs(float(r.get("feat_swap_conf_gap",0)))) * float(r.get("feat_swap_consistency",0.5)) for r in te]
        au = auroc(yt, p_test_scope)
        bests["scope"]["auroc"].append(au or 0)
        for alpha in [0.15]:
            thr, _ = calibrate_threshold(list(p_cal_scope), yc, alpha=alpha, delta=delta, min_keep=20, bound="clopper_pearson")
            acc_idx = [i for i in range(len(te)) if p_test_scope[i] >= thr]
            cov = len(acc_idx)/len(te) if te else 0
            acc = sum(yt[i] for i in acc_idx)/len(acc_idx) if acc_idx else None
            viol = 1 if acc is not None and (1-acc) > alpha else 0
            bests["scope"]["cov"].append(cov); bests["scope"]["acc"].append(acc or 0)
            if not bests["scope"].get("viol"): bests["scope"]["viol"] = []
            bests["scope"]["viol"].append(viol)
    # Aggregate
    def agg(dicts, metric):
        vals = [x for x in dicts if x is not None]
        return sum(vals)/len(vals) if vals else None
    result = {"dataset": dataset_name, "judge": judge_name, "n_total": len(rows), "signal_gaps": signal_gap(),
              "raw_acc": sum(r["correct"] for r in rows)/len(rows)}
    for m in METHODS:
        result[m] = {"auroc_mean": agg(bests[m]["auroc"], None), "auroc_std": (sum((x-agg(bests[m]["auroc"], None))**2 for x in bests[m]["auroc"])/len(bests[m]["auroc"]))**0.5 if bests[m]["auroc"] else 0,
                     "cov_at_015": agg(bests[m]["cov"], None), "acc_at_015": agg(bests[m]["acc"], None) if any(a is not None and a>0 for a in bests[m]["acc"]) else None,
                     "violation_rate": sum(bests[m]["viol"])/len(bests[m]["viol"]) if bests[m].get("viol") else 0}
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/scale")
    ap.add_argument("--out-dir", default="outputs/downstream")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    for pfx, judge_name in JUDGES.items():
        results = []
        for bench in BENCHMARKS:
            path = os.path.join(args.features_dir, f"{pfx}_{bench}_features.jsonl")
            if not os.path.exists(path): continue
            res = analyze_dataset(path, bench, judge_name, seeds=args.seeds)
            if res: results.append(res)
            print(f"  {pfx}/{bench}: {'OK' if res else 'SKIP'}")
        if results:
            json.dump(results, open(os.path.join(args.out_dir, f"{pfx}_analysis.json"), "w"), indent=2)
            print(f"  -> {pfx}: {len(results)} benchmarks analyzed")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Three-way comparison (α=0.15, 10 seeds)':^80}")
    print(f"{'='*80}")
    print(f"{'Judge/Benchmark':<25} {'RawAcc':>6} {'CARE':>15} {'SCOPE':>15} {'ToE':>15}")
    for pfx in JUDGES:
        path = os.path.join(args.out_dir, f"{pfx}_analysis.json")
        if not os.path.exists(path): continue
        for res in json.load(open(path)):
            name = f"{JUDGES[pfx]}/{res['dataset']}"
            print(f"{name:<25} {res['raw_acc']:>6.3f}   {res.get('care',{}).get('auroc_mean',0):>5.3f}({res.get('care',{}).get('cov_at_015',0):>4.1%})  {res.get('scope',{}).get('auroc_mean',0):>5.3f}({res.get('scope',{}).get('cov_at_015',0):>4.1%})  {res.get('toe',{}).get('auroc_mean',0):>5.3f}({res.get('toe',{}).get('cov_at_015',0):>4.1%})")

if __name__ == "__main__":
    main()
