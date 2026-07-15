#!/usr/bin/env python3
"""Supplementary experiments for CARE-Judge revision (方案④).

Runs three analyses on the already-collected feature data (no new API calls):
  (1) SCOPE / Trust-or-Escalate matched-budget selective comparison (coverage/acc/violation at alpha=0.15)
  (2) Ensemble baselines: majority vote + weighted vote (soft-confidence aggregation)
  (3) Feature-group ablations: protocol-only / confidence-only / length-only fusion

All reuse the strict 40/30/30 split + exact Clopper-Pearson from downstream_analysis.py.

Usage: python scripts/supplementary_experiments.py --features-dir outputs/scale --out-dir outputs/supplementary --seeds 20
"""
import argparse, json, os, random, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator

JUDGES = {"deepseek-chat": "DeepSeek-V4", "gpt-5_5": "GPT-5.5", "qwen-1.5b": "Qwen-1.5B"}
BENCHMARKS = ["judgebench", "tldr", "rewardbench", "lmaarena"]

# Feature groups
FEAT_PROTOCOL = ["feat_swap_consistency", "feat_swap_conf_gap",
                 "feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip",
                 "feat_self_vote_share", "feat_self_entropy", "feat_sim_vote_share",
                 "feat_sim_entropy", "feat_sim_flip"]
FEAT_CONF = ["confidence", "feat_base_conf", "feat_mean_conf", "feat_std_conf"]
FEAT_LENGTH = ["feat_length_gap_norm", "feat_score_margin"]
FEAT_FULL = FEAT_PROTOCOL + FEAT_CONF + FEAT_LENGTH


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


def selective_eval(p_cal, yc, p_test, yt, alpha=0.15, delta=0.10, min_keep=20):
    thr, _ = calibrate_threshold(list(p_cal), yc, alpha=alpha, delta=delta, min_keep=min_keep, bound="clopper_pearson")
    acc_idx = [i for i in range(len(yt)) if p_test[i] >= thr]
    cov = len(acc_idx)/len(yt) if yt else 0.0
    acc = sum(yt[i] for i in acc_idx)/len(acc_idx) if acc_idx else None
    risk = (1-acc) if acc is not None else None
    viol = 1 if (risk is not None and risk > alpha) else 0
    return cov, acc, risk, viol, len(acc_idx)


# ─────────────────────────────────────────────────────────
# (1) Matched-budget SCOPE / ToE selective comparison
# ─────────────────────────────────────────────────────────
def scope_scores(rows):
    # SCOPE bidirectional: high = consistent swap + confident
    return [(1.0 - abs(float(r.get("feat_swap_conf_gap",0)))) * float(r.get("feat_swap_consistency",0.5)) for r in rows]

def toe_scores(rows):
    # ToE simulated-annotator agreement: max vote-share across variants
    out = []
    for r in rows:
        rubric = float(r.get("feat_rubric_vote_share", 0.5))
        self_vs = float(r.get("feat_self_vote_share", 0.5))
        swap = float(r.get("feat_swap_consistency", 0.5))
        out.append(max(rubric, self_vs, swap))
    return out

def care_scores(tr, target):
    bundle = fit_calibrator(tr, method="logistic", feature_cols=FEAT_FULL)
    return bundle.predict_proba(target)


# ─────────────────────────────────────────────────────────
# (2) Ensemble baselines
# ─────────────────────────────────────────────────────────
def majority_vote_score(rows):
    """Majority vote over 6 calls: confidence = fraction of calls agreeing with the modal winner."""
    out = []
    for r in rows:
        pc = r.get("per_call", {})
        winners = [pc.get("base_winner","")]
        for c in pc.get("self_calls", []):
            if c.get("winner") in ("A","B"): winners.append(c["winner"])
        sw = pc.get("swap_winner","")
        if sw in ("A","B"):
            # swap winner is in swapped coords; map back: if swap says A in (B,A) order, original prefers B
            winners.append("B" if sw == "A" else "A")
        for c in pc.get("rubric_calls", []):
            if c.get("winner") in ("A","B"): winners.append(c["winner"])
        winners = [w for w in winners if w in ("A","B")]
        if not winners:
            out.append(0.5); continue
        a = winners.count("A"); b = winners.count("B")
        out.append(max(a,b)/len(winners))
    return out

def weighted_vote_score(rows):
    """Soft confidence aggregation: mean verbalized confidence across calls that agree with modal winner."""
    out = []
    for r in rows:
        pc = r.get("per_call", {})
        calls = [("base", pc.get("base_winner",""), pc.get("base_conf",0.5))]
        for c in pc.get("self_calls", []):
            calls.append(("self", c.get("winner",""), c.get("confidence",0.5)))
        sw = pc.get("swap_winner","")
        if sw in ("A","B"):
            calls.append(("swap", "B" if sw=="A" else "A", pc.get("swap_conf",0.5)))
        for c in pc.get("rubric_calls", []):
            calls.append(("rubric", c.get("winner",""), c.get("confidence",0.5)))
        winners = [w for _,w,_ in calls if w in ("A","B")]
        if not winners:
            out.append(0.5); continue
        a = winners.count("A"); b = winners.count("B")
        modal = "A" if a>=b else "B"
        agreeing_confs = [c for _,w,c in calls if w==modal]
        out.append(sum(agreeing_confs)/len(agreeing_confs) if agreeing_confs else 0.5)
    return out


# ─────────────────────────────────────────────────────────
# (3) Feature-group ablations
# ─────────────────────────────────────────────────────────
def fusion_group_scores(tr, target, group_cols):
    bundle = fit_calibrator(tr, method="logistic", feature_cols=group_cols)
    return bundle.predict_proba(target)


# ─────────────────────────────────────────────────────────
# Main per-dataset analysis
# ─────────────────────────────────────────────────────────
def analyze(features_path, dataset, judge, seeds=20, alpha=0.15, delta=0.10):
    rows = read_jsonl(features_path)
    if len(rows) < 50:
        return None
    raw_acc = sum(r["correct"] for r in rows)/len(rows)

    # collectors
    methods = ["care", "scope", "toe", "maj_vote", "wtd_vote",
               "ablation_protocol", "ablation_conf", "ablation_length"]
    agg = {m: {"auroc": [], "cov": [], "acc": [], "risk": [], "viol": [], "n_acc": []} for m in methods}
    raw_metrics = {"cov": 1.0, "acc": raw_acc, "risk": 1-raw_acc}

    for seed in range(seeds):
        tr, ca, te = split3(rows, seed)
        if len(te) < 10: continue
        yt = [int(r["correct"]) for r in te]; yc = [int(r["correct"]) for r in ca]

        # (1) matched-budget systems
        for name, scorefn in [("care", lambda t: care_scores(tr,t)),
                              ("scope", scope_scores),
                              ("toe", toe_scores),
                              ("maj_vote", majority_vote_score),
                              ("wtd_vote", weighted_vote_score)]:
            p_cal = scorefn(ca); p_test = scorefn(te)
            au = auroc(yt, p_test)
            cov, acc, risk, viol, n_acc = selective_eval(p_cal, yc, p_test, yt, alpha, delta)
            agg[name]["auroc"].append(au or 0.0)
            agg[name]["cov"].append(cov); agg[name]["acc"].append(acc or 0.0)
            agg[name]["risk"].append(risk if risk is not None else 1.0)
            agg[name]["viol"].append(viol); agg[name]["n_acc"].append(n_acc)

        # (3) feature-group ablations (all use logistic fusion)
        for name, cols in [("ablation_protocol", FEAT_PROTOCOL),
                           ("ablation_conf", FEAT_CONF),
                           ("ablation_length", FEAT_LENGTH)]:
            p_cal = fusion_group_scores(tr, ca, cols)
            p_test = fusion_group_scores(tr, te, cols)
            au = auroc(yt, p_test)
            cov, acc, risk, viol, n_acc = selective_eval(p_cal, yc, p_test, yt, alpha, delta)
            agg[name]["auroc"].append(au or 0.0)
            agg[name]["cov"].append(cov); agg[name]["acc"].append(acc or 0.0)
            agg[name]["risk"].append(risk if risk is not None else 1.0)
            agg[name]["viol"].append(viol); agg[name]["n_acc"].append(n_acc)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs)/len(xs) if xs else None
    def std(xs):
        xs = [x for x in xs if x is not None]
        m = mean(xs)
        return (sum((x-m)**2 for x in xs)/len(xs))**0.5 if xs and m is not None else 0.0

    out = {"dataset": dataset, "judge": judge, "n_total": len(rows), "raw_acc": raw_acc,
           "seeds": seeds, "alpha": alpha}
    for m in methods:
        out[m] = {
            "auroc_mean": mean(agg[m]["auroc"]), "auroc_std": std(agg[m]["auroc"]),
            "cov_mean": mean(agg[m]["cov"]), "cov_std": std(agg[m]["cov"]),
            "acc_mean": mean(agg[m]["acc"]), "acc_std": std(agg[m]["acc"]),
            "risk_mean": mean(agg[m]["risk"]),
            "violation_rate": mean(agg[m]["viol"]),
            "n_acc_mean": mean(agg[m]["n_acc"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/scale")
    ap.add_argument("--out-dir", default="outputs/supplementary")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for pfx, judge_name in JUDGES.items():
        all_results[pfx] = []
        for bench in BENCHMARKS:
            path = os.path.join(args.features_dir, f"{pfx}_{bench}_features.jsonl")
            if not os.path.exists(path):
                print(f"  [SKIP] {pfx}/{bench}: no file"); continue
            print(f"  running {pfx}/{bench} ({args.seeds} seeds)...", end=" ", flush=True)
            res = analyze(path, bench, judge_name, seeds=args.seeds, alpha=args.alpha)
            if res:
                all_results[pfx].append(res)
                print(f"OK raw={res['raw_acc']:.3f} care_au={res['care']['auroc_mean']:.3f} "
                      f"scope_au={res['scope']['auroc_mean']:.3f} toe_au={res['toe']['auroc_mean']:.3f}")
            else:
                print("SKIP (too few rows)")

    out_path = os.path.join(args.out_dir, "supplementary_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")

    # Print summary tables
    print("\n" + "="*100)
    print("(1) MATCHED-BUDGET SELECTIVE COMPARISON (alpha=0.15, %d seeds)" % args.seeds)
    print("="*100)
    print(f"{'Judge/Benchmark':<24} {'RawAcc':>6} | {'CARE au/cov/acc':>22} | {'SCOPE au/cov/acc':>22} | {'ToE au/cov/acc':>22}")
    for pfx in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{JUDGES[pfx]}/{res['dataset']}"
            c = res['care']; s = res['scope']; t = res['toe']
            print(f"{nm:<24} {res['raw_acc']:>6.3f} | {c['auroc_mean']:>5.3f}/{c['cov_mean']:>4.1%}/{c['acc_mean']:>4.1%}      "
                  f"| {s['auroc_mean']:>5.3f}/{s['cov_mean']:>4.1%}/{s['acc_mean']:>4.1%}      "
                  f"| {t['auroc_mean']:>5.3f}/{t['cov_mean']:>4.1%}/{t['acc_mean']:>4.1%}")

    print("\n" + "="*100)
    print("(2) ENSEMBLE BASELINES vs CARE (alpha=0.15, %d seeds)" % args.seeds)
    print("="*100)
    print(f"{'Judge/Benchmark':<24} {'CARE au':>8} | {'MajVote au':>10} | {'WtdVote au':>10}")
    for pfx in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{JUDGES[pfx]}/{res['dataset']}"
            print(f"{nm:<24} {res['care']['auroc_mean']:>8.3f} | {res['maj_vote']['auroc_mean']:>10.3f} | {res['wtd_vote']['auroc_mean']:>10.3f}")

    print("\n" + "="*100)
    print("(3) FEATURE-GROUP ABLATION (logistic fusion, alpha=0.15, %d seeds)" % args.seeds)
    print("="*100)
    print(f"{'Judge/Benchmark':<24} {'FULL au':>8} | {'protocol-only':>14} | {'conf-only':>11} | {'length-only':>13}")
    for pfx in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{JUDGES[pfx]}/{res['dataset']}"
            print(f"{nm:<24} {res['care']['auroc_mean']:>8.3f} | {res['ablation_protocol']['auroc_mean']:>14.3f} "
                  f"| {res['ablation_conf']['auroc_mean']:>11.3f} | {res['ablation_length']['auroc_mean']:>13.3f}")


if __name__ == "__main__":
    main()
