#!/usr/bin/env python3
"""Optimized supplementary experiments — numpy + sklearn + vectorized Clopper-Pearson.

Runs 3 analyses on collected feature data (no new API calls):
  (1) SCOPE / ToE / ensemble baselines matched-budget comparison
  (2) Feature-group ablations: protocol-only / confidence-only / length-only

All use sklearn LogisticRegression directly (no pure-Python calibrator overhead).
"""
import json, os, random, sys, argparse
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression

JUDGES = [("deepseek-chat", "DeepSeek-V4"), ("gpt-5_5", "GPT-5.5"), ("qwen-1.5b", "Qwen-1.5B")]
BENCHMARKS = ["judgebench", "tldr", "rewardbench", "lmaarena"]

FEAT_PROTOCOL = ["feat_swap_consistency", "feat_swap_conf_gap",
                 "feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip",
                 "feat_self_vote_share", "feat_self_entropy", "feat_sim_vote_share",
                 "feat_sim_entropy", "feat_sim_flip"]
FEAT_CONF = ["confidence", "feat_base_conf", "feat_mean_conf", "feat_std_conf"]
FEAT_LENGTH = ["feat_length_gap_norm", "feat_score_margin"]
FEAT_FULL = FEAT_PROTOCOL + FEAT_CONF + FEAT_LENGTH


def read_jsonl(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try:
            r = json.loads(l)
            if r.get("correct") is not None: rows.append(r)
        except: pass
    return rows


def to_matrix(rows, cols):
    return np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rows], dtype=np.float64)


def split3(n, seed, tr=0.4, ca=0.3):
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    nt, nc = int(n*tr), int(n*ca)
    return idx[:nt], idx[nt:nt+nc], idx[nt+nc:]


def auroc(y, s):
    y = np.array(y); s = np.array(s)
    p = s[y==1]; n = s[y==0]
    if len(p)==0 or len(n)==0: return 0.5
    return (np.sum(p[:,None] > n[None,:]) + 0.5*np.sum(p[:,None] == n[None,:])) / (len(p)*len(n))


def fit_predict(X_train, y_train, X_target):
    if len(set(y_train)) < 2:
        return np.full(len(X_target), float(np.mean(y_train)))
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_target)[:, 1]


def calibrate_threshold_vec(p_cal, y_cal, alpha, delta=0.10, min_keep=20):
    order = np.argsort(p_cal)[::-1]
    p_sorted = p_cal[order]; y_sorted = y_cal[order]
    n = len(p_sorted)
    if n < min_keep: return float('inf')
    cum_err = np.cumsum(y_sorted == 0)
    prefixes = np.arange(min_keep, n + 1)
    e_vals = cum_err[prefixes - 1]
    mask = e_vals < prefixes
    cp_vals = np.ones(len(prefixes))
    if np.any(mask):
        cp_vals[mask] = beta_dist.ppf(1 - delta, e_vals[mask] + 1, prefixes[mask] - e_vals[mask])
    valid = cp_vals <= alpha
    if not np.any(valid): return float('inf')
    best_i = prefixes[valid][-1]
    return float(p_sorted[best_i - 1])


def selective_eval(p_cal, y_cal, p_test, y_test, alpha=0.15, delta=0.10):
    thr = calibrate_threshold_vec(p_cal, y_cal, alpha, delta)
    acc_mask = p_test >= thr
    cov = float(np.mean(acc_mask))
    if np.any(acc_mask):
        acc = float(np.mean(y_test[acc_mask]))
        risk = 1 - acc
    else:
        acc = 0.0; risk = 0.0
    viol = 1 if risk > alpha else 0
    return cov, acc, risk, viol


def scope_scores(rows):
    return np.array([(1.0 - abs(float(r.get("feat_swap_conf_gap",0)))) * float(r.get("feat_swap_consistency",0.5)) for r in rows])

def toe_scores(rows):
    return np.array([max(float(r.get("feat_rubric_vote_share",0.5)), float(r.get("feat_self_vote_share",0.5)), float(r.get("feat_swap_consistency",0.5))) for r in rows])

def majority_vote_scores(rows):
    out = []
    for r in rows:
        pc = r.get("per_call", {})
        winners = [pc.get("base_winner","")]
        for c in pc.get("self_calls", []):
            if c.get("winner") in ("A","B"): winners.append(c["winner"])
        sw = pc.get("swap_winner","")
        if sw in ("A","B"): winners.append("B" if sw=="A" else "A")
        for c in pc.get("rubric_calls", []):
            if c.get("winner") in ("A","B"): winners.append(c["winner"])
        winners = [w for w in winners if w in ("A","B")]
        if not winners: out.append(0.5); continue
        out.append(max(winners.count("A"),winners.count("B"))/len(winners))
    return np.array(out)

def weighted_vote_scores(rows):
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
        if not winners: out.append(0.5); continue
        modal = "A" if winners.count("A") >= winners.count("B") else "B"
        confs = [c for _,w,c in calls if w == modal]
        out.append(np.mean(confs) if confs else 0.5)
    return np.array(out)


def analyze(features_path, dataset, judge, seeds=20, alpha=0.15, delta=0.10):
    rows = read_jsonl(features_path)
    if len(rows) < 50: return None
    raw_acc = float(np.mean([r["correct"] for r in rows]))
    n = len(rows)

    # Pre-compute matrices
    X_full = to_matrix(rows, FEAT_FULL)
    X_proto = to_matrix(rows, FEAT_PROTOCOL)
    X_conf = to_matrix(rows, FEAT_CONF)
    X_len = to_matrix(rows, FEAT_LENGTH)
    y_full = np.array([int(r["correct"]) for r in rows])

    methods = ["care", "scope", "toe", "maj_vote", "wtd_vote",
               "ablation_protocol", "ablation_conf", "ablation_length"]
    agg = {m: {"auroc": [], "cov": [], "acc": [], "risk": [], "viol": []} for m in methods}

    for seed in range(seeds):
        tr_idx, ca_idx, te_idx = split3(n, seed)
        if len(te_idx) < 10: continue
        y_cal = y_full[ca_idx]; y_test = y_full[te_idx]

        # Pre-compute non-fitted scores (same for all seeds — but rows differ per split)
        # Actually these are per-row scores, we just slice them
        scope_all = scope_scores(rows)
        toe_all = toe_scores(rows)
        maj_all = majority_vote_scores(rows)
        wtd_all = weighted_vote_scores(rows)

        # CARE: logistic fusion on full features
        p_cal = fit_predict(X_full[tr_idx], y_full[tr_idx], X_full[ca_idx])
        p_test = fit_predict(X_full[tr_idx], y_full[tr_idx], X_full[te_idx])
        agg["care"]["auroc"].append(auroc(y_test, p_test))
        c, a, rk, v = selective_eval(p_cal, y_cal, p_test, y_test, alpha, delta)
        agg["care"]["cov"].append(c); agg["care"]["acc"].append(a); agg["care"]["risk"].append(rk); agg["care"]["viol"].append(v)

        # Non-fitted baselines
        for name, scores_all in [("scope", scope_all), ("toe", toe_all), ("maj_vote", maj_all), ("wtd_vote", wtd_all)]:
            p_cal_b = scores_all[ca_idx]; p_test_b = scores_all[te_idx]
            agg[name]["auroc"].append(auroc(y_test, p_test_b))
            c, a, rk, v = selective_eval(p_cal_b, y_cal, p_test_b, y_test, alpha, delta)
            agg[name]["cov"].append(c); agg[name]["acc"].append(a); agg[name]["risk"].append(rk); agg[name]["viol"].append(v)

        # Ablations: logistic fusion on feature subsets
        for name, X_sub in [("ablation_protocol", X_proto), ("ablation_conf", X_conf), ("ablation_length", X_len)]:
            p_cal_a = fit_predict(X_sub[tr_idx], y_full[tr_idx], X_sub[ca_idx])
            p_test_a = fit_predict(X_sub[tr_idx], y_full[tr_idx], X_sub[te_idx])
            agg[name]["auroc"].append(auroc(y_test, p_test_a))
            c, a, rk, v = selective_eval(p_cal_a, y_cal, p_test_a, y_test, alpha, delta)
            agg[name]["cov"].append(c); agg[name]["acc"].append(a); agg[name]["risk"].append(rk); agg[name]["viol"].append(v)

    def m(x): return float(np.mean(x)) if x else None
    def s(x): return float(np.std(x)) if x else 0.0
    out = {"dataset": dataset, "judge": judge, "n_total": n, "raw_acc": raw_acc, "seeds": seeds, "alpha": alpha}
    for meth in methods:
        out[meth] = {"auroc_mean": m(agg[meth]["auroc"]), "auroc_std": s(agg[meth]["auroc"]),
                     "cov_mean": m(agg[meth]["cov"]), "acc_mean": m(agg[meth]["acc"]),
                     "risk_mean": m(agg[meth]["risk"]), "violation_rate": m(agg[meth]["viol"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/scale")
    ap.add_argument("--out-dir", default="outputs/supplementary")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for pfx, jname in JUDGES:
        all_results[pfx] = []
        for bench in BENCHMARKS:
            path = os.path.join(args.features_dir, f"{pfx}_{bench}_features.jsonl")
            if not os.path.exists(path):
                print(f"  [SKIP] {pfx}/{bench}"); continue
            print(f"  {pfx}/{bench}...", end=" ", flush=True)
            res = analyze(path, bench, jname, seeds=args.seeds)
            if res:
                all_results[pfx].append(res)
                print(f"OK care={res['care']['auroc_mean']:.3f} scope={res['scope']['auroc_mean']:.3f} "
                      f"maj={res['maj_vote']['auroc_mean']:.3f} wtd={res['wtd_vote']['auroc_mean']:.3f} "
                      f"proto={res['ablation_protocol']['auroc_mean']:.3f} conf={res['ablation_conf']['auroc_mean']:.3f}", flush=True)
            else:
                print("SKIP")

    out_path = os.path.join(args.out_dir, "supplementary_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")

    # Print summary tables
    print("\n" + "="*110)
    print("(1) ENSEMBLE BASELINES vs CARE (AUROC, %d seeds)" % args.seeds)
    print("="*110)
    print(f"{'Judge/Benchmark':<24} {'CARE':>7} {'SCOPE':>7} {'ToE':>7} {'MajVote':>8} {'WtdVote':>8}")
    for pfx, jname in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{jname}/{res['dataset']}"
            print(f"{nm:<24} {res['care']['auroc_mean']:>7.3f} {res['scope']['auroc_mean']:>7.3f} "
                  f"{res['toe']['auroc_mean']:>7.3f} {res['maj_vote']['auroc_mean']:>8.3f} {res['wtd_vote']['auroc_mean']:>8.3f}")

    print("\n" + "="*110)
    print("(2) FEATURE-GROUP ABLATION (AUROC, %d seeds)" % args.seeds)
    print("="*110)
    print(f"{'Judge/Benchmark':<24} {'FULL':>7} {'protocol':>9} {'conf':>7} {'length':>7}")
    for pfx, jname in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{jname}/{res['dataset']}"
            print(f"{nm:<24} {res['care']['auroc_mean']:>7.3f} {res['ablation_protocol']['auroc_mean']:>9.3f} "
                  f"{res['ablation_conf']['auroc_mean']:>7.3f} {res['ablation_length']['auroc_mean']:>7.3f}")

    print("\n" + "="*110)
    print("(3) SELECTIVE METRICS at alpha=0.15 (cov/acc/viol)")
    print("="*110)
    print(f"{'Judge/Benchmark':<24} {'CARE':>20} {'SCOPE':>20} {'MajVote':>20}")
    for pfx, jname in JUDGES:
        for res in all_results.get(pfx, []):
            nm = f"{jname}/{res['dataset']}"
            c = res['care']; s = res['scope']; m = res['maj_vote']
            print(f"{nm:<24} {c['cov_mean']:>5.1%}/{c['acc_mean']:>4.1%}/{c['violation_rate']:>.0%}    "
                  f"{s['cov_mean']:>5.1%}/{s['acc_mean']:>4.1%}/{s['violation_rate']:>.0%}    "
                  f"{m['cov_mean']:>5.1%}/{m['acc_mean']:>4.1%}/{m['violation_rate']:>.0%}")


if __name__ == "__main__":
    main()
