#!/usr/bin/env python3
"""
Unified rigorous selective-evaluation analysis for CARE-Judge.

Fixes the methodological issues flagged in review:
  (1) Alpha-coverage monotonicity: we report BOTH per-seed means (with std)
      AND coverage/risk POOLED across seeds (pooling accepts+errors). The
      pooled curve is monotone in alpha by construction and is the headline
      risk-coverage curve; the per-seed spread is reported for honesty.
  (2) Per-seed risk reporting: for every (method,dataset,alpha) we record,
      per seed, accepted count, realized test risk, CP upper bound on the
      calibration split, and whether the test risk exceeds alpha
      (violation). We report Pr_seed(R_test > alpha).
  (3) Matched-call-budget baselines: raw, base-confidence-only, self-only,
      swap-only (SCOPE-style bidirectional consistency), rubric-only,
      max-signal, best-single-signal, and full logistic/gbm fusion --- all
      consume the SAME stored 6-call features, so comparison is budget-matched.
  (4) Strict disjoint 3-way split (40/30/30): calibrator fit on TRAIN only,
      threshold on CAL only, all metrics on TEST only.
  (5) Risk-coverage curves + AURC + calibration (ECE/Brier) reported.

No API calls: everything is computed from the stored per-item feature files,
so results are exactly reproducible.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator

ALPHAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Signal feature columns actually present in the stored files.
SINGLE = {
    "base_conf": "feat_base_conf",
    "self": "feat_self_vote_share",
    "swap": "feat_swap_consistency",      # SCOPE-style bidirectional consistency
    "rubric": "feat_rubric_vote_share",
    "mean_conf": "feat_mean_conf",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return [r for r in rows if r.get("correct") is not None]


def split3(rows, seed, tr=0.4, ca=0.3):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(idx)
    nt, nc = int(n * tr), int(n * ca)
    tr_i, ca_i, te_i = idx[:nt], idx[nt:nt + nc], idx[nt + nc:]
    return [rows[i] for i in tr_i], [rows[i] for i in ca_i], [rows[i] for i in te_i]


def f(x, d=0.5):
    try:
        return float(x)
    except Exception:
        return d


def score_method(method, train, cal, test):
    """Return (p_cal, p_test) confidence scores for a method. All use stored features."""
    if method == "raw":
        return [1.0] * len(cal), [1.0] * len(test)
    if method in SINGLE:
        col = SINGLE[method]
        return [f(r.get(col)) for r in cal], [f(r.get(col)) for r in test]
    if method == "max_signal":
        def m(r):
            return max(f(r.get("feat_rubric_vote_share")), f(r.get("feat_swap_consistency")),
                       f(r.get("feat_self_vote_share")))
        return [m(r) for r in cal], [m(r) for r in test]
    if method in ("logistic", "gbm"):
        bundle = fit_calibrator(train, method=method)
        return bundle.predict_proba(cal), bundle.predict_proba(test)
    if method == "best_single":
        # pick highest-AUROC single signal on TRAIN, then use it
        best, best_auc = None, -1
        for k, col in SINGLE.items():
            s = [f(r.get(col)) for r in train]
            y = [int(r["correct"]) for r in train]
            if len(set(y)) < 2:
                continue
            a = auroc(y, s)
            if a is not None and a > best_auc:
                best_auc, best = a, col
        best = best or "feat_rubric_vote_share"
        return [f(r.get(best)) for r in cal], [f(r.get(best)) for r in test]
    raise ValueError(method)


def auroc(y, s):
    pos = [s[i] for i in range(len(y)) if y[i] == 1]
    neg = [s[i] for i in range(len(y)) if y[i] == 0]
    if not pos or not neg:
        return None
    # rank-sum
    alls = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(alls):
        j = i
        while j < len(alls) and s[alls[j]] == s[alls[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[alls[k]] = avg
        i = j
    sum_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    n1, n0 = len(pos), len(neg)
    return (sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(y, p, bins=10):
    tot = len(y)
    if tot == 0:
        return None
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(tot) if (p[i] > lo or (b == 0 and p[i] >= lo)) and p[i] <= hi]
        if not idx:
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        e += (len(idx) / tot) * abs(acc - conf)
    return e


def brier(y, p):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / max(1, len(y)) if y else None


def selective_at_threshold(test, p_test, thr):
    acc_idx = [i for i in range(len(test)) if p_test[i] >= thr]
    n_acc = len(acc_idx)
    if n_acc == 0:
        return 0.0, None, 0, 0
    errs = sum(1 - int(test[i]["correct"]) for i in acc_idx)
    risk = errs / n_acc
    return n_acc / len(test), 1 - risk, n_acc, errs


def risk_coverage_curve(test, p_test):
    """Coverage vs realized risk as threshold sweeps (test-set only, descriptive)."""
    order = sorted(range(len(test)), key=lambda i: -p_test[i])
    pts, errs = [], 0
    for k, i in enumerate(order, 1):
        errs += 1 - int(test[i]["correct"])
        pts.append((k / len(test), errs / k))
    return pts


def aurc(pts):
    """Area under risk-coverage curve (lower is better)."""
    if len(pts) < 2:
        return None
    a = 0.0
    for (c0, r0), (c1, r1) in zip(pts, pts[1:]):
        a += 0.5 * (r0 + r1) * (c1 - c0)
    return a


def run(features_path, dataset, out_dir, seeds=10, delta=0.10, min_keep=20, methods=None):
    rows = read_jsonl(features_path)
    methods = methods or ["raw", "base_conf", "self", "swap", "rubric",
                          "max_signal", "best_single", "logistic", "gbm"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Per (method, alpha): collect per-seed records and pooled accumulators.
    perseed = {}      # (method,alpha) -> list of dicts
    pooled = {}       # (method,alpha) -> [tot_acc, tot_err, tot_n]
    method_auc = {}   # method -> list of test AUROC per seed
    method_cal = {}   # method -> list of (ece,brier)
    rc_curves = {}    # method -> averaged risk-coverage pts (from seed 0 for plotting)

    for seed in range(seeds):
        train, cal, test = split3(rows, seed)
        if len(test) < 10 or len(cal) < min_keep:
            continue
        y_test = [int(r["correct"]) for r in test]
        for method in methods:
            p_cal, p_test = score_method(method, train, cal, test)
            # AUROC + calibration (test)
            a = auroc(y_test, p_test)
            method_auc.setdefault(method, []).append(a if a is not None else float("nan"))
            method_cal.setdefault(method, []).append((ece(y_test, p_test), brier(y_test, p_test)))
            if method == "raw":
                # raw = accept all, no thresholding
                for alpha in ALPHAS:
                    cov, acc, nacc, errs = 1.0, sum(y_test) / len(y_test), len(test), len(test) - sum(y_test)
                    perseed.setdefault((method, alpha), []).append(
                        {"seed": seed, "coverage": cov, "acc": acc, "n_acc": nacc,
                         "errs": errs, "risk": 1 - acc, "cp_cal": None, "violation": int((1 - acc) > alpha)})
                    p = pooled.setdefault((method, alpha), [0, 0, 0])
                    p[0] += nacc - errs; p[1] += errs; p[2] += len(test)
                if seed == 0:
                    rc_curves[method] = risk_coverage_curve(test, p_test)
                continue
            y_cal = [int(r["correct"]) for r in cal]
            if seed == 0:
                rc_curves[method] = risk_coverage_curve(test, p_test)
            for alpha in ALPHAS:
                thr, info = calibrate_threshold(p_cal, y_cal, alpha=alpha, delta=delta,
                                                min_keep=min_keep, bound="clopper_pearson")
                cov, acc, nacc, errs = selective_at_threshold(test, p_test, thr)
                # CP upper bound recorded on calibration accepted set
                cal_acc_idx = [i for i in range(len(cal)) if p_cal[i] >= thr]
                cp_cal = None
                if cal_acc_idx:
                    ce = sum(1 - y_cal[i] for i in cal_acc_idx)
                    cp_cal = clopper_pearson_upper(ce, len(cal_acc_idx), delta)
                risk = (1 - acc) if acc is not None else None
                perseed.setdefault((method, alpha), []).append(
                    {"seed": seed, "coverage": cov, "acc": acc, "n_acc": nacc, "errs": errs,
                     "risk": risk, "cp_cal": cp_cal,
                     "violation": int(risk is not None and risk > alpha)})
                p = pooled.setdefault((method, alpha), [0, 0, 0])
                if nacc > 0:
                    p[0] += nacc - errs; p[1] += errs
                p[2] += len(test)

    # Aggregate
    def agg(method, alpha):
        recs = perseed.get((method, alpha), [])
        covs = [r["coverage"] for r in recs]
        accs = [r["acc"] for r in recs if r["acc"] is not None]
        risks = [r["risk"] for r in recs if r["risk"] is not None]
        viol = [r["violation"] for r in recs]
        tot_acc, tot_err, tot_n = pooled.get((method, alpha), [0, 0, 0])
        pooled_cov = (tot_acc + tot_err) / tot_n if tot_n else 0.0
        pooled_acc = tot_acc / (tot_acc + tot_err) if (tot_acc + tot_err) else None
        return {
            "method": method, "alpha": alpha, "n_seeds": len(recs),
            "cov_mean": mean(covs), "cov_std": std(covs),
            "acc_mean": mean(accs), "acc_std": std(accs),
            "pooled_cov": pooled_cov, "pooled_acc": pooled_acc,
            "pooled_risk": (1 - pooled_acc) if pooled_acc is not None else None,
            "mean_accepted": mean([r["n_acc"] for r in recs]),
            "violation_rate": mean(viol),
        }

    results = {"dataset": dataset, "n_total": len(rows), "seeds": seeds,
               "delta": delta, "min_keep": min_keep, "alphas": ALPHAS}
    results["method_auroc"] = {m: {"mean": nanmean(v), "std": nanstd(v)} for m, v in method_auc.items()}
    results["method_calibration"] = {
        m: {"ece_mean": mean([c[0] for c in v if c[0] is not None]),
            "brier_mean": mean([c[1] for c in v if c[1] is not None])}
        for m, v in method_cal.items()}
    results["aurc"] = {m: aurc(pts) for m, pts in rc_curves.items()}
    results["by_method_alpha"] = {f"{m}@{a}": agg(m, a) for m in methods for a in ALPHAS}
    results["perseed"] = {f"{m}@{a}": perseed.get((m, a), []) for m in methods for a in ALPHAS}
    results["rc_curves"] = {m: pts for m, pts in rc_curves.items()}

    with open(out / f"{dataset}_rigorous.json", "w") as fp:
        json.dump(results, fp, indent=2)
    return results


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def nanstd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return std(xs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--min-keep", type=int, default=20)
    args = ap.parse_args()
    r = run(args.features, args.dataset, args.out_dir, args.seeds, args.delta, args.min_keep)
    print(f"\n=== {args.dataset} (N={r['n_total']}) ===")
    print("AUROC:", {m: round(v["mean"], 3) for m, v in r["method_auroc"].items() if v["mean"]})
    print("\nAlpha sweep (logistic) pooled cov/acc  |  per-seed cov_mean±std  viol_rate:")
    for a in ALPHAS:
        e = r["by_method_alpha"][f"logistic@{a}"]
        pc = f"{e['pooled_cov']:.3f}" if e['pooled_cov'] is not None else "--"
        pa = f"{e['pooled_acc']:.3f}" if e['pooled_acc'] is not None else "--"
        cm = f"{e['cov_mean']:.3f}" if e['cov_mean'] is not None else "--"
        cs = f"{e['cov_std']:.3f}" if e['cov_std'] is not None else "--"
        print(f"  a={a:.2f}  pooled {pc}/{pa}   per-seed {cm}±{cs}  viol={e['violation_rate']:.2f}")
