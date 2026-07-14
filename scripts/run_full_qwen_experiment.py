from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold, clopper_pearson_upper
from care_judge.calibration.models import fit_calibrator, CalibratorBundle
from care_judge.evaluation.metrics import calibration_report, bootstrap_ci, expected_calibration_error, _auroc, _auprc
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.utils import read_jsonl, write_jsonl


# ── helpers ──────────────────────────────────────────────────────────────────

SINGLE_SIGNALS = {
    "rubric_vote_share": "feat_rubric_vote_share",
    "self_vote_share": "feat_self_vote_share",
    "swap_consistency": "feat_swap_consistency",
    "sim_vote_share": "feat_sim_vote_share",
    "base_conf": "feat_base_conf",
    "mean_conf": "feat_mean_conf",
}


def split3(rows: List[Dict], seed: int, tr: float, ca: float):
    labeled = [r for r in rows if r.get("correct") is not None]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n = len(labeled)
    nt = max(2, int(n * tr))
    nc = max(2, int(n * ca))
    train = labeled[:nt]
    cal = labeled[nt:nt+nc]
    test = labeled[nt+nc:]
    if not test:
        test = cal
    return train, cal, test


def safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def run_logistic_fusion(train, cal, test, alpha, delta, min_keep, bound, method="logistic"):
    """Full CARE: logistic fusion of all features."""
    bundle = fit_calibrator(train, method=method)
    p_cal = bundle.predict_proba(cal)
    cal_pairs = [(p, int(r["correct"])) for r, p in zip(cal, p_cal)]
    thr, _ = calibrate_threshold([x[0] for x in cal_pairs], [x[1] for x in cal_pairs], alpha=alpha, delta=delta, min_keep=min_keep, bound=bound)
    p_test = bundle.predict_proba(test)
    return p_test, thr, bundle


def run_single_signal(train, cal, test, signal_key, alpha, delta, min_keep, bound):
    """Single-signal: use the raw feature value as confidence."""
    feat_col = SINGLE_SIGNALS[signal_key]
    p_cal = [safe_float(r.get(feat_col, 0.5)) for r in cal]
    cal_pairs = [(p, int(r["correct"])) for r, p in zip(cal, p_cal)]
    thr, _ = calibrate_threshold([x[0] for x in cal_pairs], [x[1] for x in cal_pairs], alpha=alpha, delta=delta, min_keep=min_keep, bound=bound)
    p_test = [safe_float(r.get(feat_col, 0.5)) for r in test]
    return p_test, thr


def run_best_single_signal(train, cal, test, alpha, delta, min_keep, bound):
    """Best-single-signal gating: pick the signal with highest AUROC on train,
    then use it on cal/test. This avoids logistic fusion destroying signals."""
    # Evaluate each signal's AUROC on train
    best_signal = None
    best_auroc = -1.0
    for sig_key, feat_col in SINGLE_SIGNALS.items():
        scores = [safe_float(r.get(feat_col, 0.5)) for r in train]
        y = [int(r["correct"]) for r in train]
        if len(set(y)) < 2:
            continue
        try:
            auc = _auroc(y, scores)
            if not math.isnan(auc) and auc > best_auroc:
                best_auroc = auc
                best_signal = sig_key
        except Exception:
            pass
    if best_signal is None:
        best_signal = "rubric_vote_share"
    p_test, thr = run_single_signal(train, cal, test, best_signal, alpha, delta, min_keep, bound)
    return p_test, thr, best_signal, best_auroc


def run_max_signal(train, cal, test, alpha, delta, min_keep, bound):
    """Max-signal: take the maximum of rubric + swap + self vote shares."""
    def max_score(r):
        vals = [safe_float(r.get("feat_rubric_vote_share", 0)),
                safe_float(r.get("feat_swap_consistency", 0)),
                safe_float(r.get("feat_self_vote_share", 0))]
        return max(vals)
    p_cal = [max_score(r) for r in cal]
    cal_pairs = [(p, int(r["correct"])) for r, p in zip(cal, p_cal)]
    thr, _ = calibrate_threshold([x[0] for x in cal_pairs], [x[1] for x in cal_pairs], alpha=alpha, delta=delta, min_keep=min_keep, bound=bound)
    p_test = [max_score(r) for r in test]
    return p_test, thr


def eval_method(test, p_test, thr):
    selected = apply_threshold(test, p_test, thr)
    sel = summarize_selective(selected)
    cal = calibration_report(test, p_test)
    acc_vals = [float(r["correct"]) for r in selected if r.get("accepted") and r.get("correct") is not None]
    ci = bootstrap_ci(acc_vals) if acc_vals else (None, None)
    return {**sel, **{f"cal_{k}": v for k, v in cal.items()}, "threshold": thr, "ci_low": ci[0], "ci_high": ci[1]}


def signal_rates(rows):
    """Compute per-signal stable/unstable accuracy."""
    specs = [
        ("rubric_stable", lambda r: float(r.get("feat_rubric_flip", 1)) == 0.0),
        ("rubric_unstable", lambda r: float(r.get("feat_rubric_flip", 0)) > 0.0),
        ("position_consistent", lambda r: float(r.get("feat_swap_consistency", 0)) >= 1.0),
        ("position_inconsistent", lambda r: float(r.get("feat_swap_consistency", 0)) < 1.0),
        ("self_high", lambda r: float(r.get("feat_self_vote_share", 0)) >= 0.8),
        ("self_low", lambda r: float(r.get("feat_self_vote_share", 1)) < 0.8),
    ]
    out = []
    for name, pred in specs:
        g = [r for r in rows if r.get("correct") is not None and pred(r)]
        if g:
            acc = sum(int(r["correct"]) for r in g) / len(g)
            out.append({"signal": name, "n": len(g), "accuracy": acc, "error_rate": 1-acc})
        else:
            out.append({"signal": name, "n": 0, "accuracy": None, "error_rate": None})
    return out


# ── main experiment ──────────────────────────────────────────────────────────

def run_full_qwen_experiment(features_path, dataset_name, out_dir, seeds=10, alpha=0.15, delta=0.10, min_keep=20, bound="clopper_pearson"):
    rows = [r for r in read_jsonl(features_path) if r.get("correct") is not None]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(rows)

    methods = ["logistic_fusion", "best_single_signal", "max_signal", "rubric_only", "swap_only", "self_only", "raw"]
    all_results = []
    alpha_sweep = []

    for seed in range(seeds):
        train, cal, test = split3(rows, seed, 0.4, 0.3)
        if len(test) < 10:
            continue
        for method in methods:
            if method == "logistic_fusion":
                p_test, thr, _ = run_logistic_fusion(train, cal, test, alpha, delta, min_keep, bound)
            elif method == "best_single_signal":
                p_test, thr, best_sig, best_auc = run_best_single_signal(train, cal, test, alpha, delta, min_keep, bound)
            elif method == "max_signal":
                p_test, thr = run_max_signal(train, cal, test, alpha, delta, min_keep, bound)
            elif method == "rubric_only":
                p_test, thr = run_single_signal(train, cal, test, "rubric_vote_share", alpha, delta, min_keep, bound)
            elif method == "swap_only":
                p_test, thr = run_single_signal(train, cal, test, "swap_consistency", alpha, delta, min_keep, bound)
            elif method == "self_only":
                p_test, thr = run_single_signal(train, cal, test, "self_vote_share", alpha, delta, min_keep, bound)
            elif method == "raw":
                p_test = [0.5] * len(test)  # always accept
                thr = 0.0
            res = eval_method(test, p_test, thr)
            res.update({"dataset": dataset_name, "method": method, "seed": seed, "n_test": len(test)})
            all_results.append(res)

        # alpha sweep for logistic_fusion
        for a in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            p_test, thr, _ = run_logistic_fusion(train, cal, test, a, delta, min_keep, bound)
            res = eval_method(test, p_test, thr)
            res.update({"dataset": dataset_name, "method": "logistic_fusion", "alpha": a, "seed": seed})
            alpha_sweep.append(res)

    # signal rates (full data, no split needed)
    sig = signal_rates(rows)

    # aggregate by method
    agg = {}
    for method in methods:
        mres = [r for r in all_results if r["method"] == method]
        if not mres:
            continue
        numeric = ["coverage", "accuracy_accepted", "risk_accepted", "raw_accuracy", "cal_auroc", "cal_ece", "cal_brier", "cal_auprc"]
        entry = {"method": method, "n_folds": len(mres)}
        for k in numeric:
            vals = [float(r[k]) for r in mres if r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
            if vals:
                entry[k] = sum(vals)/len(vals)
                entry[f"{k}_std"] = (sum((x-entry[k])**2 for x in vals)/len(vals))**0.5
        agg[method] = entry

    # aggregate alpha sweep
    sweep_agg = {}
    for a in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        sres = [r for r in alpha_sweep if r.get("alpha") == a]
        if not sres:
            continue
        entry = {"alpha": a, "n_folds": len(sres)}
        for k in ["coverage", "accuracy_accepted", "risk_accepted"]:
            vals = [float(r[k]) for r in sres if r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
            if vals:
                entry[k] = sum(vals)/len(vals)
                entry[f"{k}_std"] = (sum((x-entry[k])**2 for x in vals)/len(vals))**0.5
        sweep_agg[str(a)] = entry

    report = {
        "dataset": dataset_name,
        "n_total": n,
        "seeds": seeds,
        "alpha": alpha,
        "delta": delta,
        "bound": bound,
        "splits": "train=0.4 cal=0.3 test=0.3",
        "aggregated_results": agg,
        "alpha_sweep": sweep_agg,
        "signal_rates": sig,
        "all_fold_results": all_results,
    }
    with open(out / f"{dataset_name}_full_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # CSV for easy reading
    fieldnames = sorted({k for r in all_results for k in r})
    with open(out / f"{dataset_name}_fold_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_results)

    with open(out / f"{dataset_name}_aggregated.csv", "w", newline="") as f:
        fieldnames = sorted({k for v in agg.values() for k in v})
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for v in agg.values():
            w.writerow(v)

    with open(out / f"{dataset_name}_alpha_sweep.csv", "w", newline="") as f:
        fieldnames = sorted({k for v in sweep_agg.values() for k in v})
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for v in sweep_agg.values():
            w.writerow(v)

    with open(out / f"{dataset_name}_signal_rates.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in sig for k in r})
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(sig)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--min-keep", type=int, default=20)
    ap.add_argument("--bound", default="clopper_pearson")
    args = ap.parse_args()
    report = run_full_qwen_experiment(args.features, args.dataset, args.out_dir, args.seeds, args.alpha, args.delta, args.min_keep, args.bound)
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  N={report['n_total']}  seeds={report['seeds']}")
    print(f"{'='*60}")
    for method, vals in report["aggregated_results"].items():
        cov = vals.get("coverage", 0)
        acc = vals.get("accuracy_accepted", 0)
        auc = vals.get("cal_auroc", 0)
        ece = vals.get("cal_ece", 0)
        print(f"  {method:22s}  cov={cov:.3f}  acc={acc:.3f}  auroc={auc:.3f}  ece={ece:.3f}")
    print(f"\n  Signal rates:")
    for s in report["signal_rates"]:
        a = s.get("accuracy")
        print(f"    {s['signal']:22s}  n={s['n']:>4d}  acc={a:.3f}" if a else f"    {s['signal']:22s}  n={s['n']:>4d}  acc=N/A")
    print(f"\n  Alpha sweep (logistic_fusion):")
    for a, vals in report["alpha_sweep"].items():
        cov = vals.get("coverage", 0)
        acc = vals.get("accuracy_accepted", 0)
        print(f"    alpha={a:>5s}  cov={cov:.3f}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
