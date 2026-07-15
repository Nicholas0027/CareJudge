#!/usr/bin/env python3
"""Optimized figure generation — caches calibrator fits, uses numpy for predictions.

Key optimization: fit calibrator ONCE per (judge, benchmark, seed), reuse across alphas.
Reduces 1440 calibrator fittings to 120 (12x speedup).
Uses numpy vectorized prediction instead of pure Python loops.
"""
import json, os, random, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CB = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}

JUDGES = [("deepseek-chat", "DeepSeek-V4", CB["blue"]),
          ("gpt-5_5", "GPT-5.5", CB["green"]),
          ("qwen-1.5b", "Qwen-1.5B", CB["vermillion"])]
BENCHMARKS = [("judgebench", "JudgeBench"), ("tldr", "TL;DR"),
              ("rewardbench", "RewardBench"), ("lmaarena", "LMArena")]
FEAT_COLS = ["feat_swap_consistency", "feat_swap_conf_gap",
             "feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip",
             "feat_self_vote_share", "feat_self_entropy", "feat_sim_vote_share",
             "feat_sim_entropy", "feat_sim_flip",
             "confidence", "feat_base_conf", "feat_mean_conf", "feat_std_conf",
             "feat_length_gap_norm", "feat_score_margin"]


def read_jsonl(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
            if r.get("correct") is not None:
                rows.append(r)
        except Exception:
            pass
    return rows


def to_matrix(rows, cols):
    return np.array([[float(r.get(c, 0.0) or 0.0) for c in cols] for r in rows], dtype=np.float64)


def split3(n, seed, tr=0.4, ca=0.3):
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    nt, nc = int(n*tr), int(n*ca)
    return idx[:nt], idx[nt:nt+nc], idx[nt+nc:]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_predict_logistic(X_train, y_train, X_target):
    """Fit sklearn LogisticRegression, return predicted probabilities."""
    from sklearn.linear_model import LogisticRegression
    if len(set(y_train)) < 2:
        return np.full(len(X_target), float(sum(y_train)/len(y_train)))
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_target)[:, 1]


def clopper_pearson_upper(e, n, delta=0.10):
    """Exact one-sided CP upper bound via beta.ppf (no bisection needed)."""
    from scipy.stats import beta as beta_dist
    if n == 0:
        return 1.0
    if e == n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, e + 1, n - e))


def calibrate_threshold_vec(p_cal, y_cal, alpha, delta=0.10, min_keep=20):
    """Vectorized prefix search: compute CP bounds for all prefixes at once."""
    from scipy.stats import beta as beta_dist
    order = np.argsort(p_cal)[::-1]
    p_sorted = p_cal[order]
    y_sorted = y_cal[order]
    n = len(p_sorted)
    if n < min_keep:
        return float('inf')
    # Cumulative error counts for prefixes [min_keep, n]
    cum_err = np.cumsum(y_sorted == 0)
    prefixes = np.arange(min_keep, n + 1)
    e_vals = cum_err[prefixes - 1]
    # Vectorized CP upper bound: beta.ppf(1-delta, e+1, n-e)
    # Handle e==n case (CP=1.0)
    mask = e_vals < prefixes
    cp_vals = np.ones(len(prefixes))
    if np.any(mask):
        cp_vals[mask] = beta_dist.ppf(1 - delta, e_vals[mask] + 1, prefixes[mask] - e_vals[mask])
    # Find largest prefix where CP <= alpha
    valid = cp_vals <= alpha
    if not np.any(valid):
        return float('inf')
    best_i = prefixes[valid][-1]
    return float(p_sorted[best_i - 1])


def calibrate_threshold(p_cal, y_cal, alpha, delta=0.10, min_keep=20):
    """Empirical prefix search: largest certified prefix."""
    order = np.argsort(p_cal)[::-1]
    p_sorted = p_cal[order]
    y_sorted = y_cal[order]
    best_thr = float('inf')
    for i in range(min_keep, len(p_sorted) + 1):
        e = int(np.sum(y_sorted[:i] == 0))
        cp = clopper_pearson_upper(e, i, delta)
        if cp <= alpha:
            best_thr = float(p_sorted[i - 1])
    return best_thr


# ── Figure 1: Signal gap ──
def make_signal_gap(features_dir, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bar_data = {}
    for pfx, jname, _ in JUDGES:
        for bench, bname in BENCHMARKS:
            path = os.path.join(features_dir, f"{pfx}_{bench}_features.jsonl")
            if not os.path.exists(path):
                continue
            rows = read_jsonl(path)
            if len(rows) < 50:
                continue
            stable = [r for r in rows if float(r.get("feat_rubric_flip", 1)) == 0.0]
            unstable = [r for r in rows if float(r.get("feat_rubric_flip", 0)) > 0.0]
            def acc(xs):
                return sum(r["correct"] for r in xs)/len(xs) if xs else None
            key = f"{jname}/{bname}"
            bar_data[key] = (acc(stable), acc(unstable), len(stable), len(unstable))

    labels = list(bar_data.keys())
    stable_vals = [bar_data[k][0] or 0 for k in labels]
    unstable_vals = [bar_data[k][1] or 0 for k in labels]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, stable_vals, w, label="Rubric-stable", color=CB["green"], edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, unstable_vals, w, label="Rubric-unstable", color=CB["orange"], edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, ls=":", color=CB["black"], lw=0.7, alpha=0.5)
    for i, k in enumerate(labels):
        s, u = stable_vals[i], unstable_vals[i]
        if s > 0 and u > 0:
            gap = s - u
            color = CB["blue"] if gap > 0 else CB["vermillion"]
            ax.annotate(f"{gap:+.1%}", xy=(i, max(s, u) + 0.03), ha="center", fontsize=6.5,
                        color=color, fontweight="bold")
    ax.set_ylabel("Judge accuracy", fontsize=9)
    ax.set_xticks(list(x))
    short_labels = [k.replace("JudgeBench", "JB").replace("RewardBench", "RB").replace("LMArena", "Arena")
                    for k in labels]
    ax.set_xticklabels(short_labels, fontsize=6.5, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2, lw=0.4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "signal_gap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote signal_gap.pdf", flush=True)
    for k, v in bar_data.items():
        if v[0] and v[1]:
            print(f"  {k}: stable={v[0]:.3f}(n={v[2]}) unstable={v[1]:.3f}(n={v[3]}) gap={v[0]-v[1]:+.3f}", flush=True)


# ── Figure 2: Risk-coverage (optimized) ──
def make_risk_coverage(features_dir, out_dir, seeds=10):
    alphas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    axes = axes.flatten()

    for ax_idx, (bench, bname) in enumerate(BENCHMARKS):
        ax = axes[ax_idx]
        for pfx, jname, color in JUDGES:
            path = os.path.join(features_dir, f"{pfx}_{bench}_features.jsonl")
            if not os.path.exists(path):
                continue
            rows = read_jsonl(path)
            if len(rows) < 50:
                continue

            # Pre-compute full feature matrix and labels
            X_full = to_matrix(rows, FEAT_COLS)
            y_full = np.array([int(r["correct"]) for r in rows])
            n = len(rows)

            cov_avg = []
            risk_avg = []
            # Fit per seed, then evaluate all alphas (cached predictions)
            seed_results = []
            for seed in range(seeds):
                tr_idx, ca_idx, te_idx = split3(n, seed)
                if len(te_idx) < 10:
                    continue
                X_train = X_full[tr_idx]
                y_train = y_full[tr_idx]
                X_cal = X_full[ca_idx]
                y_cal = y_full[ca_idx]
                X_test = X_full[te_idx]
                y_test = y_full[te_idx]
                p_cal = fit_predict_logistic(X_train, y_train, X_cal)
                p_test = fit_predict_logistic(X_train, y_train, X_test)
                seed_results.append((p_cal, y_cal, p_test, y_test))

            for alpha in alphas:
                covs, risks = [], []
                for p_cal, y_cal, p_test, y_test in seed_results:
                    thr = calibrate_threshold_vec(p_cal, y_cal, alpha)
                    acc_mask = p_test >= thr
                    cov = np.mean(acc_mask)
                    if np.any(acc_mask):
                        acc = np.mean(y_test[acc_mask])
                        risk = 1 - acc
                    else:
                        risk = 0.0
                    covs.append(cov)
                    risks.append(risk)
                cov_avg.append(np.mean(covs) if covs else 0)
                risk_avg.append(np.mean(risks) if risks else 0)

            pts = sorted(zip(cov_avg, risk_avg))
            cov_sorted = [p[0] for p in pts]
            risk_sorted = [p[1] for p in pts]
            ax.plot(cov_sorted, risk_sorted, label=jname, color=color, lw=1.8, marker="o", markersize=3)
            print(f"  {jname}/{bname}: {len(seed_results)} seeds, "
                  f"cov range [{min(cov_avg):.2f},{max(cov_avg):.2f}]", flush=True)

        ax.axhline(0.15, ls="--", color=CB["black"], lw=0.8, alpha=0.5, label=r"$\alpha=0.15$")
        ax.set_title(bname, fontsize=10)
        ax.set_xlabel("Coverage", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 0.55)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Selective risk\n(accepted error)", fontsize=8)
    axes[2].set_ylabel("Selective risk\n(accepted error)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "risk_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote risk_coverage.pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/scale")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_signal_gap(args.features_dir, args.out_dir)
    make_risk_coverage(args.features_dir, args.out_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()
