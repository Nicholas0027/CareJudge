#!/usr/bin/env python3
"""Generate paper figures from raw feature data (self-contained, no external deps beyond matplotlib).

Figures:
  1. signal_gap.pdf — rubric-stable vs unstable accuracy, grouped by judge × benchmark
  2. risk_coverage.pdf — risk-coverage curves for 3 judges, 4 benchmarks (2×2 panels)

Usage: python scripts/make_figures_v2.py --features-dir outputs/scale --out-dir outputs/figures
"""
import argparse, json, os, random, sys
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator

# Okabe-Ito colorblind-safe palette
CB = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
      "yellow": "#F0E442", "grey": "#999999"}

JUDGES = [("deepseek-chat", "DeepSeek-V4", CB["blue"]),
          ("gpt-5_5", "GPT-5.5", CB["green"]),
          ("qwen-1.5b", "Qwen-1.5B", CB["vermillion"])]
BENCHMARKS = [("judgebench", "JudgeBench"), ("tldr", "TL;DR"),
              ("rewardbench", "RewardBench"), ("lmaarena", "LMArena")]
FEAT_FULL = ["feat_swap_consistency", "feat_swap_conf_gap",
             "feat_rubric_vote_share", "feat_rubric_entropy", "feat_rubric_flip",
             "feat_self_vote_share", "feat_self_entropy", "feat_sim_vote_share",
             "feat_sim_entropy", "feat_sim_flip",
             "confidence", "feat_base_conf", "feat_mean_conf", "feat_std_conf",
             "feat_length_gap_norm", "feat_score_margin"]


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get("correct") is not None]


def split3(rows, seed, tr=0.4, ca=0.3):
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    n = len(idx); nt, nc = int(n*tr), int(n*ca)
    return [rows[i] for i in idx[:nt]], [rows[i] for i in idx[nt:nt+nc]], [rows[i] for i in idx[nt+nc:]]


# ── Figure 1: Signal gap bar chart ──
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

    # Plot: group by judge, 4 benchmarks each
    labels = list(bar_data.keys())
    stable_vals = [bar_data[k][0] or 0 for k in labels]
    unstable_vals = [bar_data[k][1] or 0 for k in labels]
    x = np.arange(len(labels))
    w = 0.38
    bars_s = ax.bar(x - w/2, stable_vals, w, label="Rubric-stable", color=CB["green"], edgecolor="white", linewidth=0.3)
    bars_u = ax.bar(x + w/2, unstable_vals, w, label="Rubric-unstable", color=CB["orange"], edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, ls=":", color=CB["black"], lw=0.7, alpha=0.5)

    # Annotate gaps
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
    print("wrote signal_gap.pdf")
    for k, v in bar_data.items():
        print(f"  {k}: stable={v[0]:.3f}(n={v[2]}) unstable={v[1]:.3f}(n={v[3]}) gap={v[0]-v[1]:+.3f}" if v[0] and v[1] else f"  {k}: incomplete")


# ── Figure 2: Risk-coverage curves ──
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
            # Compute risk-coverage curve averaged over seeds
            cov_avg = []
            risk_avg = []
            for alpha in alphas:
                covs, risks = [], []
                for seed in range(seeds):
                    tr, ca, te = split3(rows, seed)
                    if len(te) < 10:
                        continue
                    yc = [int(r["correct"]) for r in ca]
                    yt = [int(r["correct"]) for r in te]
                    bundle = fit_calibrator(tr, method="logistic", feature_cols=FEAT_FULL)
                    p_cal = bundle.predict_proba(ca)
                    p_test = bundle.predict_proba(te)
                    thr, _ = calibrate_threshold(list(p_cal), yc, alpha=alpha, delta=0.10,
                                                 min_keep=20, bound="clopper_pearson")
                    acc_idx = [i for i in range(len(te)) if p_test[i] >= thr]
                    cov = len(acc_idx)/len(te)
                    if acc_idx:
                        acc = sum(yt[i] for i in acc_idx)/len(acc_idx)
                        risk = 1 - acc
                    else:
                        risk = 0.0
                    covs.append(cov)
                    risks.append(risk)
                cov_avg.append(np.mean(covs) if covs else 0)
                risk_avg.append(np.mean(risks) if risks else 0)
            # Sort by coverage for clean curve
            pts = sorted(zip(cov_avg, risk_avg))
            cov_sorted = [p[0] for p in pts]
            risk_sorted = [p[1] for p in pts]
            ax.plot(cov_sorted, risk_sorted, label=jname, color=color, lw=1.8, marker="o", markersize=3)
        ax.axhline(0.15, ls="--", color=CB["black"], lw=0.8, alpha=0.5, label=r"$\alpha=0.15$")
        ax.set_title(bname, fontsize=10)
        ax.set_xlabel("Coverage", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 0.55)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Selective risk\n(accepted error)", fontsize=8)
    axes[2].set_ylabel("Selective risk\n(accepted error)", fontsize=8)
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "risk_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote risk_coverage.pdf")


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
