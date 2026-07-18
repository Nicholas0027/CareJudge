#!/usr/bin/env python3
"""Generate risk-coverage curves and signal-gap bar chart for the paper (PDF, colorblind-safe)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RIG = Path("outputs/rigorous")
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
CB = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}


def rc_figure():
    """Risk-coverage curves for the three judges on JudgeBench + TL;DR."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    panels = [("JudgeBench", ["qwen_judgebench", "deepseek_judgebench", "gpt55_judgebench"]),
              ("TL;DR", ["qwen_tldr", "deepseek_tldr", "gpt55_tldr"])]
    labels = {"qwen": "Qwen-1.5B", "deepseek": "DeepSeek-V4", "gpt55": "GPT-5.5"}
    colors = {"qwen": CB["vermillion"], "deepseek": CB["blue"], "gpt55": CB["green"]}
    for ax, (title, ds) in zip(axes, panels):
        for d in ds:
            r = json.load(open(RIG / f"{d}_rigorous.json"))
            pts = r["rc_curves"].get("logistic", [])
            if not pts:
                continue
            cov = [p[0] for p in pts]; risk = [p[1] for p in pts]
            key = d.split("_")[0]
            ax.plot(cov, risk, label=labels[key], color=colors[key], lw=1.8)
        ax.axhline(0.15, ls="--", color=CB["black"], lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Coverage", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Selective risk\n(accepted error)", fontsize=9)
    axes[1].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.pdf", bbox_inches="tight")
    print("wrote risk_coverage.pdf")


def gap_figure():
    """Signal-gap bar chart: stable vs unstable accuracy, grouped by model."""
    data = {
        "Qwen-1.5B/JB": (0.472, 0.561), "Qwen/TL;DR": (0.932, 0.620),
        "DeepSeek/JB": (0.700, 0.540), "DeepSeek/TL;DR": (0.689, 0.538),
        "GPT-5.5/JB": (0.936, 0.811), "GPT-5.5/TL;DR": (0.775, 0.345),
    }
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    labels = list(data.keys())
    stable = [data[k][0] for k in labels]
    unstable = [data[k][1] for k in labels]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w/2 for i in x], stable, w, label="Rubric-stable", color=CB["green"])
    ax.bar([i + w/2 for i in x], unstable, w, label="Rubric-unstable", color=CB["orange"])
    ax.axhline(0.5, ls=":", color=CB["black"], lw=0.8, alpha=0.6)
    ax.set_ylabel("Judge accuracy", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.5, rotation=15)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "signal_gap.pdf", bbox_inches="tight")
    print("wrote signal_gap.pdf")


if __name__ == "__main__":
    rc_figure()
    gap_figure()
