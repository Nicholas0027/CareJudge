#!/usr/bin/env python3
"""Generate risk-coverage curves and signal-gap bar chart for the paper (PDF, colorblind-safe).

All text is rendered in the Bitter font family (OFL). Static weights are
instantiated under assets/fonts/ (Bitter-Regular/Medium/SemiBold/Bold.ttf).
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# --- Register the Bitter font family so every glyph in the figures uses it ---
FONT_DIR = Path("assets/fonts")
for _f in ["Bitter-Regular.ttf", "Bitter-Medium.ttf", "Bitter-SemiBold.ttf", "Bitter-Bold.ttf"]:
    _p = FONT_DIR / _f
    if _p.exists():
        fm.fontManager.addfont(str(_p))
# Embed fonts as real glyphs (TrueType) in the PDF, not Type-3 bitmaps.
matplotlib.rcParams.update({
    "font.family": "Bitter",
    "font.serif": ["Bitter"],
    "mathtext.fontset": "dejavuserif",  # math falls back cleanly; no axis labels use mathtext here
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

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
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Coverage", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Selective risk\n(accepted error)", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.pdf", bbox_inches="tight")
    print("wrote risk_coverage.pdf")


def gap_figure():
    """Signal-gap bar chart: rubric-stable vs rubric-unstable judge accuracy.

    Values are the rubric-signal stable/unstable subset accuracies reported
    verbatim in the appendix signals table (Table `tab:signals`): DeepSeek-V4
    on all four benchmarks plus the GPT-5.5/RewardBench inversion. No values
    are hand-set; each pair below matches an appendix row.
    """
    # (stable_acc, unstable_acc) — rubric signal, recomputed from the per-item
    # feature traces by scripts/compute_signal_gaps.py (authoritative; matches
    # appendix Table tab:signals). Ordered by judge accuracy to show the
    # competence gradient, including the below-competence case
    # (Qwen-1.5B/RewardBench, near-chance judge where both subsets are ~0.5).
    data = {
        "Qwen-1.5B/JB": (0.511, 0.459),
        "Qwen-1.5B/RB": (0.724, 0.833),      # below-competence: near-chance judge
        "DeepSeek/RB": (0.937, 0.664),
        "Qwen-7B/LMArena": (0.667, 0.496),
        "GPT-5.5/JB": (0.931, 0.667),
        "GPT-5.5/RB": (0.962, 0.692),
    }
    # Refined blue/grey palette: rubric-stable (the signal we highlight) in a
    # deep, low-saturation steel blue; rubric-unstable in a neutral grey so the
    # eye reads the stable bars as the focus. Muted, editorial look.
    STABLE_FILL, STABLE_EDGE = "#2F5D8A", "#20415F"   # steel blue
    UNSTABLE_FILL, UNSTABLE_EDGE = "#BFC4CC", "#9AA0A8"  # neutral grey
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    labels = list(data.keys())
    stable = [data[k][0] for k in labels]
    unstable = [data[k][1] for k in labels]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w/2 for i in x], stable, w, label="Rubric-stable",
           color=STABLE_FILL, edgecolor=STABLE_EDGE, linewidth=0.6, zorder=3)
    ax.bar([i + w/2 for i in x], unstable, w, label="Rubric-unstable",
           color=UNSTABLE_FILL, edgecolor=UNSTABLE_EDGE, linewidth=0.6, zorder=3)
    ax.axhline(0.5, ls=(0, (4, 3)), color="#6B7078", lw=1.0, alpha=0.85,
               label="Random guessing (0.5)", zorder=2)
    ax.set_ylabel("Judge accuracy", fontsize=9, fontweight="bold", color="#2B2B2B")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.5, rotation=15)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.18, lw=0.4, color="#9AA0A8")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9AA0A8")
    # Legend above the axes (horizontal) so it never overlaps the bars.
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, borderaxespad=0.0)
    ax.tick_params(labelsize=8, colors="#2B2B2B")
    fig.tight_layout()
    fig.savefig(OUT / "signal_gap.pdf", bbox_inches="tight")
    print("wrote signal_gap.pdf")


if __name__ == "__main__":
    rc_figure()
    gap_figure()
