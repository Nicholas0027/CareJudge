#!/usr/bin/env python3
"""Redraw Figure 1 (paper/figures/overview.pdf).

Top row: the CARE-Judge pipeline (feature extraction -> calibration & risk
control -> selective decision).

Bottom row: three discrete regime cards (no curve, no numeric axis) that state
the paper's framing: protocol-stability fusion is broadly effective on
competent judges, and on a judge below the competence floor CARE-Judge abstains
automatically. The only quantities shown are AUROC / coverage figures taken
verbatim from Table 1 (tab:main); nothing here is fit or inferred.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# palette (consistent with Fig. 2: blue = informative/accept, grey = neutral)
BLUE_D = "#2F5C8F"
BLUE_M = "#5B8AC0"
BLUE_L = "#DCE7F3"
GREY_D = "#6E6E6E"
GREY_M = "#9E9E9E"
GREY_L = "#ECECEC"
INK = "#1A1A1A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig = plt.figure(figsize=(7.4, 4.15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.2, rad=0.02):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rad*100}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )


def arrow(x0, y0, x1, y1, color=INK, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=13,
            linewidth=lw, color=color, zorder=3,
        )
    )


def txt(x, y, s, ha="center", va="center", size=9, weight="normal", color=INK):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight, color=color, zorder=4)


# ----------------------------------------------------------------- title strip
txt(50, 97, "CARE-Judge: selective LLM-as-a-judge from protocol-stability signals",
    size=10.5, weight="bold")

# =========================================================== TOP: pipeline row
py, ph = 61, 26
w1, w2, w3 = 27, 30, 24
x1 = 3
x2 = x1 + w1 + 8
x3 = x2 + w2 + 8

# 1. feature extraction
box(x1, py, w1, ph, BLUE_L, BLUE_D)
txt(x1 + w1 / 2, py + ph - 3.4, "1. Feature extraction", weight="bold", color=BLUE_D, size=9.2)
txt(x1 + w1 / 2, py + ph - 8.0,
    "6 judge calls per item\n$\\rightarrow$ 3 protocol-stability families\n(rubric, position, self-consistency)\n+ confidence, length $=\\phi(x)$",
    size=7.6, va="top")

# 2. calibration & risk control
box(x2, py, w2, ph, BLUE_L, BLUE_D)
txt(x2 + w2 / 2, py + ph - 3.4, "2. Calibration & risk control", weight="bold", color=BLUE_D, size=9.2)
txt(x2 + w2 / 2, py + ph - 8.5,
    "correctness calibrator\n$\\hat{p}(\\mathrm{correct}\\mid\\phi)$ on TRAIN;\nexact Clopper–Pearson\nthreshold $\\hat{\\lambda}$ on CAL",
    size=7.8, va="top")

# 3. selective decision
box(x3, py, w3, ph, BLUE_L, BLUE_D)
txt(x3 + w3 / 2, py + ph - 3.4, "3. Selective decision", weight="bold", color=BLUE_D, size=9.2)
txt(x3 + w3 / 2, py + ph - 8.5,
    "on held-out TEST:\naccept if $\\hat{p}\\geq\\hat{\\lambda}$,\nelse abstain\n(risk budget $\\alpha$)",
    size=7.8, va="top")

arrow(x1 + w1, py + ph / 2, x2, py + ph / 2)
arrow(x2 + w2, py + ph / 2, x3, py + ph / 2)

# ============================================ BOTTOM: three discrete regime cards
txt(50, 57.5, "When is the signal useful? Judge task competence, left to right:",
    size=9.0, weight="bold", color=INK)

cy, ch = 6, 44
cw = 30
gap = 3
cx = [3, 3 + cw + gap, 3 + 2 * (cw + gap)]

# card 1: below competence floor (abstain) -- grey, the "flag a weak judge" value
box(cx[0], cy, cw, ch, GREY_L, GREY_D, lw=1.4)
txt(cx[0] + cw / 2, cy + ch - 5, "Below competence floor", weight="bold", color=GREY_D, size=9.2)
txt(cx[0] + cw / 2, cy + ch - 11.5, "near-random judge\n(e.g. Qwen2.5-1.5B)", size=8.2, color=GREY_D)
txt(cx[0] + cw / 2, cy + ch - 24,
    "Cannot tell better from\nworse, so verdicts sit at\nchance and their stability\ncarries no information.",
    size=8.0)
box(cx[0] + 2.5, cy + 3.0, cw - 5, 9.5, "#FFFFFF", GREY_D, lw=1.0)
txt(cx[0] + cw / 2, cy + 7.7, "CARE-Judge abstains:\ncoverage $\\rightarrow$ 0\n(weak judge flagged)",
    size=7.6, weight="bold", color=GREY_D)

# card 2: mid-accuracy (largest gain) -- strong blue
box(cx[1], cy, cw, ch, BLUE_L, BLUE_D, lw=1.4)
txt(cx[1] + cw / 2, cy + ch - 5, "Mid-accuracy", weight="bold", color=BLUE_D, size=9.2)
txt(cx[1] + cw / 2, cy + ch - 11.5, "raw agreement 0.65–0.80\n(Qwen2.5-7B, DeepSeek-V4)", size=8.2, color=BLUE_D)
txt(cx[1] + cw / 2, cy + ch - 24,
    "Instability tracks error;\nfusing the three families\nadds the most usable\nsignal here.",
    size=8.0)
box(cx[1] + 2.5, cy + 3.0, cw - 5, 9.5, "#FFFFFF", BLUE_D, lw=1.0)
txt(cx[1] + cw / 2, cy + 7.7, "Largest gains: up to\n$+7.6$ AUROC over\nbest single signal",
    size=7.6, weight="bold", color=BLUE_D)

# card 3: frontier (redundant, matches) -- light blue
box(cx[2], cy, cw, ch, "#EEF3FA", BLUE_M, lw=1.4)
txt(cx[2] + cw / 2, cy + ch - 5, "Frontier (saturated)", weight="bold", color=BLUE_M, size=9.2)
txt(cx[2] + cw / 2, cy + ch - 11.5, "already strong judge\n(GPT-5.5)", size=8.2, color=BLUE_M)
txt(cx[2] + cw / 2, cy + ch - 24,
    "Little instability is left\nto exploit; base confidence\nis already reliable on most\nitems.",
    size=8.0)
box(cx[2] + 2.5, cy + 3.0, cw - 5, 9.5, "#FFFFFF", BLUE_M, lw=1.0)
txt(cx[2] + cw / 2, cy + 7.7, "Fusion matches\nstrong base\nconfidence",
    size=7.6, weight="bold", color=BLUE_M)

# left-to-right competence direction arrow (ordering cue, not a numeric axis)
arrow(cx[0] + 2, cy - 2.2, cx[2] + cw - 2, cy - 2.2, color=GREY_M, lw=1.3)

fig.savefig("paper/figures/overview.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("paper/figures/overview.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote paper/figures/overview.pdf and .png")
