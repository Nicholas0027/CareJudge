#!/usr/bin/env python3
"""Figure 1 (paper/figures/overview.pdf): CARE-Judge architecture, 21:9 banner.

Layout: Input | CARE-Judge method pipeline | Selective-verdict Output, in the
style of recent ML systems figures. The figure describes the architecture only
(no per-model results). All quantities are fixed design constants (6 calls,
K=3, S=3, 40/30/30 split, budget alpha/delta).

Typeface: Bitter (loaded from assets/fonts) to match the paper body font.
Small hand-drawn vector glyphs (no emoji dependency) add visual anchors, and
light callouts annotate the flow.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

# ---- Bitter font (matches the paper body) --------------------------------
FONTDIR = "assets/fonts"
for f in ("Bitter-Regular", "Bitter-Medium", "Bitter-SemiBold", "Bitter-Bold"):
    p = os.path.join(FONTDIR, f + ".ttf")
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Bitter"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---- palette --------------------------------------------------------------
INK = "#1E1E1E"
GREY_D = "#6E6E6E"
PANEL = "#EDEFF2"
PANEL_E = "#C7CCD2"
CORE = "#EAF1FA"
CORE_E = "#9DBBDD"
BLUE_D = "#2F5C8F"
BLUE_M = "#5B8AC0"
FEAT = "#FCF3E2"
FEAT_E = "#E4C892"
AMBER_D = "#9A6B1E"
CAL = "#E7F0FA"
GREEN = "#3E7D5A"
GREEN_L = "#E1EFE7"
GREEN_E = "#AFCBBB"

fig = plt.figure(figsize=(10.5, 4.5))  # 21:9
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.1, rad=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rad}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def arrow(x0, y0, x1, y1, color=INK, lw=1.7, ms=12, z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=ms, linewidth=lw, color=color, zorder=z,
                 connectionstyle=f"arc3,rad={rad}"))


def T(x, y, s, ha="center", va="center", size=8.5, w="normal", c=INK, st="normal", z=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=w, color=c, style=st, zorder=z)


# ---- tiny vector glyphs (drawn, so they never depend on emoji fonts) -----
def g_doc(cx, cy, s, c):  # document: judge under audit
    ax.add_patch(Rectangle((cx - s*0.55, cy - s*0.7), s*1.1, s*1.4, fill=False,
                 edgecolor=c, lw=1.3, zorder=7))
    for k in range(3):
        ax.add_line(Line2D([cx - s*0.3, cx + s*0.3], [cy + s*0.25 - k*s*0.35]*2,
                    color=c, lw=1.0, zorder=7))


def g_copies(cx, cy, s, c):  # stacked copies: perturbations
    for i, off in enumerate((0.5, 0.0, -0.5)):
        ax.add_patch(Rectangle((cx - s*0.5 + off*s*0.6, cy - s*0.5 - off*s*0.6),
                     s, s, fill=(i == 2), facecolor=c if i == 2 else "none",
                     edgecolor=c, lw=1.1, zorder=7))


def g_bars(cx, cy, s, c):  # feature bars
    for i, h in enumerate((0.5, 1.0, 0.75, 1.15)):
        ax.add_patch(Rectangle((cx - s*0.8 + i*s*0.45, cy - s*0.6), s*0.3, s*h*1.0,
                     facecolor=c, edgecolor="none", zorder=7))


def g_sigmoid(cx, cy, s, c):  # calibrator: fitted curve
    import numpy as np
    xs = np.linspace(-1, 1, 40)
    ys = 1 / (1 + np.exp(-4 * xs))
    ax.add_line(Line2D(cx + xs * s * 0.9, cy + (ys - 0.5) * s * 1.6,
                color=c, lw=1.6, zorder=7))


def g_gate(cx, cy, s, c):  # threshold: gate / bar with split
    ax.add_line(Line2D([cx, cx], [cy - s*0.7, cy + s*0.7], color=c, lw=1.8, zorder=7))
    ax.add_line(Line2D([cx - s*0.8, cx - s*0.15], [cy + s*0.35]*2, color=c, lw=1.3, zorder=7))
    ax.add_line(Line2D([cx + s*0.15, cx + s*0.8], [cy - s*0.35]*2, color=c, lw=1.3, zorder=7))


def g_check(cx, cy, s, c):  # accept
    ax.add_line(Line2D([cx - s*0.55, cx - s*0.1, cx + s*0.7],
                [cy, cy - s*0.5, cy + s*0.6], color=c, lw=2.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=7))


def g_stop(cx, cy, s, c):  # abstain: no-entry
    ax.add_patch(Circle((cx, cy), s*0.62, fill=False, edgecolor=c, lw=1.8, zorder=7))
    ax.add_line(Line2D([cx - s*0.42, cx + s*0.42], [cy, cy], color=c, lw=1.8, zorder=7))


# ============================================================== INPUT panel
ix, iw = 1.5, 17.5
box(ix, 12, iw, 74, PANEL, PANEL_E, 1.2, 2.2)
T(ix + iw / 2, 81.5, "Input", w="bold", size=11.5)
inp = [("LLM judge $f$", "model under audit", g_doc),
       ("Item $x$", "one pairwise comparison", None),
       ("Reference labels", "calibration / test", None),
       ("Risk budget $\\alpha,\\delta$", "target selective risk", None)]
iy = 72
for name, sub, glyph in inp:
    box(ix + 1.4, iy - 6.8, iw - 2.8, 6.2, "#FFFFFF", PANEL_E, 0.9, 1.2)
    if glyph:
        glyph(ix + 4.2, iy - 3.7, 1.7, BLUE_D)
        T(ix + iw/2 + 2.2, iy - 2.7, name, w="bold", size=8.0)
        T(ix + iw/2 + 2.2, iy - 5.4, sub, size=6.5, c=GREY_D)
    else:
        T(ix + iw / 2, iy - 2.7, name, w="bold", size=8.0)
        T(ix + iw / 2, iy - 5.4, sub, size=6.5, c=GREY_D)
    iy -= 8.7

# ============================================================== CORE panel
cx, cw = 22, 52
box(cx, 8, cw, 84, CORE, CORE_E, 1.4, 2.6)
T(cx + cw / 2, 88, "CARE-Judge", w="bold", size=12)
T(cx + cw / 2, 83.6, "from protocol perturbations to a risk-controlled verdict",
  size=7.8, c=BLUE_D, st="italic")

# stage 1: perturbations
p1x, p1w = cx + 2.5, 20
box(p1x, 44, p1w, 33, "#FFFFFF", BLUE_M, 1.2, 1.6)
g_copies(p1x + 3.2, 73.2, 1.6, BLUE_D)
T(p1x + p1w/2 + 2.0, 74, "Protocol perturbations", w="bold", size=8.6, c=BLUE_D)
T(p1x + p1w / 2, 70.6, "6 judge calls per item", size=6.7, c=GREY_D)
for i, c in enumerate(["base verdict  ($\\tau{=}0$)", "position swap",
                       "rubric paraphrases ($K{=}3$)", "self-consistency ($S{=}3$)"]):
    yy = 66 - i * 4.4
    box(p1x + 1.6, yy - 3.4, p1w - 3.2, 3.3, CORE, CORE_E, 0.7, 0.9)
    T(p1x + p1w / 2, yy - 1.75, c, size=6.8)

# stage 2: feature families
p2x, p2w = cx + 2.5, 20
box(p2x, 12.5, p2w, 26, FEAT, FEAT_E, 1.2, 1.6)
g_bars(p2x + 3.2, 34.6, 1.5, AMBER_D)
T(p2x + p2w/2 + 2.2, 35.3, "Feature families $\\phi(x)$", w="bold", size=8.5, c=AMBER_D)
for i, f in enumerate(["rubric stability", "position stability",
                       "self-consistency", "confidence, length"]):
    T(p2x + p2w / 2, 30 - i * 3.9, "$\\bullet$  " + f, size=7.0)

arrow(p1x + p1w / 2, 44, p2x + p2w / 2, 38.5, BLUE_D, 1.8)
T(p1x + p1w/2 + 6.6, 41.2, "per-item\ntraces", size=6.0, c=GREY_D, st="italic")

# stage 3: calibrator
p3x, p3w = cx + 26.5, 23
box(p3x, 52, p3w, 25, CAL, BLUE_M, 1.2, 1.6)
g_sigmoid(p3x + 4.0, 72.5, 2.0, BLUE_D)
T(p3x + p3w/2 + 2.6, 73.3, "Correctness calibrator", w="bold", size=8.6, c=BLUE_D)
T(p3x + p3w / 2, 66.5, "$\\hat{p}(\\mathrm{correct}\\mid\\phi(x))$", size=8.4)
T(p3x + p3w / 2, 61.5, "logistic / isotonic / GBM", size=6.6, c=GREY_D)
T(p3x + p3w / 2, 57, "fit on TRAIN", size=7.2, w="bold", c=BLUE_D)

# stage 4: threshold
box(p3x, 20, p3w, 25, CAL, BLUE_M, 1.2, 1.6)
g_gate(p3x + 4.0, 40.5, 2.0, BLUE_D)
T(p3x + p3w/2 + 2.6, 41.3, "Risk-controlled threshold", w="bold", size=8.6, c=BLUE_D)
T(p3x + p3w / 2, 34.5, "$\\hat{\\lambda}$ : exact Clopper–Pearson", size=7.4)
T(p3x + p3w / 2, 30, "prefix search at level $(\\alpha,\\delta)$", size=6.6, c=GREY_D)
T(p3x + p3w / 2, 25.5, "selected on CAL", size=7.2, w="bold", c=BLUE_D)

arrow(p2x + p2w, 27, p3x, 60, BLUE_D, 1.8, rad=-0.16)
arrow(p1x + p1w, 60.5, p3x, 64.5, BLUE_D, 1.8, rad=-0.05)
arrow(p3x + p3w / 2, 52, p3x + p3w / 2, 45, BLUE_D, 1.8)
T(p3x + p3w/2 + 8.5, 48.5, "calibrated\nscores", size=6.0, c=GREY_D, st="italic")

# ============================================================== OUTPUT panel
ox, ow = 77.5, 21
box(ox, 12, ow, 74, GREEN_L, GREEN_E, 1.2, 2.2)
T(ox + ow / 2, 81.5, "Selective verdict", w="bold", size=10.5)
T(ox + ow / 2, 77.2, "on held-out TEST", size=6.9, c=GREY_D)

box(ox + 1.6, 56, ow - 3.2, 15, "#FFFFFF", GREEN, 1.2, 1.4)
g_check(ox + 4.6, 63.5, 2.0, GREEN)
T(ox + ow/2 + 2.4, 67, "ACCEPT", w="bold", size=9.4, c=GREEN)
T(ox + ow/2 + 2.4, 63, "$\\hat{p}\\geq\\hat{\\lambda}$", size=8.0, c=GREEN)
T(ox + ow / 2, 58.5, "certified verdict", size=6.7, c=GREY_D)

box(ox + 1.6, 34, ow - 3.2, 16.5, "#FFFFFF", GREY_D, 1.2, 1.4)
g_stop(ox + 4.6, 43.5, 2.0, GREY_D)
T(ox + ow/2 + 2.4, 47, "ABSTAIN", w="bold", size=9.4, c=GREY_D)
T(ox + ow/2 + 2.4, 43, "$\\hat{p}<\\hat{\\lambda}$", size=8.0, c=GREY_D)
T(ox + ow / 2, 37.8, "risky item, or a judge\ntoo weak to certify", size=6.5, c=GREY_D)

box(ox + 1.6, 15, ow - 3.2, 15, "#FFFFFF", GREEN_E, 0.9, 1.2)
T(ox + ow / 2, 25.8, "Reported", w="bold", size=7.4)
T(ox + ow / 2, 20.5, "coverage, selective risk,\nviolation rate", size=6.6, c=GREY_D)

# ============================================================== flow arrows
arrow(ix + iw, 49, cx, 49, INK, 2.0, ms=15)
arrow(cx + cw, 49, ox, 49, INK, 2.0, ms=15)

fig.savefig("paper/figures/overview.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("paper/figures/overview.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
print("wrote paper/figures/overview.pdf and .png (Bitter font)")
