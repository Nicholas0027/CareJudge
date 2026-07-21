#!/usr/bin/env python
"""Recompute the alpha-sweep risk-coverage table (Appendix Table `tab:alpha`)
from the authoritative per-item feature traces in ``outputs/scale/``.

For each API judge (DeepSeek-V4, GPT-5.5) and benchmark this reruns the exact
selective-evaluation pipeline of ``final_analysis.py`` (strict 40/30/30 split,
20 seeds, logistic-regression fusion over the 13-dim feature vector, exact
Clopper--Pearson prefix threshold) at each risk budget
``alpha in {0.05,0.10,0.15,0.20,0.25,0.30}`` and reports coverage / accepted
accuracy pooled across seeds. The alpha=0.15 column matches the main table.

Usage::

    python scripts/compute_alpha_sweep.py

Emits the LaTeX data rows used in ``paper/sections/appendix.tex``.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "final_analysis", os.path.join(os.path.dirname(__file__), "final_analysis.py")
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
JUDGES = [("deepseek-chat", "DeepSeek-V4"), ("gpt-5_5", "GPT-5.5")]
BENCHES = [
    ("judgebench", "JudgeBench"),
    ("tldr", "TL;DR"),
    ("rewardbench", "RewardBench"),
    ("lmaarena", "LMArena"),
]


def sweep(path: str, alphas=ALPHAS, seeds: int = 20) -> dict:
    rows = fa.read_jsonl(path)
    n = len(rows)
    X = fa.to_matrix(rows, fa.FEAT_FULL)
    y = np.array([int(r["correct"]) for r in rows])
    res = {}
    for a in alphas:
        n_test = n_correct = n_accepted = 0
        for seed in range(seeds):
            tr, ca, te = fa.split3(n, seed)
            if len(te) < 10:
                continue
            p_cal = fa.fit_predict(X[tr], y[tr], X[ca])
            p_test = fa.fit_predict(X[tr], y[tr], X[te])
            thr = fa.cal_thr(p_cal, y[ca], a)
            accept = p_test >= thr
            n_test += len(te)
            n_accepted += int(accept.sum())
            n_correct += int(y[te][accept].sum())
        cov = n_accepted / n_test if n_test else 0.0
        acc = n_correct / n_accepted if n_accepted else 0.0
        res[a] = (cov, acc, n_accepted)
    return res


def cell(cov: float, acc: float, n_accepted: int) -> str:
    if n_accepted == 0:
        return "--/--"
    cov_str = f"{cov:.2f}" if round(cov, 2) > 0 else "$<$.01"
    return f"{cov_str}/{acc:.2f}".replace("0.", ".")


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "outputs", "scale")
    for tag, disp in JUDGES:
        first = True
        for bkey, bdisp in BENCHES:
            path = os.path.join(base, f"{tag}_{bkey}_features.jsonl")
            if not os.path.exists(path):
                continue
            r = sweep(path)
            cells = " & ".join(cell(*r[a]) for a in ALPHAS)
            prefix = f"\\multirow{{4}}{{*}}{{{disp}}}" if first else ""
            first = False
            print(f"{prefix} & {bdisp} & {cells} \\\\")
        print("\\midrule")


if __name__ == "__main__":
    main()
