#!/usr/bin/env python3
"""Rebuttal-only: SConU-style conformal selective baseline for LLM-as-a-judge.

This is NOT part of the paper's reported tables. It is kept so that, if a
reviewer asks for a comparison against a conformal-prediction judge baseline, we
can produce it on demand from the authoritative per-item feature traces without
touching outputs/final_analysis.json.

Method. For each (judge, benchmark) we form a per-item CONFORMITY score equal to
the mean of the judge's four agreement signals -- base confidence,
self-consistency vote share, swap consistency, and rubric vote share -- and take
a distribution-free split-conformal acceptance threshold at the (alpha)-quantile
of the conformity score over the CORRECT calibration items (SConU-style). This
uses exactly the same 40/30/30 split, alpha, and 20 seeds as CARE, and the same
features, so the AUROC of the conformity score is directly comparable to CARE,
SCOPE, and ToE in Tables tab:threeway / tab:ensemble.

Result summary (20 seeds, alpha=0.15): the conformal baseline is competitive --
it matches or edges CARE on JudgeBench and on the saturated GPT-5.5 judge (where
every signal is near-redundant), but it collapses in the below-threshold reversal
regime (Qwen2.5-1.5B / RewardBench: conformal AUROC ~0.45 vs CARE ~0.71), which
is the competence-gating behaviour the paper analyses. CARE remains best in the
mid-accuracy x subjective/RewardBench cells where the stability signal is
informative.

Usage::

    python scripts/conformal_baseline.py            # prints per-pair AUROC/cov/acc
    python scripts/conformal_baseline.py --out outputs/conformal_baseline.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "final_analysis", os.path.join(_HERE, "final_analysis.py")
)
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def evaluate(path: str, seeds: int = 20, alpha: float = 0.15) -> dict:
    rows = fa.read_jsonl(path)
    if len(rows) < 50:
        return {}
    n = len(rows)
    y = np.array([int(r["correct"]) for r in rows])
    s = fa.conformal_scores(rows)  # per-item conformity in [0,1]
    aurocs, covs, accs = [], [], []
    for seed in range(seeds):
        tr, ca, te = fa.split3(n, seed)
        if len(te) < 10:
            continue
        aurocs.append(fa.auroc(y[te], s[te]))
        cal_correct = s[ca][y[ca] == 1]
        if len(cal_correct) < 20:
            covs.append(0.0)
            accs.append(0.0)
            continue
        # accept if conformity >= alpha-quantile of conformity on correct cal items
        thr = float(np.quantile(cal_correct, alpha))
        accept = s[te] >= thr
        covs.append(float(np.mean(accept)))
        accs.append(float(np.mean(y[te][accept])) if accept.sum() > 0 else 0.0)
    return {
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_std": float(np.std(aurocs)) if aurocs else 0.0,
        "cov": float(np.mean(covs)) if covs else 0.0,
        "acc": float(np.mean(accs)) if accs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=os.path.join(_HERE, "..", "outputs", "scale"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    out: dict = {}
    print(f"{'judge':13}{'bench':11}{'AUROC':>8}{'cov':>7}{'acc':>7}")
    for tag, disp, _kind in fa.JUDGES:
        for b in fa.BENCHMARKS:
            path = os.path.join(args.features_dir, f"{tag}_{b}_features.jsonl")
            if not os.path.exists(path):
                continue
            r = evaluate(path, seeds=args.seeds)
            if not r:
                continue
            out.setdefault(tag, {})[b] = r
            print(f"{disp:13}{b:11}{r['auroc_mean']:>8.3f}{r['cov']:>7.3f}{r['acc']:>7.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
