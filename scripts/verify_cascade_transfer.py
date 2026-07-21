#!/usr/bin/env python3
"""Reproduce the exploratory cascade (Table `tab:cascade`) and cross-dataset
transfer (Table `tab:transfer`) numbers from the AUTHORITATIVE per-item feature
traces in ``outputs/scale/``.

Both tables are produced by the strict-split routines in
``rigorous_cascade_transfer.py``; this driver simply points those routines at
the authoritative ``outputs/scale/{judge}_{bench}_features.jsonl`` files (5
judges x 4 benchmarks) rather than the older ``outputs/dual_api_full`` dumps,
and averages over 10 seeds.

Cascade: Qwen2.5-1.5B -> DeepSeek-V4 -> GPT-5.5 with paid-API token-cost weights
(local Qwen = 0, DeepSeek = 0.14, GPT-5.5 = 1.0 relative), reported at
alpha in {0.15, 0.30}. Transfer: GPT-5.5 only, calibrator fit on the source
train+cal split, evaluated on the target test split, alpha = 0.15.

Usage::

    python scripts/verify_cascade_transfer.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "rct", os.path.join(_HERE, "rigorous_cascade_transfer.py")
)
rct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rct)

BENCH = {
    "JudgeBench": "judgebench",
    "TL;DR": "tldr",
    "RewardBench": "rewardbench",
    "LMArena": "lmaarena",
}
SCALE = os.path.join(_HERE, "..", "outputs", "scale")
# Relative paid-API token cost per tier (local GPU tiers are free).
COSTS = {"qwen": 0.0, "deepseek": 0.14, "gpt": 1.0}
SEEDS = 10


def feat(judge: str, bench: str) -> str:
    return os.path.join(SCALE, f"{judge}_{BENCH[bench]}_features.jsonl")


def run_cascade() -> None:
    print("=== CASCADE (Qwen1.5B -> DeepSeek -> GPT5.5, 10 seeds) ===")
    for b in ["JudgeBench", "TL;DR", "RewardBench", "LMArena"]:
        tiers = [
            ("qwen", feat("qwen-1.5b", b)),
            ("deepseek", feat("deepseek-chat", b)),
            ("gpt", feat("gpt-5_5", b)),
        ]
        agg: dict[str, list] = {}
        for seed in range(SEEDS):
            r = rct.cascade(tiers, COSTS, seed=seed)
            for a in ("alpha0.15", "alpha0.3"):
                if a in r:
                    agg.setdefault(a, []).append(r[a])
        for a in ("alpha0.15", "alpha0.3"):
            if a not in agg:
                continue
            cov = np.mean([x["coverage"] for x in agg[a]])
            acc = np.mean([(x["accuracy_accepted"] or 0.0) for x in agg[a]])
            sv = np.mean([x["cost_savings_pct"] for x in agg[a]])
            print(f"{b:11} {a:9} cov={cov:.3f} acc={acc:.3f} save={sv:.0f}%")


def run_transfer() -> None:
    print("=== TRANSFER (GPT-5.5, alpha=0.15, 10 seeds) ===")
    order = ["TL;DR", "JudgeBench", "RewardBench", "LMArena"]
    pairs = [
        ("TL;DR", "JudgeBench"), ("TL;DR", "RewardBench"), ("TL;DR", "LMArena"),
        ("JudgeBench", "TL;DR"), ("JudgeBench", "RewardBench"), ("JudgeBench", "LMArena"),
        ("RewardBench", "JudgeBench"), ("RewardBench", "TL;DR"), ("RewardBench", "LMArena"),
        ("LMArena", "JudgeBench"), ("LMArena", "TL;DR"), ("LMArena", "RewardBench"),
    ]
    _ = order
    for s, t in pairs:
        covs, accs = [], []
        for seed in range(SEEDS):
            r = rct.transfer(feat("gpt-5_5", s), feat("gpt-5_5", t), seed=seed)["alpha0.15"]
            covs.append(r["coverage"])
            accs.append(r["accuracy_accepted"] or 0.0)
        print(f"{s:11}-> {t:11} cov={np.mean(covs):.3f} acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    run_transfer()
    run_cascade()
