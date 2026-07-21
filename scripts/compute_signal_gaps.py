#!/usr/bin/env python3
"""Recompute stable-vs-unstable subset accuracy for every judge/benchmark/signal
directly from the per-item feature traces (outputs/scale/*_features.jsonl).

This is the authoritative source for the signal-gap table (tab:signals),
Figure 2 (signal_gap.pdf), and the stable-but-wrong analysis (Sec. 5.4).
Every number the paper reports at the signal/subset level can be reproduced by
running this script; nothing is hand-entered.

Signal definitions (matching Sec. 3):
  rubric-stable   : feat_rubric_flip == 0     (winner unchanged across K=3 rubrics)
  position-stable : feat_swap_consistency == 1 (winner unchanged after order swap)
  self-stable     : feat_self_vote_share == 1  (unanimous across S=3 samples)
"""
import json, os, sys, statistics as st

D = "outputs/scale"
JUDGES = [("qwen-1.5b", "Qwen2.5-1.5B"), ("deepseek-chat", "DeepSeek-V4"),
          ("qwen-7b", "Qwen2.5-7B"), ("gpt-5_5", "GPT-5.5"), ("qwen-14b", "Qwen2.5-14B")]
BENCHES = [("judgebench", "JudgeBench"), ("tldr", "TL;DR"),
           ("rewardbench", "RewardBench"), ("lmaarena", "LMArena")]


def load(j, b):
    p = f"{D}/{j}_{b}_features.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def split(rows, key, stable_val):
    st_ = [r for r in rows if r[key] == stable_val]
    un_ = [r for r in rows if r[key] != stable_val]
    return st_, un_


def main():
    print(f"{'judge':13s}{'bench':12s}{'signal':9s} stable  unstable   gap    n_st/n_un")
    for jk, jn in JUDGES:
        for bk, bn in BENCHES:
            rows = load(jk, bk)
            if not rows:
                continue
            for sig, key, sv in [("rubric", "feat_rubric_flip", 0),
                                  ("position", "feat_swap_consistency", 1),
                                  ("self", "feat_self_vote_share", 1.0)]:
                s_, u_ = split(rows, key, sv)
                if not s_ or not u_:
                    continue
                print(f"{jn:13s}{bn:12s}{sig:9s} {acc(s_):.3f}  {acc(u_):.3f}   "
                      f"{acc(s_)-acc(u_):+.3f}  {len(s_)}/{len(u_)}")

    # Stable-but-wrong drill-down for the below-threshold judge (Sec. 5.4)
    rows = load("qwen-1.5b", "judgebench")
    if rows:
        wrong = [r for r in rows if r["correct"] == 0]
        sw = [r for r in wrong if r["feat_rubric_flip"] == 0]
        stable = [r for r in rows if r["feat_rubric_flip"] == 0]
        print("\n[Sec 5.4] Qwen2.5-1.5B / JudgeBench stable-but-wrong:")
        print(f"  stable-wrong / all-wrong = {len(sw)}/{len(wrong)} = {len(sw)/len(wrong)*100:.1f}%")
        print(f"  rubric-stable subset accuracy = {acc(stable)*100:.1f}% (n={len(stable)})")
        conf_sw = st.mean(r["confidence"] for r in sw)
        conf_cor = st.mean(r["confidence"] for r in rows if r["correct"] == 1)
        print(f"  mean confidence: stable-wrong={conf_sw:.3f} vs correct={conf_cor:.3f}")
        so = sum(1 for r in rows if r["feat_swap_consistency"] == 0) / len(rows)
        sw_so = sum(1 for r in sw if r["feat_swap_consistency"] == 0) / len(sw)
        print(f"  swap-inconsistent: overall={so*100:.1f}%  among stable-wrong={sw_so*100:.1f}%")


if __name__ == "__main__":
    main()
