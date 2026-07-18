#!/usr/bin/env python3
"""collect_6call.py — Faithful reconstruction of the CARE-Judge 6-call protocol
described in paper/sections/method.tex (§3.1, Eq. 16 caption, Figure 1 caption).

Protocol (exactly 6 judge calls per item):
  1. base    : R[0], τ=0   -> counts as self_votes[0] AND rubric_votes[0]
  2. self #1 : R[0], τ=0.7
  3. self #2 : R[0], τ=0.7   (S=3 self votes total)
  4. swap    : R[0], τ=0   on swap_item(item)
  5. rubric#1: R[1], τ=0
  6. rubric#2: R[2], τ=0   (K=3 rubric votes total: base + 2 paraphrases)

K=3 rubrics = [BASE_RUBRIC, RUBRIC_VARIANTS[1], RUBRIC_VARIANTS[2]] — the two
paraphrases the paper itself quotes as examples in method.tex §3.1.

Output JSONL schema is byte-identical to scripts/collect_uncertainty.py so that
scripts/rigorous_analysis.py consumes it without modification. Simulated
annotators are disabled (sim_annotators=0), matching the paper's main tables.

This script exists because the stock collect_uncertainty.py runs 9 calls
(K=5, base not reused) and the paper/__paperguru_tmp/ 6-call scripts import a
non-existent care_judge.uncertainty.collector_extended module. This is the
reproducible reconstruction.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, random
from typing import Any, Dict, List

from care_judge.data.loaders import load_jsonl_pairs
from care_judge.judges.local_hf import LocalHFJudge
from care_judge.judges.base import swap_item, unswap_winner
from care_judge.judges.prompts import BASE_RUBRIC, RUBRIC_VARIANTS
from care_judge.utils import entropy_binary, majority, vote_share, write_jsonl

RUBRICS_3 = [BASE_RUBRIC, RUBRIC_VARIANTS[1], RUBRIC_VARIANTS[2]]
assert len(RUBRICS_3) == 3, "K must be 3"


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def collect_6call(item, judge, temperature: float = 0.7) -> Dict[str, Any]:
    """Run the 6-call protocol and return a feature row matching the stock schema."""
    R = RUBRICS_3
    raw: Dict[str, Any] = {"base": None, "self": [], "rubric": [], "swap": None, "simulated_annotators": []}

    # Call 1: base (τ=0, R[0]) — reused as self_votes[0] AND rubric_votes[0]
    base = judge.judge(item, rubric=R[0], temperature=0.0)
    base_w = base.normalized_winner()
    base_c = base.confidence
    raw["base"] = base.__dict__

    self_votes: List[str] = [base_w]
    self_confs: List[float] = [base_c]

    # Calls 2-3: self-consistency (τ=0.7, R[0])
    for _ in range(2):
        j = judge.judge(item, rubric=R[0], temperature=temperature)
        self_votes.append(j.normalized_winner())
        self_confs.append(j.confidence)
        raw["self"].append(j.__dict__)

    self_pred = majority(self_votes)
    if self_pred == "tie":
        self_pred = base_w
    self_vote_share = vote_share(self_votes)

    # Call 4: position-swap (τ=0, R[0]) on swapped item
    swapped = judge.judge(swap_item(item), rubric=R[0], temperature=0.0)
    mapped = unswap_winner(swapped.normalized_winner())
    swap_consistency = 1.0 if mapped == base_w else 0.0
    swap_conf_gap = abs(base_c - swapped.confidence)
    raw["swap"] = {**swapped.__dict__, "mapped_winner": mapped}

    # Rubric votes: base (already have) + calls 5-6 (R[1], R[2] at τ=0)
    rubric_votes: List[str] = [base_w]
    rubric_confs: List[float] = [base_c]
    for k in (1, 2):
        j = judge.judge(item, rubric=R[k], temperature=0.0)
        rubric_votes.append(j.normalized_winner())
        rubric_confs.append(j.confidence)
        raw["rubric"].append({**j.__dict__, "rubric": R[k]})

    rubric_share = vote_share(rubric_votes)
    p_a_rubric = max(1e-9, min(1 - 1e-9, rubric_votes.count("A") / max(1, len(rubric_votes))))
    rubric_entropy = entropy_binary(p_a_rubric)
    rubric_flip = 1.0 if len(set(rubric_votes)) > 1 else 0.0

    # Simulated annotators disabled (N=0) — degenerate, matches paper main tables.
    sim_votes = [base_w]
    sim_confs = [base_c]
    sim_share = vote_share(sim_votes)
    p_a_sim = max(1e-9, min(1 - 1e-9, sim_votes.count("A") / max(1, len(sim_votes))))
    sim_entropy = entropy_binary(p_a_sim)
    sim_flip = 0.0

    pred = majority(self_votes + rubric_votes + sim_votes)
    if pred == "tie":
        pred = base_w

    score_margin = 0.0
    if getattr(base, "score_a", None) is not None and getattr(base, "score_b", None) is not None:
        score_margin = abs(base.score_a - base.score_b)

    all_confs = self_confs + rubric_confs + sim_confs
    features = {
        "base_conf": float(base_c),
        "mean_conf": float(sum(all_confs) / max(1, len(all_confs))),
        "std_conf": float(_std(all_confs)),
        "self_vote_share": float(self_vote_share),
        "self_entropy": float(entropy_binary(self_vote_share)),
        "adaptive_calls": float(len(self_votes)),  # = 3
        "swap_consistency": float(swap_consistency),
        "swap_conf_gap": float(swap_conf_gap),
        "rubric_vote_share": float(rubric_share),
        "rubric_entropy": float(rubric_entropy),
        "rubric_flip": float(rubric_flip),
        "sim_vote_share": float(sim_share),
        "sim_entropy": float(sim_entropy),
        "sim_flip": float(sim_flip),
        "score_margin": float(score_margin),
        "length_gap_norm": float(abs(len(item.response_a) - len(item.response_b)) / max(1, len(item.response_a) + len(item.response_b))),
        "cost": float((base.cost or 0.0) + (swapped.cost or 0.0)
                      + sum(float(x.get("cost", 0.0) or 0.0) for x in raw["self"])
                      + sum(float(x.get("cost", 0.0) or 0.0) for x in raw["rubric"])),
    }
    correct = None if item.label is None else int(pred == item.label)
    confidence = float(sum([features["base_conf"], features["self_vote_share"], features["rubric_vote_share"], features["swap_consistency"], features["sim_vote_share"]]) / 5)

    row = {
        "id": item.id,
        "pred": pred,
        "label": item.label,
        "correct": correct,
        "confidence": confidence,
        "raw": raw,
        "domain": item.domain,
    }
    row.update({f"feat_{k}": v for k, v in features.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    random.seed(args.seed)
    items = load_jsonl_pairs(args.input, limit=args.limit)
    n = len(items)
    print(f"[collect_6call] {n} items from {args.input}", flush=True)

    judge = LocalHFJudge(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    rows = []
    n_invalid = 0
    for i, item in enumerate(items):
        try:
            row = collect_6call(item, judge, temperature=args.temperature)
        except Exception as e:
            print(f"  [warn] item {item.id} failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            row = {"id": item.id, "pred": "abstain", "label": item.label,
                   "correct": (None if item.label is None else 0), "confidence": 0.5,
                   "raw": {"error": str(e)}, "domain": item.domain}
            for k in ["base_conf","mean_conf","std_conf","self_vote_share","self_entropy","adaptive_calls","swap_consistency","swap_conf_gap","rubric_vote_share","rubric_entropy","rubric_flip","sim_vote_share","sim_entropy","sim_flip","score_margin","length_gap_norm","cost"]:
                row[f"feat_{k}"] = 0.5 if k in ("base_conf","mean_conf","self_vote_share","rubric_vote_share","sim_vote_share","swap_consistency") else 0.0
        rows.append(row)
        if row.get("pred") == "abstain":
            n_invalid += 1
        if (i + 1) % args.log_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{n}] {rate:.1f} items/s  eta {eta/60:.1f}m  invalid={n_invalid}  ({row.get('pred')})", flush=True)

    write_jsonl(args.out, rows)
    elapsed = time.time() - t0
    print(f"[collect_6call] DONE {n} items in {elapsed/60:.1f}m  ({n/max(elapsed,1e-6):.1f} items/s)  invalid={n_invalid}", flush=True)
    print(f"[collect_6call] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
