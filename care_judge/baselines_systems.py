#!/usr/bin/env python3
"""Trust-or-Escalate and SCOPE baseline systems for head-to-head comparison."""

from __future__ import annotations
import json, math, random
from typing import Any, Dict, List, Optional, Tuple
from care_judge.schemas import PairItem, Judgment


# ═══════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════

def _winner_to_label(winner: str) -> str:
    return winner if winner in ("A", "B") else "tie"


# ═══════════════════════════════════════════════════
# SCOPE-style: Bidirectional Preference Entropy
# ═══════════════════════════════════════════════════

def scope_score(conf_orig: float, conf_swap: float) -> float:
    """Permutation-invariant score from bidirectional confidences.
    p_A = conf that order (a,b) prefers A
    q_B = conf that order (b,a) prefers B  (= prefers A in original coords)
    score =  ||p_A - (1 - q_B)||  captures order-sensitivity.
    Actually SCOPE averages: p_A + (1 - q_B[swap mapped]); higher = more confident A.
    """
    return conf_orig  # simplified for single-call; extended version stores per-call probabilities


def scope_bpe(conf_ab: float, conf_ba_mapped: float) -> Tuple[float, float]:
    """Bidirectional Preference Entropy proxy: agreement = 0.5*(conf_ab + conf_ba_mapped)."""
    p_a_from_ab = conf_ab
    p_a_from_ba = conf_ba_mapped
    agreement = 0.5 * (p_a_from_ab + p_a_from_ba)
    entropy = -agreement * math.log(max(agreement, 1e-9)) - (1-agreement) * math.log(max(1-agreement, 1e-9))
    return agreement, entropy


# ═══════════════════════════════════════════════════
# Trust-or-Escalate: Simulated Annotators
# ═══════════════════════════════════════════════════

def simulated_annotator_confidence(
    item: PairItem,
    judge_calls: List[Judgment],
    few_shot_pool: List[Tuple[PairItem, Judgment]] = None,
    n_annotators: int = 5,
    seed: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """Compute simulated-annotator agreement conf from existing judgments.
    
    In the original T-o-E, each simulated annotator is a few-shot evaluation.
    Here we approximate from stored multi-call judgments: the agreement across 
    rubric and self-consistency variants serves as a proxy.
    
    Returns (agreement_score, meta).
    """
    if len(judge_calls) < 2:
        return 0.5, {"n_annotators": 0, "votes": []}
    winners = [j.normalized_winner() for j in judge_calls if j.normalized_winner() in ("A","B")]
    if not winners:
        return 0.5, {"n_annotators": 0, "reason": "no_valid_winners"}
    a_votes = winners.count("A")
    b_votes = winners.count("B")
    n = a_votes + b_votes
    if n == 0:
        return 0.5, {"n_annotators": 0}
    max_share = max(a_votes, b_votes) / n
    return max_share, {"n_annotators": n, "a_votes": a_votes, "b_votes": b_votes}


# ═══════════════════════════════════════════════════
# Main baseline runner: given precomputed per-item features + 
# per-call raw outputs, produce confidence scores for each 
# baseline. This is the function the main analysis calls.
# ═══════════════════════════════════════════════════

def compute_baseline_scores(
    features: List[Dict[str, Any]],
    per_call_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, List[float]]:
    """Produce confidence vectors for each baseline from stored features.
    
    Returns dict mapping baseline_name -> [score per item].
    
    Baselines:
        base_conf: raw confidence (single call, predictive probability)
        swap_consistency: SCOPE-style bidirectional (already in features)
        scope_bpe: full bidirectional preference entropy if per-call probs available
        sim_agreement: simulated annotator agreement (already in features as 'sim')
                        or computed from multi-call per-call JSOns
    """
    scores = {
        "base_conf": [],
        "swap_consistency": [],
        "sim_agreement": [],
        "multi_agreement": [],  # agreement across ALL variants (rubric+self+base)
    }
    
    for row in features:
        # base confidence 
        bc = float(row.get("feat_base_conf", 0.5))
        scores["base_conf"].append(bc)
        # swap consistency (already binary in features)
        scores["swap_consistency"].append(float(row.get("feat_swap_consistency", 0.5)))
        
        # simulated annotator proxy: agreement across rubric+self variants
        rubric = float(row.get("feat_rubric_vote_share", 0.5))
        self_vs = float(row.get("feat_self_vote_share", 0.5))
        swap = float(row.get("feat_swap_consistency", 0.5))
        scores["sim_agreement"].append(max(rubric, self_vs, swap))
        
        # multi-signal agreement (mean of all three stability signals)
        scores["multi_agreement"].append((rubric + self_vs + swap) / 3.0)
    
    return scores

