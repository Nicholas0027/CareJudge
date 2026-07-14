#!/usr/bin/env python3
"""Extended feature collector that persists per-call confidence values.

This enables:
  - Full SCOPE bidirectional preference entropy (needs conf_orig and conf_swap)
  - Trust-or-Escalate simulated annotator agreement from multi-call outputs
  - Per-item cost accounting

Run this instead of the original collector when collecting features at scale.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from care_judge.schemas import PairItem


def collect_with_call_trace(
    item: PairItem,
    judge,
    rubrics: List[str],
    k_self: int = 3,
    temperature: float = 0.7,
    use_swap: bool = True,
    use_rubrics: bool = True,
) -> Dict[str, Any]:
    """Collect features AND persist per-call confidences for baseline reproduction."""
    from care_judge.uncertainty.feature_builder import collect_uncertainty_features, record_to_row
    
    record = collect_uncertainty_features(
        item, judge,
        rubrics=rubrics,
        k_self=k_self,
        temperature=temperature,
        use_swap=use_swap,
        use_rubrics=use_rubrics,
        sim_annotators=0,  # main runs: N=0, no leakage
        sim_shots=0,
    )
    row = record_to_row(record)
    
    # Persist per-call confidences from the raw attribute
    raw = record.raw
    per_call = {
        "id": item.id,
        "base_winner": raw.get("base", {}).get("winner"),
        "base_conf": raw.get("base", {}).get("confidence"),
        "base_score_a": raw.get("base", {}).get("score_a"),
        "base_score_b": raw.get("base", {}).get("score_b"),
    }
    
    # Self-consistency samples
    self_calls = raw.get("self", [])
    per_call["self_calls"] = [
        {"winner": c.get("winner"), "confidence": c.get("confidence"),
         "score_a": c.get("score_a"), "score_b": c.get("score_b")}
        for c in self_calls
    ]
    
    # Swap judgment
    swap = raw.get("swap", {})
    if swap:
        per_call["swap"] = {
            "winner": swap.get("winner"), "confidence": swap.get("confidence"),
            "mapped_winner": swap.get("mapped_winner"),
            "score_a": swap.get("score_a"), "score_b": swap.get("score_b"),
        }
    
    # Rubric variants
    rubric_calls = raw.get("rubric", [])
    per_call["rubric_calls"] = [
        {"rubric_idx": i, "winner": c.get("winner"), "confidence": c.get("confidence"),
         "score_a": c.get("score_a"), "score_b": c.get("score_b"),
         "rubric": c.get("rubric")}
        for i, c in enumerate(rubric_calls)
    ]
    
    row["per_call"] = per_call
    return row


def collect_features_at_scale(
    items: List[PairItem],
    judge,
    rubrics: List[str],
    output_path: str,
    k_self: int = 3,
    temperature: float = 0.7,
    max_items: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Collect features for a list of items, with per-call trace, saving as we go."""
    import json
    results = []
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w") as f:
        for idx, item in enumerate(items[:max_items] if max_items else items):
            try:
                row = collect_with_call_trace(item, judge, rubrics,
                                              k_self=k_self, temperature=temperature)
                f.write(json.dumps(row) + "\n")
                f.flush()
                results.append(row)
                if verbose and (idx + 1) % 50 == 0:
                    print(f"  [{idx+1}/{len(items)}] {item.id} base_conf={row.get('feat_base_conf','NA')}")
            except Exception as e:
                print(f"  [ERROR] item {idx} {item.id}: {e}")
                continue
    
    if verbose:
        print(f"  Done: {len(results)} items -> {out}")
    return results

