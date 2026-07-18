from __future__ import annotations

from typing import Any, Dict, List, Tuple

from care_judge.judges.base import BaseJudge
from care_judge.schemas import PairItem
from care_judge.uncertainty.feature_builder import collect_uncertainty_features, record_to_row


def run_cascade_item(
    item: PairItem,
    judges: List[Tuple[str, BaseJudge]],
    calibrators: Dict[str, Any],
    thresholds: Dict[str, float],
    k_self: int = 3,
    sim_annotators: int = 0,
    sim_shots: int = 3,
    sim_examples: List[PairItem] | None = None,
) -> Dict[str, Any]:
    trace = []
    for name, judge in judges:
        rec = collect_uncertainty_features(item, judge, k_self=k_self, sim_examples=sim_examples, sim_annotators=sim_annotators, sim_shots=sim_shots)
        row = record_to_row(rec)
        p = float(calibrators[name].predict_proba([row])[0]) if name in calibrators else row["confidence"]
        trace.append({"judge": name, "pred": row["pred"], "p_correct": p, "row": row})
        if p >= thresholds.get(name, 1.01):
            return {"id": item.id, "pred": row["pred"], "label": item.label, "correct": None if item.label is None else int(row["pred"] == item.label), "accepted_by": name, "p_correct": p, "abstained": False, "trace": trace}
    return {"id": item.id, "pred": None, "label": item.label, "correct": None, "accepted_by": "abstain", "p_correct": 0.0, "abstained": True, "trace": trace}
