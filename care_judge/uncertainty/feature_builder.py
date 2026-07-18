from __future__ import annotations

from typing import Any, Dict, List
import math
import random

from care_judge.judges.base import BaseJudge, swap_item, unswap_winner
from care_judge.judges.prompts import BASE_RUBRIC, RUBRIC_VARIANTS
from care_judge.schemas import FeatureRecord, PairItem
from care_judge.utils import entropy_binary, majority, vote_share


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def collect_uncertainty_features(
    item: PairItem,
    judge: BaseJudge,
    rubrics: List[str] | None = None,
    k_self: int = 3,
    temperature: float = 0.7,
    use_swap: bool = True,
    use_rubrics: bool = True,
    sim_examples: List[PairItem] | None = None,
    sim_annotators: int = 0,
    sim_shots: int = 3,
    adaptive_k: bool = False,
    adaptive_tau: float = 0.85,
    min_self: int = 2,
) -> FeatureRecord:
    rubrics = rubrics or RUBRIC_VARIANTS
    base = judge.judge(item, rubric=BASE_RUBRIC, temperature=0.0)
    votes: List[str] = [base.normalized_winner()]
    confs: List[float] = [base.confidence]
    raw: Dict[str, Any] = {"base": base.__dict__, "self": [], "rubric": [], "swap": None, "simulated_annotators": []}

    # Self-consistency: repeated stochastic judgments under the base rubric.
    calls_used = 1
    for _ in range(max(0, k_self - 1)):
        j = judge.judge(item, rubric=BASE_RUBRIC, temperature=temperature)
        votes.append(j.normalized_winner())
        confs.append(j.confidence)
        raw["self"].append(j.__dict__)
        calls_used += 1
        if adaptive_k and calls_used >= min_self and vote_share(votes) >= adaptive_tau:
            break

    self_pred = majority(votes)
    if self_pred == "tie":
        self_pred = base.normalized_winner()
    self_vote_share = vote_share(votes)

    # Position-swap stability: swapped prediction should map back to same winner.
    swap_consistency = 1.0
    swap_conf_gap = 0.0
    if use_swap:
        swapped = judge.judge(swap_item(item), rubric=BASE_RUBRIC, temperature=0.0)
        mapped = unswap_winner(swapped.normalized_winner())
        swap_consistency = 1.0 if mapped == base.normalized_winner() else 0.0
        swap_conf_gap = abs(base.confidence - swapped.confidence)
        raw["swap"] = swapped.__dict__ | {"mapped_winner": mapped}

    # Rubric perturbation stability: semantically equivalent rubrics should agree.
    rubric_votes: List[str] = []
    rubric_confs: List[float] = []
    if use_rubrics:
        for rubric in rubrics:
            j = judge.judge(item, rubric=rubric, temperature=0.0)
            rubric_votes.append(j.normalized_winner())
            rubric_confs.append(j.confidence)
            raw["rubric"].append(j.__dict__ | {"rubric": rubric})
    else:
        rubric_votes = [base.normalized_winner()]
        rubric_confs = [base.confidence]

    rubric_share = vote_share(rubric_votes)
    p_a_rubric = max(1e-9, min(1 - 1e-9, rubric_votes.count("A") / max(1, len(rubric_votes))))
    rubric_entropy = entropy_binary(p_a_rubric)

    # Simulated annotators: create N few-shot evaluator personas from labeled
    # calibration examples. This approximates Trust-or-Escalate's simulated
    # annotator confidence without requiring per-annotator labels.
    sim_votes: List[str] = []
    sim_confs: List[float] = []
    if sim_examples and sim_annotators > 0 and sim_shots > 0:
        pool = [x for x in sim_examples if x.label in {"A", "B"} and x.id != item.id]
        for annotator_idx in range(sim_annotators):
            rng = random.Random(f"{item.id}:{annotator_idx}:{sim_shots}")
            shots = rng.sample(pool, k=min(sim_shots, len(pool))) if pool else []
            rubric = _simulated_annotator_rubric(BASE_RUBRIC, shots, annotator_idx)
            j = judge.judge(item, rubric=rubric, temperature=temperature)
            sim_votes.append(j.normalized_winner())
            sim_confs.append(j.confidence)
            raw["simulated_annotators"].append(j.__dict__ | {"annotator": annotator_idx, "shots": [s.id for s in shots]})
    if not sim_votes:
        sim_votes = [base.normalized_winner()]
        sim_confs = [base.confidence]

    sim_share = vote_share(sim_votes)
    p_a_sim = max(1e-9, min(1 - 1e-9, sim_votes.count("A") / max(1, len(sim_votes))))
    sim_entropy = entropy_binary(p_a_sim)

    pred = majority(votes + rubric_votes + sim_votes)
    if pred == "tie":
        pred = base.normalized_winner()

    score_margin = 0.0
    if base.score_a is not None and base.score_b is not None:
        score_margin = abs(base.score_a - base.score_b)

    features = {
        "base_conf": float(base.confidence),
        "mean_conf": float(sum(confs + rubric_confs + sim_confs) / max(1, len(confs + rubric_confs + sim_confs))),
        "std_conf": float(_std(confs + rubric_confs + sim_confs)),
        "self_vote_share": float(self_vote_share),
        "self_entropy": float(entropy_binary(self_vote_share)),
        "adaptive_calls": float(calls_used),
        "swap_consistency": float(swap_consistency),
        "swap_conf_gap": float(swap_conf_gap),
        "rubric_vote_share": float(rubric_share),
        "rubric_entropy": float(rubric_entropy),
        "rubric_flip": float(1.0 if len(set(rubric_votes)) > 1 else 0.0),
        "sim_vote_share": float(sim_share),
        "sim_entropy": float(sim_entropy),
        "sim_flip": float(1.0 if len(set(sim_votes)) > 1 else 0.0),
        "score_margin": float(score_margin),
        "length_gap_norm": float(abs(len(item.response_a) - len(item.response_b)) / max(1, len(item.response_a) + len(item.response_b))),
        "cost": float(sum(_as_float(x.get("cost")) for group in [raw["self"], raw["rubric"], raw["simulated_annotators"]] for x in group) + base.cost + (_as_float(raw["swap"].get("cost")) if raw["swap"] else 0.0)),
    }
    correct = None if item.label is None else int(pred == item.label)
    confidence = float(sum([features["base_conf"], features["self_vote_share"], features["rubric_vote_share"], features["swap_consistency"], features["sim_vote_share"]]) / 5)
    return FeatureRecord(id=item.id, pred=pred, label=item.label, correct=correct, confidence=confidence, features=features, raw=raw)


def _simulated_annotator_rubric(base_rubric: str, shots: List[PairItem], annotator_idx: int) -> str:
    examples = []
    for s in shots:
        examples.append(
            f"Example prompt: {s.prompt}\nResponse A: {s.response_a}\nResponse B: {s.response_b}\nHuman preference: {s.label}"
        )
    joined = "\n\n".join(examples) if examples else "No examples available."
    return f"""{base_rubric}

You are simulated annotator #{annotator_idx}. Match the pattern of the human preference examples below, while still judging the new item independently.

Human preference examples:
{joined}
"""


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def record_to_row(record: FeatureRecord) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": record.id,
        "pred": record.pred,
        "label": record.label,
        "correct": record.correct,
        "confidence": record.confidence,
        "raw": record.raw,
    }
    row.update({f"feat_{k}": v for k, v in record.features.items()})
    return row


def feature_columns(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted([k for k in rows[0] if k.startswith("feat_")] + (["confidence"] if "confidence" in rows[0] else []))
