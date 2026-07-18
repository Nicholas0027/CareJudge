from __future__ import annotations

from care_judge.schemas import PairItem


BASE_RUBRIC = "Prefer the response that is more correct, helpful, faithful to the prompt, and safe. Ignore response order and verbosity unless they affect quality."


RUBRIC_VARIANTS = [
    BASE_RUBRIC,
    "Choose the answer a careful human evaluator would prefer after checking factual correctness, instruction following, and usefulness.",
    "Select the response with fewer factual or reasoning errors. If both are plausible, prefer the one that better addresses the user request.",
    "Evaluate quality by correctness first, then completeness, clarity, and safety. Do not reward unnecessary length.",
    "Pick the answer that is most reliable and best supported by the prompt; penalize hallucinations, omissions, and unsafe advice.",
]


def pairwise_prompt(item: PairItem, rubric: str = BASE_RUBRIC, ask_confidence: bool = True) -> str:
    conf = '"confidence": a number from 0 to 1,' if ask_confidence else '"confidence": 0.5,'
    return f"""You are an impartial evaluator. Judge which response is better.

Rubric:
{rubric}

User prompt:
{item.prompt}

Response A:
{item.response_a}

Response B:
{item.response_b}

Return only valid JSON with keys:
{{"winner": "A" or "B", {conf} "reason": "brief explanation"}}
"""
