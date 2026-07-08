from __future__ import annotations

from abc import ABC, abstractmethod

from care_judge.schemas import Judgment, PairItem


class BaseJudge(ABC):
    name: str
    cost_per_call: float = 0.0

    @abstractmethod
    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        raise NotImplementedError


def swap_item(item: PairItem) -> PairItem:
    return PairItem(
        id=item.id,
        prompt=item.prompt,
        response_a=item.response_b,
        response_b=item.response_a,
        label=("A" if item.label == "B" else "B" if item.label == "A" else None),
        domain=item.domain,
        metadata=item.metadata,
    )


def unswap_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return winner
