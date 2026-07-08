from __future__ import annotations

import hashlib
import random

from care_judge.judges.base import BaseJudge
from care_judge.schemas import Judgment, PairItem


class MockJudge(BaseJudge):
    """Deterministic, label-aware-ish judge for smoke tests.

    If an item has a label, it returns it with configurable accuracy. Otherwise it
    prefers the longer response with noise. This makes the full pipeline runnable
    without API keys while preserving nontrivial calibration behavior.
    """

    def __init__(self, name: str = "mock", accuracy: float = 0.72, cost_per_call: float = 0.0):
        self.name = name
        self.accuracy = accuracy
        self.cost_per_call = cost_per_call

    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        key = f"{self.name}|{item.id}|{rubric}|{temperature}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        if item.label in {"A", "B"}:
            correct = rng.random() < self.accuracy
            winner = item.label if correct else ("B" if item.label == "A" else "A")
            conf = self.accuracy if correct else 1 - self.accuracy
            conf = min(0.99, max(0.51, conf + rng.uniform(-0.12, 0.12)))
        else:
            winner = "A" if len(item.response_a) >= len(item.response_b) else "B"
            if rng.random() < 0.25:
                winner = "B" if winner == "A" else "A"
            conf = rng.uniform(0.55, 0.9)
        return Judgment(winner=winner, confidence=conf, reason="mock judgment", cost=self.cost_per_call)
