from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PairItem:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: Optional[str] = None  # "A", "B", or None for unlabeled eval
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Judgment:
    winner: str
    confidence: float
    reason: str = ""
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    raw: Any = None
    cost: float = 0.0

    def normalized_winner(self) -> str:
        if self.winner not in {"A", "B", "tie", "abstain"}:
            return "abstain"
        return self.winner


@dataclass
class FeatureRecord:
    id: str
    pred: str
    label: Optional[str]
    correct: Optional[int]
    confidence: float
    features: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)
