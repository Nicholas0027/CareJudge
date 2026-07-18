from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)


def entropy_binary(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def majority(votes: List[str]) -> str:
    a = sum(v == "A" for v in votes)
    b = sum(v == "B" for v in votes)
    if a == b:
        return "tie"
    return "A" if a > b else "B"


def vote_share(votes: List[str]) -> float:
    valid = [v for v in votes if v in {"A", "B"}]
    if not valid:
        return 0.5
    return max(valid.count("A"), valid.count("B")) / len(valid)


def normalize_label(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"A", "1", "LEFT", "RESPONSE_A"}:
        return "A"
    if s in {"B", "2", "RIGHT", "RESPONSE_B"}:
        return "B"
    return None
