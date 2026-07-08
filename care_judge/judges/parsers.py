from __future__ import annotations

import json
import re
from typing import Any, Dict

from care_judge.schemas import Judgment


def parse_judgment(text: str, raw: Any = None) -> Judgment:
    text = text.strip()
    obj: Dict[str, Any] = {}
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
    winner = str(obj.get("winner", "")).strip().upper()
    if winner not in {"A", "B"}:
        if re.search(r"\bA\b", text, flags=re.I) and not re.search(r"\bB\b", text, flags=re.I):
            winner = "A"
        elif re.search(r"\bB\b", text, flags=re.I) and not re.search(r"\bA\b", text, flags=re.I):
            winner = "B"
        else:
            winner = "abstain"
    try:
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = min(max(conf, 0.0), 1.0)
    return Judgment(winner=winner, confidence=conf, reason=str(obj.get("reason", "")), raw=raw or text)
