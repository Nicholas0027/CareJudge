from __future__ import annotations

from care_judge.judges.base import BaseJudge
from care_judge.judges.parsers import parse_judgment
from care_judge.judges.prompts import pairwise_prompt
from care_judge.schemas import Judgment, PairItem


class LiteLLMJudge(BaseJudge):
    def __init__(self, model: str, name: str | None = None, cost_per_call: float = 0.0, max_tokens: int = 512):
        self.model = model
        self.name = name or model
        self.cost_per_call = cost_per_call
        self.max_tokens = max_tokens

    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("Install API extras with `pip install -e .[api]`") from exc
        prompt = pairwise_prompt(item, rubric=rubric, ask_confidence=True)
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        text = resp["choices"][0]["message"]["content"]
        j = parse_judgment(text, raw=resp)
        j.cost = self.cost_per_call
        return j
