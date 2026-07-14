from __future__ import annotations
import json, os, time, urllib.request, urllib.error
from care_judge.judges.base import BaseJudge
from care_judge.judges.parsers import parse_judgment
from care_judge.judges.prompts import pairwise_prompt
from care_judge.schemas import Judgment, PairItem

class DeepSeekJudge(BaseJudge):
    def __init__(self, model: str, name: str | None = None, cost_per_call: float = 0.0, max_tokens: int = 200, timeout: int = 90):
        self.model = model
        self.name = name or f"deepseek:{model}"
        self.cost_per_call = cost_per_call
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY")

    def _post(self, payload):
        key = str(self.api_key)
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), method="POST", headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        prompt = pairwise_prompt(item, rubric=rubric, ask_confidence=True)
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": self.max_tokens}
        last_err = None
        for attempt in range(3):
            try:
                out = self._post(payload)
                text = out.get("choices", [{}])[0].get("message", {}).get("content", "")
                j = parse_judgment(text, raw=text)
                j.cost = self.cost_per_call
                return j
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:200]
                last_err = f"HTTP {e.code}: {body}"
                if attempt == 0 and e.code == 400:
                    payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        return Judgment(winner="abstain", confidence=0.5, reason=f"api_error: {last_err}", cost=self.cost_per_call)
