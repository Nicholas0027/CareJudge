from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error

from care_judge.judges.base import BaseJudge
from care_judge.judges.parsers import parse_judgment
from care_judge.judges.prompts import pairwise_prompt
from care_judge.schemas import Judgment, PairItem


class OpenAICompatJudge(BaseJudge):
    """Judge backed by any OpenAI-compatible /chat/completions endpoint.

    Reads base URL and key from env (OPENAI_BASE_URL / OPENAI_API_KEY, or the
    POLOAI_* aliases). No third-party deps; uses urllib so it runs anywhere.
    Never logs the key.
    """

    def __init__(self, model: str, name: str | None = None, cost_per_call: float = 0.0,
                 max_tokens: int = 200, timeout: int = 90, base_url: str | None = None,
                 api_key: str | None = None):
        self.model = model
        self.name = name or f"openai:{model}"
        self.cost_per_call = cost_per_call
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("POLOAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("POLOAI_API_KEY")
        if not self.base_url:
            raise RuntimeError("Set OPENAI_BASE_URL (or POLOAI_BASE_URL) for OpenAICompatJudge")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY (or POLOAI_API_KEY) for OpenAICompatJudge")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        key = str(self.api_key)
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=data, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        prompt = pairwise_prompt(item, rubric=rubric, ask_confidence=True)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
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
                # Some models reject temperature/max_tokens; retry minimal payload once.
                if attempt == 0 and ("temperature" in body or "max_tokens" in body or e.code == 400):
                    payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        # Graceful abstain on persistent failure so a run can continue.
        return Judgment(winner="abstain", confidence=0.5, reason=f"api_error: {last_err}", cost=self.cost_per_call)
