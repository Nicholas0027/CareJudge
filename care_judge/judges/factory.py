from __future__ import annotations

from care_judge.judges.base import BaseJudge
from care_judge.judges.litellm_api import LiteLLMJudge
from care_judge.judges.mock import MockJudge


def make_judge(spec: str) -> BaseJudge:
    """Create a judge from a compact spec.

    Examples:
      mock
      mock:0.8
      litellm:gpt-4o-mini
      litellm:openai/gpt-4o-mini
      local_hf:Qwen/Qwen2.5-1.5B-Instruct
      local_hf:meta-llama/Llama-3.2-1B-Instruct
    """
    if spec.startswith("mock"):
        parts = spec.split(":")
        acc = float(parts[1]) if len(parts) > 1 else 0.72
        return MockJudge(name=spec, accuracy=acc)
    if spec.startswith("litellm:"):
        return LiteLLMJudge(model=spec.split(":", 1)[1])
    if spec.startswith("local_hf:"):
        from care_judge.judges.local_hf import LocalHFJudge
        model_name = spec.split(":", 1)[1]
        return LocalHFJudge(model_name=model_name)
    raise ValueError(f"Unknown judge spec: {spec}")
