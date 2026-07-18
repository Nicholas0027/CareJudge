from __future__ import annotations

import logging
import time
from typing import Any

from care_judge.judges.base import BaseJudge
from care_judge.judges.parsers import parse_judgment
from care_judge.judges.prompts import pairwise_prompt
from care_judge.schemas import Judgment, PairItem

logger = logging.getLogger(__name__)


class LocalHFJudge(BaseJudge):
    """A local HuggingFace transformers judge.

    Loads a causal-LM on GPU (or CPU fallback) and runs pairwise judging.
    Parses JSON output; falls back to regex extraction if JSON is malformed.
    """

    def __init__(
        self,
        model_name: str,
        name: str | None = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        cost_per_call: float = 0.0,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.name = name or model_name
        self.cost_per_call = cost_per_call
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "auto": torch.float16 if device == "cuda" else torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        logger.info("Loading model %s on %s (%s)", model_name, device, dtype)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs", elapsed)
        print(f"[LocalHFJudge] Loaded {model_name} on {device} in {elapsed:.1f}s")

    def judge(self, item: PairItem, rubric: str, temperature: float = 0.0) -> Judgment:
        import torch

        prompt = pairwise_prompt(item, rubric=rubric, ask_confidence=True)
        messages = [{"role": "user", "content": prompt}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.01
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 0.01) if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        j = parse_judgment(response, raw=response)
        j.cost = self.cost_per_call
        return j

    def __del__(self):
        try:
            import torch
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
