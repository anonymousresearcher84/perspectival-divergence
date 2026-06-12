"""
Used for the GPT-5.1 judge robustness analysis.
The prompt template and JSON contract are identical to the local judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from ..utils import coerce_unit_score, extract_json, require_api_key
from .prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class GPTJudgeConfig:
    system_prompt: str
    model: str = "gpt-5.1"
    temperature: float = 0.0
    max_output_tokens: int = 256
    reasoning_effort: str | None = None


class GPTJudge:
    def __init__(self, cfg: GPTJudgeConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=require_api_key("OPENAI_API_KEY"))

    def judge(self, language: str, sentence_a: str, sentence_b: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            system=self.cfg.system_prompt.format(language=language),
            sentence_a=sentence_a.strip(),
            sentence_b=sentence_b.strip(),
        )

        kwargs = dict(
            model=self.cfg.model,
            input=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_output_tokens,
        )
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}

        response = self.client.responses.create(**kwargs)
        text = (getattr(response, "output_text", "") or "").strip()
        obj = extract_json(text)
        return {"score": coerce_unit_score(obj["score"]), "explanation": str(obj.get("explanation", "")).strip()}
