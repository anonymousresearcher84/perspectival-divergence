from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from ..utils import require_api_key


_PROVIDERS = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "uses_responses_api": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "uses_responses_api": False,
    },
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "uses_responses_api": False,
    },
}


@dataclass
class OpenAITranslatorConfig:
    provider: str
    model: str
    language: str
    system_prompt: str
    temperature: float = 0.4
    max_tokens: int = 8192
    reasoning_effort: str | None = None
    throttle_seconds: float = 0.0


class OpenAITranslator:
    def __init__(self, cfg: OpenAITranslatorConfig):
        if cfg.provider not in _PROVIDERS:
            raise KeyError(f"unknown provider {cfg.provider!r}; known: {sorted(_PROVIDERS)}")
        self.cfg = cfg
        self.spec = _PROVIDERS[cfg.provider]

        api_key = require_api_key(self.spec["api_key_env"])
        self.client = OpenAI(api_key=api_key, base_url=self.spec["base_url"])

    def _user_prompt(self, source: str) -> str:
        return (
            f"Translate the following text into {self.cfg.language}.\n"
            f"Text: {source.strip()}\n"
            "Output only the translation.\n"
        )

    def translate(self, source: str) -> str:
        if self.cfg.throttle_seconds > 0:
            time.sleep(self.cfg.throttle_seconds)

        if self.spec["uses_responses_api"]:
            kwargs = dict(
                model=self.cfg.model,
                input=[
                    {"role": "developer", "content": self.cfg.system_prompt},
                    {"role": "user", "content": self._user_prompt(source)},
                ],
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
            )
            if self.cfg.reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}
            response = self.client.responses.create(**kwargs)
            text = getattr(response, "output_text", "") or ""
        else:
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": self._user_prompt(source)},
                ],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            text = response.choices[0].message.content or ""

        return text.strip().strip('"').strip()

    def translate_batch(self, sources: list[str], batch_size: int = 1) -> list[str]:
        return [self.translate(s) for s in sources]
