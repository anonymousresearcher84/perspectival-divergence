"""Build a translator from a model id and a per-side prompt slot"""

from __future__ import annotations

from .openai_translator import OpenAITranslator, OpenAITranslatorConfig
from .translator import HFTranslator, TranslatorConfig


def _provider_for(model_id: str) -> str | None:
    name = model_id.lower()
    if "gpt" in name and "5" in name:
        return "openai"
    if "gemini" in name:
        return "gemini"
    if "moonshot" in name or "kimi" in name:
        return "moonshot"
    return None


def build_translator(
    model_id: str,
    *,
    language: str,
    country: str,
    prompt_template: str,
    throttle_seconds: float = 0.0,
):
    system_prompt = prompt_template.format(language=language, country=country)
    provider = _provider_for(model_id)

    if provider is None:
        return HFTranslator(
            TranslatorConfig(model_path=model_id, language=language, system_prompt=system_prompt)
        )

    return OpenAITranslator(
        OpenAITranslatorConfig(
            provider=provider,
            model=model_id,
            language=language,
            system_prompt=system_prompt,
            throttle_seconds=throttle_seconds,
        )
    )


def build_pivot_translator(model_id: str, *, source_language: str):
    """Build the Llama pivot-English back-translator"""

    system_prompt = (
        "Translate the following text into English as faithfully as possible.\n"
        "Preserve names, stance-bearing wording, and source-specific phrasing.\n"
        "Do not summarize or neutralize.\n"
        "Output only the translation."
    )
    return HFTranslator(
        TranslatorConfig(
            model_path=model_id,
            language=source_language,
            system_prompt=system_prompt,
            do_sample=False,
            temperature=0.0,
        )
    )
