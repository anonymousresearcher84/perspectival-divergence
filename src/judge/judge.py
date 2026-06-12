"""
Local HF distance judge.

Implements the within-language semantic distance described in Section 3.3.1.
The judge returns a score in [0, 1] interpreted as the distance (d) between two snippets.
The corresponding similarity is s = 1 - d.

The same class is used both for the primary judge (Llama-3.1-8B-Instruct) and for the Falcon-7B-Instruct additional-judge robustness check.
The judge's behaviour is controlled by the system prompt and the model path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..utils import coerce_unit_score, extract_json
from .prompts import JUDGE_PROMPT_TEMPLATE


@dataclass
class JudgeConfig:
    model_path: str
    system_prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.0
    dtype: str = "auto"


_DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Judge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg

        _local = cfg.model_path.startswith("/")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=True, local_files_only=_local
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=_DTYPES[cfg.dtype],
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=_local,
        ).eval()

        self.tokenizer = tokenizer
        self.model = model
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=cfg.temperature > 0.0,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            **({"temperature": cfg.temperature} if cfg.temperature > 0.0 else {}),
        )

    def _prompt(self, language: str, a: str, b: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            system=self.cfg.system_prompt.format(language=language),
            sentence_a=a,
            sentence_b=b,
        )

    def judge(self, language: str, sentence_a: str, sentence_b: str) -> dict:
        prompt = self._prompt(language, sentence_a, sentence_b)
        with torch.inference_mode():
            raw = self.pipe(prompt)[0]["generated_text"]
        obj = extract_json(raw)
        return {"score": coerce_unit_score(obj["score"]), "explanation": str(obj.get("explanation", "")).strip()}

    def judge_batch(
        self,
        language: str,
        sentences_a: list[str],
        sentences_b: list[str],
        batch_size: int = 4,
    ) -> list[dict]:
        if len(sentences_a) != len(sentences_b):
            raise ValueError("sentences_a and sentences_b must have the same length")

        prompts = [self._prompt(language, a, b) for a, b in zip(sentences_a, sentences_b)]
        with torch.inference_mode():
            raw_outputs = self.pipe(prompts, batch_size=batch_size)

        if raw_outputs and isinstance(raw_outputs[0], list):
            raw_outputs = [x[0] for x in raw_outputs]

        results: list[dict] = []
        for out in raw_outputs:
            try:
                obj = extract_json(out["generated_text"])
                results.append(
                    {"score": coerce_unit_score(obj["score"]), "explanation": str(obj.get("explanation", "")).strip()}
                )
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                results.append({"score": None, "explanation": f"judge_failed: {e}"})
        return results
