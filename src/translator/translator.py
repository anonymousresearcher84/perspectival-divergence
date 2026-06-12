from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class TranslatorConfig:
    model_path: str
    language: str
    system_prompt: str
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.4
    top_p: float = 0.85
    top_k: int = 50


_MEMORY_HINT = {0: "46GiB", 1: "46GiB", "cpu": "0GiB"}


class HFTranslator:
    def __init__(self, cfg: TranslatorConfig):
        self.cfg = cfg

        _local = cfg.model_path.startswith("/")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=True, trust_remote_code=True,
            local_files_only=_local,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        max_memory = _MEMORY_HINT if torch.cuda.device_count() >= 2 else None
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=_local,
        ).eval()

        self.tokenizer = tokenizer
        self.model = model
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    def _build_prompt(self, source: str) -> str:
        system = self.cfg.system_prompt.strip()
        user = f"Text: {source.strip()}\n"
        if self.has_chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{system}\n\n{user}" if system else user

    def _decode_kwargs(self) -> dict:
        return dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            use_cache="internlm2" not in self.cfg.model_path,
        )

    def translate(self, source: str) -> str:
        prompt = self._build_prompt(source)
        out = self.pipe(prompt, **self._decode_kwargs())[0]["generated_text"]
        if out.startswith(prompt):
            out = out[len(prompt):]
        return out.strip().strip('"').strip()

    def translate_batch(self, sources: list[str], batch_size: int = 8) -> list[str]:
        prompts = [self._build_prompt(s) for s in sources]
        with torch.inference_mode():
            outputs = self.pipe(prompts, batch_size=batch_size, **self._decode_kwargs())
        if outputs and isinstance(outputs[0], list):
            outputs = [x[0] for x in outputs]
        results: list[str] = []
        for prompt, out in zip(prompts, outputs):
            text = out["generated_text"]
            if text.startswith(prompt):
                text = text[len(prompt):]
            results.append(text.strip().strip('"').strip())
        return results
