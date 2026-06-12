"""Small shared helpers used across the pipeline."""

from __future__ import annotations

import json
import math
import os
import re

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJ.search(text)
    if match is None:
        raise ValueError(f"no json object in output: {text[:200]!r}")
    return json.loads(match.group(0))


def coerce_unit_score(value) -> float:
    score = float(value)
    if math.isnan(score) or math.isinf(score):
        raise ValueError("score is not finite")
    return max(0.0, min(1.0, score))


def require_api_key(env_name: str) -> str:
    key = (os.environ.get(env_name, "") or "").strip().strip("\"'")
    if not key:
        raise RuntimeError(f"{env_name} is not set")
    return key


def free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
