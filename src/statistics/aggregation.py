"""
Shared aggregation helpers used by the statistics notebooks.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.conflicts import Conflict, Side


def latest_run_dir(parent: str, side_code: str) -> str | None:
    if not os.path.isdir(parent):
        return None
    prefix = f"{side_code}_run_"
    runs = sorted(d for d in os.listdir(parent) if d.startswith(prefix))
    return os.path.join(parent, runs[-1]) if runs else None


def load_scores(jsonl_path: str, score_key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            value = rec.get(score_key)
            if value is None:
                continue
            out[str(rec["id"])] = float(value)
    return out


def load_run_scores(
    *,
    results_root: str,
    conflict: Conflict,
    judge_label: str,
    judge_prompt: str,
    translator_label: str,
    prompt_name: str,
    side: Side,
) -> dict[str, float]:
    parent = os.path.join(
        results_root,
        conflict.code,
        f"{judge_label}_judge",
        judge_prompt,
        translator_label,
        prompt_name,
    )
    run_dir = latest_run_dir(parent, side.code)
    if run_dir is None:
        return {}
    score_key = f"score_{side.code}_mt_vs_{side.code}_human"
    return load_scores(os.path.join(run_dir, f"{side.code}_results.jsonl"), score_key)


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    ids = sorted(set(a) & set(b))
    return (
        np.array([a[i] for i in ids], dtype=float),
        np.array([b[i] for i in ids], dtype=float),
    )


def bh_fdr(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    adjusted = np.empty_like(q)
    adjusted[order] = q
    return adjusted
