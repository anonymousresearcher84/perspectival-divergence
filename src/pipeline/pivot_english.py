"""
Translation-mediated English judging.

Given a previous within-language run, translate every generated output into English with Llama-3.1-8B-Instruct
and pair it with the pre-built English translation of the corresponding side-oriented reference
(loaded from ``data/<conflict>/english_neutral_data.json``), then re-judge the two English snippets
with the same judge. As described in the paper, the pivot-English version of the evaluation
data is constructed once by translating each side-oriented reference into English a single time;
only the per-run generated outputs are translated at runtime.
This produces the pivot-English distances used to validate that the main results are not an artifact of direct cross-language judging.

Outputs follow the same layout as the primary pipeline::
<out_dir>/<conflict>/<judge>_judge/<judge_prompt>/<translator>/<prompt>/<side>_run_<ts>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.conflicts import Conflict, get_conflict
from src.data import Item
from src.judge import JUDGE_PROMPTS, Judge, JudgeConfig
from src.translator.factory import build_pivot_translator
from src.utils import free_cuda


ENGLISH_DATA_PATHS: dict[str, str] = {
    "ru_ua": "data/ru_ua/english_neutral_data.json",
    "il_ps": "data/il_ps/english_neutral_data.json",
}


def _load_english_items(conflict: Conflict, root: str | Path) -> list[Item]:
    path = Path(root) / ENGLISH_DATA_PATHS[conflict.code]
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    keys = [s.reference_key for s in conflict.sides]
    items: list[Item] = []
    for idx, raw in enumerate(payload["messages"]):
        if "neutral" not in raw or any(k not in raw for k in keys):
            continue
        items.append(
            Item(
                id=raw.get("id", f"ex_{idx:05d}"),
                neutral=raw["neutral"].strip(),
                references={k: raw[k].strip() for k in keys},
            )
        )
    return items


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_prev_mt(prev_jsonl: str, side_code: str) -> dict[str, str]:
    mt_by_id: dict[str, str] = {}
    mt_key = f"{side_code}_mt"
    for rec in _read_jsonl(prev_jsonl):
        rid = rec.get("id")
        mt = rec.get(mt_key) or rec.get("mt")
        if rid and mt:
            mt_by_id[str(rid)] = str(mt)
    return mt_by_id


def run_side(
    *,
    conflict_code: str,
    side_code: str,
    prev_jsonl: str,
    pivot_model: str,
    judge_model_path: str,
    judge_label: str,
    judge_prompt_name: str,
    translator_label: str,
    prompt_name: str,
    data_root: str,
    out_dir: str,
    limit: int | None,
) -> None:
    conflict = get_conflict(conflict_code)
    side = conflict.side(side_code)

    mt_by_id = _load_prev_mt(prev_jsonl, side.code)
    english_items = _load_english_items(conflict, root=data_root)
    items = [it for it in english_items if it.id in mt_by_id]
    if not items:
        raise KeyError(
            f"no overlap between prev_jsonl ids and {ENGLISH_DATA_PATHS[conflict.code]} ids"
        )
    if limit is not None:
        items = items[:limit]

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        out_dir,
        conflict.code,
        f"{judge_label}_judge",
        judge_prompt_name,
        translator_label,
        prompt_name,
        f"{side.code}_run_{run_ts}",
    )
    os.makedirs(run_dir, exist_ok=True)
    out_jsonl = os.path.join(run_dir, f"{side.code}_results.jsonl")
    out_summary = os.path.join(run_dir, f"{side.code}_summary.json")

    pivot = build_pivot_translator(pivot_model, source_language=side.language)
    try:
        mt_en = pivot.translate_batch([mt_by_id[it.id] for it in items])
    finally:
        del pivot
        free_cuda()

    ref_en = [it.references[side.reference_key] for it in items]

    judge = Judge(JudgeConfig(model_path=judge_model_path, system_prompt=JUDGE_PROMPTS[judge_prompt_name]))
    try:
        judged = [judge.judge("English", a, b) for a, b in zip(mt_en, ref_en)]
    finally:
        del judge
        free_cuda()

    mt_key = f"{side.code}_mt"
    mt_en_key = f"{side.code}_mt_en"
    ref_en_key = f"{side.code}_human_english"
    score_key = f"score_{side.code}_mt_vs_{side.code}_human"
    expl_key = f"explanation_{side.code}_mt_vs_{side.code}_human"
    mean_key = f"mean_score_{side.code}_MT_vs_{side.code}_POV"

    scores: list[float] = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it, mt_en_i, ref_en_i, jr in zip(items, mt_en, ref_en, judged):
            if jr.get("score") is None:
                continue
            score = float(jr["score"])
            scores.append(score)
            row = {
                "id": it.id,
                "en": it.neutral,
                mt_key: mt_by_id[it.id],
                mt_en_key: mt_en_i,
                ref_en_key: ref_en_i,
                score_key: score,
                expl_key: str(jr.get("explanation", "")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n": len(scores),
        mean_key: round(sum(scores) / len(scores), 4) if scores else None,
        "convention": (
            f"Pivot-English distance: {side.code.upper()} MT and {side.code.upper()} "
            "POV reference both translated into English before judging."
        ),
        "source_prev_jsonl": prev_jsonl,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pivot-English robustness pipeline.")
    ap.add_argument("--conflict", required=True, choices=sorted(("ru_ua", "il_ps")))
    ap.add_argument("--side", required=True)
    ap.add_argument("--prev_jsonl", required=True, help="JSONL produced by distance.py for this side+prompt")
    ap.add_argument("--pivot_model", required=True, help="HF path for the pivot translator (Llama-3.1-8B-Instruct)")
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--judge_label", required=True)
    ap.add_argument("--judge_prompt", default="primary", choices=sorted(JUDGE_PROMPTS))
    ap.add_argument("--translator_label", required=True, help="bucket used in the output path")
    ap.add_argument("--prompt_name", required=True, help="prompt template bucket used in the output path")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_dir", default="results_pivot_english")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_side(
        conflict_code=args.conflict,
        side_code=args.side,
        prev_jsonl=args.prev_jsonl,
        pivot_model=args.pivot_model,
        judge_model_path=args.judge_model,
        judge_label=args.judge_label,
        judge_prompt_name=args.judge_prompt,
        translator_label=args.translator_label,
        prompt_name=args.prompt_name,
        data_root=args.data_root,
        out_dir=args.out_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
