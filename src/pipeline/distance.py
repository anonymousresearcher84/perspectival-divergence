"""
Within-language distance pipeline (Sections 3.2 and 3.3.1).

For a chosen conflict, generator model, judge and prompt template,
the pipeline renders each neutral English statement into the target language of each side,
scores the output against the corresponding side-oriented reference with the judge, and writes per-example results and a summary.

Runs are stored under::
<out_dir>/<conflict>/<judge>_judge/<judge_prompt>/<translator>/<prompt>/<side>_run_<ts>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.conflicts import Side, get_conflict
from src.data import Item, load_items
from src.judge import GPTJudge, GPTJudgeConfig, JUDGE_PROMPTS, Judge, JudgeConfig
from src.translator.factory import build_translator
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB
from src.utils import free_cuda


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values))) if values else 0.0


def _chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _translate_all(translator, items: list[Item], batch_size: int) -> list[str]:
    texts = [it.neutral for it in items]
    if hasattr(translator, "translate_batch"):
        out: list[str] = []
        for batch in _chunks(texts, batch_size):
            out.extend(translator.translate_batch(batch, batch_size=batch_size))
        return out
    return [translator.translate(t) for t in texts]


def _judge_all(judge, language: str, hyps: list[str], refs: list[str], batch_size: int) -> list[dict]:
    if hasattr(judge, "judge_batch"):
        results: list[dict] = []
        for sl in (slice(i, i + batch_size) for i in range(0, len(hyps), batch_size)):
            results.extend(
                judge.judge_batch(
                    language=language,
                    sentences_a=hyps[sl],
                    sentences_b=refs[sl],
                    batch_size=batch_size,
                )
            )
        return results
    return [judge.judge(language, a, b) for a, b in zip(hyps, refs)]


def _build_judge(judge_model_path: str, system_prompt: str, judge_backend: str):
    if judge_backend == "gpt":
        return GPTJudge(GPTJudgeConfig(system_prompt=system_prompt, model=judge_model_path))
    return Judge(JudgeConfig(model_path=judge_model_path, system_prompt=system_prompt))


def _load_prev_mt(prev_jsonl: str, side_code: str) -> dict[str, str]:
    mt_by_id: dict[str, str] = {}
    mt_key = f"{side_code}_mt"
    with open(prev_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            mt = rec.get(mt_key) or rec.get("mt")
            if rid and mt:
                mt_by_id[str(rid)] = str(mt)
    return mt_by_id


def run_prompt(
    *,
    conflict_code: str,
    side_code: str,
    translator_model: str,
    translator_label: str,
    prompt_name: str,
    judge_model_path: str,
    judge_label: str,
    judge_prompt_name: str,
    judge_backend: str,
    data_root: str,
    out_dir: str,
    limit: int | None,
    batch_size: int,
    prev_jsonl: str | None = None,
) -> None:
    conflict = get_conflict(conflict_code)
    side: Side = conflict.side(side_code)

    system_prompt = JUDGE_PROMPTS[judge_prompt_name]

    all_items = load_items(conflict, root=data_root)
    if prev_jsonl is not None:
        mt_by_id = _load_prev_mt(prev_jsonl, side.code)
        if not mt_by_id:
            raise ValueError(f"no usable rows with '{side.code}_mt' in {prev_jsonl}")
        items = [it for it in all_items if it.id in mt_by_id]
        if not items:
            raise KeyError(f"no overlap between prev_jsonl ids and {conflict.data_path} ids")
    else:
        items = all_items
    if limit is not None:
        items = items[:limit]

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        out_dir,
        conflict_code,
        f"{judge_label}_judge",
        judge_prompt_name,
        translator_label,
        prompt_name,
        f"{side.code}_run_{run_ts}",
    )
    os.makedirs(run_dir, exist_ok=True)
    out_jsonl = os.path.join(run_dir, f"{side.code}_results.jsonl")
    out_summary = os.path.join(run_dir, f"{side.code}_summary.json")

    if prev_jsonl is not None:
        mt_outputs = [mt_by_id[it.id] for it in items]
    else:
        template = SYSTEM_PROMPTS_LIB[prompt_name]
        translator = build_translator(
            translator_model,
            language=side.language,
            country=side.country,
            prompt_template=template,
        )
        try:
            mt_outputs = _translate_all(translator, items, batch_size)
        finally:
            del translator
            free_cuda()

    references = [it.references[side.reference_key] for it in items]
    judge = _build_judge(judge_model_path, system_prompt, judge_backend)
    try:
        judge_results = _judge_all(judge, side.language, mt_outputs, references, batch_size)
    finally:
        del judge
        free_cuda()

    mt_key = f"{side.code}_mt"
    score_key = f"score_{side.code}_mt_vs_{side.code}_human"
    expl_key = f"explanation_{side.code}_mt_vs_{side.code}_human"
    mean_key = f"mean_score_{side.code}_MT_vs_{side.code}_POV"

    scores: list[float] = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item, mt, jr in zip(items, mt_outputs, judge_results):
            if jr.get("score") is None:
                continue
            score = float(jr["score"])
            scores.append(score)
            row = {
                "id": item.id,
                "en": item.neutral,
                mt_key: mt,
                score_key: score,
                expl_key: str(jr.get("explanation", "")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n": len(scores),
        mean_key: round(_mean(scores), 4),
        "convention": (
            f"Lower score = closer to {side.code.upper()} POV reference "
            f"(within-language distance in {side.language})."
        ),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {out_jsonl}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Within-language distance pipeline.")
    ap.add_argument("--conflict", required=True, choices=sorted(("ru_ua", "il_ps")))
    ap.add_argument("--sides", default=None, help="comma-separated sides; defaults to both sides of the conflict")
    ap.add_argument("--prompts", default=None, help="comma-separated prompt names; defaults to all templates")
    ap.add_argument("--translator_model", default=None, help="ignored when --prev_jsonl is given")
    ap.add_argument("--translator_label", required=True, help="bucket used in the output path")
    ap.add_argument(
        "--prev_jsonl",
        default=None,
        help=(
            "Skip translation and re-judge an existing <side>_results.jsonl "
            "produced by a prior run. Requires a single --sides and --prompts value."
        ),
    )
    ap.add_argument("--judge_model", required=True, help="HF path for Judge, or OpenAI model id for GPTJudge")
    ap.add_argument("--judge_label", required=True)
    ap.add_argument("--judge_prompt", default="primary", choices=sorted(JUDGE_PROMPTS))
    ap.add_argument("--judge_backend", default="hf", choices=("hf", "gpt"))
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    conflict = get_conflict(args.conflict)

    sides = [s.strip() for s in args.sides.split(",")] if args.sides else [s.code for s in conflict.sides]
    prompts = [p.strip() for p in args.prompts.split(",")] if args.prompts else list(SYSTEM_PROMPTS_LIB)

    if args.prev_jsonl is not None and (len(sides) != 1 or len(prompts) != 1):
        raise SystemExit("--prev_jsonl requires exactly one --sides value and one --prompts value")
    if args.prev_jsonl is None and not args.translator_model:
        raise SystemExit("--translator_model is required unless --prev_jsonl is given")

    for side_code in sides:
        for prompt_name in prompts:
            print(f"\n== {conflict.code} / {side_code} / {prompt_name} ==", flush=True)
            run_prompt(
                conflict_code=conflict.code,
                side_code=side_code,
                translator_model=args.translator_model,
                translator_label=args.translator_label,
                prompt_name=prompt_name,
                judge_model_path=args.judge_model,
                judge_label=args.judge_label,
                judge_prompt_name=args.judge_prompt,
                judge_backend=args.judge_backend,
                data_root=args.data_root,
                out_dir=args.out_dir,
                limit=args.limit,
                batch_size=args.batch_size,
                prev_jsonl=args.prev_jsonl,
            )


if __name__ == "__main__":
    main()
