# Perspectival Divergence in AI-Mediated Multilingual Political Communication

This repository contains the **data**, **prompt templates**, **evaluation pipeline**, and **statistical analyses** used in the submitted manuscript:

> *Perspectival Divergence in AI-Mediated Multilingual Political Communication*

The code implements a multilingual, conflict-aware evaluation framework for measuring **perspectival divergence**: the extent to which an LLM output preserves a shared event core while moving closer to one of two side-oriented narrative ecosystems.

The repository supports two conflict-aware datasets:

- **`ru_ua`** - Russia / Ukraine, target languages Russian and Ukrainian.
- **`il_ps`** - Israel / Palestine, target languages Hebrew and Arabic.

Adding a third conflict only requires adding one entry to `src/conflicts.py` and a matching `data/<conflict>/neutral_data.json`.

---

## Repository structure

```
data/
  ru_ua/
    neutral_data.json              # event-core triples (neutral, ru, ua)
    english_neutral_data.json      # English pivot references for the robustness check (Supplementary Information Section 4)
  il_ps/
    neutral_data.json              # event-core triples (neutral, il, ps)
    english_neutral_data.json      # English pivot references for the robustness check (Supplementary Information Section 4)

src/
  conflicts.py                     # Conflict / Side registry
  data.py                          # triple loader
  judge/
    judge.py                       # local HF judge
    gpt_judge.py                   # OpenAI Responses API judge (GPT-5.1)
    prompts.py                     # primary + robustness-variant judge system prompts
  translator/
    translator.py                  # local HF generator / pivot back-translator
    openai_translator.py           # OpenAI / Gemini / Moonshot OpenAI-compatible client
    factory.py                     # build_translator, build_pivot_translator
    prompts/prompts.py
  pipeline/
    distance.py
    pivot_english.py
    category_labeler.py            # mechanism-level cue annotator (Methods Section 4.5 / Supplementary Information Section 5)
  statistics/
    aggregation.py                 # shared for statistics
    within_language_tests.ipynb
    inter_judge_kappa.ipynb
    pivot_english_robustness.ipynb
    category_distribution.ipynb
```

### `data/`

Each conflict ships an event-core triple file and a pre-translated English pivot file, row-aligned with N = 100 events per conflict.

- `neutral_data.json`: `{"messages": [{"neutral": ..., "<side_a>": ..., "<side_b>": ...}, ...]}`
  - `neutral`: neutral English event statement
  - `<side_a>`, `<side_b>`: side-oriented reference texts in the target languages
- `english_neutral_data.json`: the same event list with the side-oriented references translated into English. This is the fixed pivot-English dataset used for the robustness check described in Supplementary Information Section 4. It is consumed *only* by `src/pipeline/pivot_english.py`.

### `src/conflicts.py`

`Conflict` and `Side` dataclasses describing a conflict as a pair of opposing information ecosystems. A `Side` fixes:
- the target `language` used to produce and judge a generation,
- the `country` slot used to instantiate country-aligned prompt templates,
- the `reference_key` used to read the side-oriented reference from the dataset.

### `src/translator/`

Translator backends that produce a translation from the neutral English sentence:

- **Local HF translator** (`translator.py`): runs a HuggingFace CausalLM via `transformers` pipeline. Also used as the pivot back-translator (Llama-3.1-8B-Instruct) for the English-pivot robustness check.
- **API translators** (`openai_translator.py`), selected automatically by model id in `factory.py:_provider_for`:
  - `gpt-*` - OpenAI Responses API (set `OPENAI_API_KEY`)
  - `gemini-*` - Google Gemini via OpenAI-compatible endpoint (set `GEMINI_API_KEY`)
  - `moonshot-*` - Moonshot (set `MOONSHOT_API_KEY`)
- All other models evaluated in the paper (Mistral, Qwen, LLaMA, DeepSeek, Falcon) run via the local HF translator.

Prompt templates for translation live in `src/translator/prompts/prompts.py` as `SYSTEM_PROMPTS_LIB` (13 templates grouped into: baseline/neutral, social influence, news, journalist voice, country-aligned).

### `src/judge/`

Implements the **semantic-distance judge** that scores how close a model output is to a POV reference, producing structured JSON (`score`, `explanation`):

- `judge.py` - **`Judge`** (primary): local HuggingFace judge. Llama-3.1-8B-Instruct is the primary; Falcon-7B-Instruct is available as an additional judge for inter-judge validation.
- `gpt_judge.py` - **OpenAI GPT-5.1 judge** (Responses API; set `OPENAI_API_KEY`). Used as an additional judge for inter-judge agreement.
- `prompts.py` - `JUDGE_PROMPTS` dict. Contains the primary prompt and robustness-phrasing variants used for judge-prompt robustness checks.

### `src/pipeline/`

- **`distance.py`** - end-to-end within-language pipeline. For each (conflict, side, generator model, prompt template, judge) combination, it translates each neutral sentence into the target language and judges it against the side-oriented reference.
- **`pivot_english.py`** - English-pivot robustness check. Given a previous within-language run, it translates the generated outputs into English with Llama-3.1-8B-Instruct and pairs them with the pre-translated English references from `english_neutral_data.json`, then re-judges the two English snippets.
- **`category_labeler.py`** - GPT-5.2 multi-label annotator applying the propaganda-inspired taxonomy to each generated output.

### `src/statistics/`

Jupyter notebooks that consume the run tree and reproduce paper artefacts:

| Notebook | Paper artefact | Tests / analysis |
|---|---|---|
| `within_language_tests.ipynb` | Fig. 2 and Supplementary similarity/distance tables | paired t-test per model/prompt cell, BH-FDR over 40 cells per conflict |
| `inter_judge_kappa.ipynb` | Judge-robustness analysis | Cohen's kappa between the primary judge and additional judges on closer-reference decisions |
| `pivot_english_robustness.ipynb` | English-pivot robustness analysis | count of model/prompt cells whose favored side changes under pivot-English judging |
| `category_distribution.ipynb` | Fig. 3 and mechanism-distribution tables | per-model percentage distribution across the five mechanism families |

Each notebook has a `CONFLICT` variable at the top - switch between `'ru_ua'` and `'il_ps'` to reproduce the corresponding table.

---

## Setup

Minimum dependencies depend on what you run:

- **Local HF translation / local judge**: `torch`, `transformers`, `langchain`, `pydantic`
- **OpenAI / Gemini / Moonshot APIs**: `openai` (and the corresponding API key)

Example:

```bash
pip install -U torch transformers langchain langchain-community pydantic openai
```

---

## Running the pipeline

All stages are conflict-agnostic. Select the conflict with `--conflict {ru_ua,il_ps}`. `--sides` and `--prompts` default to *all* sides of the conflict and *all* 13 templates; pass comma-separated lists to restrict.

### 1. Within-language distance

```bash
python -m src.pipeline.distance \
  --conflict ru_ua \
  --translator_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --translator_label llama3-8b \
  --judge_model /path/to/llama3-8b \
  --judge_label llama3-8b \
  --judge_prompt primary
```

With an API generator (automatically routed by model id):

```bash
export OPENAI_API_KEY="..."
python -m src.pipeline.distance \
  --conflict il_ps \
  --translator_model gpt-5.1 \
  --translator_label gpt-5.1 \
  --judge_model /path/to/llama3-8b \
  --judge_label llama3-8b \
  --judge_prompt primary
```

Outputs land at `<out_dir>/<conflict>/<judge>_judge/<judge_prompt>/<translator>/<prompt>/<side>_run_<ts>/`, containing `<side>_results.jsonl` (per example) and `<side>_summary.json`.

JSONL schema per row: `id`, `en`, `<side>_mt`, `score_<side>_mt_vs_<side>_human`, `explanation_<side>_mt_vs_<side>_human`.
Summary: `n`, `mean_score_<side>_MT_vs_<side>_POV`, `convention`.

**Re-judge an existing run with a different judge.** Pass `--prev_jsonl <path>` to skip translation and score the saved `<side>_mt` outputs with any other judge (HF or GPT). Because a prev file is tied to one side and one prompt, `--sides` and `--prompts` must each be a single value; `--translator_model` is ignored in this mode, but `--translator_label` and the chosen `--prompts` value are still used as the output-tree buckets so the new judge branch lines up with the original run. Point `--judge_model` / `--judge_label` at the new judge:

```bash
python -m src.pipeline.distance \
  --conflict ru_ua --sides ua --prompts neutral_1 \
  --prev_jsonl results/ru_ua/llama3-8b_judge/primary/llama3-8b/neutral_1/ua_run_.../ua_results.jsonl \
  --translator_label llama3-8b \
  --judge_model tiiuae/Falcon-7B-Instruct --judge_label falcon-7b \
  --judge_prompt primary
```


### 2. Pivot-English robustness

Run after a within-language run exists for the same (conflict, side, prompt). The per-run MT outputs are translated into English at runtime with Llama-3.1-8B-Instruct; the side-oriented references are read from `data/<conflict>/english_neutral_data.json` and are *not* re-translated:

```bash
python -m src.pipeline.pivot_english \
  --conflict ru_ua --side ru \
  --prev_jsonl results/ru_ua/llama3-8b_judge/primary/llama3-8b/neutral_1/ru_run_.../ru_results.jsonl \
  --pivot_model /path/to/Meta-Llama-3.1-8B-Instruct \
  --judge_model /path/to/llama3-8b --judge_label llama3-8b \
  --translator_label llama3-8b --prompt_name neutral_1
```

Default `--out_dir` is `results_pivot_english`; the run layout inside it is identical to the direct pipeline.

### 3. Category labeling

```bash
python -m src.pipeline.category_labeler \
  --conflict il_ps --side il \
  --prev_jsonl results/il_ps/.../il_results.jsonl \
  --translator_label llama3-8b --prompt_name neutral_1
```

Writes `<side>_labels.jsonl` under `<out_dir>/<conflict>/<labeler>_labeler/<translator>/<prompt>/<side>_run_<ts>/` (default `--out_dir` is `results_categories`).

---

## Outputs

Each run writes:

- `*_results.jsonl` - one line per example: `{id, en, <side>_mt, score_<side>_mt_vs_<side>_human, explanation_<side>_mt_vs_<side>_human, ...}`
- `*_summary.json` - aggregate stats: `n`, `mean_score_<side>_MT_vs_<side>_POV`, `convention`

Run directories are timestamped and nested by:
`<out_dir>/<conflict>/<judge>_judge/<judge_prompt>/<translator>/<prompt_name>/<side>_run_<timestamp>/`

The pivot-English pipeline adds two extra fields per row (`<side>_mt_en`, `<side>_human_english`) and keeps the same directory layout (default under `results_pivot_english/`).

---

## Extending the framework

- **New conflict**: add a `Conflict` entry to `CONFLICTS` in `src/conflicts.py`, drop a `data/<code>/neutral_data.json` with matching reference keys, and (for English-pivot robustness support) an `english_neutral_data.json` row-aligned with it.
- **New language / side**: each `Side` carries its own `language`, `country`, and `reference_key`. Prompts use the `{language}` and `{country}` slots.
- **New judge prompt**: add entries to `JUDGE_PROMPTS` in `src/judge/prompts.py`; pass their name via `--judge_prompt`. The judge-prompt robustness check uses the prebuilt `variant` entry.
- **New generator prompt**: edit `SYSTEM_PROMPTS_LIB` in `src/translator/prompts/prompts.py`. The registry is flat; prompt groupings are declared locally in the statistics notebooks.
- **New generator backend**: drop a new branch in `src/translator/factory.py:_provider_for`, or rely on the default HF path by passing a local model path.
- **New judge backend**: anything HF-backed works with `src/judge/judge.py` as-is. API-backed judges should extend `src/judge/gpt_judge.py` (the prompt and JSON contract are language-agnostic).
