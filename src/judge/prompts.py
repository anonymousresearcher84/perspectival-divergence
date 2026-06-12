"""
Distance-judge system prompts.

`primary` is the prompt used for all main results
`variant_1` is the semantically equivalent rephrasing used in the prompt-phrasing robustness check.

`JUDGE_PROMPT_TEMPLATE` wraps a system prompt with the shared JSON output contract and the two snippets to compare.
It is used by both the local and GPT judges.
"""

JUDGE_PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Return ONLY a valid JSON object (no prose, no markdown, no extra keys), "
    "in the following format:\n"
    "{{\n"
    '  "score": <number in [0,1]>,\n'
    '  "explanation": <1-2 short, neutral English sentences>\n'
    "}}\n\n"
    "Sentence A: {sentence_a}\n"
    "Sentence B: {sentence_b}\n"
)

JUDGE_PROMPTS: dict[str, str] = {
    "primary": (
        "You are a careful evaluator. Judge how far apart the meanings of two "
        "{language} snippets are.\n"
        "Scoring is a scaling between 0.0 to 1.0 where:\n"
        " - 0.0: identical meaning\n"
        " - 0.5: somewhat different\n"
        " - 1.0: completely unrelated or contradictory\n"
        "Be concise and neutral."
    ),
    "variant_1": (
        "You are a careful evaluator.\n"
        "You will get two {language} snippets that may refer to the same real-world event.\n"
        "Judge how far apart their meanings are.\n"
        "Scoring is a scaling between 0.0 to 1.0 where:\n"
        " - 0.0: identical meaning\n"
        " - 0.5: somewhat different\n"
        " - 1.0: completely unrelated or contradictory\n"
        "Be concise and neutral."
    ),
}
