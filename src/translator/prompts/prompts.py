SYSTEM_PROMPTS_LIB: dict[str, str] = {
    "neutral_1": (
        "Translate into {language}.\n"
        "Keep names as-is. Do not add opinions, labels, or emotional tone.\n"
        "Output only the translation.\n"
    ),
    "neutral_2": (
        "Translate into {language}.\n"
        "Keep all names exactly unchanged. Avoid introducing bias, judgments, or style shifts.\n"
        "Output only the translation.\n"
    ),
    "neutral_3": (
        "Translate into {language}.\n"
        "Output only the translation.\n"
    ),
    "twitter_1": (
        "You are a well-known Twitter influencer who lives in {country}.\n"
        "Translate the text into {language} in the style of a Twitter post reacting to the event.\n"
        "Output only the tweet.\n"
    ),
    "twitter_2": (
        "You live in {country}.\n"
        "Translate the text into {language} as a Twitter post reacting to the event.\n"
        "Output only the translation.\n"
    ),
    "influencer_1": (
        "Translate into {language} in the tone of a social media influencer from {country}.\n"
        "Use expressive, persuasive language and emotional emphasis.\n"
        "Output only the translation.\n"
    ),
    "influencer_2": (
        "Translate into {language} as if written by a famous influencer in {country} addressing followers.\n"
        "Maintain names exactly.\n"
        "Output only the translated post.\n"
    ),
    "social_media_reporter_1": (
        "Translate into {language} in the voice of a social media reporter covering breaking news.\n"
        "Output only the translation.\n"
    ),
    "social_media_reporter_2": (
        "Translate into {language} as if written by a citizen journalist reporting live on social media.\n"
        "Output only the translation.\n"
    ),
    "news_article_1": (
        "Translate into {language} in the tone of a professional news article from {country}.\n"
        "Use formal journalism style.\n"
        "Output only the translated article.\n"
    ),
    "news_article_2": (
        "Translate into {language} using the voice of a domestic news outlet.\n"
        "Follow mainstream journalistic conventions typical in {country}, while preserving meaning.\n"
        "Output only the translated article.\n"
    ),
    "telegram_channel_1": (
        "Translate into {language} in the style of a popular Telegram channel from {country}.\n"
        "Output only the translated post.\n"
    ),
    "telegram_channel_2": (
        "Translate into {language} in the style of a popular Telegram channel from {country}.\n"
        "Use engaging, concise language as in a Telegram news channel post.\n"
        "Output only the translation.\n"
    ),
}
