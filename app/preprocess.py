"""
Text preprocessing for headless-ts-pipeline.

This module cleans raw social media text so that the toxicity model
and policy rules receive clean, consistent input.
"""

import re


def remove_urls(text: str) -> str:
    """Remove web links from text.

    Social media posts often contain links like https://example.com.
    Links do not help us decide if text is toxic, so we remove them.

    Example:
        "Check this out https://example.com" -> "Check this out "
    """
    url_pattern = r"http\S+|www\S+|https\S+"
    return re.sub(url_pattern, "", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML markup like <b>, <div>, <a>, etc.

    Sometimes posts contain HTML code copied from elsewhere.
    We keep only the visible words, not the markup.

    Example:
        "Hello <b>world</b>" -> "Hello world"
    """
    html_pattern = r"<.*?>"
    return re.sub(html_pattern, "", text)


def remove_emojis(text: str) -> str:
    """Remove emoji symbols from text.

    Emojis like 🔥 or 😂 do not carry meaning for toxicity checking,
    and they can confuse the model. We strip them out.

    Example:
        "Hello 🔥🔥" -> "Hello "
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces, tabs, or newlines into a single space.

    People often type multiple spaces by accident. We normalise them.

    Example:
        "Hello    world" -> "Hello world"
    """
    return " ".join(text.split())


def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase.

    "HELLO" and "hello" mean the same thing. Lowercasing helps the model
    treat them identically.

    Example:
        "HELLO World" -> "hello world"
    """
    return text.lower()


def clean_text(text: str) -> str:
    """Run all cleaning steps in order.

    This is the main function. Other parts of the project (model, policy, API)
    will call this function before doing anything with the text.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove emojis
        5. Remove extra whitespace
        6. Trim leading and trailing spaces
    """
    if not text:
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_extra_whitespace(text)

    return text.strip()
