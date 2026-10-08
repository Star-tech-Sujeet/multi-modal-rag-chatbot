"""
Markdown presentation and sanitization utility for Aura AI.
Removes internal SVG/anchor artifacts and localhost implementation URLs
while strictly preserving valid Markdown syntax (headings, code blocks, lists, valid external links).
"""

import re


def clean_markdown_display(text: str) -> str:
    """
    Sanitize assistant markdown output for presentation.

    Guarantees:
    1. Removes internal SVG/anchor artifacts like:
       - [svg](http://localhost:8501/#managing-transactions)
       - [svg](http://localhost:8501/#jdbc-example)
       - [svg](#some-anchor)
       - Bare [svg](...)
    2. Strips localhost URLs from links while preserving genuine anchor labels:
       - [Dashboard](http://localhost:8501/overview) -> Dashboard
    3. Preserves legitimate external markdown links:
       - [Python](https://python.org) -> [Python](https://python.org)
    4. Preserves fenced code blocks (```...```) and inline code (`...`) untouched.
    5. Preserves headings (#, ##, ###), lists, blockquotes, bold/italic, tables.
    6. Removes leftover double spaces or awkward spacing before punctuation.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # Protect code blocks (fenced by ``` or inline `) so code examples remain unchanged
    code_blocks = []

    def save_code(match):
        code_blocks.append(match.group(0))
        return f"__AURA_CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(r"(```[\s\S]*?```|`[^`\n]+`)", save_code, text)

    # 1. Remove [svg](...) or [SVG](...) or [anchor](...) links entirely
    text = re.sub(r"\[\s*(?:svg|SVG|anchor|link)\s*\]\([^)]*\)", "", text, flags=re.IGNORECASE)

    # 2. Remove [svg/anchor/link](http://localhost...) or empty label links to localhost
    text = re.sub(
        r"\[\s*(?:svg|SVG|anchor|link)?\s*\]\(\s*https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 3. For [Label](http://localhost:...), strip the localhost URL and keep just Label
    text = re.sub(r"\[([^\]]+)\]\(\s*https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)[^)]*\)", r"\1", text)

    # 4. Remove bare localhost anchor URLs
    text = re.sub(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0)(?::\d+)?(?:/[^\s)]*)?#[a-zA-Z0-9_\-]+", "", text)

    # 5. Clean up redundant spaces and whitespace before punctuation
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)

    # Restore code blocks
    for idx, cb in enumerate(code_blocks):
        text = text.replace(f"__AURA_CODE_BLOCK_{idx}__", cb)

    return text.strip()
