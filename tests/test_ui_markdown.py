"""
Regression tests for Aura AI Markdown presentation sanitization.

Verifies:
1. Removal of [svg](http://localhost:8501/#...) and other SVG/anchor artifacts.
2. Stripping of localhost implementation URLs while keeping genuine text.
3. Preservation of legitimate external Markdown links.
4. Preservation of Markdown headings, code blocks, lists, and tables.
"""

import pytest
from ui.markdown_utils import clean_markdown_display


def test_svg_localhost_anchor_removal():
    """Verify that [svg](http://localhost:8501/#...) artifacts are stripped."""
    t1 = "Here is the architecture [svg](http://localhost:8501/#managing-transactions) for your review."
    cleaned1 = clean_markdown_display(t1)
    assert "[svg]" not in cleaned1
    assert "localhost:8501" not in cleaned1
    assert cleaned1 == "Here is the architecture for your review."

    t2 = "### Managing Transactions [svg](http://localhost:8501/#managing-transactions)\nDetails here."
    cleaned2 = clean_markdown_display(t2)
    assert "[svg]" not in cleaned2
    assert "### Managing Transactions" in cleaned2
    assert "Details here." in cleaned2

    t3 = "Refer to [svg](http://localhost:8501/#jdbc-example) for the code."
    cleaned3 = clean_markdown_display(t3)
    assert "[svg]" not in cleaned3
    assert "localhost:8501" not in cleaned3


def test_bare_and_varied_svg_artifacts():
    """Verify handling of case-insensitive and relative SVG/anchor artifacts."""
    t1 = "Step 1 [SVG](#step-1): Do this."
    assert "[SVG]" not in clean_markdown_display(t1)
    assert "#step-1" not in clean_markdown_display(t1)

    t2 = "Anchor link: [anchor](http://127.0.0.1:8501/#sec1) done."
    assert "[anchor]" not in clean_markdown_display(t2)
    assert "127.0.0.1" not in clean_markdown_display(t2)


def test_localhost_url_stripping_preserves_label():
    """Verify that localhost URLs are removed while user-facing text labels remain."""
    text = "Visit the [Cluster Health Dashboard](http://localhost:8501/health) for details."
    cleaned = clean_markdown_display(text)
    assert "http://localhost:8501/health" not in cleaned
    assert "Visit the Cluster Health Dashboard for details." in cleaned


def test_safe_external_links_preserved():
    """Verify that genuine external links are strictly preserved."""
    text = "Check the [Official Documentation](https://docs.oracle.com/en/java) or [Python](https://python.org)."
    cleaned = clean_markdown_display(text)
    assert "[Official Documentation](https://docs.oracle.com/en/java)" in cleaned
    assert "[Python](https://python.org)" in cleaned


def test_code_blocks_preserved():
    """Verify that code blocks containing svg or localhost strings are not mangled."""
    code_text = (
        "Here is an example:\n\n"
        "```python\n"
        "# Local endpoint test\n"
        "url = 'http://localhost:8501/#test'\n"
        "artifact = '[svg](http://localhost:8501/#test)'\n"
        "print(artifact)\n"
        "```\n\n"
        "Outside code: [svg](http://localhost:8501/#managing-transactions) should be gone."
    )
    cleaned = clean_markdown_display(code_text)
    assert "url = 'http://localhost:8501/#test'" in cleaned
    assert "artifact = '[svg](http://localhost:8501/#test)'" in cleaned
    assert "Outside code: should be gone." in cleaned or "Outside code: should be gone" in cleaned


def test_headings_lists_and_tables_preserved():
    """Verify standard Markdown formatting remains fully functional."""
    md = (
        "# Title\n\n"
        "## Subtitle\n\n"
        "- Item 1\n"
        "- Item 2\n\n"
        "| Col 1 | Col 2 |\n"
        "| --- | --- |\n"
        "| Val 1 | Val 2 |\n\n"
        "> Blockquote citation [S1]."
    )
    cleaned = clean_markdown_display(md)
    assert "# Title" in cleaned
    assert "## Subtitle" in cleaned
    assert "- Item 1" in cleaned
    assert "| Col 1 | Col 2 |" in cleaned
    assert "> Blockquote citation [S1]." in cleaned


def test_empty_and_none_input():
    """Verify graceful handling of empty or non-string inputs."""
    assert clean_markdown_display("") == ""
    assert clean_markdown_display(None) == ""
    assert clean_markdown_display("   ") == ""
