"""
Regression tests for Aura AI UI Background Image resolution and CSS generation.

Verifies:
1. Robust resolution of default provided anime galaxy AVIF image relative to project structure.
2. Graceful fallback when non-existent background path is configured.
3. Configurable override via APP_BACKGROUND_IMAGE.
4. Correct MIME type detection for AVIF and common formats.
5. Inlining of AVIF as base64 data URI in CSS.
6. Fallback gradient when no background image is available.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from ui.streamlit_app import (
    get_background_css,
    get_image_mime_type,
    resolve_background_image_path,
)


def test_default_background_resolution():
    """Verify that default provided AVIF background image is resolved robustly."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("APP_BACKGROUND_IMAGE", None)
        path = resolve_background_image_path()
        assert path is not None, "Failed to resolve default background image"
        assert path.exists(), f"Resolved path does not exist: {path}"
        assert path.is_file(), f"Resolved path is not a file: {path}"
        assert path.name == "anime-style-galaxy-background_23-2151133974.avif"


def test_configured_background_override(tmp_path):
    """Verify that an explicitly configured valid background path takes precedence."""
    dummy_img = tmp_path / "custom_bg.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    # Direct argument
    path = resolve_background_image_path(str(dummy_img))
    assert path == dummy_img

    # Via environment variable
    with patch.dict(os.environ, {"APP_BACKGROUND_IMAGE": str(dummy_img)}):
        path_env = resolve_background_image_path()
        assert path_env == dummy_img


def test_invalid_configured_background_fallback():
    """Verify that a non-existent configured path falls back to the default AVIF image."""
    with patch.dict(os.environ, {"APP_BACKGROUND_IMAGE": "non_existent_image_path.jpg"}):
        path = resolve_background_image_path()
        assert path is not None
        assert path.name == "anime-style-galaxy-background_23-2151133974.avif"


def test_avif_mime_type_detection():
    """Verify correct MIME types for AVIF, PNG, JPG, and WEBP."""
    assert get_image_mime_type(Path("test.avif")) == "image/avif"
    assert get_image_mime_type(Path("test.png")) == "image/png"
    assert get_image_mime_type(Path("test.jpg")) == "image/jpeg"
    assert get_image_mime_type(Path("test.jpeg")) == "image/jpeg"
    assert get_image_mime_type(Path("test.webp")) == "image/webp"


def test_get_background_css_contains_avif_data_uri():
    """Verify generated CSS embeds AVIF as base64 with translucent overlay."""
    css = get_background_css()
    assert "data:image/avif;base64," in css
    assert ".stApp, [data-testid=\"stAppViewContainer\"]" in css
    assert "radial-gradient" in css
    assert "linear-gradient" in css


def test_fallback_gradient_css_when_no_image_available():
    """Verify gradient fallback when image resolver returns None."""
    with patch("ui.streamlit_app.resolve_background_image_path", return_value=None):
        css = get_background_css()
        assert "radial-gradient" in css
        assert "#05080d" in css.lower()
        assert "data:image" not in css


def test_default_background_video_resolution():
    """Verify that default provided MP4 background video is resolved robustly."""
    from ui.streamlit_app import resolve_background_video_path

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("APP_BACKGROUND_VIDEO", None)
        path = resolve_background_video_path()
        assert path is not None, "Failed to resolve default background video"
        assert path.exists(), f"Resolved path does not exist: {path}"
        assert path.is_file(), f"Resolved path is not a file: {path}"
        assert path.name == "gemini_generated_video_44ac85bd_gwr_video_mvp.mp4"


def test_configured_background_video_override(tmp_path):
    """Verify that an explicitly configured video path takes precedence."""
    from ui.streamlit_app import resolve_background_video_path

    dummy_vid = tmp_path / "custom_bg.mp4"
    dummy_vid.write_bytes(b"\x00\x00\x00 ftypisom")

    path = resolve_background_video_path(str(dummy_vid))
    assert path == dummy_vid

    with patch.dict(os.environ, {"APP_BACKGROUND_VIDEO": str(dummy_vid)}):
        path_env = resolve_background_video_path()
        assert path_env == dummy_vid


def test_invalid_configured_background_video_fallback():
    """Verify that a non-existent configured video path falls back to default video."""
    from ui.streamlit_app import resolve_background_video_path

    with patch.dict(os.environ, {"APP_BACKGROUND_VIDEO": "non_existent_video.mp4"}):
        path = resolve_background_video_path()
        assert path is not None
        assert path.name == "gemini_generated_video_44ac85bd_gwr_video_mvp.mp4"


def test_get_background_video_url_streaming_not_base64():
    """Verify that background video URL is a streamable path, never base64 data URI."""
    from ui.streamlit_app import get_background_video_url

    url = get_background_video_url()
    assert url is not None
    assert "data:video" not in url, "Background video should NOT be base64 encoded into HTML"
    assert url.startswith("/media/")


def test_glass_ui_hierarchy_rendered():
    """Verify that apply_custom_styles generates proper glass surface hierarchy and video background."""
    from ui.streamlit_app import apply_custom_styles

    rendered_markdowns = []
    with patch("streamlit.markdown", side_effect=lambda content, *args, **kwargs: rendered_markdowns.append(content)):
        apply_custom_styles()

    combined_output = "".join(rendered_markdowns)
    # Video container
    assert "id=\"aura-persistent-bg\"" in combined_output
    assert "id=\"aura-bg-video\"" in combined_output
    # Glass UI opacities
    assert "rgba(7, 9, 18, 0.88)" in combined_output  # Sidebar
    assert "rgba(6, 8, 17, 0.75)" in combined_output  # Header/topbar
    assert "rgba(18, 25, 34, 0.82)" in combined_output  # User message
    assert "rgba(12, 18, 26, 0.78)" in combined_output  # Assistant message
    assert "rgba(12, 18, 26, 0.90)" in combined_output  # Composer
    assert "rgba(10, 16, 24, 0.75)" in combined_output  # Source cards
    # Base colors
    assert "#05080d" in combined_output.lower()
    assert "#080c12" in combined_output.lower()
    assert "#0b1118" in combined_output.lower()

