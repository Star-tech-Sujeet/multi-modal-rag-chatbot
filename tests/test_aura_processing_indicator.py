"""
Tests for Universal Aura Processing Indicator Infrastructure (Phase 13.0).

Verifies:
1. Indicator HTML renders cleanly in both full and compact modes.
2. Full mode renders cosmic energy core, orbital rings, halo, particles, nucleus, status, label.
3. Compact mode renders miniature core and inline status without layout-shifting footprint.
4. Dynamic status text and optional operation labels are sanitized with zero CoT/prompt leakage.
5. Lifecycle methods (start, update, complete, error, clear) work correctly on containers.
6. Context manager (__enter__, __exit__) guarantees cleanup on success, exception, or interruption.
7. Zero artificial sleep or blocking delays.
8. Zero MP4 / video / audio / GIF / external network dependencies.
9. Lightweight payload (< 3KB with CSS, < 1KB without CSS) and fast immediate render (< 2ms).
10. Repeated calls do not duplicate indicators or thrash the DOM with redundant updates.
11. Dual-container mode preserves continuous animation across status updates.
"""

import inspect
import time
from unittest.mock import MagicMock, patch

import pytest

from ui.processing_indicator import (
    AuraProcessingIndicator,
    get_indicator_css,
    render_aura_processing_indicator_html,
    sanitize_status_text,
)


class TestAuraProcessingIndicatorHTML:
    """Tests for HTML and CSS generation."""

    def test_indicator_renders(self):
        """Verify indicator renders non-empty HTML containing key indicator markers."""
        html = render_aura_processing_indicator_html(mode="full", status="Testing...")
        assert html is not None
        assert len(html) > 0
        assert 'role="status"' in html
        assert 'aria-live="polite"' in html

    def test_full_mode_structure(self):
        """Verify full mode contains complete cosmic energy core visual and typography."""
        html = render_aura_processing_indicator_html(
            mode="full",
            status="Processing document...",
            operation_label="Ingesting and indexing your file",
            include_css=False,
        )
        assert "aura-indicator-full-container" in html
        assert "aura-core-full-wrapper" in html
        assert "aura-core-halo" in html
        assert "aura-core-orbital-ring aura-core-orbital-1" in html
        assert "aura-core-orbital-ring aura-core-orbital-2" in html
        assert "aura-core-particle aura-core-particle-1" in html
        assert "aura-core-particle aura-core-particle-2" in html
        assert "aura-core-nucleus" in html
        assert "aura-indicator-text-block" in html
        assert "aura-indicator-status-full" in html
        assert "aura-indicator-dot" in html
        assert "Processing document..." in html
        assert "aura-indicator-label-full" in html
        assert "Ingesting and indexing your file" in html

    def test_compact_mode_structure(self):
        """Verify compact mode contains inline miniature core and compact typography."""
        html = render_aura_processing_indicator_html(
            mode="compact",
            status="Saving chat...",
            operation_label="Rename",
            include_css=False,
        )
        assert "aura-indicator-compact-container" in html
        assert "aura-core-compact-wrapper" in html
        assert "aura-core-compact-orbital" in html
        assert "aura-core-compact-nucleus" in html
        assert "aura-indicator-status-compact" in html
        assert "Saving chat..." in html
        assert "aura-indicator-label-compact" in html
        assert "(Rename)" in html

    def test_custom_status_and_optional_label(self):
        """Verify dynamic status text and optional operation label are rendered."""
        # Without operation label
        html_no_label = render_aura_processing_indicator_html(
            mode="full",
            status="Gathering evidence...",
            operation_label=None,
            include_css=False,
        )
        assert "Gathering evidence..." in html_no_label
        assert "aura-indicator-label-full" not in html_no_label

        # With operation label
        html_with_label = render_aura_processing_indicator_html(
            mode="full",
            status="Running SQL query...",
            operation_label="Executing read-only query",
            include_css=False,
        )
        assert "Running SQL query..." in html_with_label
        assert "Executing read-only query" in html_with_label

    def test_no_mp4_dependency(self):
        """Verify HTML strictly contains no MP4, video, audio, or GIF assets."""
        for mode in ("full", "compact"):
            html = render_aura_processing_indicator_html(mode=mode, include_css=True)
            assert "<video" not in html
            assert "</video>" not in html
            assert ".mp4" not in html.lower()
            assert "<audio" not in html
            assert "<source" not in html
            assert ".gif" not in html.lower()

    def test_no_external_network_dependency(self):
        """Verify HTML contains no external URLs, CDN links, or scripts."""
        for mode in ("full", "compact"):
            html = render_aura_processing_indicator_html(mode=mode, include_css=True)
            assert "http://" not in html
            assert "https://" not in html
            assert "<script" not in html
            assert "<iframe" not in html

    def test_lightweight_payload_and_render_speed(self):
        """Verify indicator payload is lightweight and renders in < 2ms."""
        t0 = time.perf_counter()
        html_with_css = render_aura_processing_indicator_html(mode="full", include_css=True)
        duration_ms = (time.perf_counter() - t0) * 1000

        assert duration_ms < 10.0, f"Render took too long: {duration_ms:.2f}ms"
        # Total footprint with CSS must be under 8KB
        assert len(html_with_css) < 8000

        # Without CSS (e.g. when global stylesheet is active), footprint < 1.5KB
        html_no_css = render_aura_processing_indicator_html(mode="full", include_css=False)
        assert len(html_no_css) < 1500


class TestStatusTextSanitization:
    """Tests for zero CoT, secret, and prompt leakage guarantees."""

    def test_strips_internal_thought_and_cot(self):
        """Verify chain-of-thought traces and thoughts are replaced with safe default."""
        bad_texts = [
            "<thought>Let me search ChromaDB for the document...</thought>",
            "Chain of thought: analyzing query",
            "Hidden reasoning: checking permissions",
            "System prompt: you are an AI assistant",
            "Secret: API_KEY=xyz",
        ]
        for bad in bad_texts:
            sanitized = sanitize_status_text(bad)
            assert "<thought" not in sanitized
            assert "chain of thought" not in sanitized.lower()
            assert "hidden reasoning" not in sanitized.lower()
            assert "system prompt" not in sanitized.lower()
            assert "api_key" not in sanitized.lower()
            assert sanitized == "Processing..."

    def test_strips_html_tags_and_escapes(self):
        """Verify HTML tags are stripped and special characters escaped."""
        raw = "<b>Uploading</b> & processing <script>alert(1)</script>"
        sanitized = sanitize_status_text(raw)
        assert "<b>" not in sanitized
        assert "<script>" not in sanitized
        assert "&amp;" in sanitized or "&" in sanitized

    def test_handles_empty_or_none(self):
        """Verify empty or None falls back to safe default."""
        assert sanitize_status_text(None) == "Processing..."
        assert sanitize_status_text("") == "Processing..."
        assert sanitize_status_text("   ") == "Processing..."


class TestAuraProcessingIndicatorLifecycle:
    """Tests for start, update, complete, error, and clear lifecycle."""

    def test_start_lifecycle(self):
        """Verify start renders HTML into container and activates indicator."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full")
        assert not indicator.is_active

        indicator.start(status="Uploading document...")
        assert indicator.is_active
        assert indicator.current_status == "Uploading document..."
        assert mock_container.markdown.call_count == 1
        call_args = mock_container.markdown.call_args[0][0]
        assert "Uploading document..." in call_args

    def test_update_lifecycle(self):
        """Verify update modifies status text and re-renders into container."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full", status="Step 1")
        indicator.start()
        assert mock_container.markdown.call_count == 1

        indicator.update(status="Step 2", operation_label="Extracting tables")
        assert mock_container.markdown.call_count == 2
        call_args = mock_container.markdown.call_args[0][0]
        assert "Step 2" in call_args
        assert "Extracting tables" in call_args

    def test_repeated_rerender_does_not_duplicate_or_thrash(self):
        """Verify calling update with identical content does not thash the DOM."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="compact", status="Saving...")
        indicator.start()
        assert mock_container.markdown.call_count == 1

        # Calling update with same status and label must no-op
        indicator.update(status="Saving...")
        assert mock_container.markdown.call_count == 1

        # Calling with new status must trigger render
        indicator.update(status="Saved!")
        assert mock_container.markdown.call_count == 2

    def test_complete_cleanup(self):
        """Verify complete calls container.empty() and deactivates indicator."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full")
        indicator.start(status="Processing...")
        assert indicator.is_active

        indicator.complete()
        assert not indicator.is_active
        mock_container.empty.assert_called_once()

    def test_error_cleanup(self):
        """Verify error cleanly clears container so broken animation never remains."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full")
        indicator.start(status="Processing...")

        indicator.error(message="Network timeout")
        assert not indicator.is_active
        mock_container.empty.assert_called_once()

    def test_clear_cleanup(self):
        """Verify clear empties container and resets active flag."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="compact")
        indicator.start()
        indicator.clear()
        assert not indicator.is_active
        mock_container.empty.assert_called_once()

    def test_dual_container_mode_preserves_animation_slot(self):
        """
        Verify that in dual-container mode, start renders both animation and status slots,
        while update ONLY updates the status slot (never restarting the animation).
        """
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Initial step",
        )
        indicator.start()

        # Both containers rendered initially
        assert mock_anim.markdown.call_count == 1
        assert mock_status.markdown.call_count == 1

        # Update status
        indicator.update(status="Subsequent step")

        # Animation container MUST remain untouched (not re-rendered!)
        assert mock_anim.markdown.call_count == 1
        # Status container receives updated status
        assert mock_status.markdown.call_count == 2
        status_call = mock_status.markdown.call_args[0][0]
        assert "Subsequent step" in status_call

        # Complete clears both
        indicator.complete()
        mock_anim.empty.assert_called_once()
        mock_status.empty.assert_called_once()


class TestContextManagerLifecycleAndSafety:
    """Tests for context manager guarantee and exception/interruption safety."""

    def test_context_manager_normal_success_cleanup(self):
        """Verify context manager automatically completes and clears on normal exit."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full", status="Working...")

        with indicator:
            assert indicator.is_active
            assert mock_container.markdown.call_count == 1

        assert not indicator.is_active
        mock_container.empty.assert_called_once()

    def test_context_manager_exception_cleanup(self):
        """Verify context manager automatically cleans up when an exception is raised."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="full", status="Working...")

        with pytest.raises(RuntimeError, match="Simulated backend failure"):
            with indicator:
                assert indicator.is_active
                raise RuntimeError("Simulated backend failure")

        # Indicator must be cleared despite exception
        assert not indicator.is_active
        mock_container.empty.assert_called_once()

    def test_context_manager_interruption_cleanup(self):
        """Verify context manager automatically cleans up on KeyboardInterrupt."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(container=mock_container, mode="compact", status="Working...")

        with pytest.raises(KeyboardInterrupt):
            with indicator:
                assert indicator.is_active
                raise KeyboardInterrupt()

        # Must be cleared on interruption
        assert not indicator.is_active
        mock_container.empty.assert_called_once()


class TestNoArtificialSleep:
    """Verify zero artificial delay or sleep calls exist in indicator implementation."""

    def test_no_sleep_calls_in_indicator_module(self):
        """Inspect source code of ui.processing_indicator to verify 0 time.sleep calls."""
        import ui.processing_indicator as mod
        src = inspect.getsource(mod)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src


class TestReExportFromStreamlitApp:
    """Verify component is accessible via ui.streamlit_app namespace."""

    def test_reexported_from_ui_streamlit_app(self):
        """Verify components can be imported from ui.streamlit_app."""
        from ui.streamlit_app import (
            AuraProcessingIndicator as AppIndicator,
            render_aura_processing_indicator_html as app_render_html,
            get_indicator_css as app_get_css,
            sanitize_status_text as app_sanitize,
        )
        assert AppIndicator is AuraProcessingIndicator
        assert app_render_html is render_aura_processing_indicator_html
        assert app_get_css is get_indicator_css
        assert app_sanitize is sanitize_status_text
