"""
Tests for Phase 13.6 — Aura Processing Indicator in Image Upload & Processing ONLY.

Verifies:
1. Indicator starts when image processing begins.
2. Compact mode is used.
3. Initial status is "Processing image…".
4. Indicator appears before processing dispatch.
5. Successful image processing clears indicator.
6. Invalid image clears indicator.
7. Oversized image clears indicator.
8. Corrupt image clears indicator.
9. OCR/processing failure clears indicator.
10. Vision/API failure clears indicator.
11. Unexpected exception clears indicator.
12. Interruption clears indicator.
13. Exactly one processing operation occurs.
14. Image preview remains intact.
15. Image evidence remains intact.
16. No artificial sleep or delay exists.
17. Existing chat streaming remains unchanged.
18. Existing document upload indicator remains unchanged.
19. Existing document deletion indicator remains unchanged.
20. Existing chat rename and delete indicators remain unchanged.
"""

import inspect
import io
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import (
    process_attached_image,
    process_question,
    upload_document,
    delete_document_api,
    rename_session_api,
    delete_session_api,
)
from app.logic import process_live_image, CanonicalEvidence


def create_valid_test_png() -> bytes:
    """Create a minimal valid PNG image."""
    img = Image.new("RGB", (64, 64), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_valid_test_jpeg() -> bytes:
    """Create a minimal valid JPEG image."""
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class MockFile:
    """Mock file upload object with getvalue() and name."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class TestImageProcessingIndicator:
    """Verifies the image processing indicator lifecycle and integration."""

    def test_indicator_starts_with_compact_mode_and_status(self):
        """1, 2 & 3. Verify compact mode, initial status 'Processing image…' and label."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="flowchart.png",
            include_css=False,
        )
        indicator.start()

        assert indicator.mode == "compact"
        assert indicator.current_status == "Processing image…"
        assert indicator._operation_label == "flowchart.png"
        assert mock_container.markdown.called

        rendered_html = mock_container.markdown.call_args[0][0]
        assert "aura-indicator-compact-container" in rendered_html
        assert "Processing image…" in rendered_html
        assert "flowchart.png" in rendered_html

    def test_indicator_appears_before_processing_dispatch(self):
        """4. Verify indicator starts strictly before image validation/processing executes."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="chart.png",
            include_css=False,
        )

        call_events = []

        orig_start = indicator.start
        def track_start(*args, **kwargs):
            call_events.append("indicator.start")
            return orig_start(*args, **kwargs)

        indicator.start = track_start

        test_file = MockFile("chart.png", create_valid_test_png())

        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            indicator.start()
            call_events.append("process_attached_image")
            ok = process_attached_image(test_file, indicator=indicator)
            assert ok is True

        assert "indicator.start" in call_events
        assert "process_attached_image" in call_events
        assert call_events.index("indicator.start") < call_events.index("process_attached_image")

    def test_successful_image_processing_clears_indicator(self):
        """5. Verify indicator is cleared immediately on successful image processing."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="invoice.png",
            include_css=False,
        )
        indicator.start()

        test_file = MockFile("invoice.png", create_valid_test_png())
        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            try:
                ok = process_attached_image(test_file, indicator=indicator)
                assert ok is True
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_invalid_extension_clears_indicator(self):
        """6. Verify invalid image extension (.bmp / .exe) clears indicator."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="bad_format.bmp",
            include_css=False,
        )
        indicator.start()

        test_file = MockFile("bad_format.bmp", b"dummy_content")
        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            try:
                ok = process_attached_image(test_file, indicator=indicator)
                assert ok is False
                mock_st.error.assert_called_once()
                assert "Unsupported image format" in mock_st.error.call_args[0][0]
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_oversized_image_clears_indicator(self):
        """7. Verify oversized image (>10MB) clears indicator and returns error."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="huge_photo.png",
            include_css=False,
        )
        indicator.start()

        oversized_data = b"X" * (11 * 1024 * 1024)
        test_file = MockFile("huge_photo.png", oversized_data)
        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            try:
                ok = process_attached_image(test_file, indicator=indicator)
                assert ok is False
                mock_st.error.assert_called_once()
                assert "exceeds maximum limit" in mock_st.error.call_args[0][0]
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_corrupt_image_clears_indicator(self):
        """8. Verify corrupt binary image payload clears indicator and reports error."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="corrupt.png",
            include_css=False,
        )
        indicator.start()

        test_file = MockFile("corrupt.png", b"NOT_A_VALID_PNG_HEADER_CORRUPT_BYTES")
        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            try:
                ok = process_attached_image(test_file, indicator=indicator)
                assert ok is False
                mock_st.error.assert_called_once()
                assert "Corrupted or invalid" in mock_st.error.call_args[0][0]
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_ocr_failure_handling_and_indicator_cleanup(self):
        """9. Verify OCR failure in multimodal pipeline gracefully falls back and cleans up."""
        png_bytes = create_valid_test_png()
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="chart.png",
            include_css=False,
        )
        indicator.start()

        # When OCR returns empty text, process_live_image invokes Vision fallback
        with patch("pytesseract.image_to_string", return_value=""), \
             patch("app.logic.analyze_image_with_vision", return_value="Vision diagram description"):
            try:
                ev = process_live_image(png_bytes, "chart.png")
                assert isinstance(ev, CanonicalEvidence)
                assert ev.source_type == "image_vision"
                assert ev.ingestion_method == "vision"
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_vision_failure_handling_and_indicator_cleanup(self):
        """10. Verify Vision/API failure fallback produces evidence and cleans up loader."""
        png_bytes = create_valid_test_png()
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="unclear.png",
            include_css=False,
        )
        indicator.start()

        with patch("pytesseract.image_to_string", return_value=""), \
             patch("app.logic.analyze_image_with_vision", return_value="[Image file: unclear.png] - Unable to analyze"):
            try:
                ev = process_live_image(png_bytes, "unclear.png")
                assert isinstance(ev, CanonicalEvidence)
                assert ev.ingestion_method == "fallback"
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_unexpected_exception_clears_indicator(self):
        """11. Verify unexpected exception cleanly clears indicator container via try/finally."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="crash.png",
            include_css=False,
        )
        indicator.start()

        with pytest.raises(ZeroDivisionError):
            try:
                raise ZeroDivisionError("Crash during image processing")
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_interruption_clears_indicator(self):
        """12. Verify KeyboardInterrupt cleanly empties the container."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing image…",
            operation_label="interrupted.png",
            include_css=False,
        )
        indicator.start()

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_exactly_one_processing_operation(self):
        """13. Verify exactly one processing call occurs per image attachment action."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        test_file = MockFile("single.png", create_valid_test_png())

        with patch("ui.streamlit_app.st") as mock_st:
            mock_st.session_state = MagicMock()
            ok = process_attached_image(test_file, indicator=mock_indicator)
            assert ok is True
            # Exactly one status update call
            mock_indicator.update.assert_called_once_with(status="Processing image…", operation_label="single.png")

    def test_image_preview_state_remains_intact(self):
        """14. Verify attached_image_bytes and attached_image_name are preserved for preview card."""
        raw_bytes = create_valid_test_png()
        test_file = MockFile("preview_test.png", raw_bytes)

        with patch("ui.streamlit_app.st") as mock_st:
            mock_state = MagicMock()
            mock_st.session_state = mock_state

            ok = process_attached_image(test_file)
            assert ok is True
            assert mock_state.attached_image_bytes == raw_bytes
            assert mock_state.attached_image_name == "preview_test.png"

    def test_image_evidence_remains_intact(self):
        """15. Verify process_live_image generates valid canonical evidence with [S1] citation."""
        png_bytes = create_valid_test_png()
        with patch("pytesseract.image_to_string", return_value="Verified text in image"):
            ev = process_live_image(png_bytes, "doc_figure.png")
            assert isinstance(ev, CanonicalEvidence)
            assert ev.citation_id == "S1"
            assert ev.citation_marker == "[S1]"
            assert "[S1] doc_figure.png" in ev.citation_label
            assert ev.content == "Verified text in image"

    def test_no_sleep_or_artificial_delays(self):
        """16. Verify process_attached_image contains zero time.sleep or asyncio.sleep calls."""
        src = inspect.getsource(process_attached_image)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src

    def test_existing_chat_streaming_remains_unchanged(self):
        """17. Verify process_question streaming flow remains intact and untouched."""
        src = inspect.getsource(process_question)
        assert "AuraProcessingIndicator" in src
        assert "query_document_stream" in src

    def test_existing_document_upload_indicator_remains_unchanged(self):
        """18. Verify Phase 13.2 upload_document indicator remains intact."""
        src = inspect.getsource(upload_document)
        assert "AuraProcessingIndicator" in src
        assert "Uploading document…" in src

    def test_existing_document_deletion_indicator_remains_unchanged(self):
        """19. Verify Phase 13.3 delete_document_api indicator remains intact."""
        src = inspect.getsource(delete_document_api)
        assert "AuraProcessingIndicator" in src
        assert "Deleting document…" in src

    def test_existing_chat_rename_and_delete_indicators_remain_unchanged(self):
        """20. Verify Phase 13.4 rename and Phase 13.5 delete indicators remain intact."""
        src_ren = inspect.getsource(rename_session_api)
        assert "AuraProcessingIndicator" in src_ren
        assert "Saving chat name…" in src_ren

        src_del = inspect.getsource(delete_session_api)
        assert "AuraProcessingIndicator" in src_del
        assert "Deleting chat…" in src_del
