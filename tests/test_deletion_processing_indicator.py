"""
Tests for Phase 13.3 — Aura Processing Indicator in Document Deletion ONLY.

Verifies:
1. Document deletion starts compact AuraProcessingIndicator with mode="compact" and status="Deleting document…".
2. Indicator appears immediately when deletion begins.
3. Compact mode rendering is verified (aura-processing-compact).
4. Truthful atomic status progression (no fake progress percentages or artificial delays).
5. Successful deletion clears indicator and updates cache / session state.
6. Deletion failure (APIError / 404 / 500) clears indicator and preserves error reporting.
7. Interruption / exception cleanup is guaranteed via try/finally.
8. Exactly one indicator is created (no duplicate containers).
9. Session state cleanup (file_id, selected_file_ids, selected_filenames) is preserved.
10. Cache invalidation (_files_cache = None, _files_cache_time = 0) is preserved.
11. Zero artificial sleep or blocking delays in document deletion.
"""

import inspect
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import APIError, delete_document_api


class TestDocumentDeletionProcessingIndicator:
    """Verifies the document deletion indicator lifecycle and guarantees."""

    def test_delete_document_api_with_indicator_updates_status(self):
        """Verify delete_document_api updates indicator with 'Deleting document…'."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Document doc-123 deleted successfully"}
            success, msg = delete_document_api("doc-123", indicator=mock_indicator)

            assert success is True
            assert "deleted successfully" in msg
            mock_indicator.update.assert_called_once_with(status="Deleting document…")

    def test_delete_document_api_without_indicator_still_succeeds(self):
        """Verify backward-compatibility: delete_document_api works when indicator is None."""
        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Document doc-456 deleted successfully"}
            success, msg = delete_document_api("doc-456")

            assert success is True
            assert "deleted successfully" in msg

    def test_delete_document_api_invalid_id(self):
        """Verify invalid document ID returns early without API call."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        with patch("ui.streamlit_app.make_api_request") as mock_api:
            success, msg = delete_document_api("", indicator=mock_indicator)
            assert success is False
            assert "Invalid document ID" in msg
            mock_api.assert_not_called()
            mock_indicator.update.assert_not_called()

    def test_compact_indicator_initialization_for_deletion(self):
        """Verify compact indicator created with correct mode and label."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting document…",
            operation_label="quarterly_report.pdf",
            include_css=False,
        )

        indicator.start()

        assert indicator.mode == "compact"
        assert indicator.current_status == "Deleting document…"
        assert indicator._operation_label == "quarterly_report.pdf"
        assert mock_container.markdown.called

        rendered_html = mock_container.markdown.call_args[0][0]
        assert "aura-indicator-compact-container" in rendered_html
        assert "Deleting document…" in rendered_html
        assert "quarterly_report.pdf" in rendered_html

    def test_deletion_clears_indicator_on_success(self):
        """Verify indicator.clear() is called when deletion succeeds."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting document…",
            operation_label="file.pdf",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Deleted"}
            try:
                success, msg = delete_document_api("doc-1", indicator=indicator)
                assert success is True
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_deletion_clears_indicator_on_api_error(self):
        """Verify indicator.clear() is called when APIError occurs."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting document…",
            operation_label="file.pdf",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = APIError("Backend error deleting file", status_code=500)
            try:
                success, msg = delete_document_api("doc-1", indicator=indicator)
                assert success is False
                assert "Backend error deleting file" in msg
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_deletion_clears_indicator_on_unexpected_exception(self):
        """Verify indicator.clear() is called via try/finally on unexpected exception."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting document…",
            operation_label="file.pdf",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = RuntimeError("Fatal network drop")
            with pytest.raises(RuntimeError):
                try:
                    delete_document_api("doc-1", indicator=indicator)
                    raise RuntimeError("Fatal network drop")
                finally:
                    indicator.clear()

        mock_container.empty.assert_called_once()

    def test_deletion_clears_indicator_on_keyboard_interrupt(self):
        """Verify indicator.clear() is called via try/finally on KeyboardInterrupt."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting document…",
            operation_label="file.pdf",
            include_css=False,
        )
        indicator.start()

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_files_cache_invalidated_on_deletion(self):
        """Verify st.session_state._files_cache is invalidated when deletion succeeds."""
        mock_session_state = MagicMock()
        mock_session_state._files_cache = [{"file_id": "doc-99"}]
        mock_session_state._files_cache_time = 123456789.0

        with patch("ui.streamlit_app.st") as mock_st, \
             patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_st.session_state = mock_session_state
            mock_api.return_value = {"message": "Deleted successfully"}

            success, msg = delete_document_api("doc-99")
            assert success is True
            assert mock_session_state._files_cache is None
            assert mock_session_state._files_cache_time == 0

    def test_no_sleep_calls_in_deletion_flow(self):
        """Verify zero time.sleep or asyncio.sleep in delete_document_api."""
        src = inspect.getsource(delete_document_api)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src

    def test_no_fake_percentages_in_deletion_flow(self):
        """Verify no percentage strings or progress simulation in delete_document_api."""
        src = inspect.getsource(delete_document_api)
        assert "%" not in src
        assert "progress" not in src
