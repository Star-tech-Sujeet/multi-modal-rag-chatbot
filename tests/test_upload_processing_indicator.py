"""
Tests for Phase 13.2 — Aura Processing Indicator in Document Upload & Ingestion.

Verifies:
1. Document upload immediately starts the AuraProcessingIndicator.
2. Indicator appears immediately before network transfer.
3. Truthful status progression ("Uploading document…", "Processing document…", "Finalizing…").
4. Processing indicator remains stable during updates.
5. Successful ingestion clears the indicator and refreshes document state.
6. Upload error (e.g. 400 invalid type) clears the indicator.
7. Ingestion error (e.g. 500 processing failure) clears the indicator.
8. Interruption / exception cleanup is guaranteed via try/finally.
9. Exactly one indicator is created per upload (no duplicate containers).
10. Existing document metadata behavior and session state are preserved.
11. Zero artificial sleep or blocking delays in upload_document.
"""

import inspect
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import APIError, upload_document


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, name: str = "annual_report.pdf", content: bytes = b"%PDF-1.4 test data", mime: str = "application/pdf"):
        self.name = name
        self._content = content
        self.type = mime

    def getvalue(self) -> bytes:
        return self._content


class TestUploadProcessingIndicatorFlow:
    """Verifies the upload & ingestion indicator lifecycle."""

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_upload_starts_indicator(self, mock_make_api, mock_st):
        """Verify upload immediately initializes and starts the indicator."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.return_value = {
            "file_id": "test-uuid-123",
            "filename": "annual_report.pdf",
            "message": "File processed successfully with 12 chunks.",
        }

        file = MockUploadedFile()
        res = upload_document(file)

        assert res is True
        # Verify container was allocated and markdown called
        assert mock_st.empty.called
        assert mock_container.markdown.called

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_indicator_appears_immediately(self, mock_make_api, mock_st):
        """Verify indicator starts with initial status before API call occurs."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        call_order = []

        def track_empty():
            call_order.append("st.empty")
            return mock_container

        def track_api(*args, **kwargs):
            call_order.append("make_api_request")
            return {
                "file_id": "test-uuid-123",
                "filename": "doc.pdf",
                "message": "Success",
            }

        mock_st.empty.side_effect = track_empty
        mock_make_api.side_effect = track_api

        upload_document(MockUploadedFile())

        # st.empty must be called before make_api_request
        assert "st.empty" in call_order
        assert "make_api_request" in call_order
        assert call_order.index("st.empty") < call_order.index("make_api_request")

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_truthful_status_progression(self, mock_make_api, mock_st):
        """Verify status progression follows truthful stages without fake percentages."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        captured_renders = []

        def capture_markdown(html_content, **kwargs):
            captured_renders.append(html_content)

        mock_container.markdown.side_effect = capture_markdown

        mock_make_api.return_value = {
            "file_id": "test-uuid-123",
            "filename": "doc.pdf",
            "message": "Indexed 8 chunks",
        }

        upload_document(MockUploadedFile())

        all_rendered_text = " ".join(captured_renders)
        # Verify truthful stages
        assert "Uploading document…" in all_rendered_text
        assert "Processing document…" in all_rendered_text
        assert "Finalizing…" in all_rendered_text
        # Strict: no fake progress percentage
        assert "%" not in all_rendered_text.replace("&#37;", "").replace("%PDF", "")

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_successful_ingestion_clears_indicator(self, mock_make_api, mock_st):
        """Verify indicator is cleaned up on successful upload and success toast shown."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.return_value = {
            "file_id": "file-abc",
            "filename": "guide.pdf",
            "message": "Document ingested successfully",
        }

        res = upload_document(MockUploadedFile(name="guide.pdf"))

        assert res is True
        # Container must be emptied
        mock_container.empty.assert_called_once()
        # Success message displayed
        mock_st.success.assert_called_once()
        assert "Document ingested successfully" in mock_st.success.call_args[0][0]

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_upload_error_clears_indicator(self, mock_make_api, mock_st):
        """Verify APIError (e.g. 400 invalid type) clears indicator and displays error."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.side_effect = APIError("Invalid file type: .exe not supported", status_code=400)

        res = upload_document(MockUploadedFile(name="malicious.exe"))

        assert res is False
        # Container must be emptied in finally
        mock_container.empty.assert_called_once()
        mock_st.error.assert_called_once()
        assert "Upload failed" in mock_st.error.call_args[0][0]

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_ingestion_error_clears_indicator(self, mock_make_api, mock_st):
        """Verify 500 ingestion error clears indicator cleanly."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.side_effect = APIError("ChromaDB connection timeout during embedding", status_code=500)

        res = upload_document(MockUploadedFile())

        assert res is False
        mock_container.empty.assert_called_once()
        mock_st.error.assert_called_once()

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_interruption_exception_cleanup(self, mock_make_api, mock_st):
        """Verify unexpected exception or interruption unconditionally cleans up loader."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            upload_document(MockUploadedFile())

        mock_container.empty.assert_called_once()

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_no_duplicate_indicator(self, mock_make_api, mock_st):
        """Verify exactly one st.empty container is allocated for upload indicator."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.return_value = {
            "file_id": "file-1",
            "filename": "doc.pdf",
            "message": "Done",
        }

        upload_document(MockUploadedFile())

        assert mock_st.empty.call_count == 1

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_existing_document_metadata_behavior_unchanged(self, mock_make_api, mock_st):
        """Verify document metadata and cache invalidation are preserved."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.return_value = {
            "file_id": "file-uuid-999",
            "filename": "sales_q3.csv",
            "message": "Processed 100 rows",
        }

        upload_document(MockUploadedFile(name="sales_q3.csv"))

        assert mock_st.session_state.file_id == "file-uuid-999"
        assert mock_st.session_state.uploaded_filename == "sales_q3.csv"
        assert mock_st.session_state.chat_history == []
        assert mock_st.session_state._files_cache is None
        assert mock_st.session_state._files_cache_time == 0

    def test_no_sleep_calls_in_upload_document(self):
        """Strictly verify zero time.sleep or asyncio.sleep calls exist in upload_document."""
        src = inspect.getsource(upload_document)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src


class TestCompactDocumentUploadProcessingUI:
    """Verifies the compact, inline document upload processing presentation requirements."""

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_upload_uses_compact_mode_not_full_card(self, mock_make_api, mock_st):
        """Verify indicator uses compact mode and never renders large/full card."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        captured_renders = []
        mock_container.markdown.side_effect = lambda html, **kw: captured_renders.append(html)

        mock_make_api.return_value = {
            "file_id": "test-uuid-456",
            "filename": "jarvis-agent-architecture-v2.docx",
            "message": "Indexed successfully",
        }

        # 42.7 KB payload (42.7 * 1024 = 43725 bytes)
        content_42k = b"x" * int(42.7 * 1024)
        file = MockUploadedFile(
            name="jarvis-agent-architecture-v2.docx",
            content=content_42k,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        res = upload_document(file)
        assert res is True

        all_html = " ".join(captured_renders)

        # 1 & 2. Compact mode container must be present
        assert "aura-indicator-compact-container" in all_html
        assert "aura-upload-processing-container" in all_html
        assert "aura-core-compact-wrapper" in all_html

        # 3. Large/full card must NOT be present
        assert "aura-indicator-full-container" not in all_html
        assert "aura-core-full-wrapper" not in all_html

        # 4 & 5. Processing text and secondary indexing text must be present
        assert "Processing document…" in all_html
        assert "Indexing content and embeddings" in all_html

        # 6 & 7. Filename and file size must remain visible
        assert "jarvis-agent-architecture-v2.docx" in all_html
        assert "42.7 KB" in all_html

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_compact_indicator_clears_on_success(self, mock_make_api, mock_st):
        """Verify compact indicator is cleared immediately on success."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.return_value = {"file_id": "id1", "filename": "doc.pdf", "message": "Success"}
        res = upload_document(MockUploadedFile())
        assert res is True
        mock_container.empty.assert_called_once()

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_compact_indicator_clears_on_api_error(self, mock_make_api, mock_st):
        """Verify compact indicator is cleared immediately on API error without getting stuck."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.side_effect = APIError("Ingestion failure", status_code=500)
        res = upload_document(MockUploadedFile())
        assert res is False
        mock_container.empty.assert_called_once()

    @patch("ui.streamlit_app.st")
    @patch("ui.streamlit_app.make_api_request")
    def test_compact_indicator_clears_on_interruption(self, mock_make_api, mock_st):
        """Verify compact indicator is cleared on interruption."""
        mock_container = MagicMock()
        mock_st.empty.return_value = mock_container
        mock_st.session_state = MagicMock()

        mock_make_api.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            upload_document(MockUploadedFile())
        mock_container.empty.assert_called_once()

    def test_source_guarantees_compact_and_no_sleep(self):
        """Verify upload_document source code uses compact mode and contains zero sleep delays."""
        src = inspect.getsource(upload_document)
        assert 'mode="compact"' in src
        assert 'mode="full"' not in src
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src

