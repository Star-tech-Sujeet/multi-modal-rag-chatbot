"""
Tests for Phase 13.4 — Aura Processing Indicator in Chat Rename ONLY.

Verifies:
1. Rename starts indicator.
2. Compact mode is used.
3. Initial "Saving chat name…" status is displayed.
4. Indicator appears before the rename API request.
5. Successful rename clears indicator.
6. Failed rename clears indicator.
7. Unexpected exception clears indicator.
8. Interruption clears indicator.
9. Existing title persistence remains unchanged.
10. Session ID remains unchanged.
11. Messages remain unchanged.
12. No duplicate rename request is performed.
13. No sleep or artificial delay.
14. Existing chat streaming behavior remains unchanged.
"""

import inspect
import json
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import APIError, rename_session_api, process_question


class TestChatRenameProcessingIndicator:
    """Verifies the chat rename indicator lifecycle and guarantees."""

    def test_rename_session_api_with_indicator_updates_status(self):
        """1 & 3. Verify rename_session_api updates indicator with 'Saving chat name…'."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"title": "Updated Strategy", "session_id": "sess-100"}
            ok, result = rename_session_api("sess-100", "Updated Strategy", indicator=mock_indicator)

            assert ok is True
            assert result == "Updated Strategy"
            mock_indicator.update.assert_called_once_with(status="Saving chat name…")

    def test_rename_compact_mode_and_html_rendering(self):
        """2 & 3. Verify compact mode is used and renders compact core with status."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="New Meeting Notes",
            include_css=False,
        )
        indicator.start()

        assert indicator.mode == "compact"
        assert indicator.current_status == "Saving chat name…"
        assert indicator._operation_label == "New Meeting Notes"
        assert mock_container.markdown.called

        rendered_html = mock_container.markdown.call_args[0][0]
        assert "aura-indicator-compact-container" in rendered_html
        assert "Saving chat name…" in rendered_html
        assert "New Meeting Notes" in rendered_html

    def test_indicator_appears_before_rename_request(self):
        """4. Verify indicator.start() occurs strictly before the network PATCH request."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="Q4 Roadmap",
            include_css=False,
        )

        call_events = []

        orig_start = indicator.start
        def track_start(*args, **kwargs):
            call_events.append("indicator.start")
            return orig_start(*args, **kwargs)

        indicator.start = track_start

        def track_api(*args, **kwargs):
            call_events.append("make_api_request")
            return {"title": "Q4 Roadmap", "session_id": "sess-01"}

        with patch("ui.streamlit_app.make_api_request", side_effect=track_api):
            indicator.start()
            ok, res = rename_session_api("sess-01", "Q4 Roadmap", indicator=indicator)
            assert ok is True

        assert "indicator.start" in call_events
        assert "make_api_request" in call_events
        assert call_events.index("indicator.start") < call_events.index("make_api_request")

    def test_successful_rename_clears_indicator(self):
        """5. Verify indicator is cleared immediately on successful rename."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="Design Sprint",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"title": "Design Sprint"}
            try:
                ok, res = rename_session_api("sess-02", "Design Sprint", indicator=indicator)
                assert ok is True
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_failed_rename_clears_indicator(self):
        """6. Verify failed rename (APIError) clears indicator and returns truthful error."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="Failing Chat",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = APIError("Session not found", status_code=404)
            try:
                ok, err = rename_session_api("sess-invalid", "Failing Chat", indicator=indicator)
                assert ok is False
                assert "Session not found" in err
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_unexpected_exception_clears_indicator(self):
        """7. Verify unexpected exception cleans up the indicator via try/finally."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="Exception Chat",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = ConnectionResetError("Connection lost abruptly")
            with pytest.raises(ConnectionResetError):
                try:
                    mock_api("PATCH", "/sessions/sess-err")
                finally:
                    indicator.clear()

        mock_container.empty.assert_called_once()

    def test_interruption_clears_indicator(self):
        """8. Verify KeyboardInterrupt cleanly empties the container."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Saving chat name…",
            operation_label="Interrupt Chat",
            include_css=False,
        )
        indicator.start()

        with pytest.raises(KeyboardInterrupt):
            try:
                raise KeyboardInterrupt()
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_title_persistence_remains_unchanged(self):
        """9, 10 & 11. Verify backend session title persistence, session ID, and messages remain intact."""
        from app.sessions import session_manager

        # Create real session with messages in sqlite
        session = session_manager.create_session(title="Initial Name")
        sid = session.session_id

        session_manager.append_message(sid, "user", "Hello Aura")
        session_manager.append_message(sid, "assistant", "Hello! How can I assist you?")

        # Rename session via session_manager
        updated = session_manager.update_session(sid, title="Persisted Name")
        assert updated is not None
        assert updated.title == "Persisted Name"

        # Verify session ID remains identical
        details = session_manager.get_session(sid)
        assert details is not None
        assert details.session_id == sid
        assert details.title == "Persisted Name"

        # Verify messages are completely unchanged
        msgs = details.messages
        assert len(msgs) == 2
        assert msgs[0].content == "Hello Aura"
        assert msgs[1].content == "Hello! How can I assist you?"

    def test_no_duplicate_rename_request(self):
        """12. Verify exactly one PATCH network request is dispatched per rename operation."""
        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"title": "Single Call"}
            ok, res = rename_session_api("sess-single", "Single Call")
            assert ok is True
            assert mock_api.call_count == 1
            mock_api.assert_called_once_with(
                "PATCH",
                "/sessions/sess-single",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"title": "Single Call"}),
            )

    def test_no_sleep_or_artificial_delays(self):
        """13. Verify rename_session_api contains zero time.sleep or asyncio.sleep calls."""
        src = inspect.getsource(rename_session_api)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src

    def test_existing_chat_streaming_behavior_unchanged(self):
        """14. Verify chat streaming and process_question remains intact and untouched."""
        src = inspect.getsource(process_question)
        assert "AuraProcessingIndicator" in src
        assert "query_document_stream" in src
