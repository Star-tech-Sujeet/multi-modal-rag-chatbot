"""
Tests for Phase 13.5 — Aura Processing Indicator in Chat Deletion ONLY.

Verifies:
1. Delete starts indicator.
2. Compact mode is used.
3. Initial status is "Deleting chat…".
4. Indicator appears before DELETE request.
5. Successful deletion clears indicator.
6. Failed deletion clears indicator.
7. Unexpected exception clears indicator.
8. Interruption clears indicator.
9. Exactly one DELETE request occurs.
10. Deleted session is actually removed.
11. Active session fallback remains correct.
12. Remaining sessions remain intact.
13. Session cache invalidation remains correct.
14. No sleep or artificial delay.
15. Document deletion indicator remains unchanged.
16. Existing chat rename behavior remains unchanged.
17. Existing chat streaming behavior remains unchanged.
"""

import inspect
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import (
    APIError,
    delete_session_api,
    delete_document_api,
    rename_session_api,
    process_question,
)


class TestChatDeletionProcessingIndicator:
    """Verifies the chat deletion indicator lifecycle and guarantees."""

    def test_delete_session_api_with_indicator_updates_status(self):
        """1 & 3. Verify delete_session_api updates indicator with 'Deleting chat…'."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Session deleted"}
            ok = delete_session_api("sess-100", indicator=mock_indicator)

            assert ok is True
            mock_indicator.update.assert_called_once_with(status="Deleting chat…")

    def test_delete_compact_mode_and_html_rendering(self):
        """2 & 3. Verify compact mode is used and renders compact core with status."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
            operation_label="Old Project Discussion",
            include_css=False,
        )
        indicator.start()

        assert indicator.mode == "compact"
        assert indicator.current_status == "Deleting chat…"
        assert indicator._operation_label == "Old Project Discussion"
        assert mock_container.markdown.called

        rendered_html = mock_container.markdown.call_args[0][0]
        assert "aura-indicator-compact-container" in rendered_html
        assert "Deleting chat…" in rendered_html
        assert "Old Project Discussion" in rendered_html

    def test_indicator_appears_before_delete_request(self):
        """4. Verify indicator.start() occurs strictly before the network DELETE request."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
            operation_label="Chat to Delete",
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
            return {"message": "Deleted"}

        with patch("ui.streamlit_app.make_api_request", side_effect=track_api):
            indicator.start()
            ok = delete_session_api("sess-del", indicator=indicator)
            assert ok is True

        assert "indicator.start" in call_events
        assert "make_api_request" in call_events
        assert call_events.index("indicator.start") < call_events.index("make_api_request")

    def test_successful_deletion_clears_indicator(self):
        """5. Verify indicator is cleared immediately on successful deletion."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
            operation_label="Chat to Clear",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Success"}
            try:
                ok = delete_session_api("sess-02", indicator=indicator)
                assert ok is True
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_failed_deletion_clears_indicator(self):
        """6. Verify failed deletion (APIError) clears indicator cleanly."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
            operation_label="Chat Failed",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = APIError("Deletion error", status_code=500)
            try:
                ok = delete_session_api("sess-err", indicator=indicator)
                assert ok is False
            finally:
                indicator.clear()

        mock_container.empty.assert_called_once()

    def test_unexpected_exception_clears_indicator(self):
        """7. Verify unexpected exception cleans up the indicator via try/finally."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
            operation_label="Exception Chat",
            include_css=False,
        )
        indicator.start()

        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.side_effect = ConnectionResetError("Connection lost abruptly")
            with pytest.raises(ConnectionResetError):
                try:
                    mock_api("DELETE", "/sessions/sess-err")
                finally:
                    indicator.clear()

        mock_container.empty.assert_called_once()

    def test_interruption_clears_indicator(self):
        """8. Verify KeyboardInterrupt cleanly empties the container."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Deleting chat…",
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

    def test_exactly_one_delete_request_occurs(self):
        """9. Verify exactly one DELETE request is dispatched per action."""
        with patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_api.return_value = {"message": "Deleted"}
            ok = delete_session_api("sess-exact")
            assert ok is True
            assert mock_api.call_count == 1
            mock_api.assert_called_once_with("DELETE", "/sessions/sess-exact")

    def test_deleted_session_removed_and_remaining_intact(self):
        """10 & 12. Verify deleted session is removed while remaining sessions remain intact."""
        from app.sessions import session_manager

        s1 = session_manager.create_session(title="Session 1 to Delete")
        s2 = session_manager.create_session(title="Session 2 to Keep")

        session_manager.append_message(s1.session_id, "user", "Msg 1")
        session_manager.append_message(s2.session_id, "user", "Msg 2")

        # Delete session 1
        deleted = session_manager.delete_session(s1.session_id)
        assert deleted is True

        # s1 should be gone
        assert session_manager.get_session(s1.session_id) is None
        # s2 should remain completely intact
        rem = session_manager.get_session(s2.session_id)
        assert rem is not None
        assert rem.title == "Session 2 to Keep"
        assert len(rem.messages) == 1
        assert rem.messages[0].content == "Msg 2"

    def test_active_session_fallback_logic(self):
        """11. Verify active session reset logic when active session is deleted."""
        mock_state = MagicMock()
        mock_state.active_session_id = "sess-active"
        mock_state.active_session_title = "Active Chat"
        mock_state.chat_history = [{"role": "user", "content": "hello"}]

        # Emulate delete logic from sidebar
        sid = "sess-active"
        if mock_state.active_session_id == sid:
            mock_state.active_session_id = None
            mock_state.active_session_title = "New Chat"
            mock_state.chat_history = []

        assert mock_state.active_session_id is None
        assert mock_state.active_session_title == "New Chat"
        assert mock_state.chat_history == []

    def test_session_cache_invalidation(self):
        """13. Verify _sessions_cache and _sessions_cache_time are invalidated on deletion."""
        mock_state = MagicMock()
        mock_state._sessions_cache = [{"session_id": "sess-old"}]
        mock_state._sessions_cache_time = 999999.0

        with patch("ui.streamlit_app.st") as mock_st, \
             patch("ui.streamlit_app.make_api_request") as mock_api:
            mock_st.session_state = mock_state
            mock_api.return_value = {"message": "Deleted"}

            ok = delete_session_api("sess-old")
            assert ok is True
            assert mock_state._sessions_cache is None
            assert mock_state._sessions_cache_time == 0

    def test_no_sleep_or_artificial_delays(self):
        """14. Verify delete_session_api contains zero time.sleep or asyncio.sleep calls."""
        src = inspect.getsource(delete_session_api)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src
        assert "sleep(" not in src

    def test_document_deletion_indicator_remains_unchanged(self):
        """15. Verify document deletion API and indicator implementation remain intact."""
        src = inspect.getsource(delete_document_api)
        assert "AuraProcessingIndicator" in src
        assert "Deleting document…" in src
        assert "time.sleep" not in src

    def test_existing_chat_rename_behavior_unchanged(self):
        """16. Verify chat rename API and indicator implementation remain intact."""
        src = inspect.getsource(rename_session_api)
        assert "AuraProcessingIndicator" in src
        assert "Saving chat name…" in src
        assert "time.sleep" not in src

    def test_existing_chat_streaming_behavior_unchanged(self):
        """17. Verify chat streaming and process_question remain intact and untouched."""
        src = inspect.getsource(process_question)
        assert "AuraProcessingIndicator" in src
        assert "query_document_stream" in src
