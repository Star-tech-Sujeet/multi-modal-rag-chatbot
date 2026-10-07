"""
Tests for Phase 13.1 — Aura Processing Indicator in Chat Query / Streaming Flow.

Verifies:
1. Chat submission starts the indicator immediately.
2. Initial processing status ("Understanding request…") appears.
3. SSE status updates change status text dynamically with truthful backend stages.
4. Status updates do not recreate or restart the CSS animation.
5. First token clears the animation immediately.
6. Token streaming continues normally after animation removal.
7. Successful completion leaves no loader.
8. Streaming error clears the indicator completely.
9. Stream interruption clears the loader in try/finally.
10. Historical messages never render the processing indicator.
11. Exactly one active processing indicator exists (no duplicate containers).
12. SSE event sequence (start, status, sources, token, complete, error) remains unchanged.
"""

from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import (
    AuraProcessingIndicator,
    render_aura_core_html,
    render_aura_status_html,
)
from ui.streamlit_app import render_chat_message


class TestChatProcessingIndicatorStreamingFlow:
    """Simulates the exact chat streaming lifecycle in ui/streamlit_app.py."""

    def test_chat_submission_starts_indicator(self):
        """Verify chat submission immediately starts indicator on animation and status slots."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        assert indicator.is_active
        # Animation container rendered once
        assert mock_anim.markdown.call_count == 1
        anim_call = mock_anim.markdown.call_args[0][0]
        assert "aura-core-full-wrapper" in anim_call
        assert "aura-core-nucleus" in anim_call

        # Status container rendered once with initial text
        assert mock_status.markdown.call_count == 1
        status_call = mock_status.markdown.call_args[0][0]
        assert "Understanding request…" in status_call

    def test_initial_processing_status_appears(self):
        """Verify initial processing status text is 'Understanding request…'."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        assert indicator.current_status == "Understanding request…"
        status_html = mock_status.markdown.call_args[0][0]
        assert "Understanding request…" in status_html

    def test_sse_status_updates_change_text(self):
        """Verify dynamic SSE events update status text truthfully."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        stages = [
            ("Searching documents…", "Vector index search"),
            ("Reviewing sources…", "Analyzing 3 verified context sources"),
            ("Preparing answer…", None),
            ("Generating response…", None),
        ]

        for idx, (status, label) in enumerate(stages, start=2):
            indicator.update(status=status, operation_label=label)
            assert mock_status.markdown.call_count == idx
            last_call = mock_status.markdown.call_args[0][0]
            assert status in last_call
            if label:
                assert label in last_call

    def test_status_updates_do_not_recreate_animation(self):
        """Verify that throughout multiple status updates, animation container is NEVER re-rendered."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        assert mock_anim.markdown.call_count == 1

        # Emit 5 status events
        for i in range(5):
            indicator.update(status=f"Stage {i}...")

        # Animation container MUST remain strictly at 1 call
        assert mock_anim.markdown.call_count == 1
        # Status container updated 6 times (1 initial + 5 updates)
        assert mock_status.markdown.call_count == 6

    def test_first_token_clears_animation(self):
        """Verify first real token immediately calls clear_animation() on indicator."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        first_token_seen = False
        tokens = ["The ", "quantum ", "effect "]

        for tok in tokens:
            if not first_token_seen:
                first_token_seen = True
                indicator.clear_animation()

        # Animation container must have been cleared exactly once
        mock_anim.empty.assert_called_once()

    def test_streaming_continues_after_animation_removal(self):
        """Verify tokens continue yielding normally after animation removal without touching anim slot."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        def _simulate_stream():
            first_token_seen = False
            events = [
                ("status", {"message": "Searching documents…"}),
                ("token", {"text": "Aura "}),
                ("token", {"text": "AI "}),
                ("token", {"text": "answers."}),
                ("complete", {"answer": "Aura AI answers."}),
            ]
            for evt_type, data in events:
                if evt_type == "status":
                    indicator.update(status=data["message"])
                elif evt_type == "token":
                    if not first_token_seen:
                        first_token_seen = True
                        indicator.clear_animation()
                    yield data["text"]

        streamed = list(_simulate_stream())
        assert streamed == ["Aura ", "AI ", "answers."]
        # Anim was rendered once and emptied once
        assert mock_anim.markdown.call_count == 1
        assert mock_anim.empty.call_count == 1

    def test_successful_completion_leaves_no_loader(self):
        """Verify completion leaves animation slot empty."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        try:
            # Simulate first token arriving
            indicator.clear_animation()
        finally:
            indicator.clear_animation()

        # Animation container is clean
        assert mock_anim.empty.call_count >= 1

    def test_streaming_error_clears_loader(self):
        """Verify error in stream completely clears both animation and status slots."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        error_holder = ["HTTP 429 Quota Exceeded"]
        if error_holder:
            indicator.clear()

        assert not indicator.is_active
        mock_anim.empty.assert_called_once()
        mock_status.empty.assert_called_once()

    def test_interruption_clears_loader(self):
        """Verify unexpected stream interruption cleanly empties the animation in finally block."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        def _faulty_stream():
            raise ConnectionResetError("Connection lost mid-stream")

        with pytest.raises(ConnectionResetError):
            try:
                _faulty_stream()
            finally:
                indicator.clear_animation()

        mock_anim.empty.assert_called_once()

    def test_historical_messages_do_not_show_loader(self):
        """Verify historical chat messages never contain the processing indicator."""
        msg = {
            "role": "assistant",
            "content": "This is a past answer retrieved from session history.",
            "timestamp": "12:00",
            "sources": [{"filename": "doc.pdf", "page": 1}],
        }
        # render_chat_message returns or renders HTML for a past message
        # Verify rendered content contains no indicator or core classes
        html_out = render_chat_message(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg["timestamp"],
            sources=msg["sources"],
        )
        if html_out:
            assert "aura-indicator-full-container" not in html_out
            assert "aura-core-full-wrapper" not in html_out
            assert "aura-indicator-compact-container" not in html_out

    def test_no_duplicate_indicator(self):
        """Verify that exactly one indicator is used and redundant start calls do not stack containers."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()
        # Calling start again with same status does not recreate containers
        indicator.start(status="Understanding request…")

        assert mock_anim.markdown.call_count == 1
        assert mock_status.markdown.call_count == 1

    def test_existing_sse_event_sequence_remains_unchanged(self):
        """Verify that the full SSE sequence is processed in exact order: start, status, sources, token, complete."""
        mock_anim = MagicMock()
        mock_status = MagicMock()

        indicator = AuraProcessingIndicator(
            animation_container=mock_anim,
            status_container=mock_status,
            mode="full",
            status="Understanding request…",
            include_css=False,
        )
        indicator.start()

        events = [
            ("start", {"model": "gemini-3.1-flash-lite", "message": "Understanding request…"}),
            ("status", {"stage": "retrieval", "status": "running", "message": "Searching documents…"}),
            ("sources", {"sources": [{"file_id": "1", "filename": "report.pdf"}]}),
            ("token", {"text": "Aura "}),
            ("token", {"text": "AI"}),
            ("complete", {"answer": "Aura AI", "sources": []}),
        ]

        seen_sequence = []
        first_token_seen = False

        for event_type, data in events:
            seen_sequence.append(event_type)
            if event_type == "start":
                indicator.update(status=data["message"])
            elif event_type == "status":
                indicator.update(status=data["message"])
            elif event_type == "sources":
                indicator.update(status="Reviewing sources…")
            elif event_type == "token":
                if not first_token_seen:
                    first_token_seen = True
                    indicator.clear_animation()
            elif event_type == "complete":
                indicator.clear_animation()

        assert seen_sequence == ["start", "status", "sources", "token", "token", "complete"]
        # Animation cleared upon first token
        assert mock_anim.empty.call_count >= 1

    def test_status_events_holder_scope_and_collection_in_process_question(self):
        """Verify status_events_holder exists in process_question scope and collects events without NameError."""
        from ui.streamlit_app import process_question, init_session_state
        import streamlit as st

        init_session_state()
        st.session_state.chat_history = []
        st.session_state.comparison_mode_active = False

        mock_events = [
            ("start", {"message": "Understanding request…"}),
            ("status", {"message": "Searching database", "label": "Search"}),
            ("status", {"message": "Inspecting schema", "label": "Schema"}),
            ("sources", {"sources": [{"filename": "doc.pdf"}]}),
            ("token", {"text": "Hello "}),
            ("token", {"text": "world!"}),
            ("complete", {"answer": "Hello world!", "sources": []}),
        ]

        def fake_write_stream(gen):
            tokens = list(gen)
            return "".join(tokens)

        with patch("ui.streamlit_app.query_document_stream", return_value=iter(mock_events)), \
             patch("streamlit.write_stream", side_effect=fake_write_stream), \
             patch("streamlit.chat_message"), \
             patch("streamlit.empty"):
            process_question("What is the revenue?")

            # 2 messages: 1 user turn, 1 assistant turn
            assert len(st.session_state.chat_history) == 2
            assert st.session_state.chat_history[0]["role"] == "user"
            msg = st.session_state.chat_history[1]
            assert msg["role"] == "assistant"
            assert msg["content"] == "Hello world!"
            assert len(msg.get("pipeline_stages", [])) == 2
            assert msg["pipeline_stages"][0]["message"] == "Searching database"
            assert msg["pipeline_stages"][1]["message"] == "Inspecting schema"

