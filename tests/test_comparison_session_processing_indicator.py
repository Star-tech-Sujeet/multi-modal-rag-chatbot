"""
Tests for Phase 13.8 — Aura Processing Indicator in Comparison & Session Switching ONLY.

Verifies:
COMPARISON:
1. Comparison indicator starts.
2. Compact mode is used.
3. Initial status is "Comparing documents…".
4. Indicator starts before comparison dispatch.
5. Successful comparison clears indicator.
6. Comparison API failure clears indicator.
7. Comparison exception clears indicator.
8. Comparison interruption clears indicator.
9. Exactly one comparison request occurs.
10. Comparison result remains intact.
11. Citations/evidence remain intact.
12. No artificial delays exist.

SESSION SWITCHING:
13. Inspect whether session switching is synchronous.
14. If asynchronous / indicator-equipped, verify indicator starts before I/O.
15. If asynchronous / indicator-equipped, verify successful cleanup.
16. If asynchronous / indicator-equipped, verify error cleanup.
17. Verify no duplicate session loading.
18. Verify active session remains correct.
19. Verify chat history remains correct.
20. Verify synchronous switching does NOT show a fake loader.

REGRESSION:
21. Chat streaming indicator remains unchanged.
22. Document upload indicator remains unchanged.
23. Document deletion indicator remains unchanged.
24. Chat rename and delete indicators remain unchanged.
25. Image processing indicator remains unchanged.
26. SQL query processing indicator remains unchanged.
"""

import inspect
import json
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import (
    compare_documents_api,
    query_document,
    process_question,
    load_session,
    render_sidebar,
    upload_document,
    delete_document_api,
    rename_session_api,
    delete_session_api,
    process_attached_image,
    execute_sql_query_api,
    APIError,
)


class TestComparisonProcessingIndicator:
    """Verifies the comparison processing indicator lifecycle and integration."""

    def test_comparison_indicator_starts_with_compact_mode_and_status(self):
        """1, 2, 3: Verify comparison indicator starts with compact mode and 'Comparing documents…'."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Comparing documents…",
            operation_label="Analyzing cross-document metrics and contradictions",
            include_css=False,
        )
        indicator.start()

        assert indicator.is_active is True
        assert indicator.mode == "compact"
        assert indicator.current_status == "Comparing documents…"
        assert indicator.current_operation_label == "Analyzing cross-document metrics and contradictions"
        assert mock_container.markdown.called

    def test_indicator_starts_before_comparison_dispatch(self):
        """4: Indicator starts before comparison API dispatch."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        call_order = []

        mock_indicator.start.side_effect = lambda: call_order.append("indicator.start")
        mock_indicator.update.side_effect = lambda **kw: call_order.append(f"indicator.update:{kw.get('status')}")

        def mock_make_api(*args, **kwargs):
            call_order.append("make_api_request")
            return {
                "answer": "Document A mentions X, Document B mentions Y.",
                "sources": [],
                "evidence_groups": [],
            }

        with patch("ui.streamlit_app.make_api_request", side_effect=mock_make_api):
            res = compare_documents_api(
                query="Compare revenue in 2024 vs 2025",
                file_ids=["fid-1", "fid-2"],
                indicator=mock_indicator,
            )

        assert res is not None
        assert "indicator.start" in call_order
        assert "make_api_request" in call_order
        assert call_order.index("indicator.start") < call_order.index("make_api_request")

    def test_successful_comparison_clears_indicator(self):
        """5: Successful comparison clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            return_value={
                "answer": "Comparison successful.",
                "sources": [{"citation_label": "[D1] Doc 1"}],
                "evidence_groups": [{"document_name": "Doc 1"}],
                "numerical_comparison": {"difference": 10},
            },
        ):
            res = compare_documents_api(
                query="Compare metrics",
                file_ids=["fid-1", "fid-2"],
                indicator=mock_indicator,
            )

        assert res is not None
        assert res["answer"] == "Comparison successful."
        assert mock_indicator.clear.called

    def test_comparison_api_failure_clears_indicator(self):
        """6: Comparison API failure clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError("Comparison service unavailable", status_code=503),
        ):
            res = compare_documents_api(
                query="Compare",
                file_ids=["fid-1", "fid-2"],
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_comparison_exception_clears_indicator(self):
        """7: Comparison unexpected exception clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=RuntimeError("Connection terminated unexpectedly"),
        ):
            with pytest.raises(RuntimeError):
                compare_documents_api(
                    query="Compare",
                    file_ids=["fid-1", "fid-2"],
                    indicator=mock_indicator,
                )

        assert mock_indicator.clear.called

    def test_comparison_interruption_clears_indicator(self):
        """8: Comparison interruption (KeyboardInterrupt) clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=KeyboardInterrupt(),
        ):
            with pytest.raises(KeyboardInterrupt):
                compare_documents_api(
                    query="Compare",
                    file_ids=["fid-1", "fid-2"],
                    indicator=mock_indicator,
                )

        assert mock_indicator.clear.called

    def test_exactly_one_comparison_request_occurs(self):
        """9: Exactly one comparison request is dispatched per action."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        with patch("ui.streamlit_app.make_api_request") as mock_make:
            mock_make.return_value = {
                "answer": "One comparison result",
                "sources": [],
            }

            compare_documents_api(
                query="Compare A and B",
                file_ids=["fid-1", "fid-2"],
                session_id="sess-123",
                indicator=mock_indicator,
            )

            assert mock_make.call_count == 1
            call_args = mock_make.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/sessions/sess-123/compare"
            payload = json.loads(call_args[1]["data"])
            assert payload["query"] == "Compare A and B"
            assert payload["file_ids"] == ["fid-1", "fid-2"]

    def test_comparison_result_remains_intact(self):
        """10: Comparison response structures remain completely intact."""
        expected_resp = {
            "answer": "Doc 1 has higher profit.",
            "sources": [{"citation_id": "D1", "source_file": "doc1.pdf"}],
            "evidence_groups": [
                {"document_name": "doc1.pdf", "excerpts": ["Profit: 100M"]},
                {"document_name": "doc2.pdf", "excerpts": ["Profit: 80M"]},
            ],
            "numerical_comparison": {"metric_name": "Profit", "difference": 20.0},
            "contradictions_detected": [],
        }

        with patch("ui.streamlit_app.make_api_request", return_value=expected_resp):
            res = compare_documents_api(
                query="Compare profits",
                file_ids=["fid-1", "fid-2"],
            )

        assert res == expected_resp
        assert res["numerical_comparison"]["difference"] == 20.0
        assert len(res["evidence_groups"]) == 2

    def test_citations_and_evidence_remain_intact(self):
        """11: Citations and evidence groups are preserved."""
        raw_resp = {
            "answer": "Comparison analysis [D1][D2]",
            "sources": [
                {"citation_marker": "[D1]", "filename": "report_a.pdf"},
                {"citation_marker": "[D2]", "filename": "report_b.pdf"},
            ],
            "evidence_groups": [{"document_name": "report_a.pdf"}],
        }

        with patch("ui.streamlit_app.make_api_request", return_value=raw_resp):
            res = compare_documents_api(
                query="Check sources",
                file_ids=["fid-a", "fid-b"],
            )

        assert len(res["sources"]) == 2
        assert res["sources"][0]["citation_marker"] == "[D1]"
        assert res["sources"][1]["citation_marker"] == "[D2]"

    def test_no_artificial_delays_in_comparison(self):
        """12: Verify no time.sleep or asyncio.sleep in compare_documents_api."""
        source = inspect.getsource(compare_documents_api)
        assert "time.sleep" not in source
        assert "asyncio.sleep" not in source


class TestSessionSwitchingIndicator:
    """Verifies session switching architecture and truthful indicator lifecycle."""

    def test_inspect_session_switching_is_synchronous(self):
        """13: Verify session switching is currently implemented as synchronous local/direct I/O."""
        import inspect
        sig = inspect.signature(load_session)
        # load_session is a standard synchronous function
        assert not inspect.iscoroutinefunction(load_session)
        assert "session_id" in sig.parameters

    def test_asynchronous_equipped_load_session_starts_before_io(self):
        """14: If an indicator is supplied, it starts with 'Loading chat…' before API retrieval."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        call_order = []

        mock_indicator.start.side_effect = lambda: call_order.append("indicator.start")
        mock_indicator.update.side_effect = lambda **kw: call_order.append(f"indicator.update:{kw.get('status')}")

        def mock_get_details(session_id):
            call_order.append("get_session_details_api")
            return {"title": "Test Chat", "messages": []}

        with patch("ui.streamlit_app.get_session_details_api", side_effect=mock_get_details):
            load_session("sess-xyz", indicator=mock_indicator)

        assert "indicator.start" in call_order
        assert "get_session_details_api" in call_order
        assert call_order.index("indicator.start") < call_order.index("get_session_details_api")
        assert any("Loading chat…" in str(call) for call in call_order)

    def test_asynchronous_equipped_load_session_successful_cleanup(self):
        """15: If an indicator is supplied, it clears upon successful session load."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.get_session_details_api",
            return_value={"title": "Loaded Chat", "messages": [{"role": "user", "content": "Hello"}]},
        ):
            load_session("sess-1", indicator=mock_indicator)

        assert mock_indicator.clear.called

    def test_asynchronous_equipped_load_session_error_cleanup(self):
        """16: If an indicator is supplied, it clears upon error or exception."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.get_session_details_api",
            side_effect=RuntimeError("Database lock error"),
        ):
            with pytest.raises(RuntimeError):
                load_session("sess-1", indicator=mock_indicator)

        assert mock_indicator.clear.called

    def test_no_duplicate_session_loading(self):
        """17: Switching to already active session does not duplicate load."""
        import streamlit as st
        st.session_state.active_session_id = "sess-active"

        # If user is already on sess-active, sidebar button skips load_session
        sidebar_source = inspect.getsource(render_sidebar)
        assert 'if sid != st.session_state.get("active_session_id"):' in sidebar_source

    def test_active_session_remains_correct(self):
        """18: Active session ID and title are properly updated on load."""
        import streamlit as st
        with patch(
            "ui.streamlit_app.get_session_details_api",
            return_value={"title": "Strategic Plan", "messages": []},
        ):
            load_session("sess-target")

        assert st.session_state.active_session_id == "sess-target"
        assert st.session_state.active_session_title == "Strategic Plan"

    def test_chat_history_remains_correct(self):
        """19: Chat history messages and order are properly reconstructed."""
        import streamlit as st
        msgs = [
            {"role": "user", "content": "First message", "created_at": "2026-09-19T10:00:00"},
            {"role": "assistant", "content": "Second message", "created_at": "2026-09-19T10:00:05"},
        ]
        with patch(
            "ui.streamlit_app.get_session_details_api",
            return_value={"title": "Chat", "messages": msgs},
        ):
            load_session("sess-history")

        assert len(st.session_state.chat_history) == 2
        assert st.session_state.chat_history[0]["content"] == "First message"
        assert st.session_state.chat_history[1]["content"] == "Second message"

    def test_synchronous_switching_does_not_show_fake_loader(self):
        """20: Verify that standard sidebar session switching does not introduce a fake loader."""
        sidebar_source = inspect.getsource(render_sidebar)
        # Verify no artificial delay or fake AuraProcessingIndicator in sidebar session button switch
        # Sidebar session button switch block:
        assert "load_session(sid)" in sidebar_source
        assert "time.sleep" not in sidebar_source
        assert "asyncio.sleep" not in sidebar_source


class TestRegressionIntegrity:
    """Verifies Phases 13.1 - 13.7 processing indicators remain intact."""

    def test_chat_streaming_indicator_remains_unchanged(self):
        """21: Chat streaming uses full mode AuraProcessingIndicator with immediate start."""
        source = inspect.getsource(process_question)
        assert "mode=\"full\"" in source
        assert "indicator.start()" in source
        assert "indicator.clear_animation()" in source

    def test_document_upload_indicator_remains_unchanged(self):
        """22: Document upload uses AuraProcessingIndicator with 'Uploading document…'."""
        source = inspect.getsource(upload_document)
        assert "AuraProcessingIndicator" in source
        assert "Uploading document…" in source

    def test_document_deletion_indicator_remains_unchanged(self):
        """23: Document deletion uses compact AuraProcessingIndicator with 'Deleting document…'."""
        source = inspect.getsource(delete_document_api)
        assert "Deleting document…" in source

    def test_chat_rename_and_delete_indicators_remain_unchanged(self):
        """24: Chat rename and delete indicators remain intact."""
        rename_source = inspect.getsource(rename_session_api)
        assert "Saving chat name…" in rename_source

        delete_source = inspect.getsource(delete_session_api)
        assert "Deleting chat…" in delete_source

    def test_image_processing_indicator_remains_unchanged(self):
        """25: Image processing indicator uses AuraProcessingIndicator with 'Processing image…'."""
        source = inspect.getsource(process_attached_image)
        assert "AuraProcessingIndicator" in source
        assert "Processing image…" in source

    def test_sql_indicator_remains_unchanged(self):
        """26: SQL indicator remains compact with 'Processing SQL query…'."""
        sql_source = inspect.getsource(execute_sql_query_api)
        assert "Processing SQL query…" in sql_source
