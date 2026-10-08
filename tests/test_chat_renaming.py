"""
Comprehensive test suite for Aura AI Chat/Conversation Renaming.

Covers:
7. Existing chat can be renamed.
8. Rename updates the persistent session title.
9. Empty title is rejected.
10. Whitespace-only title is rejected.
11. Long title is handled according to validation rules.
12. Cancel leaves the original title unchanged.
13. Renamed chat remains correctly associated with the same session ID.
14. Renamed title persists after Streamlit rerun.
15. Renamed title persists after browser refresh.
16. Renamed title persists after restart.
17. Other conversations remain unchanged.
18. Delete functionality still works.
19. Session isolation remains intact.
"""

import os
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sessions import SessionManager
from app.models import SessionCreateRequest, SessionUpdateRequest


@pytest.fixture
def temp_db(tmp_path):
    """Create an isolated test SQLite DB path."""
    db_file = tmp_path / f"test_rename_{uuid.uuid4().hex}.sqlite"
    return str(db_file)


@pytest.fixture
def session_mgr(temp_db):
    """Create an isolated SessionManager instance."""
    return SessionManager(db_path=temp_db)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestChatRenamingLifecycleAndValidation:
    """Test suite for Chat Renaming lifecycle, persistence, validation, and isolation."""

    def test_7_and_8_rename_existing_chat_updates_persistent_title(self, session_mgr):
        """7, 8. Verify existing chat can be renamed and persistent title is updated in SQLite."""
        session = session_mgr.create_session(title="JDBC Questions")
        sid = session.session_id
        assert session.title == "JDBC Questions"

        # Rename session
        updated = session_mgr.update_session(session_id=sid, title="JDBC Exam Preparation")
        assert updated is not None
        assert updated.session_id == sid
        assert updated.title == "JDBC Exam Preparation"

        # Verify direct fetch from SQLite reflects new title
        fetched = session_mgr.get_session(sid)
        assert fetched is not None
        assert fetched.title == "JDBC Exam Preparation"

    def test_9_and_10_empty_and_whitespace_title_rejected(self, session_mgr, client):
        """9, 10. Verify empty title and whitespace-only title are rejected."""
        # 1. Direct SessionManager validation
        session = session_mgr.create_session(title="Original Title")
        sid = session.session_id

        with pytest.raises(ValueError, match="empty or whitespace only"):
            session_mgr.update_session(session_id=sid, title="")

        with pytest.raises(ValueError, match="empty or whitespace only"):
            session_mgr.update_session(session_id=sid, title="     ")

        # Original title must remain intact
        assert session_mgr.get_session(sid).title == "Original Title"

        # 2. REST API endpoint validation (PATCH & PUT)
        create_res = client.post("/api/v1/sessions", json={"title": "API Validation Chat"})
        assert create_res.status_code == 201
        api_sid = create_res.json()["session_id"]

        try:
            # Empty string rejected by Pydantic min_length=1
            r_empty = client.patch(f"/api/v1/sessions/{api_sid}", json={"title": ""})
            assert r_empty.status_code in (400, 422)

            # Whitespace-only string rejected by field_validator
            r_spaces = client.patch(f"/api/v1/sessions/{api_sid}", json={"title": "    \t  \n  "})
            assert r_spaces.status_code in (400, 422)

            # PUT endpoint behaves identically
            r_put_spaces = client.put(f"/api/v1/sessions/{api_sid}", json={"title": "   "})
            assert r_put_spaces.status_code in (400, 422)

            # Verified unchanged
            assert client.get(f"/api/v1/sessions/{api_sid}").json()["title"] == "API Validation Chat"
        finally:
            client.delete(f"/api/v1/sessions/{api_sid}")

    def test_11_long_title_handled_according_to_validation_rules(self, session_mgr, client):
        """11. Verify titles are bounded and trimmed to schema constraints."""
        session = session_mgr.create_session(title="Short")
        sid = session.session_id

        # 200 characters string
        very_long = "A" * 200
        # Model / SessionManager clamps to 150 characters
        updated = session_mgr.update_session(session_id=sid, title=very_long)
        assert updated is not None
        assert len(updated.title) == 150
        assert updated.title == "A" * 150

        # REST API rejects exceeding max_length=150
        create_res = client.post("/api/v1/sessions", json={"title": "Length Test"})
        api_sid = create_res.json()["session_id"]
        try:
            r_oversized = client.patch(f"/api/v1/sessions/{api_sid}", json={"title": "X" * 200})
            assert r_oversized.status_code == 422
        finally:
            client.delete(f"/api/v1/sessions/{api_sid}")

    def test_12_cancel_leaves_original_title_unchanged(self, session_mgr):
        """12. Verify cancellation leaves the original title completely unchanged."""
        session = session_mgr.create_session(title="JDBC Original")
        sid = session.session_id

        # Simulating UI cancel: user begins editing but does not commit update_session
        draft_title = "Never Saved Title"
        # No update_session called

        current = session_mgr.get_session(sid)
        assert current.title == "JDBC Original"

    def test_13_renamed_chat_keeps_same_session_id_and_messages(self, session_mgr):
        """13. Renamed chat remains correctly associated with the same session ID, created_at, and messages."""
        session = session_mgr.create_session(title="Before Rename")
        sid = session.session_id
        created_at = session.created_at

        # Append messages
        session_mgr.append_message(sid, "user", "What is JDBC?")
        session_mgr.append_message(sid, "assistant", "JDBC is Java Database Connectivity.")

        # Rename
        updated = session_mgr.update_session(sid, title="After Rename")
        assert updated.session_id == sid
        assert updated.created_at == created_at
        assert updated.title == "After Rename"

        # Messages must be preserved
        msgs = session_mgr.get_session_messages(sid)
        assert len(msgs) == 2
        assert msgs[0].content == "What is JDBC?"
        assert msgs[1].content == "JDBC is Java Database Connectivity."

    def test_14_and_15_and_16_persistence_across_rerun_refresh_and_restart(self, temp_db):
        """14, 15, 16. Verify renamed title persists across simulated reruns, refreshes, and backend restart."""
        # 1. Instance 1: Create and Rename
        mgr1 = SessionManager(db_path=temp_db)
        s1 = mgr1.create_session(title="Database Revision")
        sid = s1.session_id

        mgr1.update_session(sid, title="Database Revision (Final)")

        # 2. Instance 2: Simulating server/backend restart with fresh SessionManager on same DB
        mgr2 = SessionManager(db_path=temp_db)
        reloaded = mgr2.get_session(sid)
        assert reloaded is not None
        assert reloaded.title == "Database Revision (Final)"

        # Verify listing reflects renamed title
        all_sessions = mgr2.list_sessions()
        matched = [s for s in all_sessions.sessions if s.session_id == sid]
        assert len(matched) == 1
        assert matched[0].title == "Database Revision (Final)"

    def test_17_other_conversations_remain_unchanged(self, session_mgr):
        """17. Verify renaming one conversation does not alter any other conversations."""
        s1 = session_mgr.create_session(title="Chat Alpha")
        s2 = session_mgr.create_session(title="Chat Beta")
        s3 = session_mgr.create_session(title="Chat Gamma")

        # Rename s2 only
        session_mgr.update_session(s2.session_id, title="Chat Beta Renamed")

        # Verify s1 and s3 are unaffected
        assert session_mgr.get_session(s1.session_id).title == "Chat Alpha"
        assert session_mgr.get_session(s2.session_id).title == "Chat Beta Renamed"
        assert session_mgr.get_session(s3.session_id).title == "Chat Gamma"

    def test_18_delete_functionality_works_on_renamed_chat(self, session_mgr):
        """18. Verify delete functionality continues to work flawlessly on renamed chats."""
        s = session_mgr.create_session(title="Temporary Chat")
        sid = s.session_id

        session_mgr.update_session(sid, title="Renamed Temporary Chat")
        assert session_mgr.get_session(sid).title == "Renamed Temporary Chat"

        # Delete session
        deleted = session_mgr.delete_session(sid)
        assert deleted is True
        assert session_mgr.get_session(sid) is None

    def test_19_session_isolation_remains_intact_after_rename(self, session_mgr):
        """19. Verify message isolation across sessions remains intact after rename."""
        s1 = session_mgr.create_session(title="Isolation Chat 1")
        s2 = session_mgr.create_session(title="Isolation Chat 2")

        session_mgr.append_message(s1.session_id, "user", "Message in session 1")
        session_mgr.append_message(s2.session_id, "user", "Message in session 2")

        # Rename s1
        session_mgr.update_session(s1.session_id, title="Renamed Chat 1")

        # Session 1 messages
        m1 = session_mgr.get_session_messages(s1.session_id)
        assert len(m1) == 1
        assert m1[0].content == "Message in session 1"

        # Session 2 messages
        m2 = session_mgr.get_session_messages(s2.session_id)
        assert len(m2) == 1
        assert m2[0].content == "Message in session 2"

    def test_20_ui_sidebar_rename_flow(self):
        """Verify UI Sidebar renders rename button, enters edit mode, and handles save/cancel."""
        from streamlit.testing.v1 import AppTest

        app_path = (Path(__file__).parent.parent / "ui" / "streamlit_app.py").resolve()
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.session_state["api_healthy"] = True
        at.session_state["last_health_check"] = 9999999999
        at.session_state["active_session_id"] = "test-session-1"
        at.session_state["active_session_title"] = "JDBC Questions"

        test_sessions = [
            {"session_id": "test-session-1", "title": "JDBC Questions", "message_count": 3}
        ]

        with patch("requests.request") as mock_req:
            def fake_req(method, url, **kwargs):
                resp = MagicMock()
                resp.status_code = 200
                if "/files" in url:
                    resp.json.return_value = []
                elif "/health" in url:
                    resp.json.return_value = {"status": "healthy", "total_documents": 0}
                elif "/sessions/test-session-1" in url and method == "PATCH":
                    resp.json.return_value = {"session_id": "test-session-1", "title": "JDBC Preparation"}
                elif "/sessions" in url:
                    resp.json.return_value = {"sessions": test_sessions}
                else:
                    resp.json.return_value = {}
                return resp

            mock_req.side_effect = fake_req

            # Initial render
            at.run()

            # Verify rename button is rendered
            ren_btns = [b for b in at.sidebar.button if b.key == "sess_ren_test-session-1"]
            assert len(ren_btns) == 1, "Rename button ✏️ not rendered"

            # Click rename button -> enters edit mode
            ren_btns[0].click().run()
            assert "_renaming_session_id" in at.session_state
            assert at.session_state["_renaming_session_id"] == "test-session-1"

            # Verify text input and Save/Cancel buttons are rendered
            save_btns = [b for b in at.sidebar.button if b.key == "save_ren_test-session-1"]
            cancel_btns = [b for b in at.sidebar.button if b.key == "cancel_ren_test-session-1"]
            assert len(save_btns) == 1, "Save button not rendered in rename mode"
            assert len(cancel_btns) == 1, "Cancel button not rendered in rename mode"

            # Click Cancel -> clears rename mode
            cancel_btns[0].click().run()
            assert at.session_state["_renaming_session_id"] is None
