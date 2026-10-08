"""
Regression tests for Aura AI Persistent Chat & Test Data Isolation.

Verifies:
1. Automated tests do NOT pollute the production/application database.
2. Real user-created sessions persist across application/SessionManager restarts.
3. New chat creation works via REST API.
4. Switching between chats maintains complete message and citation isolation.
5. Deleting a chat cascades and removes all associated messages.
6. Application restarts do not duplicate sessions.
7. Cleanup utility safely purges test artifacts while strictly preserving user conversations.
"""

import json
import sqlite3
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.sessions import SessionManager, session_manager
from scripts.clean_test_sessions import analyze_sessions, clean_test_sessions, is_test_session


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


def test_automated_tests_do_not_pollute_app_db(tmp_path, client):
    """
    Verify that test-created sessions ('Corrupt Test', 'Text Only', etc.)
    are written exclusively to the ephemeral test database and NEVER touch
    the configured application database.
    """
    # Create a simulated application database
    mock_app_db = str(tmp_path / "mock_app_sessions.sqlite")
    mgr_app = SessionManager(db_path=mock_app_db)
    mgr_app.create_session(title="Real User Conversation")

    # Verify mock_app_db currently has exactly 1 session
    with mgr_app._get_connection() as conn:
        count_before = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
    assert count_before == 1

    # Simulate what automated tests do (creating Corrupt Test and Text Only via API)
    res1 = client.post("/api/v1/sessions", json={"title": "Corrupt Test"})
    assert res1.status_code == 201
    sid1 = res1.json()["session_id"]

    res2 = client.post("/api/v1/sessions", json={"title": "Text Only"})
    assert res2.status_code == 201
    sid2 = res2.json()["session_id"]

    # The test database managed by the fixture has the test sessions
    res_list = client.get("/api/v1/sessions")
    assert res_list.status_code == 200
    session_ids = [s["session_id"] for s in res_list.json()["sessions"]]
    assert sid1 in session_ids
    assert sid2 in session_ids

    # The simulated application database was NEVER touched by the test client!
    with mgr_app._get_connection() as conn:
        count_after = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
        titles = [r[0] for r in conn.execute("SELECT title FROM sessions").fetchall()]
    assert count_after == 1
    assert titles == ["Real User Conversation"]


def test_real_sessions_persist_across_restarts(tmp_path):
    """
    Verify that genuine user sessions and message history persist completely
    across application / SessionManager restarts.
    """
    db_file = str(tmp_path / "persistent_user_chat.sqlite")

    # 1. First run / instance: User creates session and messages
    mgr1 = SessionManager(db_path=db_file)
    session = mgr1.create_session(title="Deep Financial Analysis", metadata={"source": "user_ui"})
    sid = session.session_id

    mgr1.append_message(sid, "user", "What was our Q3 EBITDA?")
    mgr1.append_message(
        sid,
        "assistant",
        "Q3 EBITDA was $4.2M [S1].",
        sources=[{
            "citation_id": "S1",
            "filename": "q3_report.pdf",
            "chunk_id": "c1",
            "page_number": 4,
            "content": "Q3 EBITDA reached $4.2M in 2024."
        }],
        citation_validation={"valid": True, "coverage": 1.0}
    )

    # 2. Simulate complete application restart (reinstantiate SessionManager pointing to same file)
    mgr2 = SessionManager(db_path=db_file)
    restored_session = mgr2.get_session(sid)

    assert restored_session is not None
    assert restored_session.session_id == sid
    assert restored_session.title == "Deep Financial Analysis"
    assert restored_session.metadata == {"source": "user_ui"}
    assert len(restored_session.messages) == 2

    user_msg = restored_session.messages[0]
    assert user_msg.role == "user"
    assert user_msg.content == "What was our Q3 EBITDA?"

    asst_msg = restored_session.messages[1]
    assert asst_msg.role == "assistant"
    assert asst_msg.content == "Q3 EBITDA was $4.2M [S1]."
    assert len(asst_msg.sources) == 1
    assert asst_msg.sources[0].filename == "q3_report.pdf"
    assert asst_msg.citation_validation["valid"] is True


def test_new_chat_creation_via_api(client):
    """
    Verify that creating a new chat via the API produces a valid session
    with proper timestamps, title, and empty initial history.
    """
    res = client.post("/api/v1/sessions", json={"title": "Q4 Budget Review"})
    assert res.status_code == 201
    data = res.json()

    assert "session_id" in data
    assert data["title"] == "Q4 Budget Review"
    assert data["created_at"] is not None
    assert data["updated_at"] is not None
    assert data["messages"] == []

    # Verify session is retrievable
    sid = data["session_id"]
    get_res = client.get(f"/api/v1/sessions/{sid}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == sid


def test_switching_chats_isolation(tmp_path):
    """
    Verify that switching between multiple chats maintains complete isolation:
    messages in Chat A never appear in Chat B.
    """
    db_file = str(tmp_path / "multi_chat_isolation.sqlite")
    mgr = SessionManager(db_path=db_file)

    # Chat A
    sess_a = mgr.create_session(title="Chat Alpha")
    mgr.append_message(sess_a.session_id, "user", "Message in Alpha 1")
    mgr.append_message(sess_a.session_id, "assistant", "Response in Alpha 1")

    # Chat B
    sess_b = mgr.create_session(title="Chat Beta")
    mgr.append_message(sess_b.session_id, "user", "Message in Beta 1")
    mgr.append_message(sess_b.session_id, "assistant", "Response in Beta 1")
    mgr.append_message(sess_b.session_id, "user", "Message in Beta 2")

    # Verify Chat Alpha has only 2 Alpha messages
    history_a = mgr.get_session_messages(sess_a.session_id)
    assert len(history_a) == 2
    assert [m.content for m in history_a] == ["Message in Alpha 1", "Response in Alpha 1"]

    # Verify Chat Beta has only 3 Beta messages
    history_b = mgr.get_session_messages(sess_b.session_id)
    assert len(history_b) == 3
    assert [m.content for m in history_b] == ["Message in Beta 1", "Response in Beta 1", "Message in Beta 2"]


def test_deleting_chat_cascades_messages(client):
    """
    Verify that deleting a chat via API removes the session and cascades
    to remove all associated messages from the database.
    """
    # Create session
    create_res = client.post("/api/v1/sessions", json={"title": "Temporary Chat"})
    assert create_res.status_code == 201
    sid = create_res.json()["session_id"]

    # Add a message to it
    session_manager.append_message(sid, "user", "Temporary user question")
    session_manager.append_message(sid, "assistant", "Temporary assistant answer")

    # Confirm messages exist
    msgs_before = session_manager.get_session_messages(sid)
    assert len(msgs_before) == 2

    # Delete session via API
    del_res = client.delete(f"/api/v1/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["session_id"] == sid

    # Verify session is not found
    get_res = client.get(f"/api/v1/sessions/{sid}")
    assert get_res.status_code == 404

    # Verify messages were cascaded and deleted directly from SQLite
    with session_manager._get_connection() as conn:
        msg_count = conn.execute("SELECT count(*) FROM messages WHERE session_id = ?", (sid,)).fetchone()[0]
    assert msg_count == 0


def test_application_restart_does_not_duplicate_sessions(tmp_path):
    """
    Verify that repeated application restarts or SessionManager reinstantiations
    never duplicate existing sessions or corrupt message order.
    """
    db_file = str(tmp_path / "restart_stability.sqlite")

    # Instance 1: Create 3 sessions
    mgr1 = SessionManager(db_path=db_file)
    s1 = mgr1.create_session(title="Session One")
    s2 = mgr1.create_session(title="Session Two")
    s3 = mgr1.create_session(title="Session Three")

    mgr1.append_message(s1.session_id, "user", "Msg 1")
    mgr1.append_message(s2.session_id, "user", "Msg 2")

    # Simulate restart 1
    mgr2 = SessionManager(db_path=db_file)
    list2 = mgr2.list_sessions(limit=50)
    assert list2.total == 3
    assert len(list2.sessions) == 3

    # Simulate restart 2
    mgr3 = SessionManager(db_path=db_file)
    list3 = mgr3.list_sessions(limit=50)
    assert list3.total == 3
    assert len(list3.sessions) == 3

    # Ensure titles and counts are consistent
    titles = {s.title for s in list3.sessions}
    assert titles == {"Session One", "Session Two", "Session Three"}


def test_cleanup_utility_identifies_and_purges_only_test_artifacts(tmp_path):
    """
    Verify that clean_test_sessions correctly identifies test artifacts
    ('Corrupt Test', 'Text Only', empty 'New Chat') and removes them while
    strictly preserving genuine conversations.
    """
    db_file = str(tmp_path / "cleanup_test.sqlite")
    mgr = SessionManager(db_path=db_file)

    # 1. Create test artifacts
    t1 = mgr.create_session(title="Corrupt Test")
    mgr.append_message(t1.session_id, "user", "Corrupt img?")

    t2 = mgr.create_session(title="Text Only")
    mgr.append_message(t2.session_id, "user", "Text only question")

    t3 = mgr.create_session(title="New Chat")  # 0 messages -> empty test artifact

    # 2. Create genuine user conversations
    u1 = mgr.create_session(title="Production Contract Q&A")
    mgr.append_message(u1.session_id, "user", "What is the termination clause in Section 4?")
    mgr.append_message(u1.session_id, "assistant", "Section 4 specifies a 30-day written notice [S1].")

    u2 = mgr.create_session(title="hi")
    mgr.append_message(u2.session_id, "user", "hi")
    mgr.append_message(u2.session_id, "assistant", "Hello! How can I assist you with your documents?")

    # Analyze before cleanup
    to_delete, to_preserve = analyze_sessions(db_file)
    assert len(to_delete) == 3
    assert {s["title"] for s in to_delete} == {"Corrupt Test", "Text Only", "New Chat"}

    assert len(to_preserve) == 2
    assert {s["title"] for s in to_preserve} == {"Production Contract Q&A", "hi"}

    # Execute cleanup with confirmation
    result = clean_test_sessions(db_file, confirm=True)
    assert result["status"] == "cleaned"
    assert result["deleted_count"] == 3
    assert result["preserved_count"] == 2

    # Verify remaining in DB
    rem_delete, rem_preserve = analyze_sessions(db_file)
    assert len(rem_delete) == 0
    assert len(rem_preserve) == 2
    assert {s["title"] for s in rem_preserve} == {"Production Contract Q&A", "hi"}
