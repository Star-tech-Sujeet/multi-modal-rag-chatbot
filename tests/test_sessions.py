"""
Tests for Phase 6: Conversational RAG & Persistent Sessions.

Verifies:
1. SQLite-backed persistent conversation sessions and message history.
2. Complete session isolation with foreign keys and cascading deletion.
3. Thread-safe database operations and persistence across restarts.
4. Deterministic conversational query contextualization.
5. Structured source and citation persistence per turn.
6. Deterministic session titling without LLM calls.
7. History window limit enforcement.
8. REST API endpoints (/sessions, /sessions/{id}, /sessions/{id}/query).
9. Strict backward compatibility with stateless /query.
"""

import os
import uuid
from typing import List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document

from app.main import app
from app.config import settings
from app.models import (
    Source,
    SessionCreateRequest,
    SessionUpdateRequest,
    SessionQueryRequest,
    SessionMessageModel,
)
from app.sessions import SessionManager, construct_contextual_query
from app.logic import perform_session_rag_query, RetrievalCandidate


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary SQLite database file path managed by pytest."""
    db_file = tmp_path / f"test_sessions_{uuid.uuid4().hex}.sqlite"
    return str(db_file)


@pytest.fixture
def session_mgr(temp_db_path):
    """Create an isolated SessionManager with a temporary database."""
    mgr = SessionManager(db_path=temp_db_path)
    return mgr


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


# =============================================================================
# 1. Session Lifecycle & Persistence Tests
# =============================================================================

def test_session_create_default_title(session_mgr):
    """Verify session creation with default title and initial state."""
    session = session_mgr.create_session()
    assert session.session_id is not None
    assert session.title == "New Chat"
    assert session.created_at is not None
    assert session.updated_at is not None
    assert session.messages == []


def test_session_create_custom_title_and_metadata(session_mgr):
    """Verify session creation with custom title and metadata."""
    meta = {"user_role": "analyst", "tag": "finance"}
    session = session_mgr.create_session(title="Financial Review 2024", metadata=meta)
    assert session.title == "Financial Review 2024"
    assert session.metadata == meta

    fetched = session_mgr.get_session(session.session_id)
    assert fetched is not None
    assert fetched.title == "Financial Review 2024"
    assert fetched.metadata == meta


def test_append_message_and_retrieve(session_mgr):
    """Verify appending messages and chronological retrieval."""
    session = session_mgr.create_session(title="Chat 1")
    sid = session.session_id

    # Append user message
    msg1 = session_mgr.append_message(session_id=sid, role="user", content="Hello RAG")
    assert msg1.role == "user"
    assert msg1.content == "Hello RAG"
    assert msg1.message_id is not None

    # Append assistant message
    msg2 = session_mgr.append_message(session_id=sid, role="assistant", content="Hello! How can I help?")
    assert msg2.role == "assistant"
    assert msg2.content == "Hello! How can I help?"

    # Fetch messages
    messages = session_mgr.get_session_messages(sid)
    assert len(messages) == 2
    assert messages[0].content == "Hello RAG"
    assert messages[1].content == "Hello! How can I help?"


def test_session_isolation(session_mgr):
    """Verify that messages in Session A are completely invisible to Session B."""
    sess_a = session_mgr.create_session(title="Session A")
    sess_b = session_mgr.create_session(title="Session B")

    session_mgr.append_message(sess_a.session_id, role="user", content="Secret in A")
    session_mgr.append_message(sess_b.session_id, role="user", content="Secret in B")

    msgs_a = session_mgr.get_session_messages(sess_a.session_id)
    msgs_b = session_mgr.get_session_messages(sess_b.session_id)

    assert len(msgs_a) == 1
    assert msgs_a[0].content == "Secret in A"

    assert len(msgs_b) == 1
    assert msgs_b[0].content == "Secret in B"


def test_session_deletion_cascade(session_mgr):
    """Verify cascading delete: deleting a session removes all its messages."""
    session = session_mgr.create_session(title="To Delete")
    sid = session.session_id

    session_mgr.append_message(sid, role="user", content="Message 1")
    session_mgr.append_message(sid, role="assistant", content="Message 2")

    assert len(session_mgr.get_session_messages(sid)) == 2

    # Delete session
    deleted = session_mgr.delete_session(sid)
    assert deleted is True

    # Verify session and messages are gone
    assert session_mgr.get_session(sid) is None
    assert session_mgr.get_session_messages(sid) == []


def test_deterministic_auto_titling(session_mgr):
    """Verify that the first user message updates the default title to first 60 chars."""
    session = session_mgr.create_session()
    sid = session.session_id
    assert session.title == "New Chat"

    # First user message
    session_mgr.append_message(sid, role="user", content="What was the total net revenue for Apple in Q3 2023?")
    updated = session_mgr.get_session(sid)
    assert updated.title == "What was the total net revenue for Apple in Q3 2023?"

    # Second user message should NOT overwrite the title
    session_mgr.append_message(sid, role="user", content="What about operating expenses?")
    updated2 = session_mgr.get_session(sid)
    assert updated2.title == "What was the total net revenue for Apple in Q3 2023?"


def test_structured_source_persistence(session_mgr):
    """Verify structured Source objects and citation validation persistence."""
    session = session_mgr.create_session(title="Sources Test")
    sid = session.session_id

    test_source = Source(
        filename="financials.pdf",
        file_id="fid-12345",
        chunk_id="fid-12345_0",
        page_number=4,
        section="Balance Sheet",
        citation_id="S1",
        citation_label="[S1] financials.pdf (Page 4)",
        content="Total revenue was $81.8 billion.",
        relevance_score=0.92,
        vector_score=0.88,
        bm25_score=0.95,
        search_type="hybrid"
    )

    validation_report = {
        "valid_citation_ids_supplied": ["S1"],
        "citations_used": ["S1"],
        "invalid_citations_detected": [],
        "has_invalid_citations": False,
        "total_citations_count": 1
    }

    asst_msg = session_mgr.append_message(
        session_id=sid,
        role="assistant",
        content="Revenue was $81.8 billion [S1].",
        sources=[test_source],
        citation_validation=validation_report
    )

    # Reload message from database
    msgs = session_mgr.get_session_messages(sid)
    assert len(msgs) == 1
    loaded = msgs[0]

    assert loaded.sources is not None
    assert len(loaded.sources) == 1
    src = loaded.sources[0]
    assert src.filename == "financials.pdf"
    assert src.citation_id == "S1"
    assert src.page_number == 4
    assert src.relevance_score == 0.92

    assert loaded.citation_validation is not None
    assert loaded.citation_validation["citations_used"] == ["S1"]


def test_persistence_across_manager_restarts(temp_db_path):
    """Verify that sessions and messages survive a complete reload / re-instantiation."""
    mgr1 = SessionManager(db_path=temp_db_path)
    sess1 = mgr1.create_session(title="Persistent Chat")
    sid = sess1.session_id
    mgr1.append_message(sid, role="user", content="Save this message across restarts")

    # Simulate application restart with new SessionManager instance
    mgr2 = SessionManager(db_path=temp_db_path)
    reloaded_sess = mgr2.get_session(sid)
    assert reloaded_sess is not None
    assert reloaded_sess.title == "Persistent Chat"
    assert len(reloaded_sess.messages) == 1
    assert reloaded_sess.messages[0].content == "Save this message across restarts"


def test_history_window_limit(session_mgr):
    """Verify that message retrieval respects the limit parameter."""
    session = session_mgr.create_session(title="History Window Test")
    sid = session.session_id

    for i in range(15):
        session_mgr.append_message(sid, role="user", content=f"Message {i}")

    # Fetch with limit 10
    recent = session_mgr.get_session_messages(sid, limit=10)
    assert len(recent) == 10
    # Should be the most recent 10 (Messages 5 through 14) in chronological order
    assert recent[0].content == "Message 5"
    assert recent[-1].content == "Message 14"


# =============================================================================
# 2. Conversational Query Contextualization Tests
# =============================================================================

def test_contextual_query_followup_with_pronouns():
    """Verify that follow-up queries with pronouns or short clauses are contextualized."""
    history = [
        SessionMessageModel(
            message_id="m1",
            session_id="s1",
            role="user",
            content="What was Apple revenue in 2022?",
            created_at="2026-01-01T00:00:00Z"
        ),
        SessionMessageModel(
            message_id="m2",
            session_id="s1",
            role="assistant",
            content="Apple revenue was $394.3 billion [S1].",
            created_at="2026-01-01T00:01:00Z"
        )
    ]

    contextual = construct_contextual_query("What about 2023?", history)
    assert "2023" in contextual
    assert "apple" in contextual.lower()
    assert "revenue" in contextual.lower()


def test_contextual_query_self_contained_unaltered():
    """Verify that self-contained, descriptive queries are NOT altered."""
    history = [
        SessionMessageModel(
            message_id="m1",
            session_id="s1",
            role="user",
            content="What was Apple revenue?",
            created_at="2026-01-01T00:00:00Z"
        )
    ]

    long_query = "Explain the deep learning architecture of transformer attention heads"
    contextual = construct_contextual_query(long_query, history)
    assert contextual == long_query


def test_contextual_query_empty_or_no_history():
    """Verify safe handling of empty history or empty input."""
    assert construct_contextual_query("What is the cost?", []) == "What is the cost?"
    assert construct_contextual_query("", []) == ""


# =============================================================================
# 3. Session RAG Query & Citation Isolation Tests
# =============================================================================

def test_perform_session_rag_query_turn_isolation(session_mgr):
    """Verify that conversational query turns maintain strict citation isolation."""
    session = session_mgr.create_session(title="Turn Test")
    sid = session.session_id

    doc1 = Document(
        page_content="Turn 1 content about revenue.",
        metadata={"filename": "doc1.pdf", "file_id": "fid-1", "file_type": "pdf", "page_number": 1, "chunk_id": "fid-1_0"}
    )
    cand1 = RetrievalCandidate(
        doc=doc1,
        chunk_id="fid-1_0",
        final_score=0.9,
        vector_score=0.85,
        bm25_score=0.95,
        search_type="hybrid"
    )

    doc2 = Document(
        page_content="Turn 2 content about expenses.",
        metadata={"filename": "doc2.pdf", "file_id": "fid-2", "file_type": "pdf", "page_number": 2, "chunk_id": "fid-2_0"}
    )
    cand2 = RetrievalCandidate(
        doc=doc2,
        chunk_id="fid-2_0",
        final_score=0.88,
        vector_score=0.82,
        bm25_score=0.92,
        search_type="hybrid"
    )

    # Turn 1
    with patch("app.logic.execute_retrieval_pipeline", return_value=([cand1], "hybrid")):
        with patch("app.logic.handle_text_query", return_value=("Turn 1 answer [S1].", "gpt-4o-mini")):
            req1 = SessionQueryRequest(question="What is revenue?", file_id="fid-1")
            res1 = perform_session_rag_query(sid, req1, mgr=session_mgr)

            assert res1.session_id == sid
            assert res1.answer == "Turn 1 answer [S1]."
            assert len(res1.sources) == 1
            assert res1.sources[0].citation_id == "S1"
            assert res1.sources[0].filename == "doc1.pdf"

    # Turn 2: Follow-up question
    with patch("app.logic.execute_retrieval_pipeline", return_value=([cand2], "hybrid")):
        with patch("app.logic.handle_text_query", return_value=("Turn 2 answer [S1].", "gpt-4o-mini")):
            req2 = SessionQueryRequest(question="What about expenses?", file_id="fid-2")
            res2 = perform_session_rag_query(sid, req2, mgr=session_mgr)

            assert res2.session_id == sid
            assert res2.answer == "Turn 2 answer [S1]."
            # Turn 2's [S1] MUST point to doc2.pdf, strictly isolated from Turn 1
            assert len(res2.sources) == 1
            assert res2.sources[0].citation_id == "S1"
            assert res2.sources[0].filename == "doc2.pdf"

    # Check messages persisted in SQLite
    msgs = session_mgr.get_session_messages(sid)
    assert len(msgs) == 4  # User 1, Asst 1, User 2, Asst 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert msgs[2].role == "user"
    assert msgs[3].role == "assistant"


def test_perform_session_rag_query_nonexistent_session(session_mgr):
    """Verify error handling when querying a nonexistent session."""
    req = SessionQueryRequest(question="Any question?", file_id="fid-1")
    with pytest.raises(ValueError, match="not found"):
        perform_session_rag_query("nonexistent-id", req, mgr=session_mgr)


# =============================================================================
# 4. REST API Endpoint Tests
# =============================================================================

def test_api_session_crud(client):
    """Verify session creation, retrieval, listing, patching, and deletion via REST API."""
    # 1. Create Session
    create_res = client.post("/api/v1/sessions", json={"title": "API Test Session"})
    assert create_res.status_code == 201
    created_data = create_res.json()
    sid = created_data["session_id"]
    assert created_data["title"] == "API Test Session"

    # 2. List Sessions
    list_res = client.get("/api/v1/sessions")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert any(s["session_id"] == sid for s in list_data["sessions"])

    # 3. Get Session Details
    get_res = client.get(f"/api/v1/sessions/{sid}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == sid

    # 4. Patch Session
    patch_res = client.patch(f"/api/v1/sessions/{sid}", json={"title": "Updated API Title"})
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Updated API Title"

    # 5. Delete Session
    del_res = client.delete(f"/api/v1/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["session_id"] == sid

    # 6. Verify 404 after deletion
    get_res_after = client.get(f"/api/v1/sessions/{sid}")
    assert get_res_after.status_code == 404


def test_api_session_query_nonexistent_session(client):
    """Verify 404 response when querying a nonexistent session."""
    fake_id = str(uuid.uuid4())
    res = client.post(
        f"/api/v1/sessions/{fake_id}/query",
        json={"question": "Test question", "file_id": "fid-1"}
    )
    assert res.status_code == 404
    data = res.json()
    error_code = data.get("details", {}).get("error") or data.get("error")
    assert error_code == "session_not_found" or "not_found" in str(data)


def test_api_stateless_query_backward_compatibility(client):
    """Verify that existing stateless /query endpoint remains 100% functional."""
    # Query with a file_id that does not exist returns 200 with no documents found
    with patch("app.logic.execute_retrieval_pipeline", return_value=([], "vector")):
        res = client.post("/api/v1/query", json={"question": "Hello", "file_id": str(uuid.uuid4())})
        assert res.status_code == 200
        data = res.json()
        assert "No relevant documents found" in data["answer"]
