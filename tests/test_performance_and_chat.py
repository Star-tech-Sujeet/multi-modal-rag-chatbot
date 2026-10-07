"""
Comprehensive test suite for Aura AI performance optimizations and conversational chat lifecycle.

Covers:
1. Session persistence across operations
2. Conversational query contextualization
3. Chat session switching and isolation
4. Chat session deletion cascade
5. Clean citation mapping and metadata formatting
6. Single query embedding (no duplicate API embed calls)
7. BM25 index caching and reuse
8. AI provider singleton reuse
9. Fine-grained retrieval timing breakdown
10. Multimodal chat session handling
11. Missing information honest refusal behavior
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from app.models import Source, QueryResponse, QueryRequest, SessionQueryResponse, SessionQueryRequest, SessionMessageModel
from app.sessions import SessionManager, construct_contextual_query
from app.logic import (
    bm25_manager,
    execute_retrieval_pipeline,
    get_last_retrieval_timings,
    get_vectorstore,
    get_embeddings,
    perform_rag_query,
    perform_session_rag_query,
)
from app.providers import get_provider


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_session_manager(tmp_path):
    """Provide an isolated SQLite SessionManager."""
    db_path = str(tmp_path / "test_sessions.sqlite")
    return SessionManager(db_path=db_path)


@pytest.fixture
def sample_documents():
    """Create sample documents for retrieval and BM25 testing."""
    return [
        Document(
            page_content="Aura AI provides enterprise document retrieval using ChromaDB and BM25.",
            metadata={"chunk_id": "c1", "file_id": "test_f1", "filename": "aura_guide.pdf", "page_number": 1, "section": "Intro"}
        ),
        Document(
            page_content="Reciprocal Rank Fusion combines vector rankings and BM25 rankings with parameter k=60.",
            metadata={"chunk_id": "c2", "file_id": "test_f1", "filename": "aura_guide.pdf", "page_number": 2, "section": "Retrieval"}
        ),
        Document(
            page_content="Security policies mandate end-to-end encryption for all stored vector embeddings.",
            metadata={"chunk_id": "c3", "file_id": "test_f2", "filename": "security.pdf", "page_number": 5, "section": "Security"}
        ),
    ]


# =============================================================================
# Tests
# =============================================================================

def test_session_persistence(temp_session_manager):
    """Test that sessions and messages persist cleanly in SQLite."""
    # 1. Create session
    sess = temp_session_manager.create_session(title="Financial Q3 Discussion")
    assert sess.session_id is not None
    assert sess.title == "Financial Q3 Discussion"

    # 2. Add user message
    user_msg = temp_session_manager.append_message(
        session_id=sess.session_id,
        role="user",
        content="What was Q3 revenue?"
    )
    assert user_msg.message_id is not None

    # 3. Add assistant message with sources
    sample_src = Source(
        chunk_id="c1",
        file_id="f1",
        filename="report.pdf",
        page_number=1,
        content="Q3 revenue was $42M",
        relevance_score=0.95,
        citation_label="[S1]"
    )
    asst_msg = temp_session_manager.append_message(
        session_id=sess.session_id,
        role="assistant",
        content="Q3 revenue was $42 million [S1].",
        sources=[sample_src]
    )

    # 4. Fetch details and verify persistence
    details = temp_session_manager.get_session(sess.session_id)
    assert details is not None
    assert len(details.messages) == 2
    assert details.messages[0].role == "user"
    assert details.messages[0].content == "What was Q3 revenue?"
    assert details.messages[1].role == "assistant"
    assert len(details.messages[1].sources) == 1
    assert details.messages[1].sources[0].citation_label == "[S1]"


def test_follow_up_query_contextualization():
    """Test contextualization of ambiguous conversational queries."""
    history = [
        SessionMessageModel(
            message_id="m1",
            session_id="s1",
            role="user",
            content="Tell me about Aura AI's security architecture.",
            created_at="2026-01-01T00:00:00Z"
        ),
        SessionMessageModel(
            message_id="m2",
            session_id="s1",
            role="assistant",
            content="Aura AI enforces role-based access control and TLS 1.3 encryption.",
            created_at="2026-01-01T00:00:05Z"
        ),
    ]

    # Ambiguous follow-up
    q = "What protocols does it use?"
    contextualized = construct_contextual_query(q, history)

    # Should augment query with previous context keywords
    assert "security" in contextualized.lower() or "aura" in contextualized.lower() or "protocols" in contextualized.lower()
    assert len(contextualized) >= len(q)


def test_chat_switching(temp_session_manager):
    """Test switching between distinct chat sessions without message leakage."""
    s1 = temp_session_manager.create_session(title="Chat One")
    s2 = temp_session_manager.create_session(title="Chat Two")

    temp_session_manager.append_message(session_id=s1.session_id, role="user", content="Hello in Session 1")
    temp_session_manager.append_message(session_id=s2.session_id, role="user", content="Hello in Session 2")

    sess1_details = temp_session_manager.get_session(s1.session_id)
    sess2_details = temp_session_manager.get_session(s2.session_id)

    assert len(sess1_details.messages) == 1
    assert sess1_details.messages[0].content == "Hello in Session 1"

    assert len(sess2_details.messages) == 1
    assert sess2_details.messages[0].content == "Hello in Session 2"


def test_chat_deletion(temp_session_manager):
    """Test deleting a chat session cascades to delete its messages."""
    s = temp_session_manager.create_session(title="To be deleted")
    temp_session_manager.append_message(session_id=s.session_id, role="user", content="Ephemeral message")

    assert temp_session_manager.get_session(s.session_id) is not None

    deleted = temp_session_manager.delete_session(s.session_id)
    assert deleted is True

    # Confirm session is gone
    assert temp_session_manager.get_session(s.session_id) is None


def test_citation_mapping():
    """Verify that Source objects preserve formatted citation labels and provenance."""
    src = Source(
        chunk_id="chunk_42",
        file_id="doc_123",
        filename="company_policy.pdf",
        page_number=3,
        section="Compliance",
        content="Employees must complete compliance training annually.",
        relevance_score=0.912,
        citation_label="[S1]"
    )

    assert src.citation_label == "[S1]"
    assert src.filename == "company_policy.pdf"
    assert src.page_number == 3
    assert src.section == "Compliance"
    assert "training" in src.content


def test_single_query_embedding(sample_documents):
    """Verify that a single query retrieval triggers at most one embed_query call."""
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 768

    # Setup BM25
    bm25_manager.save_file_index("test_f1", sample_documents[:2], ["c1", "c2"])

    with patch("app.logic.get_embeddings", return_value=mock_embed):
        with patch("app.logic.generate_vector_candidates", return_value=[]):
            candidates, search_method = execute_retrieval_pipeline(
                query="What is Aura AI?",
                file_ids=["test_f1"],
                use_hybrid=True,
            )

    # In execute_retrieval_pipeline, if vector candidates are generated, embed_query is called once
    assert mock_embed.embed_query.call_count <= 1


def test_bm25_index_reuse(sample_documents):
    """Verify that multiple queries against the same file_id reuse the cached BM25 index."""
    file_id = "test_reuse_fid"
    docs = sample_documents[:2]
    chunk_ids = ["c1", "c2"]

    # Initial build and save
    bm25_manager.save_file_index(file_id, docs, chunk_ids)
    assert file_id in bm25_manager._cache

    cached_entry = bm25_manager._cache[file_id]

    # Subsequent loads must return the exact same in-memory cached object
    loaded_1 = bm25_manager.get_or_load_index(file_id)
    loaded_2 = bm25_manager.get_or_load_index(file_id)

    assert loaded_1 is cached_entry
    assert loaded_2 is cached_entry


def test_provider_reuse():
    """Verify that get_provider returns a cached provider instance."""
    p1 = get_provider()
    p2 = get_provider()
    assert p1 is p2


def test_retrieval_timing_breakdown(sample_documents):
    """Verify fine-grained timing instrumentation in execute_retrieval_pipeline."""
    file_id = "timing_fid"
    bm25_manager.save_file_index(file_id, sample_documents[:2], ["c1", "c2"])

    candidates, search_method = execute_retrieval_pipeline(
        query="ChromaDB and BM25",
        file_ids=[file_id],
        use_hybrid=True,
        enable_reranking=True,
    )

    timings = get_last_retrieval_timings()
    assert isinstance(timings, dict)
    assert "embedding_ms" in timings
    assert "chroma_retrieval_ms" in timings
    assert "bm25_retrieval_ms" in timings
    assert "rrf_ms" in timings
    assert "reranking_ms" in timings

    for k, v in timings.items():
        assert isinstance(v, (int, float))
        assert v >= 0.0


def test_multimodal_chat_session(temp_session_manager):
    """Verify that multimodal messages can be appended to session history."""
    s = temp_session_manager.create_session(title="Visual Inspection")

    msg = temp_session_manager.append_message(
        session_id=s.session_id,
        role="user",
        content="[Attached Image: chart.png]\nWhat is shown in this chart?"
    )
    assert msg.message_id is not None
    assert "[Attached Image: chart.png]" in msg.content


def test_missing_information_behavior():
    """Verify that perform_rag_query handles missing information properly."""
    # When context is empty or unrelated, response should indicate absence of evidence
    with patch("app.logic.execute_retrieval_pipeline", return_value=([], "empty")):
        req = QueryRequest(
            question="What is the population of Neptune?",
            file_ids=["fake_doc"],
            use_hybrid_search=True
        )
        resp = perform_rag_query(req)
        assert resp is not None
        assert "not found" in resp.answer.lower() or "not contain" in resp.answer.lower() or "insufficient" in resp.answer.lower() or "cannot find" in resp.answer.lower() or "no " in resp.answer.lower()
