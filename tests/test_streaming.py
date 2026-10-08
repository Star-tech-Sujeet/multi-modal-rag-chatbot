"""
Tests for REAL Token Streaming via Gemini Provider, FastAPI SSE, and Streamlit Integration.

Verifies:
1. GeminiProvider.generate_text_stream native content streaming and chunk filtering.
2. GeminiProvider error handling during stream iteration (quota 429, 503).
3. OpenAIProvider.generate_text_stream fallback streaming.
4. handle_text_query_stream routing and execution.
5. format_sse_event protocol adherence.
6. perform_session_rag_query_stream SSE event sequencing (start -> sources -> token -> complete).
7. Single persistent SQLite assistant message guarantee upon stream completion.
8. Safety guarantee: No broken/incomplete assistant message saved if streaming errors midway.
9. perform_rag_query_stream standard RAG streaming.
10. FastAPI SSE endpoints: POST /sessions/{id}/query/stream and POST /query/stream.
11. UI helper query_document_stream SSE line parsing and error handling.
"""

import os
import uuid
import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models import (
    Source,
    SessionCreateRequest,
    SessionQueryRequest,
    QueryRequest,
)
from app.sessions import SessionManager
from app.providers.gemini import (
    GeminiProvider,
    GeminiQuotaExceededError,
    GeminiServiceUnavailableError,
)
from app.providers.openai_provider import OpenAIProvider
from app.logic import (
    format_sse_event,
    handle_text_query_stream,
    perform_session_rag_query_stream,
    perform_rag_query_stream,
    RetrievalCandidate,
)
from ui.streamlit_app import query_document_stream


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_db_path(tmp_path):
    db_file = tmp_path / f"test_streaming_{uuid.uuid4().hex}.sqlite"
    return str(db_file)


@pytest.fixture
def session_mgr(temp_db_path):
    mgr = SessionManager(db_path=temp_db_path)
    return mgr


from langchain.schema import Document


@pytest.fixture
def mock_candidate():
    file_id = str(uuid.uuid4())
    meta = {
        "file_id": file_id,
        "filename": "quarterly_report.pdf",
        "page_number": 3,
        "source_type": "pdf",
        "ingestion_method": "pymupdf",
        "chunk_index": 0,
    }
    doc = Document(
        page_content="The total revenue reached $10 million in Q3 with positive cash flows.",
        metadata=meta
    )
    cand = RetrievalCandidate(
        doc=doc,
        chunk_id=f"{file_id}_chunk_0",
        final_score=0.95,
        search_type="hybrid",
        metadata=meta
    )
    cand.file_id = file_id
    return cand


# =============================================================================
# 1. GeminiProvider Streaming Tests
# =============================================================================

class TestGeminiStreaming:
    """Tests for native Gemini content streaming."""

    def test_gemini_generate_text_stream_yields_chunks(self):
        """generate_text_stream should invoke generate_content_stream and yield text chunks."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # Setup mock chunks
            chunk1 = MagicMock()
            chunk1.text = "Based on "
            chunk2 = MagicMock()
            chunk2.text = "the evidence [S1], "
            chunk3 = MagicMock()
            chunk3.text = "revenue was $10M."
            mock_client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            stream_iter, model_used = provider.generate_text_stream(
                question="What was revenue?",
                context="[SOURCE S1] Revenue was $10M."
            )

            assert model_used == settings.gemini_model
            chunks = list(stream_iter)
            assert chunks == ["Based on ", "the evidence [S1], ", "revenue was $10M."]

    def test_gemini_generate_text_stream_skips_empty_chunks(self):
        """generate_text_stream should skip chunks with empty or None text."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            chunk1 = MagicMock(text="Hello ")
            chunk2 = MagicMock(text=None)
            chunk3 = MagicMock(text="")
            chunk4 = MagicMock(text="World!")
            mock_client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3, chunk4]

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            stream_iter, _ = provider.generate_text_stream(
                question="Hi",
                context="Context"
            )
            chunks = list(stream_iter)
            assert chunks == ["Hello ", "World!"]

    def test_gemini_generate_text_stream_quota_error_mapping(self):
        """generate_text_stream should map 429 quota error to GeminiQuotaExceededError."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            def error_gen():
                yield MagicMock(text="Starting...")
                raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

            mock_client.models.generate_content_stream.return_value = error_gen()

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            stream_iter, _ = provider.generate_text_stream(
                question="Hi",
                context="Context"
            )

            # First token yields fine
            assert next(stream_iter) == "Starting..."
            # Next iteration triggers mapped quota exception
            with pytest.raises(GeminiQuotaExceededError):
                next(stream_iter)

    def test_gemini_generate_text_stream_503_error_mapping(self):
        """generate_text_stream should map 503 error to GeminiServiceUnavailableError."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            def error_gen():
                yield MagicMock(text="Initial chunk ")
                raise Exception("503 Service Unavailable: high demand")

            mock_client.models.generate_content_stream.return_value = error_gen()

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            stream_iter, _ = provider.generate_text_stream(
                question="Hi",
                context="Context"
            )
            assert next(stream_iter) == "Initial chunk "
            with pytest.raises(GeminiServiceUnavailableError):
                next(stream_iter)


# =============================================================================
# 2. OpenAI Provider Streaming Tests
# =============================================================================

class TestOpenAIStreaming:
    """Tests for OpenAI provider fallback streaming."""

    def test_openai_generate_text_stream(self):
        """OpenAIProvider generate_text_stream should yield content from delta chunks."""
        with patch("app.providers.openai_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            c1 = MagicMock()
            c1.choices = [MagicMock(delta=MagicMock(content="Token1 "))]
            c2 = MagicMock()
            c2.choices = [MagicMock(delta=MagicMock(content="Token2"))]
            mock_client.chat.completions.create.return_value = [c1, c2]

            provider = OpenAIProvider(api_key="sk-testkey123456789")
            stream_iter, model_used = provider.generate_text_stream(
                question="Hello",
                context="Context"
            )
            assert list(stream_iter) == ["Token1 ", "Token2"]


# =============================================================================
# 3. Logic & SSE Formatting Tests
# =============================================================================

class TestLogicAndSSEFormatting:
    """Tests for SSE formatting and logic stream handler."""

    def test_format_sse_event(self):
        """format_sse_event should conform to the standard SSE event format."""
        evt = format_sse_event("token", {"text": "hello"})
        assert evt == 'event: token\ndata: {"text": "hello"}\n\n'

    def test_handle_text_query_stream_routing(self):
        """handle_text_query_stream should resolve provider and return stream iterator."""
        with patch("app.logic.get_provider") as mock_get_prov:
            mock_prov = MagicMock()
            mock_prov.generate_text_stream.return_value = (iter(["Streamed ", "text"]), "gemini-3.1-flash-lite")
            mock_get_prov.return_value = mock_prov

            stream_iter, model = handle_text_query_stream(
                question="Test question",
                context="Test context"
            )
            assert model == "gemini-3.1-flash-lite"
            assert list(stream_iter) == ["Streamed ", "text"]


# =============================================================================
# 4. Session RAG Query Streaming Tests
# =============================================================================

class TestSessionRagQueryStream:
    """Tests for perform_session_rag_query_stream."""

    def test_stream_session_not_found(self, session_mgr):
        """Streaming query for non-existent session yields error event."""
        req = SessionQueryRequest(question="What is this?")
        events = list(perform_session_rag_query_stream("00000000-0000-0000-0000-000000000000", req, mgr=session_mgr))
        assert any("event: error" in e for e in events)
        assert any("not found" in e for e in events)

    def test_stream_empty_question(self, session_mgr):
        """Streaming query with empty question yields error event."""
        session = session_mgr.create_session(title="Test")
        req = SessionQueryRequest(question="   ")
        events = list(perform_session_rag_query_stream(session.session_id, req, mgr=session_mgr))
        assert any("event: error" in e for e in events)
        assert any("cannot be empty" in e for e in events)

    def test_stream_no_targets(self, session_mgr):
        """Streaming query without targets yields start, token, and complete events."""
        session = session_mgr.create_session(title="Test")
        req = SessionQueryRequest(question="Hello there")
        events = list(perform_session_rag_query_stream(session.session_id, req, mgr=session_mgr))

        event_types = [
            chunk.split("\n")[0].replace("event: ", "").strip()
            for chunk in events
            if chunk.startswith("event: ")
        ]
        assert [e for e in event_types if e != "status"] == ["start", "token", "complete"]
        assert "status" in event_types

        # Check that user and assistant messages were persisted to SQLite
        messages = session_mgr.get_session_messages(session.session_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello there"
        assert messages[1].role == "assistant"
        assert "No document or image specified" in messages[1].content

    def test_stream_success_and_single_sqlite_persistence(self, session_mgr, mock_candidate):
        """Successful stream emits full sequence and persists exactly ONE completed assistant message."""
        session = session_mgr.create_session(title="Test Q3")
        fid = mock_candidate.file_id
        req = SessionQueryRequest(question="What was the revenue in Q3?", file_id=fid)

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([mock_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["According to ", "[S1], ", "revenue was $10M."]),
                "gemini-3.1-flash-lite"
            )

            raw_events = list(perform_session_rag_query_stream(session.session_id, req, mgr=session_mgr))

            # Parse event types
            event_types = []
            tokens_received = []
            complete_data = None
            sources_data = None

            for evt_str in raw_events:
                lines = evt_str.strip().split("\n")
                evt_name = lines[0].replace("event: ", "").strip()
                data_obj = json.loads(lines[1].replace("data: ", "").strip())
                event_types.append(evt_name)

                if evt_name == "token":
                    tokens_received.append(data_obj["text"])
                elif evt_name == "sources":
                    sources_data = data_obj
                elif evt_name == "complete":
                    complete_data = data_obj

            assert [e for e in event_types if e != "status"] == ["start", "sources", "token", "token", "token", "complete"]
            assert "status" in event_types
            assert tokens_received == ["According to ", "[S1], ", "revenue was $10M."]
            assert complete_data is not None
            assert complete_data["answer"] == "According to [S1], revenue was $10M."
            assert complete_data["model_used"] == "gemini-3.1-flash-lite"
            assert len(complete_data["sources"]) == 1
            assert sources_data is not None
            assert len(sources_data["sources"]) == 1

            # CRITICAL CHECK: Exactly ONE user message and ONE assistant message persisted
            messages = session_mgr.get_session_messages(session.session_id)
            assert len(messages) == 2
            assert messages[0].role == "user"
            assert messages[0].content == "What was the revenue in Q3?"
            assert messages[1].role == "assistant"
            assert messages[1].content == "According to [S1], revenue was $10M."
            assert len(messages[1].sources) == 1
            src0 = messages[1].sources[0]
            cid = src0.citation_id if hasattr(src0, "citation_id") else src0["citation_id"]
            assert cid == "S1"

    def test_stream_interrupted_error_does_not_persist_broken_assistant_message(self, session_mgr, mock_candidate):
        """If streaming errors midway, NO assistant message should be persisted to SQLite."""
        session = session_mgr.create_session(title="Test Interruption")
        fid = mock_candidate.file_id
        req = SessionQueryRequest(question="What was revenue?", file_id=fid)

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([mock_candidate], "hybrid")

            def failing_token_generator():
                yield "Revenue is "
                raise GeminiQuotaExceededError("Daily limit hit")

            mock_stream_llm.return_value = (failing_token_generator(), "gemini-3.1-flash-lite")

            raw_events = list(perform_session_rag_query_stream(session.session_id, req, mgr=session_mgr))

            event_types = [
                chunk.split("\n")[0].replace("event: ", "").strip()
                for chunk in raw_events
                if chunk.startswith("event: ")
            ]
            assert "start" in event_types
            assert "sources" in event_types
            assert "token" in event_types
            assert "error" in event_types
            assert "complete" not in event_types

            # Verify SQLite: User message exists, but NO partial assistant message exists!
            messages = session_mgr.get_session_messages(session.session_id)
            assert len(messages) == 1
            assert messages[0].role == "user"


# =============================================================================
# 5. Standard RAG Streaming Tests
# =============================================================================

class TestPerformRagQueryStream:
    """Tests for non-session perform_rag_query_stream."""

    def test_perform_rag_query_stream_success(self, mock_candidate):
        """Standard RAG query stream yields start, sources, token, complete."""
        req = QueryRequest(question="What is the revenue?", file_id=mock_candidate.file_id)

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([mock_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["Revenue ", "is $10M [S1]."]),
                "gemini-3.1-flash-lite"
            )

            raw_events = list(perform_rag_query_stream(req))
            event_types = [
                chunk.split("\n")[0].replace("event: ", "").strip()
                for chunk in raw_events
                if chunk.startswith("event: ")
            ]
            assert [e for e in event_types if e != "status"] == ["start", "sources", "token", "token", "complete"]
            assert "status" in event_types


# =============================================================================
# 6. FastAPI SSE Endpoints Tests
# =============================================================================

class TestApiStreamingEndpoints:
    """Tests for FastAPI SSE streaming endpoints."""

    def test_api_session_query_stream_success(self, client, mock_candidate):
        """POST /sessions/{id}/query/stream returns 200 with text/event-stream."""
        # 1. Create session
        res_create = client.post("/api/v1/sessions", json={"title": "Stream Test"})
        assert res_create.status_code == 201
        sid = res_create.json()["session_id"]

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([mock_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["Chunk1 ", "Chunk2 [S1]"]),
                "gemini-3.1-flash-lite"
            )

            payload = {
                "question": "What is in the report?",
                "file_id": mock_candidate.file_id
            }
            res_stream = client.post(f"/api/v1/sessions/{sid}/query/stream", json=payload)
            assert res_stream.status_code == 200
            assert "text/event-stream" in res_stream.headers["content-type"]
            body = res_stream.text
            assert "event: start" in body
            assert "event: sources" in body
            assert "event: token" in body
            assert "event: complete" in body

    def test_api_session_query_stream_404_on_missing_session(self, client):
        """POST /sessions/{id}/query/stream returns 404 for unknown session."""
        fake_sid = str(uuid.uuid4())
        res = client.post(f"/api/v1/sessions/{fake_sid}/query/stream", json={"question": "Hello"})
        assert res.status_code == 404

    def test_api_query_stream_endpoint(self, client, mock_candidate):
        """POST /query/stream returns 200 with text/event-stream."""
        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([mock_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["ChunkA ", "ChunkB"]),
                "gemini-3.1-flash-lite"
            )

            payload = {
                "question": "Query streaming test?",
                "file_id": mock_candidate.file_id
            }
            res = client.post("/api/v1/query/stream", json=payload)
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]
            assert "event: token" in res.text

    def test_api_query_stream_validation_errors(self, client):
        """POST /query/stream validates empty questions and file_ids."""
        # Empty question -> 422 or 400
        res1 = client.post("/api/v1/query/stream", json={"question": "", "file_id": str(uuid.uuid4())})
        assert res1.status_code in (400, 422)

        # Empty file_ids -> 400
        res2 = client.post("/api/v1/query/stream", json={"question": "Valid question?", "file_id": None})
        assert res2.status_code == 400

        # Invalid file_id format -> 400
        res3 = client.post("/api/v1/query/stream", json={"question": "Valid?", "file_id": "not-a-uuid"})
        assert res3.status_code == 400


# =============================================================================
# 7. Streamlit SSE Consumer Helper Tests
# =============================================================================

class MockSessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, item, value):
        self[item] = value


class TestStreamlitStreamConsumer:
    """Tests for query_document_stream helper in Streamlit UI."""

    def test_query_document_stream_parses_sse_lines(self):
        """query_document_stream parses SSE event and data lines properly."""
        mock_sse_lines = [
            'event: start',
            'data: {"session_id": "123", "model": "gemini-3.1-flash-lite"}',
            '',
            'event: token',
            'data: {"text": "Hello "}',
            '',
            'event: token',
            'data: {"text": "World"}',
            '',
            'event: complete',
            'data: {"answer": "Hello World", "sources": []}',
            ''
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = mock_sse_lines

        state = MockSessionState({
            "active_session_id": str(uuid.uuid4()),
            "file_id": str(uuid.uuid4()),
            "multi_doc_mode": False,
            "selected_file_ids": [],
            "use_hybrid_search": True,
            "max_sources": 5
        })

        with patch("ui.streamlit_app.requests.post", return_value=mock_resp), \
             patch("ui.streamlit_app.st.session_state", state):

            events = list(query_document_stream("What is this?"))
            assert len(events) == 4
            assert events[0][0] == "start"
            assert events[1] == ("token", {"text": "Hello "})
            assert events[2] == ("token", {"text": "World"})
            assert events[3][0] == "complete"
