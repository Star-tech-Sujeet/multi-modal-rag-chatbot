"""
Tests for Phase 11 — Intelligent Response Pipeline & Reasoning UX.

Validates:
1. Canonical pipeline stages, statuses, and QueryType models.
2. Safe status event schema: user-safe status events without exposing private
   chain-of-thought, internal deliberation, or hidden prompts.
3. Deterministic query classifier speed and routing (GENERAL_RAG, FOLLOW_UP,
   COMPARISON, IMAGE_QUERY, SQL_QUERY, NO_DOCUMENT_CONTEXT, GENERAL_CHAT).
4. PipelineStatusEmitter lifecycle management (running, complete, skipped, error).
5. Real-time SSE status event streaming integration and event sequence.
6. Toggle enforcement: enable_status=False / config-level suppression.
7. Adaptive pipeline execution:
   - Reranking skipped when enable_reranking=False
   - Query rewrite for follow-ups
   - Image processing for multimodal
   - SQL generation/validation for tabular queries
8. Interruption and error resilience:
   - status: error emitted cleanly on failure
   - SQLite persistence never saves partial or broken assistant answers
9. Streamlit UI helper rendering (render_pipeline_badge, render_pipeline_steps).
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.models import (
    PipelineStage,
    StageStatus,
    QueryType,
    PipelineStatusEvent,
    QueryRequest,
    SessionQueryRequest,
    Source,
)
from app.pipeline import (
    DEFAULT_STAGE_MESSAGES,
    classify_query,
    PipelineStatusEmitter,
    parse_pipeline_timestamp,
    calculate_pipeline_duration_seconds,
)
from app.logic import (
    perform_session_rag_query_stream,
    perform_rag_query_stream,
    RetrievalCandidate,
)
from app.sessions import SessionManager
from ui.streamlit_app import (
    render_pipeline_badge,
    render_pipeline_steps,
    _parse_pipeline_timestamp,
    _calculate_duration_seconds,
)


# =============================================================================
# 1. Pipeline Stages, Statuses, and Event Models
# =============================================================================

class TestPipelineModelsAndEnums:
    """Verifies enum values, defaults, and Pydantic validation."""

    def test_pipeline_stages_and_statuses(self):
        """Ensure all canonical stages and lifecycle statuses exist."""
        assert PipelineStage.INTAKE == "intake"
        assert PipelineStage.UNDERSTANDING == "understanding"
        assert PipelineStage.CLASSIFICATION == "classification"
        assert PipelineStage.RETRIEVAL == "retrieval"
        assert PipelineStage.RERANKING == "reranking"
        assert PipelineStage.GENERATION == "generation"
        assert PipelineStage.CITATION_VALIDATION == "citation_validation"
        assert PipelineStage.COMPLETION == "completion"
        assert PipelineStage.ERROR == "error"

        assert StageStatus.PENDING == "pending"
        assert StageStatus.RUNNING == "running"
        assert StageStatus.COMPLETE == "complete"
        assert StageStatus.SKIPPED == "skipped"
        assert StageStatus.ERROR == "error"

    def test_pipeline_status_event_schema(self):
        """Verify PipelineStatusEvent serialization and constraints."""
        event = PipelineStatusEvent(
            stage=PipelineStage.RETRIEVAL,
            status=StageStatus.RUNNING,
            message="Searching your documents...",
            timestamp="2026-09-15T07:10:00Z",
            metadata={"mode": "hybrid", "candidates": 10},
        )

        data = event.model_dump()
        assert data["stage"] == "retrieval"
        assert data["status"] == "running"
        assert data["message"] == "Searching your documents..."
        assert data["metadata"]["mode"] == "hybrid"
        assert isinstance(data["timestamp"], str)

        # Ensure json serialization is safe
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["stage"] == "retrieval"

    def test_default_stage_messages_are_safe(self):
        """Verify default messages contain no internal reasoning or private prompts."""
        forbidden_phrases = ["cot", "chain-of-thought", "hidden prompt", "deliberat", "system prompt", "scratchpad"]
        for stage, status_dict in DEFAULT_STAGE_MESSAGES.items():
            for status, message in status_dict.items():
                msg_lower = message.lower()
                for phrase in forbidden_phrases:
                    assert phrase not in msg_lower, f"Forbidden phrase '{phrase}' in stage '{stage}:{status}' message"


# =============================================================================
# 2. Deterministic Query Classifier
# =============================================================================

class TestDeterministicQueryClassifier:
    """Verifies fast, rule-based query classification across all intent types."""

    def test_classify_general_rag(self):
        q_type, stages = classify_query(
            "What are the key contractual obligations outlined in the agreement?",
            file_ids=["doc1"],
        )
        assert q_type == QueryType.GENERAL_RAG
        assert PipelineStage.RETRIEVAL in stages
        assert PipelineStage.GENERATION in stages

    def test_classify_follow_up(self):
        follow_ups = [
            "Can you explain that in more detail?",
            "What did you mean by it?",
            "Tell me more about them.",
            "Can you elaborate on the previous point?",
            "Why did this happen earlier?",
        ]
        for q in follow_ups:
            q_type, stages = classify_query(
                q,
                file_ids=["doc1"],
                recent_messages_count=2,
            )
            assert q_type == QueryType.FOLLOW_UP, f"Failed for query: {q}"
            assert PipelineStage.QUERY_REWRITE in stages

    def test_classify_comparison(self):
        comparison_queries = [
            "Compare document A with document B.",
            "What are the differences between both versions?",
            "Contrast the 2023 revenue vs 2024 projections.",
            "How does Q1 differ from Q2?",
        ]
        for q in comparison_queries:
            q_type, stages = classify_query(
                q,
                file_ids=["doc1", "doc2"],
            )
            assert q_type == QueryType.COMPARISON, f"Failed for query: {q}"
            assert PipelineStage.COMPARISON in stages

    def test_classify_image_query(self):
        q_type, stages = classify_query("Explain this document", has_image=True)
        assert q_type == QueryType.IMAGE_QUERY
        assert PipelineStage.IMAGE_PROCESSING in stages

    def test_classify_sql_query(self):
        q_type, stages = classify_query(
            "Show the sum of total profits across all products.",
            file_ids=["table1"],
            has_structured_files=True,
            is_structured_intent=True,
        )
        assert q_type == QueryType.SQL_QUERY
        assert PipelineStage.SQL_SCHEMA in stages
        assert PipelineStage.SQL_GENERATION in stages
        assert PipelineStage.SQL_EXECUTION in stages

    def test_classify_no_context_and_general_chat(self):
        # General chat greeting
        q_type1, stages1 = classify_query("Hello Aura, how are you today?", file_ids=[])
        assert q_type1 == QueryType.GENERAL_CHAT

        # Knowledge question without document context
        q_type2, stages2 = classify_query("Who was Leonardo da Vinci?", file_ids=[])
        assert q_type2 == QueryType.NO_DOCUMENT_CONTEXT

    def test_classifier_is_ultra_fast(self):
        """Ensure 1000 classifications take well under 100ms (zero network/LLM overhead)."""
        queries = [
            ("What are the main risks?", ["doc1"]),
            ("Compare A and B", ["doc1", "doc2"]),
            ("What does it mean?", ["doc1"]),
            ("Hello there", []),
        ] * 250

        t0 = time.perf_counter()
        for q, fids in queries:
            classify_query(q, file_ids=fids)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        assert duration_ms < 100.0, f"Classifier took {duration_ms:.2f}ms for 1000 queries, expected < 100ms"


# =============================================================================
# 3. PipelineStatusEmitter Lifecycle
# =============================================================================

class TestPipelineStatusEmitter:
    """Verifies incremental state tracking and event emission."""

    def test_emitter_chronological_lifecycle(self):
        emitter = PipelineStatusEmitter(enabled=True)

        e1 = emitter.create_event(
            stage=PipelineStage.UNDERSTANDING,
            status=StageStatus.RUNNING,
        )
        assert e1 is not None
        assert e1["stage"] == PipelineStage.UNDERSTANDING
        assert e1["status"] == StageStatus.RUNNING
        assert "Understanding" in e1["message"]

        e2 = emitter.create_event(
            stage=PipelineStage.UNDERSTANDING,
            status=StageStatus.COMPLETE,
            message="Question parsed and verified",
            metadata={"tokens": 12},
        )
        assert e2 is not None
        assert e2["status"] == StageStatus.COMPLETE
        assert e2["message"] == "Question parsed and verified"
        assert e2["metadata"]["tokens"] == 12

        e3 = emitter.create_event(
            stage=PipelineStage.RERANKING,
            status=StageStatus.SKIPPED,
            message="Reranking disabled in configuration",
        )
        assert e3 is not None
        assert e3["status"] == StageStatus.SKIPPED

        history = emitter.get_history()
        assert len(history) == 3
        assert history[0]["stage"] == "understanding"
        assert history[0]["status"] == "running"
        assert history[1]["status"] == "complete"
        assert history[2]["stage"] == "reranking"
        assert history[2]["status"] == "skipped"

    def test_emitter_disabled_behavior(self):
        """When enabled=False, emitter returns None for all calls."""
        emitter = PipelineStatusEmitter(enabled=False)
        assert emitter.create_event(PipelineStage.RETRIEVAL, StageStatus.RUNNING) is None
        assert emitter.create_event(PipelineStage.RETRIEVAL, StageStatus.COMPLETE) is None
        assert emitter.create_event(PipelineStage.RERANKING, StageStatus.SKIPPED) is None
        assert emitter.get_history() == []


# =============================================================================
# 4. SSE Stream Status Event Integration
# =============================================================================

class TestPipelineSSEStreamIntegration:
    """Verifies real SSE status event emission in the streaming pipelines."""

    @pytest.fixture
    def test_session_mgr(self, tmp_path):
        db_file = tmp_path / "test_status_sessions.db"
        mgr = SessionManager(db_path=str(db_file))
        return mgr

    @pytest.fixture
    def sample_candidate(self):
        from langchain.schema import Document
        meta = {
            "file_id": "file1",
            "filename": "annual_report.pdf",
            "page_number": 3,
            "source_type": "pdf",
            "chunk_index": 0,
        }
        doc = Document(
            page_content="In 2023, the revenue grew by 15% following strategic expansions.",
            metadata=meta
        )
        cand = RetrievalCandidate(
            doc=doc,
            chunk_id="file1_chunk_0",
            final_score=0.92,
            search_type="hybrid",
            metadata=meta
        )
        cand.file_id = "file1"
        return cand

    def test_session_stream_emits_status_events(
        self, test_session_mgr, sample_candidate
    ):
        """Verify full status event lifecycle in perform_session_rag_query_stream."""
        session = test_session_mgr.create_session(title="Status Test")
        session_id = session.session_id

        req = SessionQueryRequest(
            question="What was the revenue growth?",
            file_ids=["file1"],
            enable_status=True,
        )

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm, \
             patch("app.logic.validate_citations") as mock_val:

            mock_retrieval.return_value = ([sample_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["According to ", "[S1], ", "revenue grew 15%."]),
                settings.gemini_model
            )
            mock_val.return_value = (
                "According to [S1], revenue grew 15%.",
                {"S1"},
                {
                    "is_valid": True,
                    "has_invalid_citations": False,
                    "hallucinated_citations": [],
                    "verified_citations": ["S1"],
                    "total_expected": 1,
                }
            )

            generator = perform_session_rag_query_stream(
                session_id,
                req,
                mgr=test_session_mgr,
            )

            raw_events = list(generator)
            event_types = []
            status_stages = []

            for raw in raw_events:
                lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
                ev_type = None
                ev_data = None
                for line in lines:
                    if line.startswith("event:"):
                        ev_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        ev_data = json.loads(line[len("data:"):].strip())

                if ev_type:
                    event_types.append(ev_type)
                    if ev_type == "status":
                        status_stages.append((ev_data.get("stage"), ev_data.get("status")))

            # Assert overall event sequence
            assert "start" in event_types
            assert "status" in event_types
            assert "sources" in event_types
            assert "token" in event_types
            assert "complete" in event_types

            # Verify canonical stages were emitted
            stages_seen = [stage for stage, status in status_stages]
            assert "understanding" in stages_seen
            assert "classification" in stages_seen
            assert "retrieval" in stages_seen
            assert "evidence_selection" in stages_seen
            assert "generation" in stages_seen
            assert "citation_validation" in stages_seen
            assert "completion" in stages_seen

    def test_enable_status_false_suppresses_status_events(
        self, test_session_mgr, sample_candidate
    ):
        """When enable_status=False, NO status events should be emitted."""
        session = test_session_mgr.create_session(title="No Status")
        session_id = session.session_id

        req = SessionQueryRequest(
            question="What was the revenue growth?",
            file_ids=["file1"],
            enable_status=False,
        )

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([sample_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["Revenue was up."]),
                settings.gemini_model
            )

            generator = perform_session_rag_query_stream(
                session_id,
                req,
                mgr=test_session_mgr,
            )

            raw_events = list(generator)
            event_types = []
            for raw in raw_events:
                for line in raw.split("\n"):
                    if line.startswith("event:"):
                        event_types.append(line[len("event:"):].strip())

            assert "status" not in event_types
            assert "start" in event_types
            assert "sources" in event_types
            assert "token" in event_types
            assert "complete" in event_types

    def test_skipped_stage_when_reranking_disabled(
        self, test_session_mgr, sample_candidate
    ):
        """When enable_reranking=False, reranking stage should emit status='skipped'."""
        session = test_session_mgr.create_session(title="Skip Rerank")
        session_id = session.session_id

        req = SessionQueryRequest(
            question="What was the revenue growth?",
            file_ids=["file1"],
            enable_reranking=False,
            enable_status=True,
        )

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream") as mock_stream_llm:

            mock_retrieval.return_value = ([sample_candidate], "hybrid")
            mock_stream_llm.return_value = (
                iter(["Revenue was up."]),
                settings.gemini_model
            )

            generator = perform_session_rag_query_stream(
                session_id,
                req,
                mgr=test_session_mgr,
            )

            status_events = []
            for raw in generator:
                if "event: status" in raw:
                    for line in raw.split("\n"):
                        if line.startswith("data:"):
                            status_events.append(json.loads(line[len("data:"):].strip()))

            rerank_event = next((e for e in status_events if e.get("stage") == "reranking"), None)
            assert rerank_event is not None
            assert rerank_event["status"] == "skipped"

    def test_interruption_error_safety_no_partial_assistant_persisted(
        self, test_session_mgr, sample_candidate
    ):
        """
        When stream raises an error mid-flight:
        - status: error or event: error must be emitted
        - partial assistant message MUST NEVER be persisted into SQLite.
        """
        session = test_session_mgr.create_session(title="Error Safety")
        session_id = session.session_id

        req = SessionQueryRequest(
            question="What is the forecast?",
            file_ids=["file1"],
            enable_status=True,
        )

        def _failing_stream(*args, **kwargs):
            def _gen():
                yield "First token "
                raise RuntimeError("Mid-stream connection drop!")
            return _gen(), settings.gemini_model

        with patch("app.logic.execute_retrieval_pipeline") as mock_retrieval, \
             patch("app.logic.handle_text_query_stream", side_effect=_failing_stream):

            mock_retrieval.return_value = ([sample_candidate], "hybrid")

            generator = perform_session_rag_query_stream(
                session_id,
                req,
                mgr=test_session_mgr,
            )

            raw_events = list(generator)
            has_error_status = any("stage\": \"error\"" in r or "event: error" in r for r in raw_events)
            assert has_error_status

            # Verify SQLite consistency: user message is saved, but assistant message is NOT saved
            stored_session = test_session_mgr.get_session(session_id)
            messages = stored_session.messages
            assert len(messages) == 1
            assert messages[0].role == "user"
            assert messages[0].content == "What is the forecast?"


# =============================================================================
# 5. Streamlit UI Rendering Components
# =============================================================================

class TestStreamlitStatusComponents:
    """Verifies frontend HTML rendering helpers."""

    def test_render_pipeline_badge(self):
        stages = [
            {"stage": "retrieval", "status": "complete", "timestamp": 100.0},
            {"stage": "completion", "status": "complete", "timestamp": 101.42},
        ]
        html = render_pipeline_badge(stages, timings={"total_s": 1.42}, source_count=3)
        assert "aura-pipeline-badge" in html
        assert "Response ready" in html
        assert "1.42s" in html
        assert "3 sources verified" in html
        # Zero CoT exposure
        assert "prompt" not in html.lower()

    def test_render_pipeline_steps(self):
        stages = [
            {
                "stage": "retrieval",
                "label": "Retrieving context",
                "status": "complete",
                "message": "Found 4 relevant passages using hybrid retrieval",
                "details": {"count": 4, "mode": "hybrid"},
            },
            {
                "stage": "reranking",
                "label": "Neural reranking",
                "status": "skipped",
                "message": "Reranking disabled in configuration",
                "details": {},
            },
        ]

        with patch("streamlit.expander") as mock_expander, \
             patch("streamlit.markdown") as mock_markdown:

            render_pipeline_steps(stages)
            assert mock_expander.called
            assert mock_markdown.called
            # Ensure markdown rendered rows with classes
            args, _ = mock_markdown.call_args
            rendered_chunk = args[0]
            assert "aura-pipeline-row" in rendered_chunk
            assert "Neural reranking" in rendered_chunk

    def test_render_pipeline_badge_exact_regression_iso_strings(self):
        """
        Regression test for Phase 11 runtime TypeError:
        unsupported operand type(s) for -: 'str' and 'str'
        when first_ts and last_ts are ISO-8601 strings and timings has no total_s.
        """
        stages = [
            {"stage": "intake", "status": "complete", "timestamp": "2026-09-15T06:30:12.000000"},
            {"stage": "retrieval", "status": "complete", "timestamp": "2026-09-15T06:30:12.800000"},
            {"stage": "completion", "status": "complete", "timestamp": "2026-09-15T06:30:13.420000"},
        ]
        # timings is None: forces stages calculation using ISO string subtraction
        html = render_pipeline_badge(stages, timings=None, source_count=2)
        assert "aura-pipeline-badge" in html
        assert "Response ready in 1.42s · 2 sources verified" in html

    def test_render_pipeline_badge_with_backend_timings_dict(self):
        """
        Backend emits total_request_time in ms and processing_time_ms.
        render_pipeline_badge should prioritize/utilize these keys cleanly.
        """
        stages = [
            {"stage": "retrieval", "status": "complete", "timestamp": "2026-09-15T06:30:12.000000"},
            {"stage": "completion", "status": "complete", "timestamp": "2026-09-15T06:30:13.420000"},
        ]
        # Case A: total_request_time in milliseconds
        html_a = render_pipeline_badge(stages, timings={"total_request_time": 1850.0}, source_count=1)
        assert "1.85s" in html_a
        assert "1 source verified" in html_a

        # Case B: processing_time_ms
        html_b = render_pipeline_badge(stages, timings={"processing_time_ms": 2340}, source_count=0)
        assert "2.34s" in html_b

    def test_render_pipeline_badge_timestamp_resilience(self):
        """Ensure malformed or unusual timestamps never raise an unhandled exception."""
        # Malformed strings
        stages_bad = [
            {"stage": "intake", "status": "complete", "timestamp": "invalid-timestamp"},
            {"stage": "completion", "status": "complete", "timestamp": "corrupted"},
        ]
        html_bad = render_pipeline_badge(stages_bad, timings=None)
        assert "aura-pipeline-badge" in html_bad
        assert "Response ready" in html_bad

        # Single stage (first_ts == last_ts)
        stages_single = [
            {"stage": "intake", "status": "complete", "timestamp": "2026-09-15T06:30:12.000000"}
        ]
        html_single = render_pipeline_badge(stages_single, timings=None)
        assert "aura-pipeline-badge" in html_single

        # None / empty stages
        assert "aura-pipeline-badge" in render_pipeline_badge(None, timings=None)
        assert "aura-pipeline-badge" in render_pipeline_badge([], timings=None)


class TestPipelineTimestampHandling:
    """Verifies timestamp parsing and duration calculations across formats."""

    def test_parse_iso8601_variants(self):
        dt_naive = parse_pipeline_timestamp("2026-09-15T06:30:12.000000")
        assert dt_naive is not None
        assert dt_naive.year == 2026 and dt_naive.hour == 6 and dt_naive.minute == 30

        dt_z = parse_pipeline_timestamp("2026-09-15T06:30:12Z")
        assert dt_z is not None
        assert dt_z.tzinfo is not None

        dt_offset = parse_pipeline_timestamp("2026-09-15T06:30:12+00:00")
        assert dt_offset is not None

    def test_parse_epoch_and_numeric_strings(self):
        dt_float = parse_pipeline_timestamp(1726381812.34)
        assert dt_float is not None

        dt_str_num = parse_pipeline_timestamp("1726381812.34")
        assert dt_str_num is not None

        dt_int = parse_pipeline_timestamp(1726381812)
        assert dt_int is not None

    def test_parse_datetime_objects(self):
        now = datetime.now()
        dt = parse_pipeline_timestamp(now)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_invalid_and_edge_cases(self):
        assert parse_pipeline_timestamp(None) is None
        assert parse_pipeline_timestamp("") is None
        assert parse_pipeline_timestamp("   ") is None
        assert parse_pipeline_timestamp("not-a-date") is None

    def test_calculate_pipeline_duration_seconds(self):
        # Difference between ISO strings
        t1 = "2026-09-15T06:30:12.000000"
        t2 = "2026-09-15T06:30:13.420000"
        diff = calculate_pipeline_duration_seconds(t1, t2)
        assert diff is not None
        assert abs(diff - 1.42) < 1e-4

        # Mixed ISO with Z and naive ISO
        t1_z = "2026-09-15T06:30:12Z"
        diff_z = calculate_pipeline_duration_seconds(t1_z, t2)
        assert diff_z is not None
        assert abs(diff_z - 1.42) < 1e-4

        # Inverted timestamps (negative duration returns None)
        assert calculate_pipeline_duration_seconds(t2, t1) is None

        # None / invalid timestamps
        assert calculate_pipeline_duration_seconds(None, t2) is None
        assert calculate_pipeline_duration_seconds(t1, None) is None
        assert calculate_pipeline_duration_seconds("bad", t2) is None
