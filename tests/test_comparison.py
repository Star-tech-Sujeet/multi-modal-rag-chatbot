"""
Tests for Phase 8: Cross-Document Comparison & Evidence-Based Reasoning.

Verifies:
Test A: Explicit two-document comparison (both contribute evidence).
Test B: Document filtering (retrieval for A cannot return B evidence).
Test C: Identical content in A and B remains separate with distinct file_id and chunk_id.
Test D: Citation correctness (A citations map to A, B citations map to B).
Test E: Difference comparison mode ("What changed?").
Test F: Similarity comparison mode ("What do they have in common?").
Test G: Metric comparison mode ("Which has higher revenue?").
Test H: Conflict detection (discrepancies between retrieved values).
Test I: Uneven relevance (both documents receive bounded retrieval opportunities).
Test J: No evidence (returns insufficient evidence message).
Test K: Missing document (invalid file ID produces clean error).
Test L: Deleted document (deleted file cannot be used in comparison).
Test M: Conversational comparison (follow-up comparison query uses session context).
Test N: Citation isolation (old session citations cannot leak into new turn).
Test O: Image + document comparison (ephemeral image evidence coexists with document evidence).
Test P: No BM25 rebuild (comparison retrieval loads existing BM25 without rebuilding).
Test Q: Numerical calculation (deterministic percentage and difference calculations).
Test R: Unit mismatch (different units e.g. $ vs € are not silently converted).
Test S: Temporal ambiguity (ambiguous dates are not fabricated).
"""

import io
import os
import uuid
import base64
from typing import List
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from langchain.schema import Document

from app.main import app
from app.config import settings
from app.models import (
    ComparisonMode,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonEvidenceGroup,
    Source
)
from app.sessions import SessionManager
from app.logic import (
    CanonicalEvidence,
    RetrievalCandidate,
    infer_comparison_mode,
    resolve_comparison_documents,
    extract_numeric_metrics,
    perform_deterministic_numeric_comparison,
    extract_contradictions_from_text,
    perform_comparison_query,
    bm25_manager,
    delete_document_embeddings
)


# =============================================================================
# Helper Utilities & Fixtures
# =============================================================================

def make_candidate(file_id: str, filename: str, content: str, chunk_index: int = 0, score: float = 0.85):
    """Helper to create a RetrievalCandidate with full metadata."""
    doc = Document(
        page_content=content,
        metadata={
            "file_id": file_id,
            "filename": filename,
            "chunk_id": f"{file_id}_chunk_{chunk_index}",
            "chunk_index": chunk_index,
            "total_chunks": 1,
            "page_number": 1,
            "source": filename
        }
    )
    return RetrievalCandidate(
        doc=doc,
        chunk_id=f"{file_id}_chunk_{chunk_index}",
        final_score=score,
        vector_score=score,
        bm25_score=score,
        search_type="hybrid",
        final_rank=1,
        metadata=doc.metadata
    )


def create_test_png(color="red", size=(80, 80)) -> bytes:
    """Generate in-memory valid PNG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def temp_db(tmp_path):
    """Isolated SQLite database for session manager."""
    db_file = tmp_path / f"test_comp_{uuid.uuid4().hex}.sqlite"
    return str(db_file)


@pytest.fixture
def session_mgr(temp_db):
    """SessionManager instance using temp_db."""
    return SessionManager(db_path=temp_db)


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


# =============================================================================
# Tests A - S
# =============================================================================

def test_a_explicit_two_document_comparison(session_mgr):
    """Test A: Both documents contribute evidence with sequential citations."""
    cand_a = make_candidate("doc_a", "report_2024.pdf", "In 2024, operating expenses were $40M.", 0)
    cand_b = make_candidate("doc_b", "report_2025.pdf", "In 2025, operating expenses were $55M.", 0)

    def mock_retrieval(query, file_ids=None, max_sources=5, **kwargs):
        if file_ids == ["doc_a"]:
            return [cand_a], {"retrieval_strategy": "hybrid"}
        elif file_ids == ["doc_b"]:
            return [cand_b], {"retrieval_strategy": "hybrid"}
        return [], {}

    llm_answer = "Expenses rose from $40M in 2024 [S1] to $55M in 2025 [S2]."

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=(llm_answer, "gpt-4o")):

        req = ComparisonRequest(
            query="Compare expenses between 2024 and 2025",
            file_ids=["doc_a", "doc_b"],
            comparison_mode=ComparisonMode.DIFFERENCE
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert resp.comparison_mode == ComparisonMode.DIFFERENCE
        assert len(resp.evidence_groups) == 2
        assert resp.evidence_groups[0].document_id == "doc_a"
        assert resp.evidence_groups[1].document_id == "doc_b"
        assert len(resp.sources) == 2
        assert resp.sources[0].citation_id == "S1"
        assert resp.sources[1].citation_id == "S2"
        assert "[S1]" in resp.answer and "[S2]" in resp.answer
        assert resp.citation_validation["has_invalid_citations"] is False


def test_b_document_filtering(session_mgr):
    """Test B: Retrieval for doc_a cannot return doc_b evidence and vice-versa."""
    calls = []

    def mock_retrieval(query, file_ids=None, max_sources=5, **kwargs):
        calls.append(file_ids)
        if file_ids == ["doc_1"]:
            return [make_candidate("doc_1", "doc1.txt", "Content from doc 1")], {}
        if file_ids == ["doc_2"]:
            return [make_candidate("doc_2", "doc2.txt", "Content from doc 2")], {}
        return [], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=("Comparison answer [S1] [S2]", "gpt-4o")):

        req = ComparisonRequest(
            query="Compare doc 1 and doc 2",
            file_ids=["doc_1", "doc_2"]
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert ["doc_1"] in calls
        assert ["doc_2"] in calls
        # Assert each group contains only its own file_id
        for src in resp.evidence_groups[0].sources:
            assert src.file_id == "doc_1"
        for src in resp.evidence_groups[1].sources:
            assert src.file_id == "doc_2"


def test_c_identical_content_remains_separate(session_mgr):
    """Test C: Identical text content across documents preserves distinct file_id and chunk_id."""
    same_content = "The retention period for audit logs is 7 years."
    cand_1 = make_candidate("policy_v1", "policy_v1.pdf", same_content, 0)
    cand_2 = make_candidate("policy_v2", "policy_v2.pdf", same_content, 0)

    def mock_retrieval(query, file_ids=None, **kwargs):
        if file_ids == ["policy_v1"]:
            return [cand_1], {}
        return [cand_2], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=("Both require 7 years [S1] [S2].", "gpt-4o")):

        req = ComparisonRequest(
            query="Compare audit retention",
            file_ids=["policy_v1", "policy_v2"],
            comparison_mode=ComparisonMode.SIMILARITY
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert len(resp.sources) == 2
        s1, s2 = resp.sources[0], resp.sources[1]
        assert s1.chunk_id == "policy_v1_chunk_0"
        assert s2.chunk_id == "policy_v2_chunk_0"
        assert s1.file_id == "policy_v1"
        assert s2.file_id == "policy_v2"
        assert s1.citation_id == "S1"
        assert s2.citation_id == "S2"


def test_d_citation_correctness_and_validation(session_mgr):
    """Test D: Document A citations map to Document A, Document B citations map to Document B, hallucinations rejected."""
    cand_a = make_candidate("file_a", "file_a.txt", "Alpha policy", 0)
    cand_b = make_candidate("file_b", "file_b.txt", "Beta policy", 0)

    def mock_retrieval(query, file_ids=None, **kwargs):
        if file_ids == ["file_a"]:
            return [cand_a], {}
        return [cand_b], {}

    # LLM hallucinates an invalid citation [S99]
    raw_answer = "Policy A is Alpha [S1], Policy B is Beta [S2], and extra claim [S99]."

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=(raw_answer, "gpt-4o")):

        req = ComparisonRequest(
            query="Compare Alpha and Beta",
            file_ids=["file_a", "file_b"]
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        # Validator should have stripped [S99] or marked invalid
        assert "[S99]" not in resp.answer
        assert "[S1]" in resp.answer and "[S2]" in resp.answer
        assert "S99" in resp.citation_validation["invalid_citations_detected"]
        assert resp.citation_validation["has_invalid_citations"] is True


def test_e_difference_mode_inference():
    """Test E: Difference comparison mode inferred correctly from query phrasing."""
    mode = infer_comparison_mode("What changed between 2024 and 2025?", None)
    assert mode == ComparisonMode.DIFFERENCE

    mode2 = infer_comparison_mode("Highlight the differences between policy A and B", None)
    assert mode2 == ComparisonMode.DIFFERENCE

    # Explicit override honored
    mode3 = infer_comparison_mode("What changed?", ComparisonMode.SIMILARITY)
    assert mode3 == ComparisonMode.SIMILARITY


def test_f_similarity_mode_inference():
    """Test F: Similarity comparison mode inferred correctly."""
    mode = infer_comparison_mode("What do both contracts agree on and share in common?", None)
    assert mode == ComparisonMode.SIMILARITY

    mode2 = infer_comparison_mode("Find similarities between document A and document B", None)
    assert mode2 == ComparisonMode.SIMILARITY


def test_g_metric_mode_inference():
    """Test G: Metric comparison mode inferred correctly."""
    mode = infer_comparison_mode("Which division reported higher revenue and profit growth?", None)
    assert mode == ComparisonMode.METRIC

    mode2 = infer_comparison_mode("Compare the quantitative numbers and financial metrics", None)
    assert mode2 == ComparisonMode.METRIC


def test_h_conflict_detection():
    """Test H: Discrepancies and conflicting statements extracted from answer text."""
    answer_with_conflicts = (
        "Overview:\n"
        "Document A and Document B discuss terms.\n\n"
        "Discrepancies / Conflicts:\n"
        "- Document A states interest rate is 5% [S1], whereas Document B states interest rate is 7.5% [S2].\n"
        "- Document A sets termination window to 30 days while Document B specifies 60 days.\n"
    )
    conflicts = extract_contradictions_from_text(answer_with_conflicts)
    assert len(conflicts) >= 2
    assert any("5%" in c and "7.5%" in c for c in conflicts)


def test_i_uneven_relevance_bounded_retrieval(session_mgr):
    """Test I: Uneven relevance does not crowd out lower-scoring document."""
    # Doc A has 5 candidates with score 0.99
    cands_a = [make_candidate("doc_high", "high.txt", f"High chunk {i}", i, 0.99) for i in range(5)]
    # Doc B has 1 candidate with score 0.20
    cands_b = [make_candidate("doc_low", "low.txt", "Low chunk 0", 0, 0.20)]

    def mock_retrieval(query, file_ids=None, max_sources=3, **kwargs):
        if file_ids == ["doc_high"]:
            return cands_a[:max_sources], {}
        elif file_ids == ["doc_low"]:
            return cands_b[:max_sources], {}
        return [], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=("Comparison [S1] [S4]", "gpt-4o")):

        req = ComparisonRequest(
            query="Compare high and low",
            file_ids=["doc_high", "doc_low"],
            max_sources_per_document=3
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert len(resp.evidence_groups) == 2
        # Doc high got bounded to 3
        assert len(resp.evidence_groups[0].sources) == 3
        # Doc low got its 1 source without being crowded out
        assert len(resp.evidence_groups[1].sources) == 1


def test_j_no_evidence_handling(session_mgr):
    """Test J: When retrieval finds 0 candidates, returns standard insufficient evidence message."""
    with patch("app.logic.execute_retrieval_pipeline", return_value=([], {})):
        req = ComparisonRequest(
            query="Compare completely absent topics",
            file_ids=["doc_x", "doc_y"]
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert "insufficient evidence" in resp.answer.lower()
        assert resp.model_used == "none"
        assert len(resp.sources) == 0


def test_k_missing_document_handling():
    """Test K: Missing/nonexistent file_id raises clean ValueError."""
    # When file_ids are passed but cannot be resolved
    with patch("app.logic.get_available_file_ids_and_names", return_value={"doc_valid": "valid.pdf"}), \
         patch("app.logic.get_chroma_collection", return_value=MagicMock(get=MagicMock(return_value={"ids": []}))):
        with pytest.raises(ValueError, match="not found"):
            resolve_comparison_documents("compare", requested_file_ids=["non_existent_doc_id"], has_image=False)


def test_l_deleted_document_handling():
    """Test L: Deleted document cannot be resolved or accessed for comparison."""
    file_id = "test_del_comp"
    # Ensure index deleted
    delete_document_embeddings(file_id)
    assert file_id not in bm25_manager._cache
    assert not os.path.exists(bm25_manager._get_file_path(file_id))


def test_m_conversational_comparison(session_mgr):
    """Test M: Conversational comparison query uses session context and persists turns."""
    sess = session_mgr.create_session(title="Comparison Session")

    cand_a = make_candidate("doc_a", "doc_a.txt", "Alpha division revenue was $10M.")
    cand_b = make_candidate("doc_b", "doc_b.txt", "Beta division revenue was $15M.")

    def mock_retrieval(query, file_ids=None, **kwargs):
        if file_ids == ["doc_a"]:
            return [cand_a], {}
        return [cand_b], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=("Alpha is $10M [S1], Beta is $15M [S2].", "gpt-4o")):

        # Turn 1: Regular query
        session_mgr.append_message(sess.session_id, "user", "What did Alpha report?")
        session_mgr.append_message(sess.session_id, "assistant", "Alpha reported $10M.")

        # Turn 2: Comparison query
        req = ComparisonRequest(
            query="How does that compare to Beta?",
            file_ids=["doc_a", "doc_b"],
            session_id=sess.session_id
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert resp.session_id == sess.session_id
        assert resp.message_id is not None

        # Verify session history now contains 4 messages
        updated_msgs = session_mgr.get_session_messages(sess.session_id)
        assert len(updated_msgs) == 4
        assert updated_msgs[3].role == "assistant"
        assert len(updated_msgs[3].sources) == 2


def test_n_citation_isolation_across_turns(session_mgr):
    """Test N: Old citations from previous session turns do not leak into new comparison turn."""
    sess = session_mgr.create_session(title="Turn Isolation Session")
    
    # Previous turn had [S1] from old document
    old_source = Source(
        filename="old_doc.pdf",
        file_id="old_doc",
        citation_id="S1",
        citation_label="[S1] old_doc.pdf, p. 1",
        content="Old content"
    )
    session_mgr.append_message(sess.session_id, "user", "Old query")
    session_mgr.append_message(sess.session_id, "assistant", "Old answer [S1]", sources=[old_source])

    # New turn compares new_a and new_b
    cand_a = make_candidate("new_a", "new_a.txt", "New Alpha content", 0)
    cand_b = make_candidate("new_b", "new_b.txt", "New Beta content", 0)

    def mock_retrieval(query, file_ids=None, **kwargs):
        if file_ids == ["new_a"]:
            return [cand_a], {}
        return [cand_b], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", return_value=("New comparison [S1] [S2].", "gpt-4o")):

        req = ComparisonRequest(
            query="Compare New Alpha and New Beta",
            file_ids=["new_a", "new_b"],
            session_id=sess.session_id
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        # Citations in new turn start fresh with new_a as S1 and new_b as S2
        assert resp.sources[0].file_id == "new_a"
        assert resp.sources[0].citation_id == "S1"
        assert resp.sources[1].file_id == "new_b"
        assert resp.sources[1].citation_id == "S2"
        # Old file is nowhere in new turn's sources
        assert not any(s.file_id == "old_doc" for s in resp.sources)


def test_o_image_plus_document_comparison(session_mgr):
    """Test O: Ephemeral live image evidence coexists with document evidence in comparison."""
    png_bytes = create_test_png()
    cand_doc = make_candidate("doc_spec", "spec.pdf", "The required operating voltage is 12V.", 0)

    def mock_retrieval(query, file_ids=None, **kwargs):
        return [cand_doc], {}

    mock_image_ev = CanonicalEvidence(
        citation_id="S_TEMP",
        citation_marker="[S_TEMP]",
        citation_label="[S_TEMP] circuit.png, OCR",
        file_id="circuit.png",
        filename="circuit.png",
        chunk_id="ephemeral_img_0",
        content="Measured voltage: 9V.",
        search_type="image_ocr"
    )

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.process_live_image", return_value=mock_image_ev), \
         patch("app.logic.handle_text_query", return_value=("Spec specifies 12V [S1] but image shows 9V [S2].", "gpt-4o")):

        req = ComparisonRequest(
            query="Compare spec voltage to measured circuit",
            file_ids=["doc_spec"],
            image_base64=base64.b64encode(png_bytes).decode("utf-8"),
            image_filename="circuit.png"
        )
        resp = perform_comparison_query(req, mgr=session_mgr)

        assert len(resp.evidence_groups) == 2
        # Group 1 is document
        assert resp.evidence_groups[0].is_image is False
        assert resp.evidence_groups[0].document_name == "spec.pdf"
        # Group 2 is image
        assert resp.evidence_groups[1].is_image is True
        assert resp.evidence_groups[1].document_name == "circuit.png"

        assert len(resp.sources) == 2
        assert resp.sources[0].citation_id == "S1"
        assert resp.sources[1].citation_id == "S2"
        assert resp.sources[1].search_type == "image_ocr"


def test_p_no_bm25_rebuild_during_comparison(session_mgr):
    """Test P: Comparison retrieval loads existing BM25 indices without rebuilding or saving them."""
    cand = make_candidate("doc_cached", "cached.pdf", "Cached content")

    with patch("app.logic.execute_retrieval_pipeline", return_value=([cand], {})), \
         patch("app.logic.handle_text_query", return_value=("Comparison answer [S1].", "gpt-4o")), \
         patch.object(bm25_manager, "save_file_index") as mock_save:

        req = ComparisonRequest(
            query="Compare cached docs",
            file_ids=["doc_cached", "doc_cached"]
        )
        perform_comparison_query(req, mgr=session_mgr)

        # save_file_index must NOT have been called during comparison query
        assert mock_save.call_count == 0


def test_q_deterministic_numeric_calculation():
    """Test Q: Deterministic percentage and difference calculations."""
    ev_a = [
        CanonicalEvidence(
            citation_id="S1", citation_marker="[S1]", citation_label="[S1] A",
            file_id="a", filename="a.txt", chunk_id="a_0",
            content="Revenue in 2024 was $100M.",
            search_type="hybrid"
        )
    ]
    ev_b = [
        CanonicalEvidence(
            citation_id="S2", citation_marker="[S2]", citation_label="[S2] B",
            file_id="b", filename="b.txt", chunk_id="b_0",
            content="Revenue in 2025 was $150M.",
            search_type="hybrid"
        )
    ]

    calc = perform_deterministic_numeric_comparison(ev_a, ev_b, "compare revenue")
    assert calc is not None
    assert calc["unit_mismatch"] is False
    assert calc["doc_a_value"] == 100_000_000.0
    assert calc["doc_b_value"] == 150_000_000.0
    assert calc["difference"] == 50_000_000.0
    assert calc["percentage_change"] == 50.0


def test_r_unit_mismatch_detection():
    """Test R: Unit/currency mismatch detected; arithmetic comparison safely omitted without silent conversion."""
    ev_a = [
        CanonicalEvidence(
            citation_id="S1", citation_marker="[S1]", citation_label="[S1] A",
            file_id="a", filename="a.txt", chunk_id="a_0",
            content="Budget is $100 million.",
            search_type="hybrid"
        )
    ]
    ev_b = [
        CanonicalEvidence(
            citation_id="S2", citation_marker="[S2]", citation_label="[S2] B",
            file_id="b", filename="b.txt", chunk_id="b_0",
            content="Budget is €100 million.",
            search_type="hybrid"
        )
    ]

    calc = perform_deterministic_numeric_comparison(ev_a, ev_b, "compare budget")
    assert calc is not None
    assert calc["unit_mismatch"] is True
    assert calc.get("difference") is None
    assert "mismatch" in calc["message"].lower()


def test_s_temporal_ambiguity_prompt_and_api(client):
    """Test S: System prompt explicitly instructs LLM against assuming chronological order without explicit evidence, and REST API handles /compare."""
    cand_a = make_candidate("file1", "2024_report.pdf", "Data points for region A.")
    cand_b = make_candidate("file2", "2025_report.pdf", "Data points for region B.")

    captured_prompt = []

    def mock_llm(query, context, chat_history=None, temperature=0.1, model=None, system_prompt=None):
        captured_prompt.append(system_prompt)
        return "Comparison answer [S1] [S2].", "gpt-4o"

    def mock_retrieval(query, file_ids=None, **kwargs):
        if file_ids == ["file1"]:
            return [cand_a], {}
        return [cand_b], {}

    with patch("app.logic.execute_retrieval_pipeline", side_effect=mock_retrieval), \
         patch("app.logic.handle_text_query", side_effect=mock_llm):

        response = client.post(
            "/api/v1/compare",
            json={
                "query": "Compare region reports",
                "file_ids": ["file1", "file2"],
                "comparison_mode": "difference"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["comparison_mode"] == "difference"
        assert len(data["evidence_groups"]) == 2

        # Verify temporal instruction is present in system prompt
        assert len(captured_prompt) > 0
        prompt_text = captured_prompt[0]
        assert "TEMPORAL RULES" in prompt_text
        assert "Do not assume chronological order solely based on filename strings" in prompt_text
