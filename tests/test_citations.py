"""
Phase 5 — Evidence-Grounded Citations & Source Attribution Test Suite.

Covers Test Matrix A through N:
- Test A: PDF citation formatting (page number, section)
- Test B: DOCX section citation formatting
- Test C: DOCX table citation formatting
- Test D: CSV row citation formatting
- Test E: SQLite table and row citation formatting
- Test F: Image OCR citation formatting
- Test G: Image Vision citation formatting
- Test H: Missing metadata handling (never invents page numbers)
- Test I: Duplicate chunk_id deduplication into a single citation
- Test J: Identical text with distinct chunk_ids preserved as independent citations
- Test K: Invalid LLM citation detection and post-generation cleanup ([S99] stripped & flagged)
- Test L: Deterministic citation ordering and numbering reproducibility
- Test M: No evidence handling (grounded response without hallucinations)
- Test N: Phase 3 rich provenance preservation through canonical evidence to Source model
- Test O: Structured evidence context formatting for LLM prompt
- Test P: End-to-end perform_rag_query integration with citation metadata & validation report
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.models import QueryRequest, QueryResponse, Source
from app.logic import (
    RetrievalCandidate,
    CanonicalEvidence,
    format_citation_label,
    build_canonical_evidence,
    format_evidence_context,
    validate_citations,
    perform_rag_query,
)


def make_cand(
    content: str,
    chunk_id: str,
    filename: str,
    final_score: float = 0.9,
    **metadata
) -> RetrievalCandidate:
    base_meta = {
        "chunk_id": chunk_id,
        "filename": filename,
        "file_id": f"fid_{filename.split('.')[0]}",
        "file_type": filename.split(".")[-1].lower(),
        "source_type": "text",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    base_meta.update(metadata)
    doc = Document(page_content=content, metadata=base_meta)
    return RetrievalCandidate(
        doc=doc,
        chunk_id=chunk_id,
        final_score=final_score,
        search_type="hybrid",
        metadata=base_meta
    )


# =============================================================================
# Test A: PDF Citation Formatting
# =============================================================================

def test_citation_pdf_page_and_section():
    # p. 4 without section
    label1 = format_citation_label(
        filename="report.pdf",
        metadata={"file_type": "pdf", "page_number": 4},
        citation_id="S1"
    )
    assert label1 == '[S1] report.pdf, p. 4'

    # p. 4 with section
    label2 = format_citation_label(
        filename="report.pdf",
        metadata={"file_type": "pdf", "page_number": 4, "section": "Financial Results"},
        citation_id="S1"
    )
    assert label2 == '[S1] report.pdf, p. 4, "Financial Results"'

    # section without page
    label3 = format_citation_label(
        filename="report.pdf",
        metadata={"file_type": "pdf", "section": "Executive Summary"},
        citation_id="S1"
    )
    assert label3 == '[S1] report.pdf, "Executive Summary"'


# =============================================================================
# Test B: DOCX Section Citation Formatting
# =============================================================================

def test_citation_docx_section():
    label = format_citation_label(
        filename="proposal.docx",
        metadata={"file_type": "docx", "section": "Technical Architecture"},
        citation_id="S2"
    )
    assert label == '[S2] proposal.docx, "Technical Architecture"'


# =============================================================================
# Test C: DOCX Table Citation Formatting
# =============================================================================

def test_citation_docx_table():
    label = format_citation_label(
        filename="proposal.docx",
        metadata={"file_type": "docx", "table_name": "Table_1"},
        citation_id="S3"
    )
    assert label == '[S3] proposal.docx, Table_1'


# =============================================================================
# Test D: CSV Row Citation Formatting
# =============================================================================

def test_citation_csv_row():
    label = format_citation_label(
        filename="customers.csv",
        metadata={"file_type": "csv", "row_number": 17},
        citation_id="S4"
    )
    assert label == '[S4] customers.csv, row 17'


# =============================================================================
# Test E: SQLite Table and Row Citation Formatting
# =============================================================================

def test_citation_sqlite_table_and_row():
    label = format_citation_label(
        filename="database.db",
        metadata={"file_type": "sqlite", "table_name": "users", "row_number": 42},
        citation_id="S5"
    )
    assert label == '[S5] database.db, users, row 42'


# =============================================================================
# Test F & G: Image OCR and Vision Citation Formatting
# =============================================================================

def test_citation_image_ocr():
    label = format_citation_label(
        filename="diagram.png",
        metadata={"file_type": "png", "source_type": "image_ocr"},
        citation_id="S6"
    )
    assert label == '[S6] diagram.png, OCR'


def test_citation_image_vision():
    label = format_citation_label(
        filename="diagram.png",
        metadata={"file_type": "png", "source_type": "image_vision"},
        citation_id="S6"
    )
    assert label == '[S6] diagram.png, Vision'


# =============================================================================
# Test H: Missing Metadata Handling (Never Invent Pages)
# =============================================================================

def test_citation_missing_metadata_never_invents_page():
    label = format_citation_label(
        filename="report.pdf",
        metadata={"file_type": "pdf", "page_number": None},
        citation_id="S1"
    )
    assert label == '[S1] report.pdf'
    assert "None" not in label
    assert "p." not in label


# =============================================================================
# Test I: Duplicate Chunk Deduplication
# =============================================================================

def test_duplicate_chunk_produces_single_canonical_evidence():
    c1 = make_cand("Identical chunk content", "chk_duplicate", "file.pdf", final_score=0.95)
    c2 = make_cand("Identical chunk content", "chk_duplicate", "file.pdf", final_score=0.80)

    evidence = build_canonical_evidence([c1, c2])
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "chk_duplicate"
    assert evidence[0].citation_id == "S1"
    assert evidence[0].citation_marker == "[S1]"


# =============================================================================
# Test J: Identical Text, Distinct Chunk IDs
# =============================================================================

def test_identical_text_different_chunks_remain_separate():
    same_text = "All employees must complete compliance training by Q4."
    c1 = make_cand(same_text, "chk_doc1_p1", "policy_us.pdf", final_score=0.9, page_number=1)
    c2 = make_cand(same_text, "chk_doc2_p3", "policy_uk.pdf", final_score=0.85, page_number=3)

    evidence = build_canonical_evidence([c1, c2])
    assert len(evidence) == 2
    assert evidence[0].chunk_id == "chk_doc1_p1"
    assert evidence[0].citation_id == "S1"
    assert evidence[0].citation_label == "[S1] policy_us.pdf, p. 1"

    assert evidence[1].chunk_id == "chk_doc2_p3"
    assert evidence[1].citation_id == "S2"
    assert evidence[1].citation_label == "[S2] policy_uk.pdf, p. 3"


# =============================================================================
# Test K: Invalid LLM Citation Detection and Remediation
# =============================================================================

def test_validate_citations_strips_hallucinated_ids():
    raw_answer = "The company reported $10M in revenue [S1], and opened 5 new offices in 2024 [S99]."
    valid_ids = {"S1", "S2"}

    cleaned_answer, valid_used, report = validate_citations(raw_answer, valid_ids)

    # Valid S1 retained
    assert "[S1]" in cleaned_answer
    # Hallucinated S99 stripped cleanly
    assert "[S99]" not in cleaned_answer
    assert valid_used == ["S1"]
    assert report["has_invalid_citations"] is True
    assert report["invalid_citations_detected"] == ["S99"]
    assert report["citations_used"] == ["S1"]
    assert "opened 5 new offices in 2024." in cleaned_answer or "opened 5 new offices in 2024" in cleaned_answer


# =============================================================================
# Test L: Deterministic Ordering & Citation Numbering
# =============================================================================

def test_deterministic_citation_ordering():
    c1 = make_cand("Alpha content", "cid_a", "a.pdf", final_score=0.92)
    c2 = make_cand("Beta content", "cid_b", "b.pdf", final_score=0.85)
    c3 = make_cand("Gamma content", "cid_c", "c.pdf", final_score=0.70)

    # Shuffle input order
    ev_order_1 = build_canonical_evidence([c3, c1, c2])
    ev_order_2 = build_canonical_evidence([c2, c3, c1])

    # Both must produce identical S1 -> c1, S2 -> c2, S3 -> c3
    assert [e.chunk_id for e in ev_order_1] == ["cid_a", "cid_b", "cid_c"]
    assert [e.citation_id for e in ev_order_1] == ["S1", "S2", "S3"]
    assert [e.chunk_id for e in ev_order_2] == ["cid_a", "cid_b", "cid_c"]
    assert [e.citation_id for e in ev_order_2] == ["S1", "S2", "S3"]


# =============================================================================
# Test M: Missing / No Evidence Handling
# =============================================================================

def test_perform_rag_query_no_evidence():
    req = QueryRequest(question="What is the flux capacitor frequency?", file_id="f_empty")

    with patch("app.logic.execute_retrieval_pipeline", return_value=([], "none")):
        res = perform_rag_query(req)

    assert isinstance(res, QueryResponse)
    assert "No relevant documents found" in res.answer
    assert res.sources == []
    assert res.citation_validation is not None
    assert res.citation_validation["has_invalid_citations"] is False
    assert res.citation_validation["citations_used"] == []


# =============================================================================
# Test N: Rich Metadata Preservation into Citation Layer
# =============================================================================

def test_provenance_metadata_preserved_in_evidence_and_sources():
    cand = make_cand(
        content="Table 4 row 8: Quarterly profit was $4.5M",
        chunk_id="chunk_rich_101",
        filename="financials.sqlite",
        final_score=0.88,
        file_id="fid_999",
        chunk_index=3,
        total_chunks=12,
        file_type="sqlite",
        source_type="table",
        ingestion_method="structured",
        table_name="quarterly_profits",
        row_number=8,
        section="Q3 Results",
        page_number=None,
    )

    evidence = build_canonical_evidence([cand])[0]
    assert evidence.citation_id == "S1"
    assert evidence.citation_label == "[S1] financials.sqlite, quarterly_profits, row 8"
    assert evidence.table_name == "quarterly_profits"
    assert evidence.row_number == 8
    assert evidence.chunk_index == 3
    assert evidence.total_chunks == 12
    assert evidence.file_id == "fid_999"


# =============================================================================
# Test O: Evidence Context Formatting
# =============================================================================

def test_format_evidence_context():
    cand1 = make_cand("Apollo 11 landed in 1969.", "chk_1", "mission.pdf", page_number=11, section="Lunar Landing")
    cand2 = make_cand("Crew: Armstrong, Aldrin, Collins.", "chk_2", "crew.csv", row_number=3)

    evidence = build_canonical_evidence([cand1, cand2])
    ctx = format_evidence_context(evidence)

    assert "[SOURCE S1]" in ctx
    assert "Filename: mission.pdf" in ctx
    assert "Page: 11" in ctx
    assert "Section: Lunar Landing" in ctx
    assert "Apollo 11 landed in 1969." in ctx

    assert "[SOURCE S2]" in ctx
    assert "Filename: crew.csv" in ctx
    assert "Row: 3" in ctx
    assert "Crew: Armstrong, Aldrin, Collins." in ctx


# =============================================================================
# Test P: End-to-End Query with Citations & Validation Report
# =============================================================================

def test_perform_rag_query_with_citations_and_validation():
    cand = make_cand(
        "Superconduction occurs below 4.2 Kelvin.",
        "cand_physics_1",
        "physics.pdf",
        page_number=14,
        final_score=0.95
    )
    llm_output = "Superconduction is observed below 4.2 Kelvin [S1]. An unverified claim [S42]."

    req = QueryRequest(question="At what temperature does superconduction occur?", file_id="f_phys")

    with patch("app.logic.execute_retrieval_pipeline", return_value=([cand], "hybrid")), \
         patch("app.logic.handle_text_query", return_value=(llm_output, "gpt-4o-mini")), \
         patch("app.logic.generate_suggested_questions", return_value=[]):
        res = perform_rag_query(req)

    assert isinstance(res, QueryResponse)
    assert len(res.sources) == 1
    src = res.sources[0]
    assert src.citation_id == "S1"
    assert src.citation_label == "[S1] physics.pdf, p. 14"

    # Verify invalid citation [S42] was stripped from answer
    assert "[S1]" in res.answer
    assert "[S42]" not in res.answer

    # Verify citation_validation report
    assert res.citation_validation is not None
    assert res.citation_validation["has_invalid_citations"] is True
    assert "S42" in res.citation_validation["invalid_citations_detected"]
    assert "S1" in res.citation_validation["citations_used"]
