"""
Comprehensive Phase 4 Retrieval & Ranking Test Suite.
Tests:
1. Stage A: Vector candidate generation (mocked vectorstore)
2. Stage A: BM25 candidate generation (mocked bm25)
3. Stage B: RRF candidate fusion & strict chunk_id deduplication
4. Stage B: Identical-content distinct chunks preservation
5. Stage C: Transparent lexical reranking (token coverage, exact phrase bonus, metadata boost)
6. Stage C: Reranking enabled vs disabled
7. Graceful fallbacks: Vector failure -> BM25 fallback; BM25 failure -> Vector fallback
8. Metadata preservation across all retrieval stages and into QueryResponse.sources
9. Configurable parameters (candidates limit, weights, k)
10. Metadata-aware retrieval filtering
11. Retrieval metrics evaluation (Recall@K, Precision@K, MRR, batch evaluation)
12. Evaluation test cases (Cases A through G: Fact, Multi-chunk, Distractors, Identical text, Table/row, PDF page, Missing info)
13. Retrieval debug info generation and structure
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.config import settings, Settings
from app.models import QueryRequest, QueryResponse, Source
from app.logic import (
    normalize_query,
    RetrievalCandidate,
    generate_vector_candidates,
    generate_bm25_candidates,
    fuse_candidates,
    reciprocal_rank_fusion,
    compute_lexical_rerank_score,
    rerank_candidates,
    execute_retrieval_pipeline,
    perform_rag_query,
)
from app.retrieval_eval import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    evaluate_retrieval_batch,
)


# =============================================================================
# Helper Fixtures & Documents
# =============================================================================

def make_doc(content: str, chunk_id: str, file_id: str = "doc_1", **metadata) -> Document:
    base_meta = {
        "chunk_id": chunk_id,
        "file_id": file_id,
        "filename": f"{file_id}.pdf",
        "source_type": "text",
        "file_type": "pdf",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    base_meta.update(metadata)
    return Document(page_content=content, metadata=base_meta)


# =============================================================================
# 1. Query Normalization Tests
# =============================================================================

def test_query_normalization():
    assert normalize_query("   What is  the   CAPITAL of France???  ") == "What is the CAPITAL of France???"
    assert normalize_query('"What is RAG?"') == "What is RAG?"
    assert normalize_query("'What is RAG?'") == "What is RAG?"
    assert normalize_query("hello\tworld\n\rfoo") == "hello world foo"
    assert normalize_query("") == ""


# =============================================================================
# 2. Stage A: Vector Candidate Generation
# =============================================================================

def test_generate_vector_candidates_success():
    doc1 = make_doc("Revenue grew by 25% in Q3", "c1", "f1")
    doc2 = make_doc("Operating expenses remained flat", "c2", "f1")

    mock_vs = MagicMock()
    # similarity_search_with_score returns (doc, distance/score)
    mock_vs.similarity_search_with_score.return_value = [(doc1, 0.2), (doc2, 0.8)]

    with patch("app.logic.get_vectorstore", return_value=mock_vs):
        candidates = generate_vector_candidates("revenue growth", ["f1"], k=5)

    assert len(candidates) == 2
    doc_out, cid, score = candidates[0]
    assert cid == "c1"
    assert score > candidates[1][2]  # lower distance = higher similarity
    assert doc_out.metadata["file_id"] == "f1"


def test_generate_vector_candidates_failure_fallback():
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.side_effect = RuntimeError("Vectorstore connection failed")

    with patch("app.logic.get_vectorstore", return_value=mock_vs):
        candidates = generate_vector_candidates("query", ["f1"], k=5)

    assert candidates == []


# =============================================================================
# 3. Stage A: BM25 Candidate Generation
# =============================================================================

def test_generate_bm25_candidates_success():
    doc1 = make_doc("Net profit reached $5 million", "c1", "f1")
    doc2 = make_doc("Customer retention was 94%", "c2", "f1")

    mock_index = MagicMock()
    with patch("app.logic.bm25_manager.get_or_load_index", return_value=mock_index), \
         patch("app.logic.bm25_search_instance", return_value=[(doc1, "c1", 4.5), (doc2, "c2", 2.1)]):
        candidates = generate_bm25_candidates("net profit", ["f1"], k=5)

    assert len(candidates) == 2
    doc_out, cid, score = candidates[0]
    assert cid == "c1"
    assert score == 4.5
    assert candidates[1][2] == 2.1


def test_generate_bm25_candidates_failure_fallback():
    with patch("app.logic.bm25_manager.get_or_load_index", side_effect=RuntimeError("BM25 index missing")):
        candidates = generate_bm25_candidates("query", ["f1"], k=5)

    assert candidates == []


# =============================================================================
# 4. Stage B: RRF Candidate Fusion & Chunk ID Deduplication
# =============================================================================

def test_rrf_fusion_merges_chunks_by_chunk_id():
    doc_shared = make_doc("Shared important finding", "shared_chunk_1", "f1")
    cand_vec = (doc_shared, "shared_chunk_1", 0.95)
    cand_bm25 = (doc_shared, "shared_chunk_1", 5.0)

    fused = fuse_candidates(
        vector_candidates=[cand_vec],
        bm25_candidates=[cand_bm25],
        rrf_k=60,
        vector_weight=0.5,
        bm25_weight=0.5
    )

    assert len(fused) == 1
    assert fused[0].chunk_id == "shared_chunk_1"
    assert fused[0].search_type == "hybrid"
    assert fused[0].vector_score == 0.95
    assert fused[0].bm25_score == 5.0
    assert fused[0].fusion_score > 0


def test_rrf_fusion_preserves_identical_content_with_distinct_chunk_ids():
    """
    CRITICAL REQUIREMENT:
    Two different chunks with identical text content must NOT collide or deduplicate each other.
    They must be tracked separately by chunk_id.
    """
    same_text = "The quick brown fox jumps over the lazy dog."
    doc_a = make_doc(same_text, chunk_id="doc1_chunk_0", file_id="doc1", page_number=1)
    doc_b = make_doc(same_text, chunk_id="doc2_chunk_0", file_id="doc2", page_number=5)

    cand_a = (doc_a, "doc1_chunk_0", 0.9)
    cand_b = (doc_b, "doc2_chunk_0", 0.85)

    fused = fuse_candidates(vector_candidates=[cand_a, cand_b], bm25_candidates=[])

    assert len(fused) == 2
    fused_ids = {c.chunk_id for c in fused}
    assert fused_ids == {"doc1_chunk_0", "doc2_chunk_0"}
    assert fused[0].doc.metadata["page_number"] == 1
    assert fused[1].doc.metadata["page_number"] == 5


def test_legacy_reciprocal_rank_fusion_backward_compatibility():
    doc1 = make_doc("Content 1", "c1")
    doc2 = make_doc("Content 2", "c2")

    vec_results = [(doc1, "c1", 0.9)]
    bm25_results = [(doc1, "c1", 3.5), (doc2, "c2", 2.0)]

    fused_tuples = reciprocal_rank_fusion(vec_results, bm25_results, k=60)
    assert len(fused_tuples) == 2
    top_doc, chunk_id, combined_score, v_score, b_score = fused_tuples[0]
    assert chunk_id == "c1"
    assert top_doc.page_content == "Content 1"
    assert v_score is not None
    assert b_score is not None


# =============================================================================
# 5. Stage C: Transparent Lexical Reranker
# =============================================================================

def test_compute_lexical_rerank_score():
    score_exact = compute_lexical_rerank_score(
        query="operating profit margin",
        content="The operating profit margin increased to 18 percent.",
        metadata={}
    )
    score_partial = compute_lexical_rerank_score(
        query="operating profit margin",
        content="The operating expenses were low and the margin varied.",
        metadata={}
    )
    assert score_exact > score_partial

    score_table_match = compute_lexical_rerank_score(
        query="balance sheet assets",
        content="Current assets: $100M",
        metadata={"table_name": "balance_sheet"}
    )
    score_no_table_match = compute_lexical_rerank_score(
        query="balance sheet assets",
        content="Current assets: $100M",
        metadata={"table_name": "other_table"}
    )
    assert score_table_match > score_no_table_match


def test_reranking_enabled_vs_disabled():
    doc_generic = make_doc("Financial metrics report for Q4", "c1")
    doc_specific = make_doc("The exact net income was $42,500,000 for the fiscal year", "c2")

    cand1 = RetrievalCandidate(doc=doc_generic, chunk_id="c1", fusion_score=0.03, search_type="vector")
    cand2 = RetrievalCandidate(doc=doc_specific, chunk_id="c2", fusion_score=0.025, search_type="bm25")

    # Disabled reranking: order remains strictly determined by fusion_score (cand1 > cand2)
    reranked_off = rerank_candidates("exact net income was $42,500,000", [cand1, cand2], enable_reranking=False)
    assert reranked_off[0].chunk_id == "c1"

    # Enabled reranking: exact phrase and token coverage boost cand2 to top
    reranked_on = rerank_candidates("exact net income was $42,500,000", [cand1, cand2], enable_reranking=True)
    assert reranked_on[0].chunk_id == "c2"
    assert reranked_on[0].rerank_score is not None
    assert reranked_on[0].rerank_score > reranked_on[1].rerank_score


# =============================================================================
# 6. Fallback Behavior Tests
# =============================================================================

def test_retrieval_pipeline_bm25_fallback():
    """If BM25 fails, vector results are returned cleanly with search_method='vector'."""
    doc = make_doc("Only vector survived", "c_vec", "f1")

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [(doc, 0.1)]

    with patch("app.logic.get_vectorstore", return_value=mock_vs), \
         patch("app.logic.bm25_manager.get_or_load_index", side_effect=RuntimeError("BM25 crash")):
        candidates, method = execute_retrieval_pipeline("query", ["f1"], k=5, use_hybrid=True)

    assert method == "vector"
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c_vec"


def test_retrieval_pipeline_vector_fallback():
    """If vectorstore fails, BM25 results are returned cleanly with search_method='bm25'."""
    doc = make_doc("Only BM25 survived", "c_bm25", "f1")

    mock_index = MagicMock()
    with patch("app.logic.get_vectorstore", side_effect=RuntimeError("Vectorstore crash")), \
         patch("app.logic.bm25_manager.get_or_load_index", return_value=mock_index), \
         patch("app.logic.bm25_search_instance", return_value=[(doc, "c_bm25", 5.0)]):
        candidates, method = execute_retrieval_pipeline("query", ["f1"], k=5, use_hybrid=True)

    assert method == "bm25"
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "c_bm25"


def test_retrieval_pipeline_both_fail_gracefully():
    with patch("app.logic.get_vectorstore", side_effect=RuntimeError("Vectorstore crash")), \
         patch("app.logic.bm25_manager.get_or_load_index", side_effect=RuntimeError("BM25 crash")):
        candidates, method = execute_retrieval_pipeline("query", ["f1"], k=5, use_hybrid=True)

    assert method == "none"
    assert candidates == []


# =============================================================================
# 7. Metadata Preservation & Filtering
# =============================================================================

def test_metadata_preservation_through_pipeline():
    rich_meta = {
        "chunk_id": "rich_chunk_99",
        "file_id": "file_abc",
        "filename": "annual_report.pdf",
        "file_type": "pdf",
        "source_type": "table",
        "ingestion_method": "pdfplumber",
        "page_number": 42,
        "section": "Financial Highlights",
        "table_name": "table_revenue_2025",
        "row_number": 7,
        "chunk_index": 3,
        "total_chunks": 10,
    }
    doc = Document(page_content="Row 7: Total Revenue was $120M", metadata=rich_meta)

    cand = RetrievalCandidate(
        doc=doc,
        chunk_id="rich_chunk_99",
        vector_score=0.9,
        bm25_score=4.0,
        fusion_score=0.03,
        search_type="hybrid",
        metadata=rich_meta
    )

    debug_dict = cand.to_debug_dict()
    assert debug_dict["chunk_id"] == "rich_chunk_99"
    assert debug_dict["file_id"] == "file_abc"
    assert debug_dict["page_number"] == 42
    assert debug_dict["section"] == "Financial Highlights"
    assert debug_dict["table_name"] == "table_revenue_2025"
    assert debug_dict["row_number"] == 7
    assert debug_dict["source_type"] == "table"


def test_metadata_filtering():
    doc1 = make_doc("Section A data", "c1", "f1", section="Alpha")
    doc2 = make_doc("Section B data", "c2", "f1", section="Beta")

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [(doc1, 0.1), (doc2, 0.2)]

    with patch("app.logic.get_vectorstore", return_value=mock_vs), \
         patch("app.logic.generate_bm25_candidates", return_value=[]):
        filtered, _ = execute_retrieval_pipeline(
            "query", ["f1"], k=5, metadata_filter={"section": "Beta"}
        )

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c2"


# =============================================================================
# 8. Retrieval Metrics Tests (Recall@K, Precision@K, MRR)
# =============================================================================

def test_retrieval_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c2", "c4", "c8"]

    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=3), 0.01) == 1 / 3
    assert pytest.approx(compute_precision_at_k(retrieved, relevant, k=3), 0.01) == 1 / 3
    assert compute_mrr(retrieved, relevant) == 0.5
    assert compute_mrr(["c2", "c1"], relevant) == 1.0
    assert compute_mrr(["c99", "c100"], relevant) == 0.0
    assert compute_recall_at_k(["c99"], relevant, k=1) == 0.0


def test_evaluate_retrieval_batch():
    batch = [
        {"retrieved_ids": ["c1", "c2"], "relevant_ids": ["c1"]},
        {"retrieved_ids": ["c2", "c3"], "relevant_ids": ["c3"]},
        {"retrieved_ids": ["c4", "c5"], "relevant_ids": ["c9"]},
    ]
    summary = evaluate_retrieval_batch(batch, k=2)
    assert summary["sample_count"] == 3
    assert summary["mean_recall@2"] > 0
    assert summary["mean_mrr"] > 0


# =============================================================================
# 9. Comprehensive Evaluation Cases (Cases A through G)
# =============================================================================

def test_case_a_exact_fact_retrieval():
    """Case A: Exact fact retrieval — Chunk with exact fact is ranked #1."""
    chunk_exact = make_doc("Apollo 11 landed on the Moon on July 20, 1969.", "apollo_fact")
    chunk_other = make_doc("The Apollo program included several missions to space.", "apollo_general")

    cand_exact = RetrievalCandidate(doc=chunk_exact, chunk_id="apollo_fact", fusion_score=0.03, search_type="hybrid")
    cand_other = RetrievalCandidate(doc=chunk_other, chunk_id="apollo_general", fusion_score=0.03, search_type="hybrid")

    ranked = rerank_candidates("When did Apollo 11 land on the moon?", [cand_other, cand_exact], enable_reranking=True)
    assert ranked[0].chunk_id == "apollo_fact"


def test_case_b_multi_chunk_retrieval():
    """Case B: Multi-chunk query requires both chunks in top K."""
    chunk_part1 = make_doc("Company Alpha acquired Company Beta in January 2024.", "p1")
    chunk_part2 = make_doc("The acquisition of Company Beta cost $350 million.", "p2")
    chunk_distractor = make_doc("Company Gamma operates in retail.", "p3")

    candidates = [
        RetrievalCandidate(doc=chunk_part1, chunk_id="p1", fusion_score=0.04, search_type="hybrid"),
        RetrievalCandidate(doc=chunk_part2, chunk_id="p2", fusion_score=0.035, search_type="hybrid"),
        RetrievalCandidate(doc=chunk_distractor, chunk_id="p3", fusion_score=0.01, search_type="hybrid"),
    ]

    ranked = rerank_candidates("How much was the acquisition of Company Beta?", candidates, enable_reranking=True)
    top2_ids = [c.chunk_id for c in ranked[:2]]
    assert "p1" in top2_ids and "p2" in top2_ids


def test_case_c_distractor_rejection():
    """Case C: Distractor with high keyword overlap but wrong semantic answer is ranked lower."""
    chunk_true = make_doc("The server port for the production database is 5432.", "true_chunk")
    chunk_distractor = make_doc(
        "For database backup configuration, ensure the database port and server port are documented in ticket 5432.",
        "distractor_chunk"
    )

    cands = [
        RetrievalCandidate(doc=chunk_distractor, chunk_id="distractor_chunk", fusion_score=0.03, search_type="hybrid"),
        RetrievalCandidate(doc=chunk_true, chunk_id="true_chunk", fusion_score=0.03, search_type="hybrid"),
    ]

    ranked = rerank_candidates("What is the production database server port?", cands, enable_reranking=True)
    assert ranked[0].chunk_id == "true_chunk"


def test_case_d_identical_text_provenance():
    """Case D: Identical text across distinct documents preserves distinct chunk IDs."""
    repeated_policy = "Employees are entitled to 20 days of paid annual leave."
    doc_handbook_us = make_doc(repeated_policy, "handbook_us_0", file_id="us_handbook", filename="US_Handbook.pdf")
    doc_handbook_uk = make_doc(repeated_policy, "handbook_uk_0", file_id="uk_handbook", filename="UK_Handbook.pdf")

    cands = [
        (doc_handbook_us, "handbook_us_0", 0.9),
        (doc_handbook_uk, "handbook_uk_0", 0.9),
    ]

    fused = fuse_candidates(vector_candidates=cands, bm25_candidates=[])
    assert len(fused) == 2
    assert {c.chunk_id for c in fused} == {"handbook_us_0", "handbook_uk_0"}
    assert {c.doc.metadata["filename"] for c in fused} == {"US_Handbook.pdf", "UK_Handbook.pdf"}


def test_case_e_table_row_retrieval():
    """Case E: Table / row retrieval preserves table_name and row_number."""
    table_doc = make_doc(
        "Row 15: Product SKU-999 price is $49.99 stock 120",
        "tbl_chunk_15",
        source_type="table",
        table_name="products",
        row_number=15
    )
    cand = RetrievalCandidate(doc=table_doc, chunk_id="tbl_chunk_15", fusion_score=0.04, search_type="hybrid")

    ranked = rerank_candidates("What is the price of SKU-999 in products?", [cand], enable_reranking=True)
    top = ranked[0]
    assert top.doc.metadata["table_name"] == "products"
    assert top.doc.metadata["row_number"] == 15
    assert top.chunk_id == "tbl_chunk_15"


def test_case_f_pdf_page_retrieval():
    """Case F: PDF page retrieval preserves page_number."""
    page_doc = make_doc(
        "Chapter 3: Methodology and Experimental Setup",
        "pdf_chunk_pg12",
        source_type="pdf",
        page_number=12
    )
    cand = RetrievalCandidate(doc=page_doc, chunk_id="pdf_chunk_pg12", fusion_score=0.04, search_type="hybrid")

    ranked = rerank_candidates("Where is the experimental setup described?", [cand], enable_reranking=True)
    assert ranked[0].doc.metadata["page_number"] == 12


def test_case_g_missing_information_retrieval():
    """Case G: Query with no matching content returns empty or low-relevance results gracefully."""
    empty_cands, method = execute_retrieval_pipeline("Completely irrelevant alien spaceship query", ["nonexistent_id"])
    assert len(empty_cands) == 0


# =============================================================================
# 10. End-to-End Query Integration & Debug Info
# =============================================================================

def test_perform_rag_query_includes_retrieval_debug():
    doc = make_doc("Antigravity engine produces zero emissions.", "eng_chunk_1", "f_eng")
    cand = RetrievalCandidate(
        doc=doc,
        chunk_id="eng_chunk_1",
        fusion_score=0.03,
        vector_score=0.88,
        bm25_score=3.2,
        search_type="hybrid",
        rerank_score=0.75,
        final_score=0.82,
        metadata=doc.metadata
    )

    req = QueryRequest(
        question="How many emissions does the antigravity engine produce?",
        file_id="f_eng",
        enable_reranking=True
    )

    with patch("app.logic.execute_retrieval_pipeline", return_value=([cand], "hybrid")), \
         patch("app.logic.handle_text_query", return_value=("Zero emissions.", "gpt-4o-mini")), \
         patch("app.logic.generate_suggested_questions", return_value=[]):
        res = perform_rag_query(req)

    assert isinstance(res, QueryResponse)
    assert len(res.sources) == 1
    assert res.sources[0].chunk_id == "eng_chunk_1"
    assert res.sources[0].relevance_score == 0.82
    assert res.retrieval_debug is not None
    assert len(res.retrieval_debug) == 1
    assert res.retrieval_debug[0]["chunk_id"] == "eng_chunk_1"
    assert res.retrieval_debug[0]["search_type"] == "hybrid"
    assert res.retrieval_debug[0]["final_score"] == 0.82
