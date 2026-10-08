"""
Deterministic evaluation and adversarial testing runner for Aura AI RAG pipeline.
Measures Recall@K, Precision@K, MRR, citation validity rate, and missing info detection rate.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from app.logic import (
    bm25_manager,
    generate_bm25_candidates,
    validate_citations
)
from app.retrieval_eval import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr
)

EVAL_DIR = Path(__file__).resolve().parent
DOCS_DIR = EVAL_DIR / "documents"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
EXPECTED_FILE = EVAL_DIR / "expected.json"


def load_eval_documents() -> Dict[str, List[Document]]:
    """Load benchmark evaluation documents into Document representations."""
    docs_map = {}
    for p in DOCS_DIR.glob("*.txt"):
        fname = p.name
        content = p.read_text(encoding="utf-8").strip()
        # Split into logical sections/paragraphs
        paras = [para.strip() for para in content.split("\n\n") if para.strip()]
        chunks = []
        for idx, para in enumerate(paras):
            chunk_id = f"{fname}_chunk_{idx}"
            doc = Document(
                page_content=para,
                metadata={
                    "file_id": fname,
                    "filename": fname,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks": len(paras),
                    "file_type": "txt",
                    "source_type": "text",
                    "page_number": 1
                }
            )
            chunks.append(doc)
        docs_map[fname] = chunks

    return docs_map


def index_eval_documents(docs_map: Dict[str, List[Document]]):
    """Index evaluation chunks into persistent BM25 manager."""
    for fname, chunks in docs_map.items():
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        bm25_manager.save_file_index(fname, chunks, chunk_ids)


def run_evaluation(k: int = 3) -> Dict[str, Any]:
    """
    Execute deterministic benchmark across questions and expected outcomes.
    
    Returns:
        Structured dictionary containing aggregated metrics and per-case results.
    """
    docs_map = load_eval_documents()
    index_eval_documents(docs_map)

    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)

        with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
            expected = json.load(f)

        all_file_ids = list(docs_map.keys())
        results = []
        recalls = []
        precisions = []
        mrrs = []
        citation_validities = []
        missing_info_correct = 0
        missing_info_total = 0

        for q in questions:
            qid = q["id"]
            exp = expected.get(qid, {})
            query_text = q["question"]

            # BM25 multi-file retrieval across all evaluation documents
            retrieved_raw = generate_bm25_candidates(query_text, file_ids=all_file_ids, k=k)
            retrieved_chunks = [cand[0] for cand in retrieved_raw]
            retrieved_filenames = [c.metadata.get("filename") for c in retrieved_chunks]
            retrieved_ids = [c.metadata.get("chunk_id") for c in retrieved_chunks]

            # Ground truth relevant filenames / chunk indicators
            exp_docs = exp.get("expected_documents", [])
            is_missing_query = not exp.get("should_find_evidence", True)

            if is_missing_query:
                missing_info_total += 1
                # For missing information query, check if distractors or missing keywords are triggered
                has_strong_match = any(
                    any(kw.lower() in c.page_content.lower() for kw in q.get("keywords", []))
                    for c in retrieved_chunks
                )
                if not has_strong_match:
                    missing_info_correct += 1
                rec = 1.0 if not has_strong_match else 0.0
                prec = 1.0 if not has_strong_match else 0.0
                rr = 1.0 if not has_strong_match else 0.0
            else:
                # Check relevance based on target document matching
                relevant_chunks = []
                for doc_name in exp_docs:
                    for c in docs_map.get(doc_name, []):
                        if any(kw.lower() in c.page_content.lower() for kw in q.get("keywords", [])):
                            relevant_chunks.append(c.metadata["chunk_id"])

                if not relevant_chunks:
                    for doc_name in exp_docs:
                        relevant_chunks.extend([c.metadata["chunk_id"] for c in docs_map.get(doc_name, [])])

                rec = compute_recall_at_k(retrieved_ids, relevant_chunks, k)
                prec = compute_precision_at_k(retrieved_ids, relevant_chunks, k)
                rr = compute_mrr(retrieved_ids, relevant_chunks)

            recalls.append(rec)
            precisions.append(prec)
            mrrs.append(rr)

            # Evaluate citation validation safety on mock answer containing citations
            valid_cids = {f"S{i+1}" for i in range(len(retrieved_chunks))}
            mock_answer = " ".join([f"Fact from {fn} [S{i+1}]." for i, fn in enumerate(retrieved_filenames)])
            if qid == "q7_injection_resilience":
                mock_answer += " Injected citation [S99] and [SYSTEM]."

            cleaned_ans, valid_used, val_report = validate_citations(mock_answer, valid_cids)
            citation_valid = (not val_report["has_invalid_citations"]) if qid != "q7_injection_resilience" else (val_report["has_invalid_citations"] and "[S99]" not in cleaned_ans)
            citation_validities.append(1.0 if citation_valid else 0.0)

            results.append({
                "id": qid,
                "category": q["category"],
                "recall_at_k": rec,
                "precision_at_k": prec,
                "mrr": rr,
                "citation_valid": citation_valid,
                "retrieved_files": retrieved_filenames
            })

        summary = {
            f"mean_recall@{k}": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            f"mean_precision@{k}": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "citation_validity_rate": round(sum(citation_validities) / len(citation_validities), 4) if citation_validities else 1.0,
            "missing_info_detection_rate": round(missing_info_correct / missing_info_total, 4) if missing_info_total > 0 else 1.0,
            "total_test_cases": len(questions),
            "detailed_results": results
        }

        return summary
    finally:
        for fname in docs_map.keys():
            bm25_manager.delete_file_index(fname)


if __name__ == "__main__":
    metrics = run_evaluation(k=3)
    print("\n=== RAG Evaluation Benchmark Results ===")
    print(f"Total Test Cases: {metrics['total_test_cases']}")
    print(f"Mean Recall@3: {metrics['mean_recall@3']:.2%}")
    print(f"Mean Precision@3: {metrics['mean_precision@3']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Citation Validity Rate: {metrics['citation_validity_rate']:.2%}")
    print(f"Missing Information Detection Rate: {metrics['missing_info_detection_rate']:.2%}")
