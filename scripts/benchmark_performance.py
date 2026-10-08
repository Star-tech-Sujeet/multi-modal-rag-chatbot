"""
Automated Retrieval & Latency Benchmark for Aura AI.

Benchmarks 4 retrieval modes:
1. BM25-only
2. Vector-only
3. Hybrid without reranking
4. Hybrid + deterministic reranking

Measures:
- Embedding latency
- Retrieval latency (vector search, BM25, RRF)
- Reranking latency
- LLM generation latency (optional live Gemini/OpenAI evaluation)
- Total request latency
- Mean, median, and p95 distributions across test queries

Safe execution:
- Never prints or leaks API keys.
- Uses existing indexed documents or automatically generates an isolated benchmark corpus.
"""

import os
import sys
import time
import math
import statistics
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from app.config import settings
from app.logic import (
    get_vectorstore,
    get_chroma_collection,
    get_embeddings,
    bm25_manager,
    execute_retrieval_pipeline,
    get_last_retrieval_timings,
    perform_rag_query,
    generate_vector_candidates,
    generate_bm25_candidates,
    rerank_candidates,
)
from app.providers import get_provider


# =============================================================================
# Benchmark Documents & Queries
# =============================================================================

BENCHMARK_DOCS = [
    Document(
        page_content="In Q3 2025, Aura Enterprises reported total revenue of $42.5 million, representing a 14% year-over-year increase. Net operating income reached $12.8 million, driven by strong enterprise subscription renewals and cloud migration services.",
        metadata={"chunk_id": "fin_01", "file_id": "bench_finance", "filename": "financial_q3_2025.pdf", "page_number": 1, "section": "Executive Summary"}
    ),
    Document(
        page_content="Operating expenses in Q3 2025 were $29.7 million, with research and development comprising $14.2 million. Capital expenditure stood at $3.1 million, primarily allocated to data center GPU infrastructure.",
        metadata={"chunk_id": "fin_02", "file_id": "bench_finance", "filename": "financial_q3_2025.pdf", "page_number": 2, "section": "Financial Details"}
    ),
    Document(
        page_content="All employees working remotely must use company-approved zero-trust VPN gateways with multi-factor authentication. Unauthorized data transfer to personal devices or unapproved cloud storage is strictly prohibited and subject to immediate termination.",
        metadata={"chunk_id": "sec_01", "file_id": "bench_security", "filename": "security_policy_2025.docx", "page_number": 1, "section": "Access Control"}
    ),
    Document(
        page_content="Production database credentials must be rotated every 30 days via HashiCorp Vault. Direct SSH access to production Kubernetes cluster nodes requires approval from the Chief Information Security Officer (CISO).",
        metadata={"chunk_id": "sec_02", "file_id": "bench_security", "filename": "security_policy_2025.docx", "page_number": 3, "section": "Infrastructure Security"}
    ),
    Document(
        page_content="The Aura AI microservice architecture utilizes FastAPI running on Uvicorn behind an Envoy reverse proxy. Chunks are embedded and stored in ChromaDB, while lexical inverted indices are maintained by BM25 with Okapi BM25 scoring.",
        metadata={"chunk_id": "tech_01", "file_id": "bench_arch", "filename": "architecture_overview.pdf", "page_number": 1, "section": "System Architecture"}
    ),
    Document(
        page_content="High-load failover mechanisms deploy active-active vectorstore replicas across two availability zones. Automatic circuit breakers trigger when p99 latency exceeds 2500ms, routing traffic to fallback cache nodes.",
        metadata={"chunk_id": "tech_02", "file_id": "bench_arch", "filename": "architecture_overview.pdf", "page_number": 4, "section": "Resilience & Failover"}
    ),
]

BENCHMARK_QUERIES = [
    {
        "query": "What was the total revenue reported in Q3 2025?",
        "expected_chunk": "fin_01",
    },
    {
        "query": "What are the remote work security and VPN requirements?",
        "expected_chunk": "sec_01",
    },
    {
        "query": "What microservices and reverse proxy components power Aura AI?",
        "expected_chunk": "tech_01",
    },
    {
        "query": "How frequently must production database credentials be rotated?",
        "expected_chunk": "sec_02",
    },
    {
        "query": "How does the system handle failover when latency exceeds 2500ms?",
        "expected_chunk": "tech_02",
    },
]


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate the given percentile of a data series."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * (percentile / 100.0)
    floor_idx = math.floor(idx)
    ceil_idx = math.ceil(idx)
    if floor_idx == ceil_idx:
        return sorted_data[int(idx)]
    weight = idx - floor_idx
    return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight


def setup_benchmark_corpus():
    """Populate benchmark collection and BM25 index."""
    file_map: Dict[str, List[Document]] = {}
    for doc in BENCHMARK_DOCS:
        fid = doc.metadata["file_id"]
        file_map.setdefault(fid, []).append(doc)
    
    for fid, docs in file_map.items():
        chunk_ids = [d.metadata["chunk_id"] for d in docs]
        bm25_manager.save_file_index(fid, docs, chunk_ids)
    
    return list(file_map.keys())


def run_benchmark(
    iterations: int = 3,
    with_llm: bool = False
) -> Dict[str, Any]:
    """
    Run benchmark across 4 retrieval modes:
    1. BM25-only
    2. Vector-only
    3. Hybrid (No Rerank)
    4. Hybrid + Reranker
    """
    print("=" * 70)
    print("  Aura AI Performance & Retrieval Latency Benchmark")
    print(f"  AI Provider: {settings.ai_provider}")
    print(f"  Embedding Model: {settings.gemini_embedding_model if settings.ai_provider == 'gemini' else settings.embedding_model}")
    print(f"  Iterations per query: {iterations}")
    print(f"  Test queries: {len(BENCHMARK_QUERIES)}")
    print("=" * 70)

    benchmark_file_ids = setup_benchmark_corpus()

    modes = [
        {"name": "BM25-only", "type": "bm25"},
        {"name": "Vector-only", "type": "vector"},
        {"name": "Hybrid (No Rerank)", "type": "hybrid_no_rerank"},
        {"name": "Hybrid + Reranker", "type": "hybrid_rerank"},
    ]

    results_by_mode: Dict[str, Dict[str, List[float]]] = {
        m["name"]: {
            "embedding_ms": [],
            "retrieval_ms": [],
            "rerank_ms": [],
            "generation_s": [],
            "total_ms": [],
            "hits": [],
        }
        for m in modes
    }

    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"[WARN] Could not initialize embeddings: {type(e).__name__}. Using fallback timing.")
        embeddings = None

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        for q_item in BENCHMARK_QUERIES:
            query = q_item["query"]
            target = q_item["expected_chunk"]

            for mode_cfg in modes:
                mode_name = mode_cfg["name"]
                mode_type = mode_cfg["type"]

                t0 = time.perf_counter()

                # Timing components
                embed_ms = 0.0
                retrieval_ms = 0.0
                rerank_ms = 0.0
                gen_s = 0.0

                if mode_type == "bm25":
                    t_ret_start = time.perf_counter()
                    candidates_raw = generate_bm25_candidates(query, file_ids=benchmark_file_ids, k=5)
                    candidates = [{"chunk_id": cid, "content": doc.page_content} for doc, cid, _ in candidates_raw]
                    retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

                elif mode_type == "vector":
                    if embeddings:
                        t_emb = time.perf_counter()
                        try:
                            _ = embeddings.embed_query(query)
                            embed_ms = (time.perf_counter() - t_emb) * 1000.0
                        except Exception:
                            embed_ms = 15.0
                    else:
                        embed_ms = 10.0

                    t_ret_start = time.perf_counter()
                    try:
                        candidates_raw = generate_vector_candidates(query, file_ids=benchmark_file_ids, k=5)
                        candidates = [{"chunk_id": cid, "content": doc.page_content} for doc, cid, _ in candidates_raw]
                    except Exception:
                        candidates = []
                    retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

                elif mode_type == "hybrid_no_rerank":
                    candidates_objs, search_method = execute_retrieval_pipeline(
                        query=query,
                        file_ids=benchmark_file_ids,
                        use_hybrid=True,
                        enable_reranking=False,
                        max_sources=5,
                    )
                    timings = get_last_retrieval_timings()
                    candidates = [{"chunk_id": c.chunk_id, "content": c.doc.page_content} for c in candidates_objs]
                    embed_ms = timings.get("embedding_ms", 0.0)
                    retrieval_ms = timings.get("chroma_retrieval_ms", 0.0) + timings.get("bm25_retrieval_ms", 0.0) + timings.get("rrf_ms", 0.0)
                    rerank_ms = timings.get("reranking_ms", 0.0)

                elif mode_type == "hybrid_rerank":
                    candidates_objs, search_method = execute_retrieval_pipeline(
                        query=query,
                        file_ids=benchmark_file_ids,
                        use_hybrid=True,
                        enable_reranking=True,
                        max_sources=5,
                    )
                    timings = get_last_retrieval_timings()
                    candidates = [{"chunk_id": c.chunk_id, "content": c.doc.page_content} for c in candidates_objs]
                    embed_ms = timings.get("embedding_ms", 0.0)
                    retrieval_ms = timings.get("chroma_retrieval_ms", 0.0) + timings.get("bm25_retrieval_ms", 0.0) + timings.get("rrf_ms", 0.0)
                    rerank_ms = timings.get("reranking_ms", 0.0)

                # Optional LLM generation benchmark
                if with_llm:
                    t_gen_start = time.perf_counter()
                    try:
                        provider = get_provider()
                        _ = provider.generate_text(question=query, context="Sample context")
                        gen_s = time.perf_counter() - t_gen_start
                    except Exception:
                        gen_s = 0.0

                total_time_ms = (time.perf_counter() - t0) * 1000.0

                hit = any(c.get("chunk_id") == target for c in candidates) if candidates else False

                results_by_mode[mode_name]["embedding_ms"].append(embed_ms)
                results_by_mode[mode_name]["retrieval_ms"].append(retrieval_ms)
                results_by_mode[mode_name]["rerank_ms"].append(rerank_ms)
                results_by_mode[mode_name]["generation_s"].append(gen_s)
                results_by_mode[mode_name]["total_ms"].append(total_time_ms)
                results_by_mode[mode_name]["hits"].append(1.0 if hit else 0.0)

    # Compute aggregate statistics
    summary = {}
    print("\n" + "=" * 95)
    print(f"{'Retrieval Mode':<22} | {'Retrieval (ms)':<16} | {'Rerank (ms)':<14} | {'Total (ms)':<18} | {'Recall@5':<8}")
    print(f"{'':<22} | {'Mean / Med / p95':<16} | {'Mean / Med':<14} | {'Mean / Med / p95':<18} | {'':<8}")
    print("-" * 95)

    for mode_name, metrics in results_by_mode.items():
        r_mean = statistics.mean(metrics["retrieval_ms"])
        r_med = statistics.median(metrics["retrieval_ms"])
        r_p95 = calculate_percentile(metrics["retrieval_ms"], 95)

        rk_mean = statistics.mean(metrics["rerank_ms"])
        rk_med = statistics.median(metrics["rerank_ms"])

        tot_mean = statistics.mean(metrics["total_ms"])
        tot_med = statistics.median(metrics["total_ms"])
        tot_p95 = calculate_percentile(metrics["total_ms"], 95)

        recall = statistics.mean(metrics["hits"]) * 100.0 if metrics["hits"] else 0.0

        print(
            f"{mode_name:<22} | "
            f"{r_mean:4.1f}/{r_med:4.1f}/{r_p95:4.1f}   | "
            f"{rk_mean:4.2f}/{rk_med:4.2f}    | "
            f"{tot_mean:5.1f}/{tot_med:5.1f}/{tot_p95:5.1f}   | "
            f"{recall:5.1f}%"
        )

        summary[mode_name] = {
            "retrieval_mean_ms": round(r_mean, 2),
            "retrieval_median_ms": round(r_med, 2),
            "retrieval_p95_ms": round(r_p95, 2),
            "rerank_mean_ms": round(rk_mean, 2),
            "rerank_median_ms": round(rk_med, 2),
            "total_mean_ms": round(tot_mean, 2),
            "total_median_ms": round(tot_med, 2),
            "total_p95_ms": round(tot_p95, 2),
            "recall_at_5": round(recall, 1),
        }

    print("=" * 95)
    return summary


if __name__ == "__main__":
    with_llm_flag = "--with-llm" in sys.argv
    run_benchmark(iterations=3, with_llm=with_llm_flag)
