# Aura AI — Large-Scale RAG Evaluation: Stage 1 Report

## 1. Executive Summary

Stage 1 of the Aura AI Large-Scale RAG Evaluation is complete. An evaluation harness and benchmark dataset have been constructed to measure the frozen Aura AI system (Phases 1–10) without any modifications or optimizations to the underlying code.

The benchmark exceeds the target threshold (minimum 300 queries, preferred 500+ queries) with **520 curated questions** grounded across **60 diverse documents** spanning PDF, DOCX, TXT, CSV, SQLite, and PNG formats. The evaluation harness supports multi-baseline comparisons, machine-readable per-query logs, retrieval and answer metrics, citation audits, unanswerable query detection, and latency percentiles.

All 204 automated regression tests from Phases 1–10 remain passing (204/204, 0 failures).

---

## 2. Benchmark Architecture & Directory Structure

```text
evaluation/
├── documents/                  # 60 benchmark documents across 6 formats
│   ├── manifest.json           # Complete metadata, provenance, and SHA-256 hashes
│   ├── *.txt (18 files)        # Technical specs, policies, agreements, reports
│   ├── *.pdf (14 files)        # Financial earnings, whitepapers, clinical trials, roadmaps
│   ├── *.docx (8 files)        # Charters, SLAs, postmortems, patents
│   ├── *.csv (8 files)         # Regional sales, churn, inventory, salaries, telemetry
│   ├── *.sqlite / *.db (6 files)# E-commerce, hospital, telecom, HR, logistics, manufacturing
│   └── *.png (6 files)         # Network topology, revenue charts, rack diagrams, notices
├── questions/
│   └── benchmark_520.json      # 520 benchmark questions across 9 distinct categories
├── ground_truth/
│   └── ground_truth_520.json   # Independent ground-truth answers, supporting evidence & facts
├── results/                    # Machine-readable per-query evaluation run logs (JSON)
│   └── benchmark_results_hybrid.json
├── reports/                    # Human-readable markdown reports and summaries
│   └── evaluation_harness_report.md
├── metrics.py                  # Evaluation metrics calculation library
├── evaluator.py                # Backward-compatible evaluation runner (for test suite)
└── run.py                      # CLI evaluation harness runner
```

---

## 3. Benchmark Dataset Composition

### 3.1 Document Corpus (60 Documents)
The evaluation corpus contains 60 documents spanning 6 file types with provenance metadata recorded in `evaluation/documents/manifest.json`:

| File Type | Count | Representative Documents | Provenance & Description |
| :--- | :--- | :--- | :--- |
| **TXT** | 18 | `doc_gamma_energy.txt`, `doc_delta_aerospace.txt`, `doc_epsilon_health.txt`, `doc_zeta_supply_chain.txt`, `doc_eta_hr_policy.txt`, `doc_alpha.txt`, `doc_beta.txt`, `doc_security_policy.txt` | Synthetic enterprise policies, clinical protocols, engineering specifications, and security policies. |
| **PDF** | 14 | `pdf_q1_earnings.pdf`, `pdf_q2_earnings.pdf`, `pdf_cybersecurity_standard.pdf`, `pdf_clinical_study_summary.pdf`, `pdf_cloud_migration_plan.pdf` | Multi-page documents with headings, sections, page numbers, financial disclosures, and clinical efficacy data. |
| **DOCX** | 8 | `docx_project_titan_charter.docx`, `docx_vendor_sla_agreement.docx`, `docx_incident_retrospective.docx`, `docx_patent_application_draft.docx` | OpenXML packages with structured headings, milestone budgets, outage root-cause analyses, and legal agreements. |
| **CSV** | 8 | `csv_sales_regional_2024.csv`, `csv_customer_churn_metrics.csv`, `csv_inventory_warehouse_stock.csv`, `csv_employee_salaries.csv` | Structured tables with numerical quantities, currency amounts, dates, employee records, and server metrics. |
| **SQLite** | 6 | `sqlite_ecommerce.db`, `sqlite_hospital.db`, `sqlite_telecom.db`, `sqlite_human_resources.db`, `sqlite_logistics.db` | Relational databases with multiple tables, primary keys, foreign relations, customer orders, and telemetry logs. |
| **PNG** | 6 | `img_network_topology.png`, `img_quarterly_revenue_chart.png`, `img_server_rack_diagram.png`, `img_maintenance_sign.png` | Architecture flowcharts, quarterly bar charts, hardware rack power draw diagrams, and scanned clinical records. |

### 3.2 Question Distribution (520 Questions)
The benchmark contains 520 curated questions strictly conforming to the requested distribution:

| Category | Count | Proportion | Description |
| :--- | :--- | :--- | :--- |
| **Simple factual** | 104 | **20.0%** | Single-hop direct retrieval of explicit facts, metrics, and parameters. |
| **Specific section/page retrieval** | 78 | **15.0%** | Targeted queries citing specific sections, page numbers, or clauses. |
| **Multi-chunk reasoning** | 78 | **15.0%** | Synthesizing facts distributed across multiple paragraphs or chunks. |
| **Cross-document questions** | 78 | **15.0%** | Comparative analysis across two or more distinct documents (e.g. Q1 vs Q2 earnings). |
| **Multi-hop questions** | 52 | **10.0%** | Two-step reasoning queries linking dependent facts across systems. |
| **Tables / structured data** | 52 | **10.0%** | Analytical and aggregation queries over CSV and SQLite tables (`SUM`, `AVG`, `COUNT`, `GROUP BY`). |
| **Multimodal questions** | 26 | **5.0%** | Queries grounded in image assets (bar charts, network diagrams, server rack power draw). |
| **Missing-information questions** | 26 | **5.0%** | Deliberately unanswerable questions to evaluate missing info detection and hallucination resistance. |
| **Conversational follow-up questions** | 26 | **5.0%** | Multi-turn queries testing pronoun resolution, context persistence, and turn isolation. |
| **Total** | **520** | **100.0%** | **Target exceeded (minimum 300, preferred 500+)** |

---

## 4. Ground Truth Methodology & Leakage Prevention

To ensure evaluation integrity:
1. **Independent Formulation**: Ground truth was written directly from the source document specifications before running evaluation. At no point was Aura AI used to determine what was relevant or correct.
2. **Explicit Fact Assertions**: Each ground-truth record defines:
   - `expected_answer`: The exact factual answer or numerical value.
   - `relevant_document_ids`: List of document filenames containing the supporting evidence.
   - `supporting_evidence`: Verbatim text excerpt or SQL table row.
   - `key_facts`: List of required factual tokens/strings that must appear in a complete answer.
   - `prohibited_facts`: Tokens or false claims that indicate hallucination or false premise acceptance.
   - `answerable`: Boolean flag (`False` for missing-information queries).
3. **No Target Leakage**: Ground-truth answers and expected sources are stored in `evaluation/ground_truth/ground_truth_520.json` and are never exposed to Aura AI during retrieval or generation.

---

## 5. Metrics Implemented

All metrics are implemented in [`evaluation/metrics.py`](file:///C:/Users/Sujit%20Kumar/Downloads/aura%20ai/multi-modal-rag-chatbot/evaluation/metrics.py):

### 5.1 Retrieval Metrics (Independent of LLM Generation)
- **Recall@K** ($K \in \{1, 3, 5, 10\}$): Proportion of ground-truth relevant documents retrieved in the top $K$.
- **Precision@K** ($K \in \{1, 3, 5, 10\}$): Proportion of top-$K$ retrieved documents that are relevant.
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant document retrieved.
- **Hit Rate@K**: Binary indicator if at least one relevant document appears in top-$K$.
- **nDCG@K**: Normalized Discounted Cumulative Gain accounting for rank positions.

### 5.2 Deterministic Answer Metrics
- **Exact Match (EM)**: Strict string equality between generated and expected answers.
- **Normalized Exact Match**: Case-folded, punctuation-stripped, whitespace-normalized equality.
- **Token F1**: Precision, Recall, and harmonic mean of shared unigrams.
- **Numeric Accuracy**: Extracts all numbers/percentages and verifies they match ground truth within a 2% tolerance.

### 5.3 Citation Metrics
- **Citation Validity Rate**: Fraction of inline citations (`[S1]`, `[S2]`) that correspond to actual retrieved sources.
- **Citation Completeness Rate**: Fraction of sentences expressing key facts that include an accompanying citation.

### 5.4 Missing Information & Hallucination Metrics
- **Missing Information Detection Rate**: Fraction of unanswerable queries where the system correctly refuses or flags insufficient evidence.
- **Hallucination Rate**: Fraction of unanswerable queries where the system fabricated answers or asserted prohibited facts.

### 5.5 Latency Percentiles
- Tracks min, mean, median (p50), p90, p95, and p99 for:
  - Retrieval latency (ms)
  - Generation latency (ms)
  - Total end-to-end latency (ms)

### 5.6 LLM Judge Interface
- Provides structured scoring (0–4 scale) for:
  - **Correctness**
  - **Completeness**
  - **Faithfulness**
  - **Relevance**
- Includes a deterministic, offline proxy scoring function based on token F1, key fact coverage, and answerability rules when running without API keys.

---

## 6. Evaluation Harness CLI (`evaluation/run.py`)

The evaluation runner provides a flexible, reproducible CLI:

```bash
python -m evaluation.run \
  --dataset evaluation/questions/benchmark_520.json \
  --documents evaluation/documents \
  --ground-truth evaluation/ground_truth/ground_truth_520.json \
  --limit 500 \
  --seed 42 \
  --top-k 5 \
  --mode hybrid \
  --output evaluation/results \
  --resume
```

### Supported Modes & Baselines:
- `vector`: Vector-only retrieval
- `bm25`: BM25 lexical retrieval
- `hybrid`: Reciprocal Rank Fusion (RRF) combining Vector and BM25
- `rerank`: Hybrid retrieval + lexical reranking
- `all`: Sequentially executes all 4 baselines over the exact same benchmark and outputs `baselines_comparison_summary.json`.

---

## 7. Initial Verification Smoke Test Results

To verify the harness without running the full 520 queries prematurely, a 10-query smoke test was executed (`--limit 10`):

```text
Indexing benchmark documents from evaluation\documents...
Indexed 60 documents (136 chunks) into BM25.
Starting evaluation of 10 queries (Mode: hybrid, Top-K: 5, Seed: 42)...
Progress: [10/10] queries evaluated.

Benchmark Execution Summary:
- Total Queries Evaluated: 10
- Successful Queries: 10
- Failed Queries: 0
- Mean Recall@5: 1.0 (100%)
- Hit Rate@5: 1.0 (100%)
- Mean MRR: 0.95
- Citation Validity Rate: 1.0 (100%)
- Missing Info Detection Rate: 1.0 (100%)
- Hallucination Rate: 0.0 (0%)
- Retrieval Latency: Mean = 31.32 ms, Median = 19.16 ms, p95 = 89.02 ms
- Total Latency: Mean = 31.36 ms, Median = 19.20 ms, p95 = 89.06 ms
```

---

## 8. Status Classification Table

| Component / Capability | Status | Notes |
| :--- | :--- | :--- |
| **60 Benchmark Documents** | **IMPLEMENTED & VERIFIED** | All 60 files created across PDF, DOCX, TXT, CSV, SQLite, PNG. `manifest.json` generated. |
| **520 Questions Dataset** | **IMPLEMENTED & VERIFIED** | 520 questions across 9 categories matching exact required proportions. |
| **Ground Truth Dataset** | **IMPLEMENTED & VERIFIED** | 520 ground-truth records with evidence excerpts and unanswerable flags. |
| **Evaluation Metrics Engine** | **IMPLEMENTED & VERIFIED** | Retrieval, answer, citation, missing info, latency percentiles, and LLM judge proxy. |
| **Harness CLI (`evaluation.run`)** | **IMPLEMENTED & VERIFIED** | CLI supporting `--limit`, `--seed`, `--top-k`, `--mode`, `--output`, `--resume`. |
| **Baselines Comparison Adapter** | **IMPLEMENTED & READY TO RUN** | Supports `vector`, `bm25`, `hybrid`, `rerank`, and `all`. |
| **Per-Query Machine-Readable JSON** | **IMPLEMENTED & VERIFIED** | Output written to `evaluation/results/benchmark_results_hybrid.json`. |
| **10-Query Smoke Test** | **ACTUALLY EXECUTED** | 10/10 queries executed, 0 errors, 100% recall@5, 1.0 citation validity. |
| **Full 520-Query Benchmark** | **READY TO RUN** | Awaiting user approval to run full benchmark. |
| **LLM Online Judge Scoring** | **READY TO RUN** | Implemented with fallback proxy; ready for API key invocation. |
| **Production Code Modifications** | **ZERO (FROZEN)** | Production RAG code was not modified or tuned. |
| **Regression Test Suite** | **ACTUALLY EXECUTED** | `pytest -q`: **204 passed / 204 total**, 0 failed. |

---

## 9. Next Steps (Awaiting Approval)

Stage 1 objective is achieved. The evaluation harness and benchmark are ready. Per instructions, execution stops immediately without running the full 520 queries or modifying the Aura AI codebase.
