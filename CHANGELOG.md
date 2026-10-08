# Changelog

All notable changes to the Aura AI codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

# [1.2.0] - 2026-09-17

### Added
- **Aura AI Loading Screen & Visual Processing Experience (Phase 12)**:
  - Non-blocking visual loading animation buffer during live RAG processing.
  - Multi-tier loading asset resolution for `ui/assets/aura-loading-bubble.mp4` with in-memory base64 caching.
  - Silent, muted HTML5 playback (`autoplay`, `loop`, `muted`, `playsinline`, `disablepictureinpicture`, `pointer-events: none`).
  - Seamless dark glassmorphism integration with CSS `mix-blend-mode: screen` and fallback glowing ring.
  - Independent DOM container separation preventing video reload/recreation during SSE status updates.
  - Clean transition to streamed tokens upon first real token delivery and final completion badge.
  - Automatic `finally` cleanup ensuring no stuck loaders on error or stream interruption.
  - Complete 20-test test suite in `tests/test_loading_experience.py` (348 passing tests baseline).

- **Intelligent Response Pipeline & Reasoning UX (Phase 11)**:
  - User-safe status events communicating real backend operations (understanding, classification, retrieval, reranking, evidence review, answer preparation, generation, citation verification).
  - Deterministic adaptive execution planning and query classification.
  - Timing and reasoning drawer with non-intrusive latency badges.
  - Zero exposure of private prompts, internal thoughts, or chain-of-thought traces.

# [1.0.0] - 2026-09-09

### Added
- **Multi-Format Document Ingestion (Phases 1–3)**:
  - Comprehensive document extraction engine supporting PDF (1-indexed page numbering), DOCX (headings, paragraphs, and markdown tables), TXT (multi-encoding UTF-8/Latin-1 fallback), CSV (row-level traceability), SQLite (schema inspection and table records), and Images (Pillow with 50MP decompression guard).
  - Rich document provenance tracking with standard metadata schema sanitized for vector storage.
- **Dual-Channel Hybrid Retrieval & Reranking (Phase 4)**:
  - Chroma vector retrieval (`text-embedding-3-small`) coupled with persistent, cache-synchronized BM25 lexical inverted indices.
  - Reciprocal Rank Fusion (RRF, $k=60$) combining semantic and keyword candidates.
  - Deterministic lexical reranker scoring exact phrase coverage and metadata matches with transparent scoring.
  - Bidirectional retrieval fallbacks (Vector $\leftrightarrow$ BM25).
- **Canonical Evidence & Deterministic Citations (Phase 5)**:
  - Internal `CanonicalEvidence` layer separating evidence tracking from display formatting.
  - Deterministic citation labels formatted as `[S#] filename, "Section", p. #`.
  - Post-generation citation AST validator to detect, log, and strip hallucinated or unbacked citations (e.g., `[S99]`).
- **Conversational RAG & Persistent Sessions (Phase 6)**:
  - SQLite persistent conversation storage (`data/sessions.sqlite`) with cryptographic session isolation and cascading foreign keys.
  - Conversational query contextualization rewriting elliptical questions against recent turn history.
  - Deterministic session titling without external LLM calls.
  - RESTful endpoints for session management (`/api/v1/sessions`).
- **Live Multimodal RAG (Phase 7)**:
  - Support for image attachments in conversational queries via multipart/form-data or base64.
  - Dual-tier OCR-first processing with automatic GPT-4o Vision fallback.
  - Ephemeral memory tagging (`is_ephemeral=True`) preventing chat image leakage into permanent vector stores.
- **Cross-Document Comparison Engine (Phase 8)**:
  - Balanced multi-document retrieval ensuring equal representation across compared documents.
  - Difference, similarity, and metric comparison modes (`/api/v1/compare`).
  - Deterministic Python numerical calculation for percentage changes, deltas, and ratios.
  - Contradiction and conflict detection between documents.
- **Adversarial Security & System Hardening (Phase 9)**:
  - Prompt injection boundary defense fencing untrusted document text.
  - Filename and path traversal sanitization blocking `../../` escape.
  - Memory bounds on file sizes (50MB) and image megapixels (50MP).
  - Safe error responses masking internal tracebacks and filesystem paths.
- **Safe Text-to-SQL & Structured Data Intelligence (Phase 10)**:
  - Natural language analytical query conversion for SQLite and CSV datasets (`/api/v1/sql-query`).
  - AST SQL validation blocking DDL, DML, PRAGMAs, ATTACH, and multi-statements.
  - Low-level SQLite C-authorizer enforcing read-only access (`mode=ro`).
  - Execution timeout enforcement via `sqlite3.set_progress_handler` ($5.0\text{ s}$).
- **Large-Scale Evaluation Framework (Stages 1–5)**:
  - Benchmark suite of 60 documents and 520 queries with independent ground truth.
  - Metric calculation harness for Recall@K, Precision@K, MRR, and corrected nDCG@K.
  - Comprehensive Stage 1–5 evaluation and methodology audit reports.
