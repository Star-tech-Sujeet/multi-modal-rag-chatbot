"""
Test suite validating Phase 2 stabilization and bug fixes:
1. Configuration resilience on missing/placeholder OPENAI_API_KEY.
2. Docker Compose frontend configuration.
3. Configurable text splitter chunk size and overlap.
4. Model routing mechanism (primary vs mini vs heuristics).
5. BM25 pre-tokenized caching, query reuse, and deletion.
6. Deterministic chunk ID assignment and collision-free RRF.
7. Non-blocking FastAPI endpoints with run_in_threadpool.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from fastapi.testclient import TestClient

from app.config import Settings, is_api_key_valid, require_api_key, settings
from app.models import QueryRequest, Source
from app.logic import (
    get_text_splitter,
    resolve_model,
    reciprocal_rank_fusion,
    BM25IndexManager,
    FileBM25Index,
    bm25_manager,
    bm25_search_instance,
)
from app.main import app



# =============================================================================
# 1. Configuration Resilience Tests
# =============================================================================
class TestConfigResilience:
    """Test OPENAI_API_KEY handling and configuration safety."""

    def test_placeholder_api_key_does_not_crash(self):
        """Placeholder key should initialize cleanly without throwing validation crash."""
        test_settings = Settings(ai_provider="openai", openai_api_key="your-openai-api-key-here")
        assert test_settings.openai_api_key == "your-openai-api-key-here"
        assert is_api_key_valid(test_settings) is False

    def test_empty_api_key_does_not_crash(self):
        """Empty key should initialize cleanly."""
        test_settings = Settings(ai_provider="openai", openai_api_key="")
        assert test_settings.openai_api_key == ""
        assert is_api_key_valid(test_settings) is False

    def test_valid_api_key_passes(self):
        """Real/valid API key should pass validation."""
        test_settings = Settings(ai_provider="openai", openai_api_key="sk-proj-testkey1234567890abcdef")
        assert is_api_key_valid(test_settings) is True
        # require_api_key should not raise
        assert require_api_key(test_settings) == "sk-proj-testkey1234567890abcdef"

    def test_require_api_key_raises_helpful_error_when_missing(self):
        """require_api_key must raise descriptive ValueError when missing."""
        test_settings = Settings(ai_provider="openai", openai_api_key="your-openai-api-key-here")
        with pytest.raises(ValueError) as exc_info:
            require_api_key(test_settings)
        assert "OPENAI_API_KEY is not configured" in str(exc_info.value)


# =============================================================================
# 2. Docker Compose Configuration Tests
# =============================================================================
class TestDockerConfiguration:
    """Verify Docker Compose configuration for frontend and backend."""

    def test_docker_compose_frontend_env(self):
        """docker-compose.yml must have API_BASE_URL and no sed command hacks."""
        compose_path = Path("docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml missing"
        content = compose_path.read_text(encoding="utf-8")

        assert "API_BASE_URL=http://backend:8000/api/v1" in content
        assert "DOCKER_ENV=true" in content
        assert "sed -i" not in content, "Brittle sed string replacement found in docker-compose.yml"


# =============================================================================
# 3. Chunk Configuration Tests
# =============================================================================
class TestChunkConfiguration:
    """Verify text splitter respects configured chunk_size and chunk_overlap."""

    def test_default_splitter_uses_settings(self):
        """get_text_splitter() without args should use settings chunk_size and overlap."""
        splitter = get_text_splitter()
        assert splitter._chunk_size == settings.chunk_size
        assert splitter._chunk_overlap == settings.chunk_overlap

    def test_custom_splitter_parameters(self):
        """get_text_splitter() should allow overriding chunk_size and chunk_overlap."""
        custom_size = 350
        custom_overlap = 50
        splitter = get_text_splitter(chunk_size=custom_size, chunk_overlap=custom_overlap)
        assert splitter._chunk_size == custom_size
        assert splitter._chunk_overlap == custom_overlap

        # Test splitting text
        sample_text = "Word " * 200
        chunks = splitter.split_text(sample_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= custom_size + 10  # approximate boundary tolerance


# =============================================================================
# 4. Model Routing Tests
# =============================================================================
class TestModelRouting:
    """Verify model routing preferences and reasoning heuristics under OpenAI provider."""

    @pytest.fixture(autouse=True)
    def setup_openai_provider(self):
        with patch.object(settings, "ai_provider", "openai"):
            yield

    def test_explicit_primary_preference(self):
        """'primary' or 'complex' preferences route to settings.openai_model."""
        assert resolve_model("primary") == settings.openai_model
        assert resolve_model("complex") == settings.openai_model

    def test_explicit_mini_preference(self):
        """'mini' or 'standard' preferences route to settings.openai_mini_model."""
        assert resolve_model("mini") == settings.openai_mini_model
        assert resolve_model("standard") == settings.openai_mini_model

    def test_explicit_custom_model_name(self):
        """Direct model names starting with gpt-, o1, o3, chatgpt pass through."""
        assert resolve_model("gpt-4.5-preview") == "gpt-4.5-preview"
        assert resolve_model("o1-mini") == "o1-mini"
        assert resolve_model("o3-mini") == "o3-mini"

    def test_heuristic_routing_for_complex_query(self):
        """Complex reasoning triggers route to primary model automatically."""
        complex_q = "Please compare and contrast the methodology between the two studies."
        assert resolve_model(question=complex_q) == settings.openai_model

        detail_q = "Provide a detailed analysis of the financial statements."
        assert resolve_model(question=detail_q) == settings.openai_model

    def test_heuristic_routing_for_simple_query(self):
        """Simple questions default to mini model."""
        simple_q = "What is the revenue for 2023?"
        assert resolve_model(question=simple_q) == settings.openai_mini_model


# =============================================================================
# 5. BM25 Caching and Management Tests
# =============================================================================
class TestBM25Caching:
    """Verify BM25 caching, persistent storage, query reuse, and deletion."""

    def test_bm25_manager_lifecycle(self):
        """Test build, cache lookup, serialization, query, and deletion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = BM25IndexManager(storage_dir=tmp_dir)

            file_id = "test-doc-001"
            chunk_ids = [f"{file_id}_chunk_0", f"{file_id}_chunk_1", f"{file_id}_chunk_2"]
            docs = [
                Document(
                    page_content="Quantum computing leverages qubits and quantum superposition.",
                    metadata={"file_id": file_id, "chunk_id": chunk_ids[0], "filename": "quantum.txt"}
                ),
                Document(
                    page_content="Classical computing uses binary bits which are either 0 or 1.",
                    metadata={"file_id": file_id, "chunk_id": chunk_ids[1], "filename": "quantum.txt"}
                ),
                Document(
                    page_content="Distributed consensus protocols ensure fault tolerant states.",
                    metadata={"file_id": file_id, "chunk_id": chunk_ids[2], "filename": "quantum.txt"}
                )
            ]

            # Build and save
            idx = mgr.save_file_index(file_id, docs, chunk_ids)
            assert file_id in mgr._cache
            assert len(idx.chunk_ids) == 3

            # Verify JSON file on disk
            json_file = Path(tmp_dir) / f"{file_id}.json"
            assert json_file.exists()

            # Verify loading from disk if memory cache is cleared
            mgr._cache.clear()
            loaded_idx = mgr.get_or_load_index(file_id)
            assert loaded_idx is not None
            assert len(loaded_idx.chunk_ids) == 3

            # Test search query
            results = bm25_search_instance(
                query="qubits superposition",
                bm25_index=loaded_idx.bm25,
                documents=loaded_idx.documents,
                chunk_ids=loaded_idx.chunk_ids,
                k=2
            )
            assert len(results) >= 1
            top_doc, top_cid, top_score = results[0]
            assert "Quantum computing" in top_doc.page_content
            assert top_score > 0.0

            # Test deletion
            mgr.delete_file_index(file_id)
            assert not json_file.exists()
            assert file_id not in mgr._cache

    def test_multi_file_bm25_search(self):
        """Test merging pre-tokenized corpora across multiple documents."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = BM25IndexManager(storage_dir=tmp_dir)

            doc1_id = "doc-alpha"
            doc2_id = "doc-beta"

            mgr.save_file_index(doc1_id, [
                Document(
                    page_content="Alpha document discusses machine learning algorithms and neural networks.",
                    metadata={"file_id": doc1_id, "chunk_id": f"{doc1_id}_chunk_0"}
                )
            ], [f"{doc1_id}_chunk_0"])

            mgr.save_file_index(doc2_id, [
                Document(
                    page_content="Beta document discusses database indexing techniques and B-trees.",
                    metadata={"file_id": doc2_id, "chunk_id": f"{doc2_id}_chunk_0"}
                )
            ], [f"{doc2_id}_chunk_0"])

            idx1 = mgr.get_or_load_index(doc1_id)
            idx2 = mgr.get_or_load_index(doc2_id)
            assert idx1 is not None and idx2 is not None

            # Combined tokenized corpus for multi-document search
            from rank_bm25 import BM25Okapi
            combined_corpus = idx1.tokenized_corpus + idx2.tokenized_corpus
            combined_bm25 = BM25Okapi(combined_corpus)
            combined_docs = idx1.documents + idx2.documents
            combined_chunk_ids = idx1.chunk_ids + idx2.chunk_ids

            results = bm25_search_instance(
                "neural networks",
                combined_bm25,
                combined_docs,
                combined_chunk_ids,
                k=2
            )
            assert len(results) >= 1
            assert results[0][0].metadata["file_id"] == doc1_id


# =============================================================================
# 6. Chunk Identification and Collision-Free RRF Tests
# =============================================================================
class TestChunkIdentificationAndRRF:
    """Verify chunk_id uniqueness and collision resistance in RRF."""

    def test_rrf_with_identical_text_chunks(self):
        """
        When two chunks have identical text (e.g. repeated table headers or disclaimers),
        RRF must not collapse them into one chunk if their chunk_id is distinct.
        """
        shared_text = "Confidential and Proprietary. All Rights Reserved."

        doc_a = Document(
            page_content=shared_text,
            metadata={"file_id": "file1", "chunk_id": "file1_chunk_0", "page": 1}
        )
        doc_b = Document(
            page_content=shared_text,
            metadata={"file_id": "file1", "chunk_id": "file1_chunk_5", "page": 6}
        )

        # Vector search finds doc_a at rank 1, doc_b at rank 2
        vector_results = [(doc_a, "file1_chunk_0", 0.95), (doc_b, "file1_chunk_5", 0.90)]
        # BM25 search finds doc_b at rank 1, doc_a at rank 2
        bm25_results = [(doc_b, "file1_chunk_5", 1.5), (doc_a, "file1_chunk_0", 1.2)]

        fused = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=60
        )

        assert len(fused) == 2, "Both chunks must be preserved; no collision on identical text"
        fused_chunk_ids = [d.metadata["chunk_id"] for d, _, _, _, _ in fused]
        assert "file1_chunk_0" in fused_chunk_ids
        assert "file1_chunk_5" in fused_chunk_ids

    def test_source_model_contains_chunk_id(self):
        """Source response model must include chunk_id."""
        source = Source(
            filename="report.pdf",
            file_id="abc-123",
            chunk_id="abc-123_chunk_4",
            page_number=4,
            content="Summary content...",
            relevance_score=0.88,
            vector_score=0.85,
            bm25_score=1.2,
            search_type="hybrid"
        )
        assert source.chunk_id == "abc-123_chunk_4"
        serialized = source.model_dump()
        assert serialized["chunk_id"] == "abc-123_chunk_4"


# =============================================================================
# 7. FastAPI Endpoints & Async Threadpool Tests
# =============================================================================
class TestFastAPIEndpoints:
    """Verify FastAPI endpoints operate cleanly with run_in_threadpool."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test API root returns 200 and healthy status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_health_endpoint(self, client):
        """Test /api/v1/health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ("healthy", "degraded")
        assert "vectorstore" in data

    def test_health_endpoint_calls_stats_in_threadpool(self, client):
        """Test /api/v1/health executes stats via threadpool."""
        with patch("app.api.get_vectorstore_stats") as mock_stats:
            mock_stats.return_value = {
                "total_chunks": 42,
                "total_documents": 2,
                "embedding_model": "text-embedding-3-small",
                "status": "healthy"
            }
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            assert response.json()["vectorstore"]["total_chunks"] == 42
            mock_stats.assert_called_once()

    def test_documents_list_endpoint(self, client):
        """Test /api/v1/files returns document list."""
        response = client.get("/api/v1/files")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_query_validation_error(self, client):
        """Test /api/v1/query returns 422 on invalid question."""
        payload = {
            "question": "",  # Empty question violates validator
            "use_hybrid_search": True
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422

    def test_query_runtime_error_handling(self, client):
        """Test /api/v1/query returns 400 with message if perform_rag_query raises ValueError."""
        with patch("app.api.perform_rag_query") as mock_rag:
            mock_rag.side_effect = ValueError("OPENAI_API_KEY is not configured")
            payload = {
                "question": "What is AI?",
                "file_id": "12345678-1234-5678-1234-567812345678",
                "use_hybrid_search": True
            }
            response = client.post("/api/v1/query", json=payload)
            assert response.status_code == 400
            resp_json = response.json()
            error_text = str(resp_json.get("details") or resp_json.get("message") or "")
            assert "OPENAI_API_KEY is not configured" in error_text

    def test_delete_file_endpoint(self, client):
        """Test DELETE /api/v1/files/{file_id} calls threadpool."""
        valid_uuid = "12345678-1234-5678-1234-567812345678"
        with patch("app.api.run_in_threadpool") as mock_run:
            mock_run.return_value = True
            response = client.delete(f"/api/v1/files/{valid_uuid}")
            assert response.status_code == 200
            assert response.json()["file_id"] == valid_uuid


# =============================================================================
# 8. Ingestion & Embedding Chunk ID Assignment Tests
# =============================================================================
class TestIngestionChunkIDs:
    """Verify deterministic chunk ID assignment during embedding storage."""

    def test_create_and_store_embeddings_sets_ids(self):
        """create_and_store_embeddings must set chunk_id metadata and pass explicit ids."""
        from app.logic import create_and_store_embeddings

        file_id = "test-doc-999"
        docs = [
            Document(page_content="First chunk text"),
            Document(page_content="Second chunk text")
        ]

        with patch("app.logic.get_vectorstore") as mock_vs, \
             patch.object(bm25_manager, "save_file_index") as mock_save:
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            count = create_and_store_embeddings(docs, file_id)

            assert count == 2
            # Verify metadata on docs
            assert docs[0].metadata["chunk_id"] == f"{file_id}_chunk_0"
            assert docs[1].metadata["chunk_id"] == f"{file_id}_chunk_1"
            assert docs[0].metadata["chunk_index"] == 0
            assert docs[1].metadata["chunk_index"] == 1

            # Verify vectorstore.add_documents called with explicit ids
            mock_store.add_documents.assert_called_once()
            called_docs, called_kwargs = mock_store.add_documents.call_args
            assert called_kwargs["ids"] == [f"{file_id}_chunk_0", f"{file_id}_chunk_1"]

            # Verify BM25 index saved with explicit chunk_ids
            mock_save.assert_called_once_with(
                file_id,
                docs,
                [f"{file_id}_chunk_0", f"{file_id}_chunk_1"]
            )


# =============================================================================
# 9. End-to-End Document Ingestion & Retrieval Pipeline Test
# =============================================================================
class TestE2EPipeline:
    """Verify document processing, chunking, BM25 indexing, and retrieval flow."""

    def test_sample_document_processing_and_bm25(self):
        """Test processing real sample.txt, splitting, indexing, and querying."""
        from app.logic import process_txt, get_text_splitter

        sample_path = "sample_docs/sample.txt"
        assert Path(sample_path).exists()

        # 1. Process document
        raw_docs = process_txt(sample_path, "sample.txt")
        assert len(raw_docs) >= 1

        # 2. Chunk document
        splitter = get_text_splitter()
        chunks = splitter.split_documents(raw_docs)
        assert len(chunks) >= 1

        # 3. Assign chunk IDs
        test_fid = "sample-test-uuid"
        for i, chunk in enumerate(chunks):
            chunk.metadata["file_id"] = test_fid
            chunk.metadata["chunk_id"] = f"{test_fid}_chunk_{i}"
            chunk.metadata["chunk_index"] = i

        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = BM25IndexManager(storage_dir=tmp_dir)
            chunk_ids = [c.metadata["chunk_id"] for c in chunks]

            # 4. Save BM25 Index
            f_idx = mgr.save_file_index(test_fid, chunks, chunk_ids)
            assert mgr.get_or_load_index(test_fid) is not None

            # 5. Search for keyword
            results = bm25_search_instance(
                query="Natural Language Processing",
                bm25_index=f_idx.bm25,
                documents=f_idx.documents,
                chunk_ids=f_idx.chunk_ids,
                k=3
            )
            assert len(results) >= 1
            top_doc, top_cid, top_score = results[0]
            assert "Natural Language Processing" in top_doc.page_content

            # 6. Delete BM25 Index
            mgr.delete_file_index(test_fid)
            assert not (Path(tmp_dir) / f"{test_fid}.json").exists()


# =============================================================================
# 10. Document Deletion Robustness & Cleanup Regression Tests
# =============================================================================
class TestDocumentDeletionRobustness:
    """
    Regression tests verifying document deletion cleanup robustness:
    A. Normal deletion: raw uploaded file, ChromaDB records, and BM25 index are removed.
    B. Deletion when vector-store cleanup fails: BM25 index is STILL removed, and uploaded file cleanup verified.
    C. Deletion without an OpenAI API key: BM25 cleanup still occurs without requiring API keys.
    """

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_normal_deletion_removes_raw_file_chroma_and_bm25(self, client):
        """Test A: Normal deletion removes raw file, ChromaDB records, and BM25 index."""
        import uuid
        from app.logic import get_chroma_collection, delete_document_embeddings

        file_id = str(uuid.uuid4())
        
        # 1. Create raw uploaded file
        uploads_dir = Path(settings.uploads_path)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        raw_file = uploads_dir / f"{file_id}_sample_doc.pdf"
        raw_file.write_bytes(b"Dummy PDF content for testing")
        assert raw_file.exists()

        # 2. Create ChromaDB records
        chroma_coll = get_chroma_collection("langchain")
        chroma_coll.add(
            ids=[f"{file_id}_chunk_0", f"{file_id}_chunk_1"],
            documents=["Chunk 0 text", "Chunk 1 text"],
            metadatas=[{"file_id": file_id}, {"file_id": file_id}]
        )
        assert len(chroma_coll.get(where={"file_id": file_id})["ids"]) == 2

        # 3. Create BM25 index on disk and in cache
        bm25_dir = Path(settings.bm25_path)
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25_file = bm25_dir / f"{file_id}.json"
        bm25_manager.save_file_index(file_id, [
            Document(page_content="Chunk 0 text", metadata={"file_id": file_id}),
            Document(page_content="Chunk 1 text", metadata={"file_id": file_id})
        ], [f"{file_id}_chunk_0", f"{file_id}_chunk_1"])
        assert bm25_file.exists()
        assert file_id in bm25_manager._cache

        # 4. Perform deletion via API
        response = client.delete(f"/api/v1/files/{file_id}")
        assert response.status_code == 200

        # 5. Verify cleanup
        # - raw file removed
        assert not raw_file.exists(), "Raw uploaded file was not removed"
        # - ChromaDB records removed
        remaining_chroma = chroma_coll.get(where={"file_id": file_id})["ids"]
        assert len(remaining_chroma) == 0, "ChromaDB records were not removed"
        # - BM25 index removed from disk and cache
        assert not bm25_file.exists(), "BM25 JSON index file was orphaned on disk"
        assert file_id not in bm25_manager._cache, "BM25 index lingered in memory cache"

    def test_deletion_when_vectorstore_cleanup_fails_still_removes_bm25_and_uploaded_file(self, client):
        """Test B: When vector-store cleanup raises an error, BM25 and uploaded files are STILL removed."""
        import uuid
        from app.logic import delete_document_embeddings

        file_id = str(uuid.uuid4())
        
        # 1. Create raw uploaded file
        uploads_dir = Path(settings.uploads_path)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        raw_file = uploads_dir / f"{file_id}_faulty_chroma_doc.pdf"
        raw_file.write_bytes(b"Faulty test content")
        assert raw_file.exists()

        # 2. Create BM25 index on disk and in cache
        bm25_dir = Path(settings.bm25_path)
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25_file = bm25_dir / f"{file_id}.json"
        bm25_manager.save_file_index(file_id, [
            Document(page_content="Faulty test content", metadata={"file_id": file_id})
        ], [f"{file_id}_chunk_0"])
        assert bm25_file.exists()
        assert file_id in bm25_manager._cache

        # 3. Simulate vectorstore complete failure
        with patch("app.logic.get_chroma_collection") as mock_coll, \
             patch("app.logic.get_vectorstore") as mock_vs:
            mock_coll.side_effect = RuntimeError("Database locked or I/O error in ChromaDB")
            mock_vs.side_effect = RuntimeError("Fallback vectorstore also down")

            # Call delete via API
            response = client.delete(f"/api/v1/files/{file_id}")
            assert response.status_code == 200

        # 4. Verify cleanup occurred despite vector-store failure
        # - raw file removed
        assert not raw_file.exists(), "Raw uploaded file was not removed"
        # - BM25 index removed from disk and cache
        assert not bm25_file.exists(), "BM25 index was orphaned when vectorstore failed!"
        assert file_id not in bm25_manager._cache, "BM25 cache was not cleared"

        # Also verify direct call to delete_document_embeddings re-raises error after cleaning BM25
        file_id2 = str(uuid.uuid4())
        bm25_file2 = bm25_dir / f"{file_id2}.json"
        bm25_manager.save_file_index(file_id2, [
            Document(page_content="Content 2", metadata={"file_id": file_id2})
        ], [f"{file_id2}_chunk_0"])
        assert bm25_file2.exists()

        with patch("app.logic.get_chroma_collection") as mock_coll, \
             patch("app.logic.get_vectorstore") as mock_vs:
            mock_coll.side_effect = RuntimeError("Simulated ChromaDB error")
            mock_vs.side_effect = RuntimeError("Simulated ChromaDB error")

            with pytest.raises(Exception) as exc_info:
                delete_document_embeddings(file_id2)
            assert "Error deleting embeddings from vector store" in str(exc_info.value)

        # BM25 must be cleaned up despite the exception!
        assert not bm25_file2.exists(), "BM25 file was orphaned on direct call failure!"
        assert file_id2 not in bm25_manager._cache

    def test_deletion_without_openai_api_key_cleans_bm25_without_leaving_stale_files(self, client):
        """Test C: Deletion without OpenAI API key cleans BM25 and ChromaDB without requiring embeddings."""
        import uuid
        from app.logic import get_chroma_collection, delete_document_embeddings

        file_id = str(uuid.uuid4())

        # 1. Add records to ChromaDB
        chroma_coll = get_chroma_collection("langchain")
        chroma_coll.add(
            ids=[f"{file_id}_chunk_0"],
            documents=["Sample content"],
            metadatas=[{"file_id": file_id}]
        )
        assert len(chroma_coll.get(where={"file_id": file_id})["ids"]) == 1

        # 2. Add BM25 file
        bm25_dir = Path(settings.bm25_path)
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25_file = bm25_dir / f"{file_id}.json"
        bm25_manager.save_file_index(file_id, [
            Document(page_content="Sample content", metadata={"file_id": file_id})
        ], [f"{file_id}_chunk_0"])
        assert bm25_file.exists()

        # 3. Patch Settings.require_api_key to raise ValueError (simulating missing API key)
        with patch.object(Settings, "require_api_key", side_effect=ValueError("OPENAI_API_KEY is not configured")):
            # delete_document_embeddings must succeed without needing require_api_key
            success = delete_document_embeddings(file_id)
            assert success is True

        # 4. Verify no stale files remain
        assert not bm25_file.exists(), "BM25 index file remained on disk when API key was missing"
        assert file_id not in bm25_manager._cache
        assert len(chroma_coll.get(where={"file_id": file_id})["ids"]) == 0, "ChromaDB records remained"



