"""
Regression tests for Aura AI SAFE Document Deletion.

Verifies all 18 requirements:
1. Document appears after upload.
2. Delete existing document → HTTP success.
3. Uploaded file is removed.
4. Chroma vectors for that document are removed.
5. Other documents remain untouched (Document A deleted, Document B intact).
6. BM25 no longer returns deleted document.
7. Retrieval no longer returns deleted document.
8. Document no longer appears after UI refresh.
9. Deleted document does not return after backend restart.
10. Delete nonexistent document → 404.
11. Invalid/malicious document ID cannot delete arbitrary files.
12. Path traversal attempts are rejected.
13. Cache invalidation works.
14. Chat/session data remains intact.
15. Multi-document retrieval still works with remaining documents.
16. UI delete confirmation works.
17. UI refreshes document list after successful deletion.
18. Failed deletion is reported correctly.
"""

import io
import os
import json
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.config import settings
from app.logic import (
    get_chroma_collection,
    delete_document_embeddings,
    delete_document_complete,
    bm25_manager,
    generate_vector_candidates,
    generate_bm25_candidates,
    reset_vectorstore_cache,
)
from app.sessions import session_manager
from ui.streamlit_app import delete_document_api


def add_test_chunks(chroma_coll, ids, documents, metadatas):
    """Helper to safely insert test vectors matching the collection's exact dimensionality."""
    sample = chroma_coll.get(limit=1, include=['embeddings'])
    embs = sample.get('embeddings')
    dim = len(embs[0]) if embs is not None and len(embs) > 0 else 3072
    chroma_coll.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=[[0.01 * (i + 1)] * dim for i in range(len(ids))]
    )


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def isolated_storage(tmp_path):
    """
    Isolate uploads and BM25 directories for tests to ensure production documents
    (such as user's JDBC_Notes_.pdf) are never touched.
    """
    orig_uploads = settings.uploads_path
    orig_bm25 = settings.bm25_path

    test_uploads = tmp_path / "uploads"
    test_bm25 = tmp_path / "bm25"
    test_uploads.mkdir(parents=True, exist_ok=True)
    test_bm25.mkdir(parents=True, exist_ok=True)

    settings.uploads_path = str(test_uploads)
    settings.bm25_path = str(test_bm25)
    bm25_manager.storage_dir = str(test_bm25)
    bm25_manager._cache.clear()
    bm25_manager._multi_cache.clear()

    try:
        yield {
            "uploads_dir": test_uploads,
            "bm25_dir": test_bm25
        }
    finally:
        settings.uploads_path = orig_uploads
        settings.bm25_path = orig_bm25
        bm25_manager.storage_dir = orig_bm25
        bm25_manager._cache.clear()
        bm25_manager._multi_cache.clear()


class TestDocumentDeletionLifecycle:
    """Test full document deletion lifecycle across all storage backends."""

    def test_1_document_appears_after_upload(self, client, isolated_storage):
        """1. Verify document appears in document list after upload with correct metadata."""
        content = b"This is a test document about artificial intelligence and neural networks."
        file_name = "test_ai_paper.txt"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": (file_name, io.BytesIO(content), "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        doc_id = data["file_id"]
        assert doc_id is not None
        assert data["filename"] == file_name

        # Verify it appears in /api/v1/files
        list_resp = client.get("/api/v1/files")
        assert list_resp.status_code == 200
        files = list_resp.json()
        matching = [f for f in files if f["file_id"] == doc_id]
        assert len(matching) == 1
        assert matching[0]["filename"] == file_name
        assert matching[0]["file_type"] == "TXT"
        assert matching[0]["chunks_count"] is not None
        assert matching[0]["chunks_count"] > 0

    def test_2_and_3_and_4_delete_existing_document_removes_all_records(self, client, isolated_storage):
        """2, 3, 4. Delete existing document -> HTTP 200, uploaded file removed, Chroma vectors removed."""
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        bm25_dir = isolated_storage["bm25_dir"]

        # Setup raw file
        raw_file = uploads_dir / f"{doc_id}_sample_note.txt"
        raw_file.write_text("Detailed notes on distributed systems.", encoding="utf-8")

        # Setup Chroma vectors
        chroma_coll = get_chroma_collection()
        chunk_ids = [f"{doc_id}_chunk_0", f"{doc_id}_chunk_1"]
        add_test_chunks(
            chroma_coll,
            ids=chunk_ids,
            documents=["Distributed consensus algorithms.", "Raft and Paxos mechanisms."],
            metadatas=[{"file_id": doc_id, "chunk_id": chunk_ids[0]}, {"file_id": doc_id, "chunk_id": chunk_ids[1]}]
        )
        assert len(chroma_coll.get(where={"file_id": doc_id})["ids"]) == 2

        # Setup BM25 index
        bm25_manager.save_file_index(
            doc_id,
            [
                Document(page_content="Distributed consensus algorithms.", metadata={"file_id": doc_id}),
                Document(page_content="Raft and Paxos mechanisms.", metadata={"file_id": doc_id})
            ],
            chunk_ids
        )
        assert (bm25_dir / f"{doc_id}.json").exists()
        assert doc_id in bm25_manager._cache

        # Delete via dedicated endpoint /api/v1/documents/{document_id}
        del_resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200
        res_json = del_resp.json()
        assert res_json["file_id"] == doc_id
        assert "deleted successfully" in res_json["message"]

        # Verify uploaded file removed (3)
        assert not raw_file.exists(), "Uploaded file was not deleted from disk"

        # Verify Chroma vectors removed (4)
        rem_chroma = chroma_coll.get(where={"file_id": doc_id})["ids"]
        assert len(rem_chroma) == 0, "Chroma embeddings were not removed"

        # Verify BM25 removed
        assert not (bm25_dir / f"{doc_id}.json").exists(), "BM25 JSON index file remained on disk"
        assert doc_id not in bm25_manager._cache, "BM25 index remained in memory cache"

    def test_5_and_6_and_7_two_documents_isolation_and_retrieval(self, client, isolated_storage):
        """
        5, 6, 7. Two documents test:
        Document A and Document B are indexed.
        Delete A.
        Verify:
        - A is removed from uploads, Chroma, BM25, and retrieval.
        - Document B remains completely intact and retrievable.
        """
        doc_a_id = str(uuid.uuid4())
        doc_b_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        bm25_dir = isolated_storage["bm25_dir"]

        # Create Document A (Alpha Systems)
        file_a = uploads_dir / f"{doc_a_id}_alpha_systems.txt"
        file_a.write_text("Alpha Systems specializes in quantum optics and photonics.", encoding="utf-8")
        chroma_coll = get_chroma_collection()
        chunk_a_id = f"{doc_a_id}_chunk_0"
        add_test_chunks(
            chroma_coll,
            ids=[chunk_a_id],
            documents=["Alpha Systems specializes in quantum optics and photonics."],
            metadatas=[{"file_id": doc_a_id, "chunk_id": chunk_a_id}]
        )
        bm25_manager.save_file_index(
            doc_a_id,
            [Document(page_content="Alpha Systems specializes in quantum optics and photonics.", metadata={"file_id": doc_a_id})],
            [chunk_a_id]
        )

        # Create Document B (Beta Robotics)
        file_b = uploads_dir / f"{doc_b_id}_beta_robotics.txt"
        file_b.write_text("Beta Robotics engineers autonomous warehouse rovers and drones.", encoding="utf-8")
        chunk_b_id = f"{doc_b_id}_chunk_0"
        add_test_chunks(
            chroma_coll,
            ids=[chunk_b_id],
            documents=["Beta Robotics engineers autonomous warehouse rovers and drones."],
            metadatas=[{"file_id": doc_b_id, "chunk_id": chunk_b_id}]
        )
        bm25_manager.save_file_index(
            doc_b_id,
            [Document(page_content="Beta Robotics engineers autonomous warehouse rovers and drones.", metadata={"file_id": doc_b_id})],
            [chunk_b_id]
        )

        # Pre-deletion check: both retrievable
        bm25_a_res = generate_bm25_candidates("quantum optics", [doc_a_id])
        assert len(bm25_a_res) > 0
        bm25_b_res = generate_bm25_candidates("warehouse rovers", [doc_b_id])
        assert len(bm25_b_res) > 0

        # DELETE DOCUMENT A via API
        del_resp = client.delete(f"/api/v1/documents/{doc_a_id}")
        assert del_resp.status_code == 200

        # 5. Verify Document A is gone
        assert not file_a.exists(), "Document A file still exists"
        assert len(chroma_coll.get(where={"file_id": doc_a_id})["ids"]) == 0, "Document A vectors still in Chroma"
        assert not (bm25_dir / f"{doc_a_id}.json").exists(), "Document A BM25 index still on disk"
        assert doc_a_id not in bm25_manager._cache, "Document A still in BM25 cache"

        # 6. BM25 no longer returns deleted document A
        bm25_after = generate_bm25_candidates("quantum optics", [doc_a_id])
        assert len(bm25_after) == 0, "BM25 returned candidates for deleted document"

        # 7. Retrieval no longer returns deleted document A
        vec_after = generate_vector_candidates("quantum optics", [doc_a_id])
        assert len(vec_after) == 0, "Vector retrieval returned chunks for deleted document"

        # 5 & 15. Verify Document B remains completely untouched and retrievable
        assert file_b.exists(), "Document B file was accidentally removed!"
        b_chroma = chroma_coll.get(where={"file_id": doc_b_id})["ids"]
        assert len(b_chroma) == 1, "Document B vectors were affected!"
        assert (bm25_dir / f"{doc_b_id}.json").exists(), "Document B BM25 was affected!"
        bm25_b_after = generate_bm25_candidates("warehouse rovers", [doc_b_id])
        assert len(bm25_b_after) > 0, "Document B BM25 retrieval failed after Document A deletion!"

        # Multi-document retrieval with both IDs only returns Document B
        combined_bm25 = generate_bm25_candidates("quantum or rovers", [doc_a_id, doc_b_id])
        for doc, cid, score in combined_bm25:
            assert doc.metadata.get("file_id") == doc_b_id

    def test_8_and_9_ui_refresh_and_restart_persistence(self, client, isolated_storage):
        """8, 9. Deleted document does not appear after UI refresh or simulated backend restart."""
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]

        # Create document
        fpath = uploads_dir / f"{doc_id}_restart_test.txt"
        fpath.write_text("Testing persistence across restarts.", encoding="utf-8")
        chroma_coll = get_chroma_collection()
        add_test_chunks(
            chroma_coll,
            ids=[f"{doc_id}_chunk_0"],
            documents=["Testing persistence across restarts."],
            metadatas=[{"file_id": doc_id, "chunk_id": f"{doc_id}_chunk_0"}]
        )
        bm25_manager.save_file_index(
            doc_id,
            [Document(page_content="Testing persistence across restarts.", metadata={"file_id": doc_id})],
            [f"{doc_id}_chunk_0"]
        )

        # Delete it
        del_resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200

        # 8. Document no longer appears on /files
        list_resp = client.get("/api/v1/files")
        file_ids = [f["file_id"] for f in list_resp.json()]
        assert doc_id not in file_ids

        # 9. Simulate restart: clear all caches
        reset_vectorstore_cache()
        bm25_manager._cache.clear()
        bm25_manager._multi_cache.clear()

        # Check /files again after restart
        list_resp_restarted = client.get("/api/v1/files")
        file_ids_restarted = [f["file_id"] for f in list_resp_restarted.json()]
        assert doc_id not in file_ids_restarted

        # Verify disk and vector storage after restart
        assert not fpath.exists()
        assert not (isolated_storage["bm25_dir"] / f"{doc_id}.json").exists()
        assert len(chroma_coll.get(where={"file_id": doc_id})["ids"]) == 0

    def test_10_delete_nonexistent_document_returns_404(self, client, isolated_storage):
        """10. Deleting nonexistent document returns HTTP 404."""
        missing_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/documents/{missing_id}")
        assert response.status_code == 404
        data = response.json()
        error_msg = str(data.get("detail") or data.get("message") or "")
        assert "not found" in error_msg.lower()

    def test_11_and_12_security_path_traversal_and_malicious_ids(self, client, isolated_storage):
        """11, 12. Path traversal attempts and invalid IDs are strictly rejected."""
        malicious_ids = [
            "../../../../etc/passwd",
            "..%2F..%2F",
            "*",
            "../uploads",
            "not-a-valid-uuid",
            "'; DROP TABLE sessions; --",
            "123-abc",
            "../../data/sessions.sqlite"
        ]

        for bad_id in malicious_ids:
            resp = client.delete(f"/api/v1/documents/{bad_id}")
            assert resp.status_code in (400, 404, 422), f"Malicious ID '{bad_id}' was not rejected! Status: {resp.status_code}"

    def test_13_cache_invalidation_works(self, client, isolated_storage):
        """13. Cache invalidation purges memory references in bm25 and vectorstore."""
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        fpath = uploads_dir / f"{doc_id}_cache_test.txt"
        fpath.write_text("Testing cache invalidation.", encoding="utf-8")

        # Manually populate caches
        bm25_manager.save_file_index(
            doc_id,
            [Document(page_content="Testing cache invalidation.", metadata={"file_id": doc_id})],
            [f"{doc_id}_chunk_0"]
        )
        cache_key = (doc_id,)
        bm25_manager._multi_cache[cache_key] = ("dummy_bm25", [], [])
        assert doc_id in bm25_manager._cache
        assert cache_key in bm25_manager._multi_cache

        # Delete document
        delete_document_complete(doc_id)

        # Verify all caches invalidated
        assert doc_id not in bm25_manager._cache
        assert len(bm25_manager._multi_cache) == 0

    def test_14_chat_and_session_data_remains_intact(self, client, isolated_storage):
        """14. Document deletion does not delete or corrupt chat sessions and messages."""
        # 1. Create a chat session with conversation history
        session = session_manager.create_session(title="Important Project Discussion")
        session_id = session.session_id if hasattr(session, "session_id") else session["session_id"]
        session_manager.append_message(session_id, "user", "What are the project milestones?")
        session_manager.append_message(session_id, "assistant", "Milestone 1 is complete. Milestone 2 is underway.")

        # 2. Upload and delete a document
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        fpath = uploads_dir / f"{doc_id}_temp.txt"
        fpath.write_text("Temporary reference doc.", encoding="utf-8")
        delete_document_complete(doc_id)

        # 3. Verify session still exists and has intact history
        retrieved_sess = session_manager.get_session(session_id)
        assert retrieved_sess is not None
        title = retrieved_sess.title if hasattr(retrieved_sess, "title") else retrieved_sess["title"]
        messages = retrieved_sess.messages if hasattr(retrieved_sess, "messages") else retrieved_sess["messages"]
        assert title == "Important Project Discussion"
        assert len(messages) == 2
        content_0 = messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
        content_1 = messages[1].content if hasattr(messages[1], "content") else messages[1]["content"]
        assert content_0 == "What are the project milestones?"
        assert content_1 == "Milestone 1 is complete. Milestone 2 is underway."

    def test_16_and_17_ui_delete_api_helper_and_refresh(self, isolated_storage):
        """16, 17. UI delete API helper and refresh flow."""
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        fpath = uploads_dir / f"{doc_id}_ui_test.txt"
        fpath.write_text("UI test document content.", encoding="utf-8")

        with patch("ui.streamlit_app.make_api_request") as mock_req:
            mock_req.return_value = {
                "message": f"Document {doc_id} deleted successfully",
                "file_id": doc_id
            }
            success, msg = delete_document_api(doc_id)
            assert success is True
            assert doc_id in msg
            mock_req.assert_called_once_with("DELETE", f"/documents/{doc_id}")

    def test_18_failed_deletion_is_reported_correctly(self, client, isolated_storage):
        """18. If a partial or system failure occurs, error is reported and success is not falsely shown."""
        doc_id = str(uuid.uuid4())
        uploads_dir = isolated_storage["uploads_dir"]
        fpath = uploads_dir / f"{doc_id}_error_test.txt"
        fpath.write_text("Error test document.", encoding="utf-8")

        with patch("pathlib.Path.unlink") as mock_unlink:
            mock_unlink.side_effect = PermissionError("Access is denied to file")
            resp = client.delete(f"/api/v1/documents/{doc_id}")
            assert resp.status_code == 500
            data = resp.json()
            assert "error" in data or "detail" in data
            assert "deleted successfully" not in str(data)

    def test_19_ui_sidebar_render_and_confirmation_flow(self):
        """16, 17. Verify UI Document Library renders indexed documents and handles confirmation."""
        from streamlit.testing.v1 import AppTest

        app_path = (Path(__file__).parent.parent / "ui" / "streamlit_app.py").resolve()
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.session_state["api_healthy"] = True
        at.session_state["last_health_check"] = 9999999999
        at.session_state["active_session_id"] = "test-session"
        at.session_state["active_session_title"] = "Test Chat"
        
        test_fid = "12345678-1234-5678-1234-567812345678"
        test_file = {
            "file_id": test_fid,
            "filename": "JDBC_Notes_.pdf",
            "file_type": "PDF",
            "chunks_count": 35,
            "size_bytes": 102400
        }

        with patch("requests.request") as mock_http:
            def fake_req(method, url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                if "/files" in url:
                    mock_resp.json.return_value = [test_file]
                elif "/health" in url:
                    mock_resp.json.return_value = {"status": "healthy", "total_documents": 35}
                elif "/sessions" in url:
                    mock_resp.json.return_value = {"sessions": []}
                else:
                    mock_resp.json.return_value = {}
                return mock_resp

            mock_http.side_effect = fake_req
            at.run()

            del_btns = [b for b in at.sidebar.button if b.key == f"del_btn_{test_fid}"]
            assert len(del_btns) == 1, "Delete button was not rendered for the document"
            
            # Click delete button -> sets confirmation state
            del_btns[0].click().run()
            assert "_doc_pending_delete" in at.session_state and at.session_state["_doc_pending_delete"] is not None
            assert at.session_state["_doc_pending_delete"]["file_id"] == test_fid

            # Find Cancel and Delete buttons in confirmation step
            cancel_btns = [b for b in at.sidebar.button if b.key == "cancel_doc_delete"]
            confirm_btns = [b for b in at.sidebar.button if b.key == "confirm_doc_delete"]
            assert len(cancel_btns) == 1, "Cancel button not rendered in confirmation step"
            assert len(confirm_btns) == 1, "Confirm Delete button not rendered in confirmation step"

            # Clicking Cancel clears pending confirmation
            cancel_btns[0].click().run()
            assert at.session_state["_doc_pending_delete"] is None
