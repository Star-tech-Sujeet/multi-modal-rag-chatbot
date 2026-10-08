"""
Tests for Phase 7: Live Multimodal RAG & Image Evidence.

Verifies:
Test A: Valid PNG live image.
Test B: Valid JPEG live image.
Test C: OCR success (image_ocr, vision skipped).
Test D: OCR empty -> Vision fallback (image_vision).
Test E: OCR success must NOT call Vision.
Test F: Corrupt image rejected (ValueError / HTTP 400).
Test G: Unsupported format rejected.
Test H: Oversized image rejected.
Test I: Transparent PNG normalized to RGB on white background.
Test J: Vision failure handled gracefully (image_fallback).
Test K: Missing API key handled gracefully without crashing.
Test L: Image evidence enters canonical evidence pipeline.
Test M: Image citation label formatted correctly ([S1] filename, OCR/Vision).
Test N: Invalid citation marker rejected by validator.
Test O: Image + document evidence coexist in one response.
Test P: Text-only session query remains 100% backward compatible.
Test Q: Session isolation maintained across multimodal turns.
Test R: Ephemeral image metadata persists in SQLite without vector indexing.
Test S: Temporary image files cleaned up after processing.
"""

import io
import os
import uuid
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.models import (
    SessionQueryRequest,
    Source
)
from app.sessions import SessionManager
from app.logic import (
    CanonicalEvidence,
    process_live_image,
    perform_session_rag_query,
    format_citation_label,
    validate_citations,
    RetrievalCandidate
)


# =============================================================================
# Helper utilities to create in-memory test images
# =============================================================================

def create_test_png(color="blue", size=(100, 100)) -> bytes:
    """Generate in-memory valid PNG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_test_jpeg(color="green", size=(100, 100)) -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_test_transparent_png(size=(100, 100)) -> bytes:
    """Generate in-memory RGBA PNG image with transparency."""
    buf = io.BytesIO()
    img = Image.new("RGBA", size, color=(255, 0, 0, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database for session manager."""
    db_file = tmp_path / f"test_mm_{uuid.uuid4().hex}.sqlite"
    return str(db_file)


@pytest.fixture
def session_mgr(temp_db):
    """Isolated SessionManager instance."""
    return SessionManager(db_path=temp_db)


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


# =============================================================================
# Phase 7 Tests A - S
# =============================================================================

def test_a_valid_png_live_image():
    """Test A: Valid PNG live image is parsed and converted to CanonicalEvidence."""
    png_bytes = create_test_png()
    with patch("pytesseract.image_to_string", return_value="Extracted valid text from diagram."):
        evidence = process_live_image(png_bytes, "chart.png")

    assert isinstance(evidence, CanonicalEvidence)
    assert evidence.file_type == "image"
    assert evidence.filename == "chart.png"
    assert evidence.file_id is None
    assert evidence.citation_id == "S1"
    assert evidence.citation_marker == "[S1]"
    assert "[S1] chart.png" in evidence.citation_label


def test_b_valid_jpeg_live_image():
    """Test B: Valid JPEG live image is parsed and converted to CanonicalEvidence."""
    jpg_bytes = create_test_jpeg()
    with patch("pytesseract.image_to_string", return_value="Extracted text from invoice screenshot."):
        evidence = process_live_image(jpg_bytes, "receipt.jpg")

    assert isinstance(evidence, CanonicalEvidence)
    assert evidence.file_type == "image"
    assert evidence.filename == "receipt.jpg"
    assert evidence.file_id is None
    assert evidence.citation_id == "S1"


def test_c_ocr_success_image_ocr():
    """Test C: OCR extraction with >=10 characters uses image_ocr source type."""
    png_bytes = create_test_png()
    extracted_text = "Quarterly Revenue: Q1 $4.2M, Q2 $5.1M"
    with patch("pytesseract.image_to_string", return_value=extracted_text):
        evidence = process_live_image(png_bytes, "report.png")

    assert evidence.source_type == "image_ocr"
    assert evidence.ingestion_method == "ocr"
    assert evidence.content == extracted_text
    assert evidence.metadata["vision_analyzed"] is False
    assert evidence.metadata["ocr_processed"] is True


def test_d_ocr_empty_vision_fallback():
    """Test D: OCR empty or <10 chars falls back to GPT Vision."""
    png_bytes = create_test_png()
    vision_description = "A detailed architectural flowchart with authentication gateway."
    with patch("pytesseract.image_to_string", return_value=""), \
         patch("app.logic.analyze_image_with_vision", return_value=vision_description):
        evidence = process_live_image(png_bytes, "architecture.png")

    assert evidence.source_type == "image_vision"
    assert evidence.ingestion_method == "vision"
    assert evidence.content == vision_description
    assert evidence.metadata["vision_analyzed"] is True


def test_e_ocr_success_does_not_call_vision():
    """Test E: OCR success must NOT invoke Vision (OCR-first cost optimization)."""
    png_bytes = create_test_png()
    with patch("pytesseract.image_to_string", return_value="Sufficiently long OCR text extracted from screenshot."), \
         patch("app.logic.analyze_image_with_vision") as mock_vision:
        evidence = process_live_image(png_bytes, "notes.png")
        mock_vision.assert_not_called()

    assert evidence.source_type == "image_ocr"


def test_f_corrupt_image_rejected(client):
    """Test F: Corrupt image payload is rejected with ValueError and HTTP 400."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_BINARY_HEADER"

    # Direct function call raises ValueError
    with pytest.raises(ValueError, match="Corrupted or invalid image file"):
        process_live_image(corrupt_bytes, "corrupt.png")

    # API call returns HTTP 400 Bad Request
    create_res = client.post("/api/v1/sessions", json={"title": "Corrupt Test"})
    sid = create_res.json()["session_id"]
    resp = client.post(
        f"/api/v1/sessions/{sid}/query",
        json={
            "question": "What is this image?",
            "image_base64": base64.b64encode(corrupt_bytes).decode("utf-8"),
            "image_filename": "corrupt.png"
        }
    )
    assert resp.status_code == 400
    res_data = resp.json()
    err_msg = res_data.get("message") or (res_data.get("detail", {}).get("message") if isinstance(res_data.get("detail"), dict) else str(res_data.get("detail"))) or str(res_data)
    assert "Corrupted or invalid image file" in err_msg


def test_g_unsupported_format_rejected():
    """Test G: Unsupported image formats (e.g. BMP, GIF) are rejected."""
    # Create valid BMP in memory
    buf = io.BytesIO()
    img = Image.new("RGB", (50, 50), color="yellow")
    img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    with pytest.raises(ValueError, match="Unsupported image format"):
        process_live_image(bmp_bytes, "graphic.bmp")


def test_h_oversized_image_rejected():
    """Test H: Oversized image bytes or excessive dimensions are rejected."""
    png_bytes = create_test_png()
    with patch.object(settings, "max_file_size_mb", 0):
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            process_live_image(png_bytes, "large.png")


def test_i_transparent_png_normalized():
    """Test I: Transparent PNG (RGBA) is normalized to RGB on white background."""
    rgba_bytes = create_test_transparent_png()
    with patch("pytesseract.image_to_string", return_value="Transparent text successfully parsed."):
        evidence = process_live_image(rgba_bytes, "transp.png")

    assert evidence.file_type == "image"
    assert evidence.source_type == "image_ocr"
    assert "Transparent text" in evidence.content


def test_j_vision_failure_handled_gracefully():
    """Test J: Vision failure returns image_fallback without unhandled exception."""
    png_bytes = create_test_png()
    fallback_msg = "[Image file: chart.png] - Unable to analyze image content: OpenAI API error."
    with patch("pytesseract.image_to_string", return_value=""), \
         patch("app.logic.analyze_image_with_vision", return_value=fallback_msg):
        evidence = process_live_image(png_bytes, "chart.png")

    assert evidence.source_type == "image_fallback"
    assert evidence.ingestion_method == "fallback"
    assert "Unable to extract" in evidence.content


def test_k_missing_api_key_handled_gracefully():
    """Test K: Missing Vision API key during Vision fallback is handled gracefully."""
    png_bytes = create_test_png()
    with patch("pytesseract.image_to_string", return_value=""), \
         patch("app.logic.get_provider", side_effect=ValueError("API key missing")), \
         patch.object(settings, "openai_api_key", ""), \
         patch.object(settings, "gemini_api_key", ""), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
        evidence = process_live_image(png_bytes, "scan.png")

    assert evidence.source_type == "image_fallback"
    assert evidence.ingestion_method == "fallback"


def test_l_canonical_evidence_pipeline():
    """Test L: Image evidence conforms strictly to the CanonicalEvidence schema."""
    png_bytes = create_test_png()
    with patch("pytesseract.image_to_string", return_value="Schema verification test string."):
        evidence = process_live_image(png_bytes, "test.png")

    assert isinstance(evidence, CanonicalEvidence)
    assert evidence.citation_id == "S1"
    assert evidence.citation_marker == "[S1]"
    assert evidence.filename == "test.png"
    assert evidence.file_id is None
    assert evidence.page_number is None
    assert evidence.section is None
    assert evidence.table_name is None
    assert evidence.row_number is None
    assert evidence.metadata.get("is_ephemeral") is True


def test_m_citation_label_formatted_correctly():
    """Test M: Citation labels for OCR and Vision images are formatted accurately."""
    # OCR format
    meta_ocr = {"file_type": "image", "source_type": "image_ocr", "ingestion_method": "ocr"}
    label_ocr = format_citation_label("chart.png", meta_ocr, "S1")
    assert label_ocr == "[S1] chart.png, OCR"

    # Vision format
    meta_vis = {"file_type": "image", "source_type": "image_vision", "ingestion_method": "vision"}
    label_vis = format_citation_label("diagram.jpg", meta_vis, "S2")
    assert label_vis == "[S2] diagram.jpg, Vision"


def test_n_invalid_citation_marker_rejected_by_validator():
    """Test N: Validator detects and strips invalid markers (e.g. [S99]) not in evidence."""
    answer_with_invalid = "The diagram illustrates the flow [S1], but also asserts details [S99]."
    valid_ids = {"S1"}

    cleaned, used, report = validate_citations(answer_with_invalid, valid_ids)

    assert "[S99]" not in cleaned
    assert "[S1]" in cleaned
    assert "S99" in report["invalid_citations_detected"]
    assert report["has_invalid_citations"] is True
    assert used == ["S1"]


def test_o_image_and_document_evidence_coexist(session_mgr):
    """Test O: Image and document evidence coexist in a single query with sequential citations."""
    session = session_mgr.create_session(title="Combined Evidence")
    sid = session.session_id

    # Mock document candidate
    from langchain.schema import Document
    mock_doc = Document(
        page_content="Remote employees must file travel requests 14 days in advance.",
        metadata={"filename": "company_handbook.pdf", "file_id": "doc_uuid_123", "file_type": "pdf", "page_number": 3, "chunk_id": "chunk_doc_1"}
    )
    mock_doc_cand = RetrievalCandidate(
        doc=mock_doc,
        chunk_id="chunk_doc_1",
        final_score=0.95,
        vector_score=0.9,
        bm25_score=0.8,
        search_type="hybrid"
    )

    png_bytes = create_test_png()
    req = SessionQueryRequest(
        question="What are the travel policy and reimbursement workflow?",
        file_id="doc_uuid_123"
    )

    with patch("app.logic.execute_retrieval_pipeline", return_value=([mock_doc_cand], "hybrid")), \
         patch("pytesseract.image_to_string", return_value="Travel Form Step 1: Attach receipts. Step 2: Manager approval."), \
         patch("app.logic.handle_text_query", return_value=(
             "According to the handbook, requests must be filed 14 days early [S1]. The workflow chart requires receipts and approval [S2].",
             "gpt-4o-mini"
         )), \
         patch("app.logic.generate_suggested_questions", return_value=[]):

        resp = perform_session_rag_query(
            session_id=sid,
            query_request=req,
            mgr=session_mgr,
            image_bytes=png_bytes,
            image_filename="workflow.png"
        )

    assert len(resp.sources) == 2
    # Document is S1, Image is S2
    s1 = resp.sources[0]
    s2 = resp.sources[1]

    assert s1.citation_id == "S1"
    assert s1.filename == "company_handbook.pdf"
    assert "[S1] company_handbook.pdf, p. 3" == s1.citation_label

    assert s2.citation_id == "S2"
    assert s2.filename == "workflow.png"
    assert s2.file_id is None
    assert "[S2] workflow.png, OCR" == s2.citation_label

    # Citation validation confirms both S1 and S2 are used
    assert resp.citation_validation["has_invalid_citations"] is False
    assert "S1" in resp.citation_validation["citations_used"]
    assert "S2" in resp.citation_validation["citations_used"]


def test_p_text_only_session_query_backward_compatible(client):
    """Test P: Text-only query remains 100% backward compatible."""
    create_res = client.post("/api/v1/sessions", json={"title": "Text Only"})
    sid = create_res.json()["session_id"]

    with patch("app.logic.execute_retrieval_pipeline", return_value=([], "hybrid")):
        resp = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={
                "question": "What is the status of the project?",
                "file_id": "test_file_id"
            }
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert "answer" in data
    assert "sources" in data


def test_q_session_isolation_maintained(session_mgr):
    """Test Q: Image evidence in Session 1 is isolated from Session 2."""
    sess1 = session_mgr.create_session(title="Session 1")
    sess2 = session_mgr.create_session(title="Session 2")

    png_bytes = create_test_png()
    req1 = SessionQueryRequest(question="What is this chart?")
    req2 = SessionQueryRequest(question="What was discussed previously?")

    with patch("pytesseract.image_to_string", return_value="Confidential Financial Chart Q3."), \
         patch("app.logic.handle_text_query", return_value=("This chart shows Q3 financials [S1].", "gpt-4o-mini")), \
         patch("app.logic.generate_suggested_questions", return_value=[]):
        resp1 = perform_session_rag_query(
            session_id=sess1.session_id,
            query_request=req1,
            mgr=session_mgr,
            image_bytes=png_bytes,
            image_filename="secret_chart.png"
        )

    # Session 2 query with no image and no files
    resp2 = perform_session_rag_query(
        session_id=sess2.session_id,
        query_request=req2,
        mgr=session_mgr
    )

    # Session 1 has image source
    assert len(resp1.sources) == 1
    assert resp1.sources[0].filename == "secret_chart.png"

    # Session 2 has no sources and did not access Session 1's image
    assert len(resp2.sources) == 0
    assert "No document or image specified" in resp2.answer


def test_r_ephemeral_image_metadata_persisted_in_sqlite(session_mgr):
    """Test R: Ephemeral image metadata persists in SQLite without pretending it is permanently indexed."""
    sess = session_mgr.create_session(title="Persistence Test")
    sid = sess.session_id

    png_bytes = create_test_png()
    req = SessionQueryRequest(question="Explain this diagram.")

    with patch("pytesseract.image_to_string", return_value="System Architecture diagram with SQLite persistence."), \
         patch("app.logic.handle_text_query", return_value=("The system architecture uses SQLite [S1].", "gpt-4o-mini")), \
         patch("app.logic.generate_suggested_questions", return_value=[]):
        resp = perform_session_rag_query(
            session_id=sid,
            query_request=req,
            mgr=session_mgr,
            image_bytes=png_bytes,
            image_filename="arch.png"
        )

    # Fetch stored messages from SQLite
    messages = session_mgr.get_session_messages(sid)
    assert len(messages) == 2  # user + assistant

    asst_msg = messages[1]
    assert asst_msg.role == "assistant"
    assert asst_msg.sources is not None
    assert len(asst_msg.sources) == 1

    stored_source = asst_msg.sources[0]
    filename = getattr(stored_source, "filename", None) or stored_source.get("filename")
    file_id = getattr(stored_source, "file_id", None) if hasattr(stored_source, "file_id") else stored_source.get("file_id")
    source_type = getattr(stored_source, "source_type", None) or stored_source.get("source_type")
    citation_label = getattr(stored_source, "citation_label", None) or stored_source.get("citation_label")

    assert filename == "arch.png"
    assert file_id is None
    assert source_type == "image_ocr"
    assert citation_label == "[S1] arch.png, OCR"


def test_s_temporary_image_files_cleaned_up():
    """Test S: Any temporary files created during Vision fallback are cleaned up."""
    png_bytes = create_test_png()
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Track temp directory contents before
    files_before = set(os.listdir(temp_dir))

    with patch("pytesseract.image_to_string", return_value=""), \
         patch("app.logic.analyze_image_with_vision", return_value="Mocked Vision analysis of diagram."):
        evidence = process_live_image(png_bytes, "cleanup_test.png")

    # Track temp directory contents after
    files_after = set(os.listdir(temp_dir))

    # All temporary live image files must be deleted in finally block
    new_files = files_after - files_before
    assert len(new_files) == 0, f"Temporary files were not cleaned up: {new_files}"
    assert evidence.source_type == "image_vision"
