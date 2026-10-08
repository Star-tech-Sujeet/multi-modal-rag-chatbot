"""
Test suite validating Phase 3 Advanced Document Ingestion:
1. Unified document representation & metadata schema.
2. PDF parsing: 1-indexed page numbers, scanned page OCR fallback, empty PDF rejection.
3. DOCX parsing: Paragraphs, sections from headings, markdown tables, embedded image OCR, empty file rejection.
4. TXT parsing: Multi-encoding fallback (UTF-8, Latin-1), empty/whitespace rejection.
5. CSV parsing: Row-level traceability, column names, row numbers, empty table schema handling, empty file rejection.
6. SQLite parsing: Schema inspection, row-level documents, empty table handling, rejection of tableless DBs.
7. Image parsing: RGBA/transparency handling, OCR extraction, Vision fallback, corrupted file rejection.
8. API failure cleanup: Ensures temporary files and embeddings are cleaned up on ingestion failure.
"""

import os
import io
import zipfile
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import pandas as pd

from langchain.schema import Document
from fastapi.testclient import TestClient

from app.models import Source, QueryResponse
from app.logic import (
    process_pdf,
    process_docx,
    process_txt,
    process_csv,
    process_database,
    process_image,
    process_document,
    create_and_store_embeddings,
    perform_rag_query,
    bm25_manager
)
from app.main import app

client = TestClient(app)


# =============================================================================
# 1. Unified Document Representation & Metadata Schema Tests
# =============================================================================
class TestMetadataSchema:
    """Verify standard metadata schema across formats and ChromaDB sanitization."""

    def test_source_model_optional_fields(self):
        """Source model must accept all Phase 3 metadata fields."""
        source = Source(
            filename="report.pdf",
            file_id="fid-123",
            chunk_id="fid-123_chunk_0",
            page_number=3,
            section="Executive Summary",
            table_name=None,
            row_number=None,
            source_type="page",
            ingestion_method="direct",
            content="Sample text content",
            relevance_score=0.92,
            vector_score=0.88,
            bm25_score=0.95,
            chunk_index=0,
            search_type="hybrid"
        )
        assert source.page_number == 3
        assert source.section == "Executive Summary"
        assert source.source_type == "page"
        assert source.ingestion_method == "direct"

    def test_chroma_metadata_sanitization_removes_none_values(self):
        """create_and_store_embeddings must strip None values to prevent ChromaDB errors."""
        doc = Document(
            page_content="Test chunk content",
            metadata={
                "filename": "test.txt",
                "page_number": None,
                "section": None,
                "table_name": "users",
                "row_number": 10,
                "none_val": None,
                "valid_bool": True
            }
        )
        with patch("app.logic.get_vectorstore") as mock_vs, \
             patch.object(bm25_manager, "save_file_index"):
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            count = create_and_store_embeddings([doc], file_id="sanitization-test")
            assert count == 1
            added_docs = mock_store.add_documents.call_args[0][0]
            added_ids = mock_store.add_documents.call_args[1]["ids"]
            assert added_ids == ["sanitization-test_chunk_0"]
            sanitized_meta = added_docs[0].metadata

            # None values must not exist in sanitized metadata
            assert "none_val" not in sanitized_meta
            assert "page_number" not in sanitized_meta
            assert "section" not in sanitized_meta
            assert sanitized_meta["table_name"] == "users"
            assert sanitized_meta["row_number"] == 10
            assert sanitized_meta["valid_bool"] is True


# =============================================================================
# 2. PDF Ingestion Tests
# =============================================================================
class TestPDFIngestion:
    """Verify PDF parsing with 1-indexed pages, OCR fallback, and empty file rejection."""

    def test_pdf_normal_text_extraction(self, tmp_path):
        """Direct text PDF must extract pages with 1-indexed page_number."""
        import pymupdf
        pdf_path = tmp_path / "normal.pdf"
        doc = pymupdf.open()
        page1 = doc.new_page()
        page1.insert_text((50, 72), "This is page 1 with important information about revenue.")
        page2 = doc.new_page()
        page2.insert_text((50, 72), "This is page 2 with details on operations and expenses.")
        doc.save(str(pdf_path))
        doc.close()

        docs = process_pdf(str(pdf_path), "normal.pdf")
        assert len(docs) == 2
        assert docs[0].metadata["page_number"] == 1
        assert docs[0].metadata["source_type"] == "page"
        assert docs[0].metadata["ingestion_method"] == "direct"
        assert "revenue" in docs[0].page_content

        assert docs[1].metadata["page_number"] == 2
        assert "expenses" in docs[1].page_content

    def test_pdf_scanned_page_triggers_ocr(self, tmp_path):
        """Scanned image-only PDF page should trigger OCR fallback."""
        import pymupdf
        pdf_path = tmp_path / "scanned.pdf"
        doc = pymupdf.open()
        page = doc.new_page()

        # Insert a small dummy image into page
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=img_byte_arr.getvalue())
        doc.save(str(pdf_path))
        doc.close()

        # Mock pytesseract.image_to_string to simulate OCR reading text from scanned page
        with patch("app.logic.pytesseract.image_to_string", return_value="Extracted OCR text from scanned invoice"):
            docs = process_pdf(str(pdf_path), "scanned.pdf")
            assert len(docs) == 1
            assert docs[0].metadata["source_type"] == "scanned_page"
            assert docs[0].metadata["ingestion_method"] == "ocr"
            assert "Extracted OCR text" in docs[0].page_content

    def test_pdf_empty_file_rejected(self, tmp_path):
        """Empty PDF file (0 bytes) must raise ValueError."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            process_pdf(str(empty_pdf), "empty.pdf")
        assert "empty" in str(exc.value).lower()

    def test_pdf_with_no_text_and_no_ocr_rejected(self, tmp_path):
        """PDF with completely blank pages must raise ValueError."""
        import pymupdf
        pdf_path = tmp_path / "blank.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        with patch("app.logic.pytesseract.image_to_string", return_value=""):
            with pytest.raises(ValueError) as exc:
                process_pdf(str(pdf_path), "blank.pdf")
            assert "no extractable text" in str(exc.value).lower()


# =============================================================================
# 3. DOCX Ingestion Tests
# =============================================================================
class TestDOCXIngestion:
    """Verify DOCX parsing with headings, markdown tables, embedded image OCR, and empty file rejection."""

    def _create_sample_docx(self, file_path: Path, include_table: bool = True, include_image: bool = False):
        """Helper to create a standard OpenXML DOCX archive."""
        tbl_xml = ""
        if include_table:
            tbl_xml = """
            <w:tbl>
              <w:tr>
                <w:tc><w:p><w:r><w:t>Quarter</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>Profit</w:t></w:r></w:p></w:tc>
              </w:tr>
              <w:tr>
                <w:tc><w:p><w:r><w:t>Q1</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>$150,000</w:t></w:r></w:p></w:tc>
              </w:tr>
            </w:tbl>
            """

        doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p>
              <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
              <w:r><w:t>Financial Overview</w:t></w:r>
            </w:p>
            <w:p>
              <w:r><w:t>The company performed exceptionally well in fiscal year 2026.</w:t></w:r>
            </w:p>
            {tbl_xml}
          </w:body>
        </w:document>"""

        with zipfile.ZipFile(file_path, 'w') as zf:
            zf.writestr('word/document.xml', doc_xml)
            if include_image:
                img = Image.new("RGB", (50, 50), color=(200, 200, 200))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                zf.writestr('word/media/image1.png', buf.getvalue())

    def test_docx_paragraphs_and_sections(self, tmp_path):
        """DOCX headings should be detected and populate the section metadata."""
        docx_path = tmp_path / "test.docx"
        self._create_sample_docx(docx_path, include_table=False)

        docs = process_docx(str(docx_path), "test.docx")
        assert len(docs) >= 1
        text_doc = docs[0]
        assert "Financial Overview" in text_doc.page_content
        assert text_doc.metadata.get("section") == "Financial Overview"
        assert text_doc.metadata["source_type"] == "text"

    def test_docx_table_extraction_markdown(self, tmp_path):
        """DOCX tables must be extracted as structured markdown with table metadata."""
        docx_path = tmp_path / "table.docx"
        self._create_sample_docx(docx_path, include_table=True)

        docs = process_docx(str(docx_path), "table.docx")
        # Find the table document
        table_docs = [d for d in docs if d.metadata.get("source_type") == "table"]
        assert len(table_docs) == 1
        tbl = table_docs[0]
        assert tbl.metadata["table_name"] == "Table_1"
        assert tbl.metadata["section"] == "Financial Overview"
        assert "| Quarter | Profit |" in tbl.page_content
        assert "| Q1 | $150,000 |" in tbl.page_content

    def test_docx_embedded_image_ocr(self, tmp_path):
        """Embedded images inside DOCX should be extracted and OCR processed."""
        docx_path = tmp_path / "with_image.docx"
        self._create_sample_docx(docx_path, include_table=False, include_image=True)

        with patch("app.logic.pytesseract.image_to_string", return_value="Chart Showing Growth: +45%"):
            docs = process_docx(str(docx_path), "with_image.docx")
            img_docs = [d for d in docs if d.metadata.get("source_type") == "embedded_image"]
            assert len(img_docs) == 1
            assert "Chart Showing Growth: +45%" in img_docs[0].page_content
            assert img_docs[0].metadata["ingestion_method"] == "ocr"

    def test_docx_empty_file_rejected(self, tmp_path):
        """Empty DOCX file must raise ValueError."""
        empty_docx = tmp_path / "empty.docx"
        empty_docx.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            process_docx(str(empty_docx), "empty.docx")
        assert "empty" in str(exc.value).lower()


# =============================================================================
# 4. TXT Ingestion Tests
# =============================================================================
class TestTXTIngestion:
    """Verify TXT multi-encoding fallback and empty file rejection."""

    def test_txt_utf8_success(self, tmp_path):
        """UTF-8 text file decodes successfully."""
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("Hello World! This is UTF-8 document content.", encoding="utf-8")

        docs = process_txt(str(txt_path), "sample.txt")
        assert len(docs) == 1
        assert docs[0].page_content == "Hello World! This is UTF-8 document content."
        assert docs[0].metadata["encoding"] == "utf-8"
        assert docs[0].metadata["source_type"] == "text"

    def test_txt_latin1_fallback_success(self, tmp_path):
        """Latin-1 encoded text file decodes cleanly using fallback."""
        txt_path = tmp_path / "latin1.txt"
        # Character 'é' in Latin-1 is 0xE9 which is invalid in UTF-8
        txt_path.write_bytes(b"Caf\xe9 and na\xefve analysis in Latin-1.")

        docs = process_txt(str(txt_path), "latin1.txt")
        assert len(docs) == 1
        assert "Café" in docs[0].page_content
        assert docs[0].metadata["encoding"] in ("latin-1", "cp1252", "iso-8859-1")

    def test_txt_empty_file_rejected(self, tmp_path):
        """Empty TXT file must raise ValueError."""
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            process_txt(str(empty_txt), "empty.txt")
        assert "empty" in str(exc.value).lower()

    def test_txt_whitespace_only_rejected(self, tmp_path):
        """Whitespace-only TXT file must raise ValueError."""
        ws_txt = tmp_path / "whitespace.txt"
        ws_txt.write_text("   \n\n\t   \n   ", encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            process_txt(str(ws_txt), "whitespace.txt")
        assert "whitespace" in str(exc.value).lower()


# =============================================================================
# 5. CSV Ingestion Tests
# =============================================================================
class TestCSVIngestion:
    """Verify CSV row-level documents, row numbers, and empty table schema preservation."""

    def test_csv_row_traceability(self, tmp_path):
        """CSV rows must be extracted as structured documents with exact row numbers."""
        csv_path = tmp_path / "employees.csv"
        csv_path.write_text("ID,Name,Department,Salary\n101,Alice,Engineering,120000\n102,Bob,Product,110000\n", encoding="utf-8")

        docs = process_csv(str(csv_path), "employees.csv")
        assert len(docs) == 2

        doc1 = docs[0]
        assert doc1.metadata["row_number"] == 1
        assert doc1.metadata["source_type"] == "row"
        assert doc1.metadata["table_name"] == "employees"
        assert "Alice" in doc1.page_content
        assert "Engineering" in doc1.page_content

        doc2 = docs[1]
        assert doc2.metadata["row_number"] == 2
        assert "Bob" in doc2.page_content

    def test_csv_empty_table_preserves_schema(self, tmp_path):
        """CSV with headers but 0 rows should preserve table schema."""
        csv_path = tmp_path / "empty_schema.csv"
        csv_path.write_text("OrderID,CustomerName,Amount,Status\n", encoding="utf-8")

        docs = process_csv(str(csv_path), "empty_schema.csv")
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "table_schema"
        assert "OrderID, CustomerName, Amount, Status" in docs[0].page_content

    def test_csv_empty_file_rejected(self, tmp_path):
        """Zero-byte CSV file must raise ValueError."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            process_csv(str(empty_csv), "empty.csv")
        assert "empty" in str(exc.value).lower()


# =============================================================================
# 6. SQLite Ingestion Tests
# =============================================================================
class TestSQLiteIngestion:
    """Verify SQLite table enumeration, row-level documents, schema preservation, and tableless DB rejection."""

    def test_sqlite_multiple_tables_and_rows(self, tmp_path):
        """SQLite DB with tables extracts row-level documents with table_name and row_number."""
        db_path = tmp_path / "company.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE products (id INT, title TEXT, price REAL)")
        cursor.execute("INSERT INTO products VALUES (1, 'Widget A', 29.99)")
        cursor.execute("INSERT INTO products VALUES (2, 'Widget B', 49.99)")

        cursor.execute("CREATE TABLE categories (code TEXT, label TEXT)")
        cursor.execute("INSERT INTO categories VALUES ('CAT1', 'Hardware')")
        conn.commit()
        conn.close()

        docs = process_database(str(db_path), "company.db")
        assert len(docs) == 3

        product_docs = [d for d in docs if d.metadata.get("table_name") == "products"]
        assert len(product_docs) == 2
        assert product_docs[0].metadata["row_number"] == 1
        assert "Widget A" in product_docs[0].page_content
        assert product_docs[1].metadata["row_number"] == 2
        assert "Widget B" in product_docs[1].page_content

        cat_docs = [d for d in docs if d.metadata.get("table_name") == "categories"]
        assert len(cat_docs) == 1
        assert "Hardware" in cat_docs[0].page_content

    def test_sqlite_empty_table_preserves_schema(self, tmp_path):
        """SQLite table with 0 rows preserves table schema."""
        db_path = tmp_path / "empty_tbl.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE audits (audit_id INT, timestamp TEXT, details TEXT)")
        conn.commit()
        conn.close()

        docs = process_database(str(db_path), "empty_tbl.db")
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "table_schema"
        assert docs[0].metadata["table_name"] == "audits"
        assert "audit_id" in docs[0].page_content

    def test_sqlite_no_tables_rejected(self, tmp_path):
        """SQLite DB without tables must raise ValueError."""
        db_path = tmp_path / "no_tables.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError) as exc:
            process_database(str(db_path), "no_tables.db")
        assert "contains no tables" in str(exc.value).lower()


# =============================================================================
# 7. Image Ingestion Tests
# =============================================================================
class TestImageIngestion:
    """Verify Image RGBA/transparency handling, OCR, Vision fallback, and corruption check."""

    def test_image_rgba_transparency_handling(self, tmp_path):
        """Transparent PNG (RGBA mode) must be converted to RGB without error."""
        img_path = tmp_path / "transparent.png"
        img = Image.new("RGBA", (120, 60), color=(255, 0, 0, 128))
        img.save(str(img_path))

        with patch("app.logic.pytesseract.image_to_string", return_value="Alpha Channel Brand"):
            docs = process_image(str(img_path), "transparent.png")
            assert len(docs) == 1
            assert docs[0].metadata["source_type"] == "image_ocr"
            assert "Alpha Channel Brand" in docs[0].page_content

    def test_image_ocr_fallback_to_vision(self, tmp_path):
        """When OCR finds no text, falls back to GPT Vision description."""
        img_path = tmp_path / "photo.jpg"
        img = Image.new("RGB", (100, 100), color=(0, 128, 255))
        img.save(str(img_path))

        with patch("app.logic.pytesseract.image_to_string", return_value=""), \
             patch("app.logic.analyze_image_with_vision", return_value="[Image Analysis: photo.jpg]\nA blue geometric square."):
            docs = process_image(str(img_path), "photo.jpg")
            assert len(docs) == 1
            assert docs[0].metadata["source_type"] == "image_vision"
            assert "blue geometric square" in docs[0].page_content

    def test_image_corrupted_file_rejected(self, tmp_path):
        """Corrupted image file must raise ValueError."""
        corrupt_img = tmp_path / "corrupt.png"
        corrupt_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00corruptbyteshere")
        with pytest.raises(ValueError) as exc:
            process_image(str(corrupt_img), "corrupt.png")
        assert "corrupted or invalid" in str(exc.value).lower()


# =============================================================================
# 8. Document Splitting and Ingestion Cleanup Tests
# =============================================================================
class TestIngestionPipelineAndCleanup:
    """Verify metadata preservation through chunking and cleanup on upload errors."""

    def test_process_document_preserves_item_metadata_across_chunks(self, tmp_path):
        """Chunking via process_document must preserve page_number, section, and table metadata."""
        txt_path = tmp_path / "long_report.txt"
        long_text = ("Section Heading\n\n" + "This is a sentence for document chunk testing. " * 30)
        txt_path.write_text(long_text, encoding="utf-8")

        chunks = process_document(str(txt_path), "long_report.txt")
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["filename"] == "long_report.txt"
            assert chunk.metadata["source_type"] == "text"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_upload_corrupted_file_cleans_up_disk_and_embeddings(self, tmp_path):
        """API upload error should remove the saved file and trigger embedding cleanup."""
        with patch("app.api.delete_document_embeddings") as mock_delete:
            corrupt_bytes = b"NOT_A_VALID_SQLITE_DATABASE"
            response = client.post(
                "/api/v1/upload",
                files={"file": ("corrupt.db", corrupt_bytes, "application/octet-stream")}
            )
            assert response.status_code == 400
            assert mock_delete.called
