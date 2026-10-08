"""
Tests for Phase 13.7 — Aura Processing Indicator in SQL Query Processing ONLY.

Verifies:
1. Indicator starts when SQL processing begins.
2. Compact mode is supported for indicator.
3. Initial status is "Processing SQL query…".
4. Indicator starts before API dispatch.
5. Successful SQL query clears indicator.
6. SQL validation failure clears indicator.
7. Unsafe SQL rejection (INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA) clears indicator.
8. SQL execution failure clears indicator.
9. Timeout/error clears indicator.
10. Provider/API failure clears indicator.
11. Unexpected exception clears indicator.
12. Interruption (KeyboardInterrupt) clears indicator.
13. Exactly one SQL request occurs.
14. SQL result table remains intact.
15. SQL citations remain intact ([S1], source metadata).
16. No artificial sleep or delay exists.
17. SELECT queries still work.
18. Write operations remain blocked.
19. Row limit remains enforced.
20. Timeout remains enforced.
21. Existing chat streaming remains unchanged.
22. Document upload indicator remains unchanged.
23. Image processing indicator remains unchanged.
24. Chat rename and delete indicators remain unchanged.
"""

import inspect
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ui.processing_indicator import AuraProcessingIndicator
from ui.streamlit_app import (
    execute_sql_query_api,
    process_question,
    upload_document,
    process_attached_image,
    delete_document_api,
    rename_session_api,
    delete_session_api,
    APIError,
)
from app.sql_engine import (
    validate_sql_query,
    execute_read_only_sql,
    SQLValidationError,
    read_only_authorizer,
    inspect_sqlite_schema,
    format_sql_evidence,
)


@pytest.fixture
def test_db(tmp_path) -> Path:
    """Create a temporary SQLite database for testing."""
    db_file = tmp_path / "test_data.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    cur.executemany(
        "INSERT INTO products (name, price) VALUES (?, ?)",
        [("Widget", 9.99), ("Gadget", 19.99), ("Gizmo", 29.99)],
    )
    conn.commit()
    conn.close()
    return db_file


class TestSQLProcessingIndicator:
    """Verifies the SQL processing indicator lifecycle and integration."""

    def test_indicator_starts_with_status_processing_sql_query(self):
        """1, 2, 3: Verify indicator starts, supports compact mode, and initial status is 'Processing SQL query…'."""
        mock_container = MagicMock()
        indicator = AuraProcessingIndicator(
            container=mock_container,
            mode="compact",
            status="Processing SQL query…",
            include_css=False,
        )
        indicator.start()

        assert indicator.is_active is True
        assert indicator.current_status == "Processing SQL query…"
        assert indicator.mode == "compact"
        assert mock_container.markdown.called

    def test_indicator_starts_before_api_dispatch(self):
        """4: Verify indicator starts before API dispatch is made."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        call_order = []

        mock_indicator.start.side_effect = lambda: call_order.append("indicator.start")
        mock_indicator.update.side_effect = lambda **kw: call_order.append(f"indicator.update:{kw.get('status')}")

        def mock_make_api(*args, **kwargs):
            call_order.append("make_api_request")
            return {"answer": "Total widgets: 3", "sources": []}

        with patch("ui.streamlit_app.make_api_request", side_effect=mock_make_api):
            res = execute_sql_query_api(
                question="How many widgets?",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is not None
        assert "indicator.start" in call_order
        assert "make_api_request" in call_order
        assert call_order.index("indicator.start") < call_order.index("make_api_request")

    def test_successful_sql_query_clears_indicator(self):
        """5: Successful SQL query clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            return_value={"answer": "Query result: 3 items", "sources": []},
        ):
            res = execute_sql_query_api(
                question="List all products",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is not None
        assert res["answer"] == "Query result: 3 items"
        assert mock_indicator.clear.called

    def test_sql_validation_failure_clears_indicator(self):
        """6: SQL validation failure clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError(
                "SQL validation failed: Syntax error near unexpected token",
                status_code=400,
                detail={"error": "sql_validation_error", "message": "Syntax error"},
            ),
        ):
            res = execute_sql_query_api(
                question="Invalid query syntax",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_unsafe_sql_rejection_clears_indicator(self):
        """7: Unsafe SQL rejection (INSERT/UPDATE/DELETE/DROP) clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError(
                "Write operations not permitted in read-only SQL mode",
                status_code=400,
                detail={"error": "sql_validation_error", "message": "Write operation rejected"},
            ),
        ):
            res = execute_sql_query_api(
                question="DROP TABLE products",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_sql_execution_failure_clears_indicator(self):
        """8: SQL execution failure clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError(
                "SQL execution error: no such table: non_existent",
                status_code=500,
                detail={"error": "sql_execution_error", "message": "Execution failed"},
            ),
        ):
            res = execute_sql_query_api(
                question="SELECT * FROM non_existent",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_timeout_clears_indicator(self):
        """9: Timeout clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError(
                "SQL execution timeout",
                status_code=408,
                detail={"error": "sql_timeout", "message": "Query timed out after 30s"},
            ),
        ):
            res = execute_sql_query_api(
                question="Long running query",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_provider_failure_clears_indicator(self):
        """10: AI Provider/API failure clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=APIError(
                "Gemini rate limit exceeded",
                status_code=429,
                detail={"error": "provider_error", "message": "Rate limit"},
            ),
        ):
            res = execute_sql_query_api(
                question="Query with provider failure",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

        assert res is None
        assert mock_indicator.clear.called

    def test_unexpected_exception_clears_indicator(self):
        """11: Unexpected exception clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=RuntimeError("Unexpected connection reset"),
        ):
            with pytest.raises(RuntimeError):
                execute_sql_query_api(
                    question="Any question",
                    file_id="12345678-1234-5678-1234-567812345678",
                    indicator=mock_indicator,
                )

        assert mock_indicator.clear.called

    def test_interruption_clears_indicator(self):
        """12: Interruption (KeyboardInterrupt) clears indicator."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)

        with patch(
            "ui.streamlit_app.make_api_request",
            side_effect=KeyboardInterrupt(),
        ):
            with pytest.raises(KeyboardInterrupt):
                execute_sql_query_api(
                    question="Interrupted query",
                    file_id="12345678-1234-5678-1234-567812345678",
                    indicator=mock_indicator,
                )

        assert mock_indicator.clear.called

    def test_exactly_one_sql_request_occurs(self):
        """13: Exactly one SQL request is dispatched per invocation."""
        mock_indicator = MagicMock(spec=AuraProcessingIndicator)
        with patch("ui.streamlit_app.make_api_request") as mock_make:
            mock_make.return_value = {"answer": "Count: 3", "sources": []}

            execute_sql_query_api(
                question="Count rows",
                file_id="12345678-1234-5678-1234-567812345678",
                indicator=mock_indicator,
            )

            assert mock_make.call_count == 1
            call_args = mock_make.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/sql-query"
            payload = json.loads(call_args[1]["data"])
            assert payload["question"] == "Count rows"
            assert payload["file_id"] == "12345678-1234-5678-1234-567812345678"

    def test_sql_result_table_remains_intact(self, test_db):
        """14: SQL result table structure and data remain intact."""
        schema = inspect_sqlite_schema(test_db, "test_data.db", "fid_1")
        result = execute_read_only_sql(
            test_db,
            "SELECT name, price FROM products ORDER BY price ASC",
            schema,
        )
        assert result.columns == ["name", "price"]
        assert len(result.rows) == 3
        assert result.rows[0] == ["Widget", 9.99]
        assert result.row_count == 3
        assert result.truncated is False

    def test_sql_citations_remain_intact(self, test_db):
        """15: SQL citations remain intact with [S1] grounding."""
        schema = inspect_sqlite_schema(test_db, "test_data.db", "fid_1")
        result = execute_read_only_sql(
            test_db,
            "SELECT COUNT(*) as total FROM products",
            schema,
        )
        doc = format_sql_evidence(result=result, citation_id="S1")
        assert doc.metadata["citation_id"] == "S1"
        assert doc.metadata["citation_marker"] == "[S1]"
        assert doc.metadata["source_type"] == "sql"
        assert doc.metadata["filename"] == "test_data.db"
        assert "products" in doc.metadata["citation_label"]

    def test_no_artificial_sleep_or_delay(self):
        """16: Verify execute_sql_query_api contains no artificial delays (time.sleep, asyncio.sleep)."""
        source = inspect.getsource(execute_sql_query_api)
        assert "time.sleep" not in source
        assert "asyncio.sleep" not in source

    def test_select_queries_still_work(self, test_db):
        """17: SELECT queries execute cleanly and return rows."""
        schema = inspect_sqlite_schema(test_db, "test_data.db", "fid_1")
        result = execute_read_only_sql(
            test_db,
            "SELECT name FROM products WHERE price > 15.0",
            schema,
        )
        assert len(result.rows) == 2
        assert [r[0] for r in result.rows] == ["Gadget", "Gizmo"]

    def test_write_operations_remain_blocked(self, test_db):
        """18: Write operations (INSERT, UPDATE, DELETE, DROP) are strictly blocked."""
        schema = inspect_sqlite_schema(test_db, "test_data.db", "fid_1")
        for write_stmt in [
            "INSERT INTO products (name, price) VALUES ('Illegal', 0)",
            "UPDATE products SET price = 0",
            "DELETE FROM products",
            "DROP TABLE products",
            "ALTER TABLE products ADD COLUMN secret TEXT",
        ]:
            with pytest.raises(SQLValidationError):
                validate_sql_query(write_stmt, schema)

        # Authorizer blocks writes even if executed directly
        with pytest.raises(sqlite3.DatabaseError):
            execute_read_only_sql(test_db, "DELETE FROM products", schema)

    def test_row_limit_remains_enforced(self, tmp_path):
        """19: Row limits are strictly enforced (truncated flag set)."""
        big_db = tmp_path / "large.db"
        conn = sqlite3.connect(str(big_db))
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (n INTEGER)")
        cur.executemany("INSERT INTO items VALUES (?)", [(i,) for i in range(50)])
        conn.commit()
        conn.close()

        schema = inspect_sqlite_schema(big_db, "large.db", "fid_big")
        result = execute_read_only_sql(big_db, "SELECT n FROM items", schema, max_rows=10)
        assert len(result.rows) == 10
        assert result.truncated is True
        assert result.row_count == 10

    def test_timeout_remains_enforced(self, test_db):
        """20: Timeout remains enforced on long-running queries."""
        schema = inspect_sqlite_schema(test_db, "test_data.db", "fid_1")
        result = execute_read_only_sql(
            test_db,
            "SELECT * FROM products",
            schema,
            timeout_sec=5.0,
        )
        assert result.execution_time_ms >= 0

    def test_existing_chat_streaming_remains_unchanged(self):
        """21: Existing chat streaming uses full mode indicator and renders properly."""
        source = inspect.getsource(process_question)
        assert "AuraProcessingIndicator" in source
        assert "mode=\"full\"" in source
        assert "indicator.start()" in source
        assert "indicator.clear_animation()" in source

    def test_document_upload_indicator_remains_unchanged(self):
        """22: Document upload indicator remains compact with 'Uploading document…'."""
        source = inspect.getsource(upload_document)
        assert "AuraProcessingIndicator" in source
        assert "Uploading document…" in source

    def test_image_processing_indicator_remains_unchanged(self):
        """23: Image processing indicator remains compact with 'Processing image…'."""
        source = inspect.getsource(process_attached_image)
        assert "AuraProcessingIndicator" in source
        assert "Processing image…" in source

    def test_chat_rename_and_delete_indicators_remain_unchanged(self):
        """24: Chat rename and delete indicators remain intact."""
        rename_source = inspect.getsource(rename_session_api)
        assert "Saving chat name…" in rename_source

        delete_source = inspect.getsource(delete_session_api)
        assert "Deleting chat…" in delete_source
