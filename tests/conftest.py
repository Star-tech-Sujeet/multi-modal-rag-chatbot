"""
Global pytest configuration and fixtures for Aura AI test suite.

Ensures strict environment and data isolation across all automated tests:
1. Redirects SQLite session storage (settings.sessions_db_path and app.sessions.session_manager)
   to an ephemeral temporary database per test run.
2. Prevents test pollution from contaminating development/production conversation databases.
"""

import uuid
import pytest
from app.config import settings
from app.sessions import session_manager


@pytest.fixture(autouse=True)
def isolate_test_session_db(tmp_path):
    """
    Autouse fixture that completely isolates the SQLite conversation sessions database
    during all automated test executions.

    Redirects both the configuration setting and the singleton SessionManager
    to an isolated temporary SQLite file managed by pytest's tmp_path.
    Guarantees that no test artifacts ('Corrupt Test', 'Text Only', etc.) can leak
    into the application database (data/sessions.sqlite).
    """
    orig_settings_db_path = settings.sessions_db_path
    orig_manager_custom = session_manager._custom_db_path

    test_db = str(tmp_path / f"test_sessions_{uuid.uuid4().hex}.sqlite")

    settings.sessions_db_path = test_db
    session_manager.db_path = test_db

    try:
        yield test_db
    finally:
        settings.sessions_db_path = orig_settings_db_path
        session_manager.db_path = orig_manager_custom
