"""
Session persistence and conversational query handling module.

Implements:
1. SQLite-backed persistent conversation sessions and message history.
2. Complete session isolation with parameterized queries and cascading deletes.
3. Thread-safe database operations compatible with FastAPI threadpool execution.
4. Deterministic conversational query contextualization for follow-up questions.
5. Structured source and citation persistence.
6. Deterministic session titling without LLM overhead.
"""

import json
import sqlite3
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.config import settings
from app.models import (
    Source,
    SessionMessageModel,
    SessionSummary,
    SessionResponse,
    SessionListResponse,
)

logger = logging.getLogger(__name__)

# Common conversational stopwords to avoid polluting retrieval contextualization
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if", "then", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "also", "about", "what",
    "how", "why", "where", "who", "which", "when", "there", "here", "it", "its",
    "this", "that", "these", "those", "they", "them", "their", "he", "him",
    "she", "her", "tell", "me", "more", "please"
}


def _get_utc_now() -> str:
    """Return current ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    """
    Manages persistent SQLite storage for conversation sessions and messages.
    Supports thread-safe localized connections and clean session isolation.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._custom_db_path = db_path
        self._last_initialized_path: Optional[str] = None
        self._ensure_db_dir()
        self.init_db()

    @property
    def db_path(self) -> str:
        return self._custom_db_path or settings.sessions_db_path

    @db_path.setter
    def db_path(self, path: Optional[str]):
        self._custom_db_path = path
        if self.db_path:
            self._ensure_db_dir()
            self.init_db()

    def _ensure_db_dir(self):
        """Ensure parent directory for SQLite database exists."""
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    def _raw_connect(self):
        self._ensure_db_dir()
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextmanager
    def _get_connection(self):
        """Create, configure, and safely close a thread-safe SQLite connection."""
        if self._last_initialized_path != self.db_path:
            self.init_db()
        conn = self._raw_connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initialize SQLite database schema for sessions and messages."""
        self._last_initialized_path = self.db_path
        conn = self._raw_connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sources TEXT,
                    citation_validation TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);")
            conn.commit()
        finally:
            conn.close()

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionResponse:
        """Create a new session with an optional title or default 'New Chat'."""
        sid = session_id or str(uuid.uuid4())
        now = _get_utc_now()
        clean_title = (title.strip() if title and title.strip() else "New Chat")[:150]
        meta_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, clean_title, now, now, meta_json)
            )
            conn.commit()

        return SessionResponse(
            session_id=sid,
            title=clean_title,
            created_at=now,
            updated_at=now,
            messages=[],
            metadata=metadata
        )

    def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Retrieve a session and all its ordered messages scoped strictly by session_id."""
        with self._get_connection() as conn:
            s_row = conn.execute(
                "SELECT session_id, title, created_at, updated_at, metadata FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if not s_row:
                return None

            m_rows = conn.execute(
                """
                SELECT message_id, session_id, role, content, created_at, sources, citation_validation
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,)
            ).fetchall()

        messages = []
        for r in m_rows:
            sources_list = None
            if r["sources"]:
                try:
                    raw_sources = json.loads(r["sources"])
                    sources_list = [Source(**s) if isinstance(s, dict) else s for s in raw_sources]
                except Exception as e:
                    logger.warning(f"Failed to parse sources for message {r['message_id']}: {e}")

            val_dict = None
            if r["citation_validation"]:
                try:
                    val_dict = json.loads(r["citation_validation"])
                except Exception:
                    pass

            messages.append(SessionMessageModel(
                message_id=r["message_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                sources=sources_list,
                citation_validation=val_dict
            ))

        meta = json.loads(s_row["metadata"]) if s_row["metadata"] else None
        return SessionResponse(
            session_id=s_row["session_id"],
            title=s_row["title"],
            created_at=s_row["created_at"],
            updated_at=s_row["updated_at"],
            messages=messages,
            metadata=meta
        )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> SessionListResponse:
        """List sessions ordered by updated_at descending with message counts."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()["cnt"]

            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at, s.metadata,
                       COUNT(m.message_id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            ).fetchall()

        summaries = []
        for r in rows:
            meta = json.loads(r["metadata"]) if r["metadata"] else None
            summaries.append(SessionSummary(
                session_id=r["session_id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["msg_count"],
                metadata=meta
            ))

        return SessionListResponse(sessions=summaries, total=total)

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SessionResponse]:
        """Update session title or metadata."""
        if title is not None:
            stripped = title.strip()
            if not stripped:
                raise ValueError("Session title cannot be empty or whitespace only.")
            new_title_val = stripped[:150]
        else:
            new_title_val = None

        now = _get_utc_now()
        with self._get_connection() as conn:
            s_row = conn.execute(
                "SELECT session_id, title, created_at, updated_at, metadata FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()

            if not s_row:
                return None

            new_title = new_title_val if new_title_val is not None else s_row["title"]
            new_meta = json.dumps(metadata) if metadata is not None else s_row["metadata"]

            conn.execute(
                """
                UPDATE sessions
                SET title = ?, metadata = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (new_title, new_meta, now, session_id)
            )
            conn.commit()

        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session and all its associated messages (cascading)."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Any]] = None,
        citation_validation: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None
    ) -> SessionMessageModel:
        """Append a message to a session and update the session's updated_at timestamp."""
        mid = message_id or str(uuid.uuid4())
        now = _get_utc_now()

        # Serialize sources if present
        sources_json = None
        parsed_sources = None
        if sources:
            serialized_sources = []
            for s in sources:
                if hasattr(s, "model_dump"):
                    serialized_sources.append(s.model_dump())
                elif isinstance(s, dict):
                    serialized_sources.append(s)
            sources_json = json.dumps(serialized_sources)
            parsed_sources = [Source(**s) if isinstance(s, dict) else s for s in serialized_sources]

        val_json = json.dumps(citation_validation) if citation_validation else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, session_id, role, content, created_at, sources, citation_validation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, session_id, role, content, now, sources_json, val_json)
            )

            # If this is the first user message and session has default title, update title deterministically
            if role == "user":
                cur_sess = conn.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                if cur_sess and (cur_sess["title"] in ("New Chat", "New Session") or not cur_sess["title"]):
                    auto_title = content.strip().replace("\n", " ")[:60]
                    conn.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (auto_title, now, session_id)
                    )
                else:
                    conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            else:
                conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))

            conn.commit()

        return SessionMessageModel(
            message_id=mid,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
            sources=parsed_sources,
            citation_validation=citation_validation
        )

    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[SessionMessageModel]:
        """Get ordered messages for a session, optionally limited to the most recent N."""
        with self._get_connection() as conn:
            if limit is not None and limit > 0:
                rows = conn.execute(
                    """
                    SELECT message_id, session_id, role, content, created_at, sources, citation_validation
                    FROM (
                        SELECT message_id, session_id, role, content, created_at, sources, citation_validation
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                    ORDER BY created_at ASC
                    """,
                    (session_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT message_id, session_id, role, content, created_at, sources, citation_validation
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,)
                ).fetchall()

        messages = []
        for r in rows:
            sources_list = None
            if r["sources"]:
                try:
                    raw = json.loads(r["sources"])
                    sources_list = [Source(**s) if isinstance(s, dict) else s for s in raw]
                except Exception:
                    pass

            val_dict = json.loads(r["citation_validation"]) if r["citation_validation"] else None

            messages.append(SessionMessageModel(
                message_id=r["message_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                sources=sources_list,
                citation_validation=val_dict
            ))
        return messages


# Singleton SessionManager instance
session_manager = SessionManager()


# =============================================================================
# Conversational Query Contextualization
# =============================================================================

def construct_contextual_query(
    current_query: str,
    history_messages: List[SessionMessageModel]
) -> str:
    """
    Construct a contextual retrieval query for conversational follow-ups.
    Enables accurate candidate retrieval when users ask follow-up questions
    (e.g., 'What about 2023?' or 'Tell me more about their revenue').

    Principles:
    1. Fast and deterministic: Zero additional LLM calls or latency.
    2. Authority: Uses prior user questions to provide core subject context.
    3. Non-pollution: Does NOT copy extensive assistant answers or hallucinated text.
    4. Safety: If the query is already self-contained and descriptive, it remains unchanged.
    """
    q_clean = current_query.strip()
    if not q_clean or not history_messages:
        return q_clean

    q_lower = q_clean.lower()
    words = [w.strip(".,?!;:'\"") for w in q_lower.split() if w.strip()]

    # Follow-up indicators
    starts_followup = any(
        q_lower.startswith(prefix) for prefix in (
            "what about", "how about", "and what", "and how", "why", "where",
            "which one", "tell me more", "explain that", "what was", "what were"
        )
    )
    has_pronouns = any(
        p in words for p in ("it", "its", "they", "them", "their", "that", "this", "these", "those")
    )
    is_short = len(words) <= 5

    if not (starts_followup or has_pronouns or is_short):
        return q_clean

    # Find the most recent user query to extract salient topic keywords
    prior_user_queries = [m.content for m in history_messages if m.role == "user"]
    if not prior_user_queries:
        return q_clean

    last_user_query = prior_user_queries[-1]
    last_words = [w.strip(".,?!;:'\"()[]") for w in last_user_query.lower().split() if w.strip()]
    salient_terms = [w for w in last_words if len(w) > 2 and w not in STOPWORDS and w not in words]

    if not salient_terms:
        return q_clean

    # Append up to 5 salient topic terms to the retrieval query
    context_expansion = " ".join(salient_terms[:5])
    contextual_query = f"{q_clean} {context_expansion}".strip()
    logger.debug(f"Contextualized query: '{q_clean}' -> '{contextual_query}'")
    return contextual_query
