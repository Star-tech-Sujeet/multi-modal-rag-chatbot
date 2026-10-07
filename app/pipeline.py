"""
Intelligent Response Pipeline & Reasoning UX Engine for Aura AI (Phase 11).

Provides:
1. Structured pipeline stages and status representation.
2. User-safe status events describing real backend operations (NO private chain-of-thought,
   NO hidden prompts, NO fake delays or invented reasoning).
3. Lightweight, deterministic query classification and adaptive execution planning.
4. Pipeline status emitter helper for real-time SSE streaming.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import re
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PipelineStage:
    """Standardized identifier constants for response pipeline stages."""
    INTAKE = "intake"
    UNDERSTANDING = "understanding"
    CLASSIFICATION = "classification"
    QUERY_REWRITE = "query_rewrite"
    RETRIEVAL = "retrieval"
    VECTOR_SEARCH = "vector_search"
    LEXICAL_SEARCH = "lexical_search"
    RERANKING = "reranking"
    EVIDENCE_SELECTION = "evidence_selection"
    IMAGE_PROCESSING = "image_processing"
    COMPARISON = "comparison"
    SQL_SCHEMA = "sql_schema"
    SQL_GENERATION = "sql_generation"
    SQL_VALIDATION = "sql_validation"
    SQL_EXECUTION = "sql_execution"
    ANSWER_PREPARATION = "answer_preparation"
    GENERATION = "generation"
    CITATION_VALIDATION = "citation_validation"
    COMPLETION = "completion"
    ERROR = "error"

    ALL_STAGES = [
        INTAKE,
        UNDERSTANDING,
        CLASSIFICATION,
        QUERY_REWRITE,
        RETRIEVAL,
        VECTOR_SEARCH,
        LEXICAL_SEARCH,
        RERANKING,
        EVIDENCE_SELECTION,
        IMAGE_PROCESSING,
        COMPARISON,
        SQL_SCHEMA,
        SQL_GENERATION,
        SQL_VALIDATION,
        SQL_EXECUTION,
        ANSWER_PREPARATION,
        GENERATION,
        CITATION_VALIDATION,
        COMPLETION,
        ERROR,
    ]


class StageStatus:
    """Execution status for a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    ERROR = "error"

    ALL_STATUSES = [PENDING, RUNNING, COMPLETE, SKIPPED, ERROR]


class QueryType:
    """Classified category of query for adaptive execution flow routing."""
    GENERAL_RAG = "GENERAL_RAG"
    FOLLOW_UP = "FOLLOW_UP"
    COMPARISON = "COMPARISON"
    IMAGE_QUERY = "IMAGE_QUERY"
    SQL_QUERY = "SQL_QUERY"
    NO_DOCUMENT_CONTEXT = "NO_DOCUMENT_CONTEXT"
    GENERAL_CHAT = "GENERAL_CHAT"

    ALL_TYPES = [
        GENERAL_RAG,
        FOLLOW_UP,
        COMPARISON,
        IMAGE_QUERY,
        SQL_QUERY,
        NO_DOCUMENT_CONTEXT,
        GENERAL_CHAT,
    ]


# =============================================================================
# User-Safe Status Messages
# =============================================================================
# CRITICAL SAFETY: Concise, professional status descriptions of what the
# system is actually doing. NEVER contains model deliberation, chain-of-thought,
# internal prompts, or reasoning traces.
# =============================================================================

DEFAULT_STAGE_MESSAGES: Dict[str, Dict[str, str]] = {
    PipelineStage.INTAKE: {
        StageStatus.RUNNING: "Receiving your question...",
        StageStatus.COMPLETE: "Question received",
        StageStatus.SKIPPED: "Intake skipped",
        StageStatus.ERROR: "Failed to receive question",
    },
    PipelineStage.UNDERSTANDING: {
        StageStatus.RUNNING: "Understanding your question...",
        StageStatus.COMPLETE: "Question understood",
        StageStatus.SKIPPED: "Understanding skipped",
        StageStatus.ERROR: "Could not understand question",
    },
    PipelineStage.CLASSIFICATION: {
        StageStatus.RUNNING: "Determining how to answer...",
        StageStatus.COMPLETE: "Execution plan determined",
        StageStatus.SKIPPED: "Routing skipped",
        StageStatus.ERROR: "Routing failed",
    },
    PipelineStage.QUERY_REWRITE: {
        StageStatus.RUNNING: "Refining the search with conversation context...",
        StageStatus.COMPLETE: "Search refined",
        StageStatus.SKIPPED: "Context resolution skipped",
        StageStatus.ERROR: "Query refinement failed",
    },
    PipelineStage.RETRIEVAL: {
        StageStatus.RUNNING: "Searching your documents...",
        StageStatus.COMPLETE: "Found relevant sources",
        StageStatus.SKIPPED: "Document search skipped",
        StageStatus.ERROR: "Document search failed",
    },
    PipelineStage.VECTOR_SEARCH: {
        StageStatus.RUNNING: "Finding semantically relevant information...",
        StageStatus.COMPLETE: "Semantic matches retrieved",
        StageStatus.SKIPPED: "— Vector search skipped",
        StageStatus.ERROR: "Vector search failed",
    },
    PipelineStage.LEXICAL_SEARCH: {
        StageStatus.RUNNING: "Checking exact keyword matches...",
        StageStatus.COMPLETE: "Keyword matches retrieved",
        StageStatus.SKIPPED: "— Keyword search disabled",
        StageStatus.ERROR: "Keyword search failed",
    },
    PipelineStage.RERANKING: {
        StageStatus.RUNNING: "Ranking the most relevant sources...",
        StageStatus.COMPLETE: "Sources ranked",
        StageStatus.SKIPPED: "— Lexical reranking disabled",
        StageStatus.ERROR: "Reranking failed",
    },
    PipelineStage.EVIDENCE_SELECTION: {
        StageStatus.RUNNING: "Reviewing relevant sources...",
        StageStatus.COMPLETE: "Relevant sources reviewed",
        StageStatus.SKIPPED: "Evidence selection skipped",
        StageStatus.ERROR: "Evidence selection failed",
    },
    PipelineStage.IMAGE_PROCESSING: {
        StageStatus.RUNNING: "Analyzing the attached image...",
        StageStatus.COMPLETE: "Image analysis complete",
        StageStatus.SKIPPED: "Image processing skipped",
        StageStatus.ERROR: "Image analysis failed",
    },
    PipelineStage.COMPARISON: {
        StageStatus.RUNNING: "Comparing information across documents...",
        StageStatus.COMPLETE: "Cross-document comparison complete",
        StageStatus.SKIPPED: "Comparison skipped",
        StageStatus.ERROR: "Comparison failed",
    },
    PipelineStage.SQL_SCHEMA: {
        StageStatus.RUNNING: "Checking database structure...",
        StageStatus.COMPLETE: "Database schema inspected",
        StageStatus.SKIPPED: "Schema check skipped",
        StageStatus.ERROR: "Schema check failed",
    },
    PipelineStage.SQL_GENERATION: {
        StageStatus.RUNNING: "Preparing a database query...",
        StageStatus.COMPLETE: "Database query prepared",
        StageStatus.SKIPPED: "SQL generation skipped",
        StageStatus.ERROR: "Failed to generate database query",
    },
    PipelineStage.SQL_VALIDATION: {
        StageStatus.RUNNING: "Validating query safety...",
        StageStatus.COMPLETE: "Query validated safely",
        StageStatus.SKIPPED: "SQL validation skipped",
        StageStatus.ERROR: "Query validation failed",
    },
    PipelineStage.SQL_EXECUTION: {
        StageStatus.RUNNING: "Running read-only database query...",
        StageStatus.COMPLETE: "Query executed safely",
        StageStatus.SKIPPED: "SQL execution skipped",
        StageStatus.ERROR: "Database execution failed",
    },
    PipelineStage.ANSWER_PREPARATION: {
        StageStatus.RUNNING: "Preparing your answer...",
        StageStatus.COMPLETE: "Answer prepared",
        StageStatus.SKIPPED: "Preparation skipped",
        StageStatus.ERROR: "Preparation failed",
    },
    PipelineStage.GENERATION: {
        StageStatus.RUNNING: "Generating response...",
        StageStatus.COMPLETE: "Response generated",
        StageStatus.SKIPPED: "Generation skipped",
        StageStatus.ERROR: "Response generation failed",
    },
    PipelineStage.CITATION_VALIDATION: {
        StageStatus.RUNNING: "Verifying sources...",
        StageStatus.COMPLETE: "Sources verified",
        StageStatus.SKIPPED: "Citation validation skipped",
        StageStatus.ERROR: "Citation verification issue",
    },
    PipelineStage.COMPLETION: {
        StageStatus.RUNNING: "Finalizing response...",
        StageStatus.COMPLETE: "Answer ready",
        StageStatus.SKIPPED: "Completion skipped",
        StageStatus.ERROR: "Completion failed",
    },
    PipelineStage.ERROR: {
        StageStatus.RUNNING: "Handling error...",
        StageStatus.COMPLETE: "Error handled",
        StageStatus.SKIPPED: "Error skipped",
        StageStatus.ERROR: "Something went wrong while processing your request.",
    },
}


class PipelineStatusEvent(BaseModel):
    """Pydantic schema for SSE pipeline status events."""
    stage: str = Field(..., description="Active pipeline stage identifier")
    status: str = Field(..., description="Stage status: pending, running, complete, skipped, error")
    message: str = Field(..., description="User-safe descriptive message")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 event timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional stage metadata (e.g. counts, search method)"
    )
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional fractional progress (only when technically measurable)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "stage": "retrieval",
                "status": "running",
                "message": "Searching your documents...",
                "timestamp": "2026-09-15T07:10:00Z",
                "metadata": {"search_method": "hybrid"}
            }
        }
    }


def parse_pipeline_timestamp(value: Any) -> Optional[datetime]:
    """
    Parse a pipeline timestamp into a UTC-aware datetime object.
    Supports:
      - datetime object (tz-aware or naive; naive is assumed UTC)
      - ISO-8601 strings (e.g. '2026-09-15T06:30:12.000000', '2026-09-15T06:30:12Z', '+00:00')
      - Unix epoch numbers (int, float) or stringified numbers ('1726381812.34')
      - Returns None if value is None, empty, or unparseable.
    """
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None

        # Check if it's a numeric unix timestamp string
        try:
            numeric_val = float(v)
            return datetime.fromtimestamp(numeric_val, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass

        # Handle 'Z' suffix for ISO-8601
        iso_str = v
        if iso_str.endswith("Z") or iso_str.endswith("z"):
            iso_str = iso_str[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            pass

    return None


def calculate_pipeline_duration_seconds(
    first_ts: Any,
    last_ts: Any
) -> Optional[float]:
    """
    Calculate the elapsed duration in seconds between two timestamps.
    Returns None if timestamps are missing, invalid, or duration is negative.
    """
    dt_first = parse_pipeline_timestamp(first_ts)
    dt_last = parse_pipeline_timestamp(last_ts)
    if dt_first is None or dt_last is None:
        return None

    diff = (dt_last - dt_first).total_seconds()
    if diff < 0:
        return None
    return diff


# =============================================================================
# Lightweight Query Classifier (Deterministic & Fast - No Extra LLM Calls)
# =============================================================================

# Keywords indicating comparison intent
COMPARISON_KEYWORDS = {
    "compare", "comparison", "difference", "differences", "differ", "versus", "vs",
    "vs.", "contrasting", "contrast", "similarities", "similarity", "discrepancy",
    "discrepancies", "inconsistent", "inconsistency", "in conflict", "disagree",
    "higher than", "lower than", "better than", "worse than"
}

# Conversational greetings / general pleasantries
GENERAL_CHAT_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "who are you", "what can you do", "help", "thanks", "thank you",
    "bye", "goodbye", "how are you", "what is your name"
}

# Follow-up cues referring to previous turns
FOLLOW_UP_CUES = {
    "it", "they", "them", "that", "this", "these", "those", "above",
    "previous", "earlier", "the first", "the second", "more details",
    "tell me more", "explain further", "why", "how so", "what about",
    "and then", "continue", "elaborate"
}


def is_comparison_query_text(text: str) -> bool:
    """Deterministically detect if text expresses comparison intent."""
    t_lower = text.lower()
    tokens = set(re.findall(r'\b[a-z\.\-]+\b', t_lower))
    if any(k in t_lower for k in COMPARISON_KEYWORDS):
        return True
    return bool(tokens.intersection(COMPARISON_KEYWORDS))


def is_general_chat_text(text: str) -> bool:
    """Deterministically detect if text is a simple conversational greeting or query."""
    clean = text.strip().lower().rstrip(".!?,")
    if clean in GENERAL_CHAT_PATTERNS:
        return True
    if any(clean.startswith(p) for p in ("hi", "hello", "hey", "thanks", "thank you", "good morning", "good afternoon", "good evening")):
        if len(clean) < 35 or any(p in clean for p in ("how are you", "who are you", "what can you do", "aura")):
            return True
    return False


def is_follow_up_query_text(text: str, history_len: int = 0) -> bool:
    """Deterministically detect if text relies on prior conversational turns."""
    if history_len == 0:
        return False
    t_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', t_lower))
    # Check for pronoun/referential cues in short questions
    if len(words) <= 7 and any(cue in words for cue in ("it", "this", "that", "they", "them", "why", "how")):
        return True
    if any(cue in t_lower for cue in ("what about", "tell me more", "explain further", "elaborate", "continue", "the previous")):
        return True
    return False


def classify_query(
    question: str,
    file_ids: Optional[List[str]] = None,
    has_image: bool = False,
    recent_messages_count: int = 0,
    has_structured_files: bool = False,
    is_structured_intent: bool = False,
    is_explicit_comparison: bool = False
) -> Tuple[str, List[str]]:
    """
    Lightweight, deterministic query classifier and adaptive pipeline planner.
    
    Categorizes the query into one of:
    - GENERAL_RAG
    - FOLLOW_UP
    - COMPARISON
    - IMAGE_QUERY
    - SQL_QUERY
    - NO_DOCUMENT_CONTEXT
    - GENERAL_CHAT
    
    Returns:
        Tuple of (query_type: str, planned_stages: List[str])
    """
    clean_q = question.strip()
    fids = [fid for fid in (file_ids or []) if fid and fid.strip()]
    doc_count = len(fids)

    # 1. Image query (live attached image)
    if has_image:
        planned = [
            PipelineStage.UNDERSTANDING,
            PipelineStage.CLASSIFICATION,
            PipelineStage.IMAGE_PROCESSING,
            PipelineStage.EVIDENCE_SELECTION,
            PipelineStage.ANSWER_PREPARATION,
            PipelineStage.GENERATION,
            PipelineStage.CITATION_VALIDATION,
            PipelineStage.COMPLETION,
        ]
        return QueryType.IMAGE_QUERY, planned

    # 2. Structured Data / Text-to-SQL query
    if has_structured_files and is_structured_intent:
        planned = [
            PipelineStage.UNDERSTANDING,
            PipelineStage.CLASSIFICATION,
            PipelineStage.SQL_SCHEMA,
            PipelineStage.SQL_GENERATION,
            PipelineStage.SQL_VALIDATION,
            PipelineStage.SQL_EXECUTION,
            PipelineStage.EVIDENCE_SELECTION,
            PipelineStage.ANSWER_PREPARATION,
            PipelineStage.GENERATION,
            PipelineStage.CITATION_VALIDATION,
            PipelineStage.COMPLETION,
        ]
        return QueryType.SQL_QUERY, planned

    # 3. No document or image context
    if doc_count == 0:
        if is_general_chat_text(clean_q):
            planned = [
                PipelineStage.UNDERSTANDING,
                PipelineStage.CLASSIFICATION,
                PipelineStage.GENERATION,
                PipelineStage.COMPLETION,
            ]
            return QueryType.GENERAL_CHAT, planned
        else:
            planned = [
                PipelineStage.UNDERSTANDING,
                PipelineStage.CLASSIFICATION,
                PipelineStage.COMPLETION,
            ]
            return QueryType.NO_DOCUMENT_CONTEXT, planned

    # 4. Cross-document comparison query
    if is_explicit_comparison or (doc_count >= 2 and is_comparison_query_text(clean_q)):
        planned = [
            PipelineStage.UNDERSTANDING,
            PipelineStage.CLASSIFICATION,
            PipelineStage.COMPARISON,
            PipelineStage.RETRIEVAL,
            PipelineStage.EVIDENCE_SELECTION,
            PipelineStage.ANSWER_PREPARATION,
            PipelineStage.GENERATION,
            PipelineStage.CITATION_VALIDATION,
            PipelineStage.COMPLETION,
        ]
        return QueryType.COMPARISON, planned

    # 5. Follow-up query in multi-turn conversation
    if is_follow_up_query_text(clean_q, history_len=recent_messages_count):
        planned = [
            PipelineStage.UNDERSTANDING,
            PipelineStage.CLASSIFICATION,
            PipelineStage.QUERY_REWRITE,
            PipelineStage.RETRIEVAL,
            PipelineStage.RERANKING,
            PipelineStage.EVIDENCE_SELECTION,
            PipelineStage.ANSWER_PREPARATION,
            PipelineStage.GENERATION,
            PipelineStage.CITATION_VALIDATION,
            PipelineStage.COMPLETION,
        ]
        return QueryType.FOLLOW_UP, planned

    # 6. Default: Standard Single/Multi-Document RAG
    planned = [
        PipelineStage.UNDERSTANDING,
        PipelineStage.CLASSIFICATION,
        PipelineStage.RETRIEVAL,
        PipelineStage.RERANKING,
        PipelineStage.EVIDENCE_SELECTION,
        PipelineStage.ANSWER_PREPARATION,
        PipelineStage.GENERATION,
        PipelineStage.CITATION_VALIDATION,
        PipelineStage.COMPLETION,
    ]
    return QueryType.GENERAL_RAG, planned


# =============================================================================
# Pipeline Status Emitter Helper
# =============================================================================

class PipelineStatusEmitter:
    """
    Helper for emitting user-safe status events across the backend execution pipeline.
    Maintains an in-memory chronological history of stages executed for audit and UI rendering.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.history: List[Dict[str, Any]] = []

    def create_event(
        self,
        stage: str,
        status: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Construct a validated PipelineStatusEvent dictionary and record it in history.
        Returns the event dict if enabled, otherwise None.
        """
        if not self.enabled:
            return None

        # Resolve clean user-facing default message if none supplied
        if not message:
            stage_dict = DEFAULT_STAGE_MESSAGES.get(stage, {})
            message = stage_dict.get(status, f"{stage} {status}")

        evt = PipelineStatusEvent(
            stage=stage,
            status=status,
            message=message,
            timestamp=timestamp or datetime.utcnow().isoformat(),
            metadata=metadata,
            progress=progress
        )
        evt_dict = evt.model_dump() if hasattr(evt, "model_dump") else evt.__dict__
        self.history.append(evt_dict)
        return evt_dict

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the completed in-memory stage history."""
        return list(self.history)
