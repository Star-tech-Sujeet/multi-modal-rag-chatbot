"""
Pydantic models for request and response schemas.
Enhanced with additional fields, validation, and documentation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

from .pipeline import (
    PipelineStage,
    StageStatus,
    QueryType,
    PipelineStatusEvent
)


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    message: str = Field(..., description="Processing status message")
    chunks_created: Optional[int] = Field(None, description="Number of document chunks created")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "document.pdf",
                "message": "File processed successfully. 15 chunks created and indexed.",
                "chunks_created": 15,
                "file_size": 102400
            }
        }
    }


class ChatMessage(BaseModel):
    """Model representing a chat message for conversation history."""
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "What is the main topic of this document?"
            }
        }
    }


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="The question to ask about the document"
    )
    file_id: Optional[str] = Field(
        default=None, 
        description="UUID of the uploaded file to query (for single document mode)"
    )
    file_ids: Optional[List[str]] = Field(
        default=None,
        description="List of file UUIDs to query (for multi-document mode)"
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages for context (max 10 messages)"
    )
    max_sources: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of source documents to return"
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0=deterministic, 2=creative)"
    )
    use_hybrid_search: Optional[bool] = Field(
        default=True,
        description="Use hybrid search (vector + BM25) for better recall"
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Model routing preference: 'mini'/'standard' (default), 'primary'/'complex' (for deep reasoning), or a specific model name"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter dictionary for narrowing search scope (e.g. file_type, source_type, table_name)"
    )
    enable_reranking: Optional[bool] = Field(
        default=None,
        description="Explicitly enable or disable the reranking stage for this query (defaults to rag_enable_reranking config)"
    )
    enable_status: Optional[bool] = Field(
        default=None,
        description="Explicitly enable or disable real-time pipeline status events in SSE stream"
    )
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Ensure question is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError('Question cannot be empty or whitespace only')
        return stripped
    
    def get_file_ids(self) -> List[str]:
        """Get list of file IDs to query (handles both single and multi-document modes)."""
        if self.file_ids:
            return self.file_ids
        elif self.file_id:
            return [self.file_id]
        return []
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the main topics discussed in this document?",
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "file_ids": ["id1", "id2"],
                "use_hybrid_search": True,
                "model_preference": "mini",
                "max_sources": 5,
                "temperature": 0.1
            }
        }
    }


class SQLQueryRequest(BaseModel):
    """Request model for dedicated structured SQL query endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question to ask about the structured data"
    )
    file_id: str = Field(
        ...,
        description="UUID of the uploaded SQLite or CSV file"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session UUID to associate this query turn with"
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Model routing preference"
    )

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError('Question cannot be empty or whitespace only')
        return stripped

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the total revenue by region?",
                "file_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }


class Source(BaseModel):
    """Model representing a source document chunk."""
    filename: str = Field(..., description="Name of the source file")
    file_id: Optional[str] = Field(None, description="ID of the source file")
    chunk_id: Optional[str] = Field(None, description="Unique, collision-free identifier for the chunk")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    content: str = Field(..., description="Preview of the source content")
    relevance_score: Optional[float] = Field(None, description="Combined relevance score (0-1)")
    vector_score: Optional[float] = Field(None, description="Vector similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword match score")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in the document")
    search_type: Optional[str] = Field(None, description="How this result was found: 'vector', 'bm25', or 'hybrid'")
    section: Optional[str] = Field(None, description="Section or heading title if applicable")
    table_name: Optional[str] = Field(None, description="Table name if from a table or database")
    row_number: Optional[int] = Field(None, description="Row number if from CSV or tabular data")
    source_type: Optional[str] = Field(None, description="Source content type: 'text', 'paragraph', 'table', 'row', 'image'")
    ingestion_method: Optional[str] = Field(None, description="Ingestion method used: 'direct', 'ocr', 'vision', 'structured'")
    citation_id: Optional[str] = Field(None, description="Deterministic citation identifier e.g. 'S1'")
    citation_label: Optional[str] = Field(None, description="Formatted citation label e.g. '[S1] report.pdf, p. 4'")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "citation_id": "S1",
                "citation_label": "[S1] document.pdf, p. 5",
                "filename": "document.pdf",
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_id": "123e4567-e89b-12d3-a456-426614174000_chunk_12",
                "page_number": 5,
                "content": "This section discusses the main findings...",
                "relevance_score": 0.92,
                "vector_score": 0.89,
                "bm25_score": 0.95,
                "chunk_index": 12,
                "search_type": "hybrid"
            }
        }
    }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="AI-generated answer to the question")
    context: str = Field(..., description="Combined context used for generating the answer")
    sources: List[Source] = Field(default_factory=list, description="Source documents used")
    suggested_questions: List[str] = Field(
        default_factory=list, 
        description="AI-generated follow-up questions based on the conversation"
    )
    search_method: Optional[str] = Field(None, description="Search method used: 'vector', 'bm25', or 'hybrid'")
    documents_searched: Optional[int] = Field(None, description="Number of documents searched")
    model_used: Optional[str] = Field(None, description="AI model used for generation")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    retrieval_debug: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Internal debug information about candidate retrieval, fusion, and reranking"
    )
    citation_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Deterministic post-generation citation validation report"
    )
    timings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed execution time breakdown in milliseconds across pipeline stages"
    )
    pipeline_stages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Chronological record of completed pipeline status stages"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The document discusses three main topics: machine learning, natural language processing, and computer vision.",
                "context": "This document contains various information about artificial intelligence...",
                "sources": [
                    {
                        "filename": "document.pdf",
                        "page_number": 1,
                        "content": "This document contains various information about AI...",
                        "relevance_score": 0.95
                    }
                ],
                "suggested_questions": [
                    "What are the key differences between machine learning and deep learning?",
                    "How is NLP used in the document's context?",
                    "What computer vision applications are mentioned?"
                ],
                "model_used": "gpt-4o-mini",
                "processing_time_ms": 1250
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    message: str = Field(..., description="Health check message")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Welcome to the Multi-Modal RAG API",
                "status": "healthy"
            }
        }
    }


class VectorstoreStats(BaseModel):
    """Statistics about the vectorstore."""
    total_documents: int = Field(..., description="Total number of document chunks stored")
    status: str = Field(..., description="Vectorstore status")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_documents": 150,
                "status": "healthy"
            }
        }
    }


class FileInfo(BaseModel):
    """Information about an uploaded file."""
    file_id: str = Field(..., description="Unique identifier for the file")
    filename: str = Field(..., description="Original filename")
    upload_path: Optional[str] = Field(None, description="Server path to the file")
    size_bytes: int = Field(..., description="File size in bytes")
    uploaded_at: Optional[str] = Field(None, description="ISO timestamp of upload")
    chunks_count: Optional[int] = Field(None, description="Number of document chunks created/indexed")
    file_type: Optional[str] = Field(None, description="File type uppercase e.g. PDF, DOCX")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "document.pdf",
                "upload_path": "/data/uploads/123e4567_document.pdf",
                "size_bytes": 102400,
                "uploaded_at": "2024-01-15T10:30:00",
                "chunks_count": 35,
                "file_type": "PDF"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error code/type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "validation_error",
                "message": "The uploaded file type is not supported",
                "details": {"supported_types": [".pdf", ".docx", ".txt"]}
            }
        }
    }


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    message: str = Field(..., description="Deletion status message")
    file_id: str = Field(..., description="ID of the deleted file")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "File deleted successfully",
                "file_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }


# =============================================================================
# Phase 6: Session & Conversational RAG Models
# =============================================================================

class SessionCreateRequest(BaseModel):
    """Request model for creating a new chat session."""
    title: Optional[str] = Field(None, max_length=150, description="Optional custom session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional session metadata (e.g. file_ids)")


class SessionUpdateRequest(BaseModel):
    """Request model for updating session attributes."""
    title: Optional[str] = Field(None, min_length=1, max_length=150, description="Updated session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated session metadata")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            if not stripped:
                raise ValueError("Session title cannot be empty or whitespace only.")
            return stripped[:150]
        return v


class SessionMessageModel(BaseModel):
    """Structured representation of a single conversation turn in a session."""
    message_id: str = Field(..., description="Unique message UUID")
    session_id: str = Field(..., description="Session UUID this message belongs to")
    role: str = Field(..., description="Message sender role: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    sources: Optional[List[Source]] = Field(None, description="Structured source citations (for assistant messages)")
    citation_validation: Optional[Dict[str, Any]] = Field(None, description="Citation validation audit details")


class SessionSummary(BaseModel):
    """Summary representation of a session for listing operations."""
    session_id: str = Field(..., description="Unique session UUID")
    title: str = Field(..., description="Session title")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last-updated timestamp")
    message_count: int = Field(0, description="Total message count in the session")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")


class SessionResponse(BaseModel):
    """Full session details including ordered message history."""
    session_id: str = Field(..., description="Unique session UUID")
    title: str = Field(..., description="Session title")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last-updated timestamp")
    messages: List[SessionMessageModel] = Field(default_factory=list, description="Ordered conversation messages")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")


class SessionListResponse(BaseModel):
    """Response model for listing chat sessions."""
    sessions: List[SessionSummary] = Field(default_factory=list, description="List of sessions")
    total: int = Field(..., description="Total count of sessions")


class SessionQueryRequest(BaseModel):
    """Request model for querying within an active persistent session."""
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="User question"
    )
    file_id: Optional[str] = Field(None, description="UUID of file to query")
    file_ids: Optional[List[str]] = Field(None, description="List of file UUIDs to query")
    max_sources: Optional[int] = Field(5, ge=1, le=20, description="Max source chunks to return")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    use_hybrid_search: Optional[bool] = Field(True, description="Enable hybrid retrieval")
    model_preference: Optional[str] = Field("mini", description="Model preference ('primary', 'mini', or model name)")
    enable_reranking: Optional[bool] = Field(None, description="Reranking toggle override")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filtering criteria")
    image_base64: Optional[str] = Field(None, description="Optional base64-encoded live attached image")
    image_filename: Optional[str] = Field(None, description="Original filename of live attached image")
    enable_status: Optional[bool] = Field(None, description="Explicitly enable or disable real-time pipeline status events in SSE stream")

    def get_file_ids(self) -> List[str]:
        """Resolve file IDs from either file_id or file_ids."""
        if self.file_ids:
            return [fid for fid in self.file_ids if fid and fid.strip()]
        if self.file_id and self.file_id.strip():
            return [self.file_id.strip()]
        return []


class SessionQueryResponse(BaseModel):
    """Response model for session-scoped conversational queries."""
    session_id: str = Field(..., description="Session UUID")
    message_id: str = Field(..., description="Assistant message UUID")
    answer: str = Field(..., description="Generated answer with inline citations")
    context: str = Field(..., description="Retrieved evidence context")
    sources: List[Source] = Field(default_factory=list, description="Attributed sources with citation IDs")
    suggested_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    search_method: Optional[str] = Field(None, description="Retrieval method used")
    documents_searched: Optional[int] = Field(None, description="Number of documents searched")
    model_used: Optional[str] = Field(None, description="AI model used")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    citation_validation: Optional[Dict[str, Any]] = Field(None, description="Citation validation audit details")
    timings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed execution time breakdown in milliseconds across pipeline stages"
    )
    pipeline_stages: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Chronological record of completed pipeline status stages"
    )


# =============================================================================
# Phase 8: Cross-Document Comparison Models
# =============================================================================

class ComparisonMode:
    """Supported comparison modes."""
    DIFFERENCE = "difference"
    SIMILARITY = "similarity"
    METRIC = "metric"
    CONFLICT = "conflict"
    GENERAL = "general"

    ALL_MODES = [DIFFERENCE, SIMILARITY, METRIC, CONFLICT, GENERAL]


class ComparisonEvidenceGroup(BaseModel):
    """Grouped evidence from a single document or live image."""
    document_id: Optional[str] = Field(None, description="Document UUID (None for ephemeral images)")
    document_name: str = Field(..., description="Filename or label")
    is_image: bool = Field(False, description="True if evidence is from an attached live image")
    evidence_count: int = Field(0, description="Number of retrieved chunks in this group")
    sources: List[Source] = Field(default_factory=list, description="List of source items with citations")


class ComparisonRequest(BaseModel):
    """Request model for cross-document comparison."""
    query: str = Field(..., min_length=1, max_length=2000, description="Comparison question or prompt")
    file_ids: Optional[List[str]] = Field(None, description="List of file UUIDs to compare (typically 2)")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for context persistence")
    comparison_mode: Optional[str] = Field(None, description="Comparison mode: 'difference', 'similarity', 'metric', 'conflict', 'general'")
    max_sources_per_document: Optional[int] = Field(None, ge=1, le=10, description="Max chunks per document")
    use_hybrid_search: Optional[bool] = Field(True, description="Enable hybrid retrieval per document")
    model_preference: Optional[str] = Field("mini", description="Model preference ('mini', 'primary', or model name)")
    enable_reranking: Optional[bool] = Field(None, description="Toggle lexical reranking override")
    image_base64: Optional[str] = Field(None, description="Optional attached live image base64")
    image_filename: Optional[str] = Field(None, description="Optional attached live image filename")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")

    def get_file_ids(self) -> List[str]:
        """Resolve clean file IDs list."""
        if self.file_ids:
            return [fid.strip() for fid in self.file_ids if fid and fid.strip()]
        return []


class ComparisonResponse(BaseModel):
    """Response model for cross-document comparison."""
    query: str = Field(..., description="Original comparison query")
    comparison_mode: str = Field(..., description="Resolved comparison mode: difference, similarity, metric, conflict, general")
    answer: str = Field(..., description="Comparison analysis with grounded inline citations")
    grouped_evidence: Dict[str, List[Source]] = Field(default_factory=dict, description="Evidence grouped by document name")
    evidence_groups: List[ComparisonEvidenceGroup] = Field(default_factory=list, description="Structured per-document evidence groups")
    sources: List[Source] = Field(default_factory=list, description="Flat ordered list of all cited sources")
    contradictions_detected: List[str] = Field(default_factory=list, description="Contradictions or discrepancies detected between documents")
    numerical_comparison: Optional[Dict[str, Any]] = Field(None, description="Deterministic numeric comparisons where units match")
    session_id: Optional[str] = Field(None, description="Session ID if conversational")
    message_id: Optional[str] = Field(None, description="Assistant message ID if persisted")
    model_used: Optional[str] = Field(None, description="Model used for response generation")
    processing_time_ms: Optional[int] = Field(None, description="Total execution time in ms")
    citation_validation: Optional[Dict[str, Any]] = Field(None, description="Citation validation audit details")

