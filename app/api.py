"""
FastAPI routes for the Multi-Modal RAG application.
Enhanced with improved error handling, logging, validation, and documentation.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import base64
import json

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .models import (
    UploadResponse, 
    QueryRequest, 
    QueryResponse, 
    HealthResponse,
    FileInfo,
    ErrorResponse,
    DeleteResponse,
    SessionCreateRequest,
    SessionUpdateRequest,
    SessionResponse,
    SessionListResponse,
    SessionQueryRequest,
    SessionQueryResponse,
    ComparisonRequest,
    ComparisonResponse,
    SQLQueryRequest
)
from .logic import (
    process_document, 
    create_and_store_embeddings, 
    perform_rag_query, 
    perform_rag_query_stream,
    perform_session_rag_query,
    perform_session_rag_query_stream,
    perform_comparison_query,
    perform_sql_query,
    get_vectorstore_stats,
    delete_document_embeddings,
    delete_document_complete,
    bm25_manager
)
from .sql_engine import SQLValidationError
from .sessions import session_manager
from .config import settings
from .providers.base import ProviderError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants - Load from settings
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.png', '.jpg', '.jpeg', '.db'}


def validate_file_extension(filename: str) -> str:
    """
    Validate file extension and return it if valid.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        File extension in lowercase
        
    Raises:
        HTTPException: If file type is not supported
    """
    file_extension = Path(filename).suffix.lower()
    
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file upload with unsupported extension: {file_extension}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_file_type",
                "message": f"Unsupported file type: {file_extension}",
                "supported_types": list(ALLOWED_EXTENSIONS)
            }
        )
    
    return file_extension


def validate_file_size(file_size: int, filename: str) -> None:
    """
    Validate file size is within limits.
    
    Args:
        file_size: Size of file in bytes
        filename: Name of the file
        
    Raises:
        HTTPException: If file is too large
    """
    if file_size > settings.max_file_size_bytes:
        logger.warning(f"Rejected file upload - too large: {filename} ({file_size} bytes)")
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed ({settings.max_file_size_bytes / 1024 / 1024}MB)",
                "max_size_bytes": settings.max_file_size_bytes
            }
        )


def validate_uuid(val: str, field_name: str = "ID") -> str:
    """Validate string is a valid UUID, returning clean string or raising 400."""
    clean_val = str(val).strip() if val else ""
    if not clean_val:
        raise HTTPException(
            status_code=400,
            detail={"error": f"invalid_{field_name.lower().replace(' ', '_')}", "message": f"{field_name} cannot be empty"}
        )
    try:
        uuid.UUID(clean_val)
        return clean_val
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail={"error": f"invalid_{field_name.lower().replace(' ', '_')}", "message": f"Invalid {field_name} format: '{val}'"}
        )


@router.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Root endpoint returning API information.
    
    Returns:
        HealthResponse with welcome message and healthy status
    """
    return HealthResponse(
        message="Welcome to the Multi-Modal RAG API",
        status="healthy"
    )


@router.get("/health", response_model=dict, tags=["Health"])
async def health_check():
    """
    Detailed health check with vectorstore statistics.
    
    Returns comprehensive health information including:
    - API status
    - Vectorstore connection status
    - Document count
    - System information
    """
    try:
        stats = await run_in_threadpool(get_vectorstore_stats)
        api_key_status = "configured" if settings.is_api_key_valid() else "missing_or_placeholder"
        
        return {
            "status": "healthy" if stats.get("status") == "healthy" else "degraded",
            "message": "Multi-Modal RAG API is running",
            "timestamp": datetime.utcnow().isoformat(),
            "vectorstore": stats,
            "ai_provider": settings.ai_provider,
            "model": settings.gemini_model if settings.ai_provider == "gemini" else settings.openai_model,
            "embedding_model": settings.gemini_embedding_model if settings.ai_provider == "gemini" else settings.openai_embedding_model,
            "api_key_status": api_key_status,
            "openai_api_key": api_key_status,
            "version": "1.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "message": f"API running but vectorstore unavailable: {str(e)}",
            "timestamp": datetime.utcnow().isoformat(),
            "vectorstore": {"status": "error", "total_documents": 0}
        }


@router.post(
    "/upload", 
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Documents"]
)
async def upload_file(file: UploadFile = File(..., description="Document file to upload and process")):
    """
    Upload and process a document file.
    
    Supports multiple file formats including:
    - **PDF** (.pdf) - Portable Document Format
    - **Word** (.docx) - Microsoft Word documents
    - **Text** (.txt) - Plain text files
    - **CSV** (.csv) - Comma-separated values
    - **Images** (.png, .jpg, .jpeg) - Images processed with OCR
    - **Database** (.db) - SQLite database files
    
    Returns:
        UploadResponse with file_id, filename, and processing message
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    # Validate file extension
    file_extension = validate_file_extension(file.filename)
    
    # Read file content asynchronously
    try:
        file_content = await file.read()
        file_size = len(file_content)
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "read_error", "message": f"Failed to read file: {str(e)}"}
        )
    
    # Validate file size
    validate_file_size(file_size, file.filename)
    
    # Check for empty file
    if file_size == 0:
        logger.warning(f"Rejected empty file: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "The uploaded file is empty"}
        )
    
    # Generate unique file ID and path with strict path traversal protection
    file_id = str(uuid.uuid4())
    raw_name = os.path.basename((file.filename or "upload").replace('\\', '/'))
    safe_filename = "".join(c for c in raw_name if c.isalnum() or c in '.-_').strip('.')
    if not safe_filename:
        safe_filename = f"upload{file_extension}"
    
    uploads_dir = Path(settings.uploads_path).resolve()
    resolved_file_path = (uploads_dir / f"{file_id}_{safe_filename}").resolve()
    if not str(resolved_file_path).startswith(str(uploads_dir)):
        logger.warning(f"Path traversal detected in upload filename: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_filename", "message": "Path traversal detected in filename"}
        )
    file_path = str(resolved_file_path)
    
    def _write_to_disk(path: str, data: bytes):
        with open(path, "wb") as buffer:
            buffer.write(data)

    try:
        # Offload file save to threadpool
        await run_in_threadpool(_write_to_disk, file_path, file_content)
        logger.info(f"File saved: {file_path} ({file_size} bytes)")
        
        # Process document offloaded to threadpool
        try:
            chunks = await run_in_threadpool(process_document, file_path, file.filename)
            
            if not chunks:
                logger.warning(f"No content extracted from file: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "no_content",
                        "message": "No content could be extracted from the file. The file may be empty, corrupted, or in an unsupported format."
                    }
                )
            
            # Create and store embeddings (offloaded to threadpool)
            await run_in_threadpool(create_and_store_embeddings, chunks, file_id)
            
            logger.info(f"Successfully processed file {file.filename}: {len(chunks)} chunks created")
            
            return UploadResponse(
                file_id=file_id,
                filename=safe_filename,
                message=f"File processed successfully. {len(chunks)} chunks created and indexed.",
                chunks_created=len(chunks),
                file_size=file_size
            )
            
        except HTTPException:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            try:
                delete_document_embeddings(file_id)
            except Exception:
                pass
            raise
        except ValueError as ve:
            logger.error(f"Validation error processing file {file.filename}: {str(ve)}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            try:
                delete_document_embeddings(file_id)
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "validation_error",
                    "message": str(ve)
                }
            )
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            try:
                delete_document_embeddings(file_id)
            except Exception:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "processing_error",
                    "message": f"Error processing file: {str(e)}"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        if 'file_id' in locals():
            try:
                delete_document_embeddings(file_id)
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upload_error",
                "message": f"Unexpected error during file upload: {str(e)}"
            }
        )


@router.post(
    "/query", 
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Query processing error"}
    },
    tags=["Query"]
)
async def query_api(request: QueryRequest):
    """
    Query the RAG system with a text question.
    
    This endpoint performs semantic and keyword search on the indexed document(s)
    and generates an AI-powered response based on the relevant context.
    
    Supports single-document and multi-document modes with hybrid search and model routing.
    All retrieval and inference tasks are offloaded to run_in_threadpool to keep the event loop non-blocking.
    """
    file_ids = request.get_file_ids()
    
    logger.info(f"Received query for file_id: {request.file_id}, file_ids: {request.file_ids}, resolved: {file_ids}")
    
    # Validate question
    if not request.question or not request.question.strip():
        logger.warning("Received empty question")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_question",
                "message": "Question cannot be empty"
            }
        )
    
    # Validate that we have at least one file_id
    if not file_ids:
        logger.warning("Received empty file_id(s)")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_file_id",
                "message": "File ID(s) cannot be empty. Please upload or select document(s) first."
            }
        )
    
    # Validate file_id format (should be UUID) for all file_ids
    for fid in file_ids:
        if not fid or not fid.strip():
            continue
        try:
            uuid.UUID(fid)
        except ValueError:
            logger.warning(f"Invalid file_id format: {fid}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_file_id",
                    "message": f"Invalid file ID format: {fid}"
                }
            )
    
    try:
        # Offload CPU & I/O heavy retrieval and inference to threadpool
        response = await run_in_threadpool(perform_rag_query, request)
        
        logger.info(f"Query successful for {len(file_ids)} file(s), sources found: {len(response.sources)}")
        return response
        
    except ProviderError as pe:
        logger.error(f"AI Provider error ({pe.provider}): {pe.message}")
        raise HTTPException(
            status_code=pe.status_code,
            detail={
                "error": "provider_error",
                "provider": pe.provider,
                "message": pe.message,
                "details": pe.details
            }
        )
    except ValueError as ve:
        logger.error(f"Configuration or validation error during query: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "configuration_error",
                "message": str(ve)
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "query_error",
                "message": f"Error processing query: {str(e)}"
            }
        )


@router.post(
    "/query/stream",
    tags=["Query"]
)
async def query_stream_api(request: QueryRequest):
    """
    Query the RAG system with REAL token streaming via Server-Sent Events (SSE).
    Emits events: start, sources, token, complete, error.
    """
    file_ids = request.get_file_ids()
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_question", "message": "Question cannot be empty"}
        )
    if not file_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file_id", "message": "File ID(s) cannot be empty. Please upload or select document(s) first."}
        )
    for fid in file_ids:
        if not fid or not fid.strip():
            continue
        try:
            uuid.UUID(fid)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_file_id", "message": f"Invalid file ID format: {fid}"}
            )

    generator = perform_rag_query_stream(request)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get(
    "/files", 
    response_model=List[FileInfo],
    tags=["Documents"]
)
async def list_uploaded_files(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of files to return"),
    offset: int = Query(default=0, ge=0, description="Number of files to skip")
):
    """
    List all uploaded files with pagination support.
    Offloaded to run_in_threadpool for non-blocking file-system inspection.
    """
    def _sync_list_files(lim: int, off: int) -> List[FileInfo]:
        files = []
        uploads_dir = Path(settings.uploads_path)
        
        if uploads_dir.exists():
            all_files = sorted(
                uploads_dir.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            paginated_files = list(all_files)[off:off + lim]
            
            for file_path in paginated_files:
                if file_path.is_file():
                    filename = file_path.name
                    if '_' in filename:
                        file_id = filename.split('_')[0]
                        original_name = '_'.join(filename.split('_')[1:])
                        
                        stat = file_path.stat()
                        ftype = Path(original_name).suffix.lstrip('.').upper() or 'DOC'
                        chunks_cnt = None
                        if file_id in bm25_manager._cache:
                            chunks_cnt = len(bm25_manager._cache[file_id].chunk_ids)
                        else:
                            bm25_f = Path(settings.bm25_path) / f"{file_id}.json"
                            if bm25_f.exists():
                                try:
                                    with open(bm25_f, "r", encoding="utf-8") as jf:
                                        jdata = json.load(jf)
                                        chunks_cnt = len(jdata.get("chunks", []))
                                except Exception:
                                    pass

                        files.append(FileInfo(
                            file_id=file_id,
                            filename=original_name,
                            upload_path=str(file_path),
                            size_bytes=stat.st_size,
                            uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            chunks_count=chunks_cnt,
                            file_type=ftype
                        ))
        return files

    try:
        files = await run_in_threadpool(_sync_list_files, limit, offset)
        logger.info(f"Listed {len(files)} files (offset: {offset}, limit: {limit})")
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "list_error",
                "message": f"Error listing files: {str(e)}"
            }
        )


@router.delete(
    "/files/{file_id}",
    response_model=dict,
    tags=["Documents"]
)
async def delete_file(file_id: str):
    """
    Delete an uploaded file, its ChromaDB embeddings, and its BM25 index.
    Offloaded to run_in_threadpool for non-blocking execution.
    """
    logger.info(f"Received delete request for file_id: {file_id}")
    
    clean_id = str(file_id).strip() if file_id else ""
    try:
        uuid.UUID(clean_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_file_id", "message": f"Invalid file ID format: '{file_id}'"}
        )
    
    def _sync_delete_file(fid: str):
        return delete_document_complete(fid)

    try:
        delete_result = await run_in_threadpool(_sync_delete_file, clean_id)
        if isinstance(delete_result, dict):
            return delete_result
        elif delete_result:
            return {
                "message": f"Document {clean_id} deleted successfully",
                "file_id": clean_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"File with ID {clean_id} not found"}
            )
    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": str(fnf)}
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_file_id", "message": str(ve)}
        )
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "delete_error", "message": f"Error deleting file: {str(e)}"}
        )


@router.delete(
    "/documents/{document_id}",
    response_model=dict,
    responses={
        200: {"model": DeleteResponse, "description": "Document deleted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid document ID format"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Document deletion error"}
    },
    tags=["Documents"]
)
async def delete_document(document_id: str):
    """
    Safely and completely delete an indexed document by document_id:
    - Removes original uploaded file from data/uploads
    - Removes ChromaDB vector embeddings across all collections
    - Removes BM25 keyword index file and memory cache
    - Invalidates vectorstore and retrieval caches
    """
    return await delete_file(file_id=document_id)


# =============================================================================
# Session Endpoints (Phase 6: Conversational RAG & Persistent Sessions)
# =============================================================================

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=201,
    tags=["Sessions"]
)
async def create_new_session(body: Optional[SessionCreateRequest] = None):
    """
    Create a new persistent conversation session.
    Offloaded to run_in_threadpool for non-blocking SQLite access.
    """
    try:
        title = body.title if body else None
        metadata = body.metadata if body else None
        session = await run_in_threadpool(session_manager.create_session, title=title, metadata=metadata)
        return session
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "session_create_error", "message": str(e)}
        )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    tags=["Sessions"]
)
async def list_all_sessions(
    limit: int = Query(50, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List persistent conversation sessions ordered by most recently updated.
    Offloaded to run_in_threadpool for non-blocking SQLite access.
    """
    try:
        result = await run_in_threadpool(session_manager.list_sessions, limit=limit, offset=offset)
        return result
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "session_list_error", "message": str(e)}
        )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"]
)
async def get_session_details(session_id: str):
    """
    Retrieve session metadata and chronological message history.
    Offloaded to run_in_threadpool for non-blocking SQLite access.
    """
    valid_sid = validate_uuid(session_id, "session_id")
    try:
        session = await run_in_threadpool(session_manager.get_session, valid_sid)
        if not session:
            raise HTTPException(
                status_code=404,
                detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
            )
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "session_get_error", "message": str(e)}
        )


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"]
)
@router.put(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"]
)
async def update_session_details(session_id: str, body: SessionUpdateRequest):
    """
    Update session title or metadata.
    Offloaded to run_in_threadpool for non-blocking SQLite access.
    """
    valid_sid = validate_uuid(session_id, "session_id")
    try:
        updated = await run_in_threadpool(
            session_manager.update_session,
            session_id=valid_sid,
            title=body.title,
            metadata=body.metadata
        )
        if not updated:
            raise HTTPException(
                status_code=404,
                detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
            )
        return updated
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "session_update_error", "message": str(e)}
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=dict,
    tags=["Sessions"]
)
async def delete_session_endpoint(session_id: str):
    """
    Delete a conversation session and all its messages (cascading).
    Offloaded to run_in_threadpool for non-blocking SQLite access.
    """
    valid_sid = validate_uuid(session_id, "session_id")
    try:
        deleted = await run_in_threadpool(session_manager.delete_session, valid_sid)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
            )
        return {
            "message": f"Session '{valid_sid}' deleted successfully",
            "session_id": valid_sid
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "session_delete_error", "message": str(e)}
        )


@router.post(
    "/sessions/{session_id}/query",
    response_model=SessionQueryResponse,
    tags=["Sessions"]
)
async def query_session_endpoint(
    session_id: str,
    request: Request
):
    """
    Submit a user question within an active session, with optional live image attachment.
    Supports both application/json (with optional image_base64) and multipart/form-data (with optional image file).
    Executes conversational query contextualization, candidate retrieval, live image processing
    (OCR-first with Vision fallback), answer generation, citation validation, and persistent storage.
    Offloaded to run_in_threadpool for non-blocking execution.
    """
    valid_sid = validate_uuid(session_id, "session_id")
    existing = await run_in_threadpool(session_manager.get_session, valid_sid)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
        )

    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None
    image_filename = None

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as form_err:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": f"Malformed multipart form data: {form_err}"}
            )

        question = form.get("question")
        if not question or not str(question).strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Question cannot be empty"}
            )

        file_id = form.get("file_id")
        file_ids_val = form.get("file_ids")
        file_ids_list = None
        if file_ids_val:
            if isinstance(file_ids_val, str):
                try:
                    parsed = json.loads(file_ids_val)
                    file_ids_list = parsed if isinstance(parsed, list) else [file_ids_val]
                except Exception:
                    file_ids_list = [fid.strip() for fid in file_ids_val.split(",") if fid.strip()]
            elif isinstance(file_ids_val, list):
                file_ids_list = file_ids_val

        max_sources_val = form.get("max_sources")
        max_sources = int(max_sources_val) if max_sources_val else 5

        use_hybrid_val = form.get("use_hybrid_search")
        use_hybrid = use_hybrid_val.lower() in ("true", "1", "yes") if isinstance(use_hybrid_val, str) else True

        model_preference = form.get("model_preference") or "mini"

        img_upload = form.get("image")
        if img_upload and hasattr(img_upload, "read"):
            image_bytes = await img_upload.read()
            image_filename = getattr(img_upload, "filename", "attached_image.png")

        query_req = SessionQueryRequest(
            question=str(question),
            file_id=str(file_id) if file_id else None,
            file_ids=file_ids_list,
            max_sources=max_sources,
            use_hybrid_search=use_hybrid,
            model_preference=str(model_preference)
        )
    else:
        try:
            body_dict = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Invalid JSON body"}
            )
        try:
            query_req = SessionQueryRequest(**body_dict)
        except Exception as pydantic_err:
            raise HTTPException(
                status_code=422,
                detail={"error": "validation_error", "message": str(pydantic_err)}
            )
        if query_req.image_base64:
            try:
                image_bytes = base64.b64decode(query_req.image_base64)
                image_filename = query_req.image_filename or "attached_image.png"
            except Exception as b64_err:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "bad_request", "message": f"Invalid image base64 encoding: {b64_err}"}
                )

    try:
        response = await run_in_threadpool(
            perform_session_rag_query,
            session_id=session_id,
            query_request=query_req,
            mgr=session_manager,
            image_bytes=image_bytes,
            image_filename=image_filename
        )
        return response
    except ProviderError as pe:
        logger.error(f"AI Provider error in session query ({pe.provider}): {pe.message}")
        raise HTTPException(
            status_code=pe.status_code,
            detail={
                "error": "provider_error",
                "provider": pe.provider,
                "message": pe.message,
                "details": pe.details
            }
        )
    except ValueError as ve:
        logger.warning(f"Validation error in session query: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": str(ve)}
        )
    except Exception as e:
        logger.error(f"Error performing session query for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "query_error", "message": str(e)}
        )


@router.post(
    "/sessions/{session_id}/query/stream",
    tags=["Sessions"]
)
async def query_session_stream_endpoint(
    session_id: str,
    request: Request
):
    """
    Submit a user question within an active session with REAL token streaming via Server-Sent Events (SSE).
    Supports both application/json (with optional image_base64) and multipart/form-data (with optional image file).
    Emits structured events:
    - start: {"session_id": str, "model": str}
    - sources: {"sources": List[dict], "search_method": str}
    - token: {"text": str}
    - complete: {"session_id": str, "message_id": str, "answer": str, "sources": List[dict], "citation_validation": dict, "suggested_questions": List[str], "search_method": str, "model_used": str, "timings": dict, "processing_time_ms": int}
    - error: {"error": str, "details": dict}
    """
    valid_sid = validate_uuid(session_id, "session_id")
    existing = await run_in_threadpool(session_manager.get_session, valid_sid)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
        )

    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None
    image_filename = None

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as form_err:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": f"Malformed multipart form data: {form_err}"}
            )

        question = form.get("question")
        if not question or not str(question).strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Question cannot be empty"}
            )

        file_id = form.get("file_id")
        file_ids_val = form.get("file_ids")
        file_ids_list = None
        if file_ids_val:
            if isinstance(file_ids_val, str):
                try:
                    parsed = json.loads(file_ids_val)
                    file_ids_list = parsed if isinstance(parsed, list) else [file_ids_val]
                except Exception:
                    file_ids_list = [fid.strip() for fid in file_ids_val.split(",") if fid.strip()]
            elif isinstance(file_ids_val, list):
                file_ids_list = file_ids_val

        max_sources_val = form.get("max_sources")
        max_sources = int(max_sources_val) if max_sources_val else 5

        use_hybrid_val = form.get("use_hybrid_search")
        use_hybrid = use_hybrid_val.lower() in ("true", "1", "yes") if isinstance(use_hybrid_val, str) else True

        model_preference = form.get("model_preference") or "mini"

        img_upload = form.get("image")
        if img_upload and hasattr(img_upload, "read"):
            image_bytes = await img_upload.read()
            image_filename = getattr(img_upload, "filename", "attached_image.png")

        query_req = SessionQueryRequest(
            question=str(question),
            file_id=str(file_id) if file_id else None,
            file_ids=file_ids_list,
            max_sources=max_sources,
            use_hybrid_search=use_hybrid,
            model_preference=str(model_preference)
        )
    else:
        try:
            body_dict = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Invalid JSON body"}
            )
        try:
            query_req = SessionQueryRequest(**body_dict)
        except Exception as pydantic_err:
            raise HTTPException(
                status_code=422,
                detail={"error": "validation_error", "message": str(pydantic_err)}
            )
        if query_req.image_base64:
            try:
                image_bytes = base64.b64decode(query_req.image_base64)
                image_filename = query_req.image_filename or "attached_image.png"
            except Exception as b64_err:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "bad_request", "message": f"Invalid image base64 encoding: {b64_err}"}
                )

    generator = perform_session_rag_query_stream(
        session_id=valid_sid,
        query_request=query_req,
        mgr=session_manager,
        image_bytes=image_bytes,
        image_filename=image_filename
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# Phase 8: Cross-Document Comparison Endpoints
# =============================================================================

@router.post(
    "/compare",
    response_model=ComparisonResponse,
    tags=["Comparison"]
)
async def compare_documents_endpoint(
    request: Request
):
    """
    Execute evidence-grounded cross-document comparison with isolated per-document retrieval.
    Supports both application/json and multipart/form-data (with optional live image).
    """
    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None
    image_filename = None

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as form_err:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": f"Malformed multipart form data: {form_err}"}
            )

        query = form.get("query") or form.get("question")
        if not query or not str(query).strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Comparison query cannot be empty"}
            )

        file_ids_val = form.get("file_ids")
        file_ids_list = None
        if file_ids_val:
            if isinstance(file_ids_val, str):
                try:
                    parsed = json.loads(file_ids_val)
                    file_ids_list = parsed if isinstance(parsed, list) else [file_ids_val]
                except Exception:
                    file_ids_list = [fid.strip() for fid in file_ids_val.split(",") if fid.strip()]
            elif isinstance(file_ids_val, list):
                file_ids_list = file_ids_val

        session_id = form.get("session_id")
        comparison_mode = form.get("comparison_mode")
        max_sources_val = form.get("max_sources_per_document")
        max_sources = int(max_sources_val) if max_sources_val else None
        use_hybrid_val = form.get("use_hybrid_search")
        use_hybrid = use_hybrid_val.lower() in ("true", "1", "yes") if isinstance(use_hybrid_val, str) else True
        model_preference = form.get("model_preference") or "mini"

        img_upload = form.get("image")
        if img_upload and hasattr(img_upload, "read"):
            image_bytes = await img_upload.read()
            image_filename = getattr(img_upload, "filename", "attached_image.png")

        comp_req = ComparisonRequest(
            query=str(query),
            file_ids=file_ids_list,
            session_id=str(session_id) if session_id else None,
            comparison_mode=str(comparison_mode) if comparison_mode else None,
            max_sources_per_document=max_sources,
            use_hybrid_search=use_hybrid,
            model_preference=str(model_preference)
        )
    else:
        try:
            body_dict = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"error": "bad_request", "message": "Invalid JSON body"}
            )
        try:
            comp_req = ComparisonRequest(**body_dict)
        except Exception as pydantic_err:
            raise HTTPException(
                status_code=422,
                detail={"error": "validation_error", "message": str(pydantic_err)}
            )
        if comp_req.image_base64:
            try:
                image_bytes = base64.b64decode(comp_req.image_base64)
                image_filename = comp_req.image_filename or "attached_image.png"
            except Exception as b64_err:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "bad_request", "message": f"Invalid image base64 encoding: {b64_err}"}
                )

    try:
        response = await run_in_threadpool(
            perform_comparison_query,
            request=comp_req,
            mgr=session_manager,
            image_bytes=image_bytes,
            image_filename=image_filename
        )
        return response
    except ProviderError as pe:
        logger.error(f"AI Provider error in comparison query ({pe.provider}): {pe.message}")
        raise HTTPException(
            status_code=pe.status_code,
            detail={
                "error": "provider_error",
                "provider": pe.provider,
                "message": pe.message,
                "details": pe.details
            }
        )
    except ValueError as ve:
        logger.warning(f"Validation error in comparison query: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": str(ve)}
        )
    except Exception as e:
        logger.error(f"Error performing comparison query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "comparison_error", "message": str(e)}
        )


@router.post(
    "/sessions/{session_id}/compare",
    response_model=ComparisonResponse,
    tags=["Comparison"]
)
async def compare_session_documents_endpoint(
    session_id: str,
    request: Request
):
    """
    Execute evidence-grounded cross-document comparison within an active conversation session.
    """
    valid_sid = validate_uuid(session_id, "session_id")
    existing = await run_in_threadpool(session_manager.get_session, valid_sid)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "message": f"Session '{valid_sid}' not found"}
        )
    return await compare_documents_endpoint(request)


@router.post(
    "/sql-query",
    response_model=QueryResponse,
    tags=["Structured Data"]
)
async def execute_sql_query_endpoint(request: SQLQueryRequest):
    """
    Execute a natural language analytical query against an SQLite database or CSV file.
    Uses safe, read-only SQL generation with strict validation and citation grounding.
    """
    validate_uuid(request.file_id, "file_id")
    if request.session_id:
        validate_uuid(request.session_id, "session_id")

    try:
        response = await run_in_threadpool(
            perform_sql_query,
            question=request.question,
            file_id=request.file_id,
            session_id=request.session_id,
            model_preference=request.model_preference
        )
        return response
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": str(fnf)}
        )
    except (ValueError, SQLValidationError) as ve:
        logger.warning(f"SQL validation or parameter error: {ve}")
        raise HTTPException(
            status_code=400,
            detail={"error": "sql_validation_error", "message": str(ve)}
        )
    except TimeoutError as te:
        logger.warning(f"SQL execution timeout: {te}")
        raise HTTPException(
            status_code=408,
            detail={"error": "sql_timeout", "message": str(te)}
        )
    except ProviderError as pe:
        logger.error(f"AI Provider error in SQL query ({pe.provider}): {pe.message}")
        raise HTTPException(
            status_code=pe.status_code,
            detail={
                "error": "provider_error",
                "provider": pe.provider,
                "message": pe.message,
                "details": pe.details
            }
        )
    except Exception as e:
        logger.error(f"Structured SQL query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "sql_execution_error", "message": "An error occurred executing the structured query."}
        )

