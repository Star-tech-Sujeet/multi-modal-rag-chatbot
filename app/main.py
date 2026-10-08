"""
FastAPI main application entry point.
Enhanced with comprehensive documentation, middleware, and error handling.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .api import router
from .config import settings
from .providers.base import ProviderError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug_mode else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Multi-Modal RAG API...")
    logger.info(f"ChromaDB path: {settings.chroma_db_path}")
    logger.info(f"Uploads path: {settings.uploads_path}")
    logger.info(f"Debug mode: {settings.debug_mode}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Modal RAG API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Multi-Modal RAG API",
    description="""
## 🚀 Multi-Modal Retrieval-Augmented Generation API

A production-ready API for document processing and AI-powered question answering.

### Features

- **📄 Multi-Format Document Processing**: PDF, DOCX, TXT, CSV, Images (OCR), SQLite
- **🔍 Semantic Search**: ChromaDB vector storage with OpenAI embeddings
- **🤖 AI-Powered Answers**: GPT-4o-mini for text, GPT-4o Vision for images
- **📊 Source Attribution**: Track where answers come from

### Quick Start

1. Upload a document via `/api/v1/upload`
2. Query the document via `/api/v1/query`
3. List uploaded files via `/api/v1/files`

### Authentication

Currently, this API does not require authentication. For production use,
implement appropriate security measures.
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Abdullah Al Raiyan",
        "email": "abdullahalraiyan4@gmail.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information."""
    start_time = datetime.utcnow()
    
    # Log request
    logger.debug(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    # Log response
    logger.debug(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Time: {process_time:.2f}ms"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed messages."""
    logger.warning(f"Validation error: {exc.errors()}")
    
    # Format errors for better readability
    formatted_errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": formatted_errors
        }
    )


@app.exception_handler(ProviderError)
async def provider_exception_handler(request: Request, exc: ProviderError):
    """Handle AI Provider specific exceptions (quotas, availability, rate limits)."""
    logger.error(f"Provider exception ({exc.provider}): {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "provider_error",
            "provider": exc.provider,
            "message": exc.message,
            "details": exc.details or None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    if isinstance(exc.detail, dict):
        err_code = exc.detail.get("error", f"http_{exc.status_code}")
        msg = exc.detail.get("message", str(exc.detail))
        details = exc.detail.get("details", exc.detail)
    else:
        err_code = f"http_{exc.status_code}"
        msg = str(exc.detail)
        details = None

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": err_code,
            "message": msg,
            "details": details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "details": str(exc) if settings.debug_mode else None
        }
    )


# =============================================================================
# Routes
# =============================================================================

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["API v1"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information and useful links.
    """
    return {
        "message": "Welcome to the Multi-Modal RAG API",
        "version": "1.1.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "files": "/api/v1/files"
        },
        "status": "healthy"
    }


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
        log_level="debug" if settings.debug_mode else "info"
    )
