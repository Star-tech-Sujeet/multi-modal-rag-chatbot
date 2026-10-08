"""
Configuration module for the Multi-Modal RAG application.
Loads all configuration from environment variables with fallback to .env.example defaults.
All configurable values are centralized here - no hardcoded values elsewhere.
"""

import os
import logging
from typing import List, Optional, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, AliasChoices
from dotenv import load_dotenv

# Load environment variables from .env file (or .env.example as fallback)
load_dotenv()
if not os.path.exists('.env'):
    load_dotenv('.env.example')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded exclusively from environment variables.
    
    Priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. .env.example file
    4. Default values defined here (lowest priority)
    
    All settings can be overridden via environment variables.
    """
    
    # =========================================================================
    # OpenAI Configuration
    # =========================================================================
    openai_api_key: str = Field(
        default="sk-your-actual-api-key-here",
        description="OpenAI API key for embeddings and chat completions"
    )
    
    openai_model: str = Field(
        default="gpt-4o",
        description="Primary OpenAI chat model for text queries"
    )
    
    openai_mini_model: str = Field(
        default="gpt-4o-mini",
        description="Lightweight OpenAI model for simple tasks"
    )
    
    openai_vision_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for vision/image analysis"
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses"
    )
    openai_max_tokens: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum tokens in LLM response"
    )

    # =========================================================================
    # AI Provider Configuration
    # =========================================================================
    ai_provider: str = Field(
        default="gemini",
        description="Active AI provider for text generation and multimodal tasks: 'gemini' or 'openai'"
    )

    embedding_provider: str = Field(
        default="",
        description="Active embedding provider: 'gemini' or 'openai'. If empty, falls back to ai_provider."
    )

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key"
    )

    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Primary Gemini model"
    )

    gemini_pro_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Advanced Gemini model"
    )

    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model"
    )

    gemini_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Gemini generation temperature"
    )

    gemini_max_tokens: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum Gemini response tokens"
    )

    @field_validator("ai_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("gemini", "openai"):
            raise ValueError(
                "AI_PROVIDER must be either 'gemini' or 'openai'."
            )
        return v

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v and v not in ("gemini", "openai"):
            raise ValueError(
                "EMBEDDING_PROVIDER must be 'gemini', 'openai', or empty (to use ai_provider)."
            )
        return v
    
    # =========================================================================
    # Storage Paths
    # =========================================================================
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        description="Path to ChromaDB persistent storage"
    )
    
    uploads_path: str = Field(
        default="./data/uploads",
        description="Path for uploaded file storage"
    )

    bm25_path: str = Field(
        default="./data/bm25",
        description="Path to BM25 index storage"
    )

    sessions_db_path: str = Field(
        default="./data/sessions.sqlite",
        description="Path to SQLite sessions database"
    )
    
    # =========================================================================
    # API Server Configuration
    # =========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for the API server"
    )
    
    cors_origins: str = Field(
        default="*",
        description="Allowed CORS origins (comma-separated or *)"
    )
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size in MB"
    )
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Text chunk size for document splitting"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between text chunks"
    )
    
    # =========================================================================
    # RAG Configuration
    # =========================================================================
    default_max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of source documents to retrieve"
    )
    
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses"
    )

    rag_vector_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias=AliasChoices("VECTOR_CANDIDATES", "vector_candidates", "rag_vector_candidates"),
        description="Number of candidate chunks to retrieve from vector search"
    )

    rag_bm25_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias=AliasChoices("BM25_CANDIDATES", "bm25_candidates", "rag_bm25_candidates"),
        description="Number of candidate chunks to retrieve from BM25 search"
    )

    rag_rrf_k: int = Field(
        default=60,
        ge=1,
        le=500,
        description="Constant k used in Reciprocal Rank Fusion"
    )

    rag_vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in reciprocal rank fusion"
    )

    rag_bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in reciprocal rank fusion"
    )

    rag_enable_reranking: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_RERANKING", "enable_reranking", "rag_enable_reranking"),
        description="Whether to run the optional reranking layer after fusion"
    )

    rag_rerank_candidate_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of fused candidates to consider during reranking"
    )

    rag_rerank_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight given to reranking lexical score vs fusion score"
    )

    rag_generate_suggested_questions: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAG_GENERATE_SUGGESTED_QUESTIONS", "generate_suggested_questions", "rag_generate_suggested_questions"),
        description="Whether to generate suggested follow-up questions using an extra LLM call"
    )

    rag_history_messages: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum number of previous conversation messages to include for context"
    )

    rag_comparison_sources_per_document: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum source chunks to retrieve per document during cross-document comparison"
    )

    enable_processing_status: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_PROCESSING_STATUS", "enable_processing_status"),
        description="Whether to emit real-time processing status events in SSE streams"
    )

    rag_sql_max_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows returned by structured SQL queries"
    )

    rag_sql_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Execution timeout limit in seconds for SQL queries"
    )

    rag_sql_max_tables: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tables exposed in schema prompts"
    )
    
    # =========================================================================
    # Frontend Configuration
    # =========================================================================
    api_base_url: str = Field(
        default="http://localhost:8000/api/v1",
        description="Backend API URL for frontend"
    )
    
    request_timeout: int = Field(
        default=60,
        ge=1,
        description="HTTP request timeout in seconds"
    )
    
    health_check_timeout: int = Field(
        default=5,
        ge=1,
        description="Health check timeout in seconds"
    )
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for verbose logging"
    )
    
    environment: str = Field(
        default="development",
        description="Application environment (development or production)"
    )
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key when OpenAI is the active provider."""
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower().strip()

        if ai_provider == "gemini":
            return v

        if not v or v == "sk-your-actual-api-key-here" or v.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is not set or is still a placeholder. "
                "Please set a valid OpenAI API key in your .env file."
            )

        if not v.startswith("sk-"):
            logger.warning(
                "OpenAI API key doesn't start with 'sk-'. "
                "This may not be a valid API key format."
            )

        return v

    def require_api_key(self, provider: Optional[str] = None) -> str:
        """
        Validate and return the API key for the requested or active provider.
        
        Args:
            provider: Optional provider name ('gemini' or 'openai').
                      If omitted, defaults to self.ai_provider.
                      
        Returns:
            The valid API key string.
            
        Raises:
            ValueError: If the required API key is missing or a placeholder.
        """
        target_provider = (provider or self.ai_provider or "gemini").lower().strip()

        if target_provider == "gemini":
            key = (self.gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
            if not key or key in ("your_gemini_api_key_here", "your-gemini-api-key-here") or key.startswith("your_") or key.startswith("your-"):
                raise ValueError(
                    "GEMINI_API_KEY is not configured or is a placeholder. "
                    "Please set a valid Gemini API key in your .env file."
                )
            return key

        elif target_provider == "openai":
            key = (self.openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
            if not key or key in ("sk-your-actual-api-key-here", "your_openai_api_key_here", "your-openai-api-key-here") or key.startswith("your_") or key.startswith("your-"):
                raise ValueError(
                    "OPENAI_API_KEY is not configured or is a placeholder. "
                    "Please set a valid OpenAI API key in your .env file."
                )
            return key

        else:
            raise ValueError(f"Unknown AI provider '{target_provider}'. Must be 'gemini' or 'openai'.")

    def is_api_key_valid(self, provider: Optional[str] = None) -> bool:
        """Check whether a valid, non-placeholder API key is configured for the active or requested provider."""
        try:
            self.require_api_key(provider=provider)
            return True
        except (ValueError, Exception):
            return False

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def enable_reranking(self) -> bool:
        """Whether reranking is enabled."""
        return self.rag_enable_reranking

    @property
    def vector_candidates(self) -> int:
        """Configured vector candidates count."""
        return self.rag_vector_candidates

    @property
    def bm25_candidates(self) -> int:
        """Configured BM25 candidates count."""
        return self.rag_bm25_candidates

    @property
    def final_top_k(self) -> int:
        """Configured final top-k candidates count."""
        return self.default_max_sources
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from string to list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_embedding_provider(self) -> str:
        """The provider used for embeddings (falls back to ai_provider if embedding_provider is empty)."""
        return (self.embedding_provider or self.ai_provider or "gemini").strip().lower()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


def is_api_key_valid(settings_instance: Optional[Settings] = None, provider: Optional[str] = None) -> bool:
    """Check if the API key for the given settings and provider is valid."""
    target_settings = settings_instance or settings
    try:
        target_settings.require_api_key(provider=provider)
        return True
    except (ValueError, Exception):
        return False


def require_api_key(settings_or_provider: Optional[Any] = None, provider: Optional[str] = None) -> str:
    """
    Module-level helper to require an API key.
    Can be called as:
      require_api_key() -> uses global settings with active provider
      require_api_key(settings_obj) -> uses settings_obj with its active provider
      require_api_key("gemini") -> uses global settings with provider="gemini"
      require_api_key(provider="gemini") -> uses global settings with provider="gemini"
      require_api_key(settings_obj, provider="openai") -> uses settings_obj with provider="openai"
    """
    if isinstance(settings_or_provider, str) and provider is None:
        return settings.require_api_key(provider=settings_or_provider)
    elif isinstance(settings_or_provider, Settings):
        return settings_or_provider.require_api_key(provider=provider)
    else:
        return settings.require_api_key(provider=provider)


# Initialize global settings instance
try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"AI Provider: {settings.ai_provider}")
    if settings.ai_provider == "gemini":
        logger.info(f"Gemini Model: {settings.gemini_model}")
        logger.info(f"Gemini Embedding Model: {settings.gemini_embedding_model}")
    else:
        logger.info(f"OpenAI Model: {settings.openai_model}")
        logger.info(f"OpenAI Vision Model: {settings.openai_vision_model}")
        logger.info(f"OpenAI Embedding Model: {settings.openai_embedding_model}")
    
    if settings.debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
        
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    raise


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.chroma_db_path,
        settings.uploads_path,
        settings.bm25_path
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


# Create directories on module load
ensure_directories()
