"""
Base AI Provider Interface.
Defines the unified abstraction for LLM text generation, embeddings,
multimodal image analysis, and SQL generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Iterator


class ProviderError(Exception):
    """Base exception for AI provider errors."""
    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class ProviderQuotaExceededError(ProviderError):
    """Raised when provider quota (e.g. daily limit, project tier limit) is exhausted."""
    def __init__(
        self,
        message: str,
        provider: str = "gemini",
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message=message, provider=provider, status_code=429, details=details)
        self.model = model


class ProviderRateLimitError(ProviderError):
    """Raised when temporary rate limit (RPM/TPM) is hit."""
    def __init__(
        self,
        message: str,
        provider: str = "gemini",
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message=message, provider=provider, status_code=429, details=details)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """Raised when provider service is temporarily unavailable (e.g. 503 high demand)."""
    def __init__(
        self,
        message: str,
        provider: str = "gemini",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message=message, provider=provider, status_code=503, details=details)


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers (Google Gemini, OpenAI, etc.).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g. 'gemini', 'openai')."""
        pass

    @abstractmethod
    def get_embeddings(self) -> Any:
        """
        Return an embeddings instance conforming to the LangChain / Chroma interface:
        - embed_documents(texts: List[str]) -> List[List[float]]
        - embed_query(text: str) -> List[float]
        """
        pass

    @abstractmethod
    def get_collection_name(self) -> str:
        """
        Return the provider/model-specific ChromaDB collection name
        to guarantee vector dimension isolation between models.
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate grounded text answer using retrieved document evidence.
        Returns: (answer_text, model_used)
        """
        pass

    @abstractmethod
    def generate_text_stream(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[Iterator[str], str]:
        """
        Generate grounded text answer using retrieved document evidence as a token stream.
        Returns: (token_iterator, model_used)
        """
        pass

    @abstractmethod
    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        prompt: Optional[str] = None
    ) -> str:
        """
        Analyze image content and return comprehensive textual description.
        """
        pass

    @abstractmethod
    def generate_suggested_questions(
        self,
        question: str,
        answer: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Generate 3 contextual follow-up questions.
        """
        pass

    @abstractmethod
    def generate_sql(
        self,
        question: str,
        schema_prompt: str,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a constrained read-only SQL query for the given database schema.
        """
        pass
