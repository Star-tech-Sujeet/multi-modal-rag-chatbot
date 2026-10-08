"""
AI Provider Registry.
Resolves and caches provider instances based on the active AI provider setting.
Supports separate providers for text generation and embeddings.
"""

import threading
from typing import Optional, Dict, Any, Tuple, Iterator

from .base import BaseAIProvider, ProviderError
from .gemini import GeminiProvider, GeminiEmbeddings
from .openai_provider import OpenAIProvider
from ..config import settings


# Module-level provider caches (text generation + embeddings are separate)
_provider_cache: Dict[str, BaseAIProvider] = {}
_embedding_provider_cache: Dict[str, BaseAIProvider] = {}
_lock = threading.RLock()


def get_provider(provider: Optional[str] = None) -> BaseAIProvider:
    """
    Return (and cache) the text-generation provider for the given or active provider name.

    Args:
        provider: Optional provider name override ('gemini' or 'openai').

    Returns:
        A BaseAIProvider instance for text generation.
    """
    name = (provider or settings.ai_provider or "gemini").strip().lower()

    # Treat mock as openai for provider resolution
    try:
        from ..logic import _is_openai_mocked
        if provider is None and _is_openai_mocked():
            name = "openai"
    except Exception:
        pass

    if name not in _provider_cache:
        if name == "openai":
            _provider_cache[name] = OpenAIProvider()
        else:
            _provider_cache[name] = GeminiProvider()

    return _provider_cache[name]


def get_embedding_provider(provider: Optional[str] = None) -> BaseAIProvider:
    """
    Return (and cache) the embedding provider.

    The embedding provider defaults to the text-generation provider but can be
    overridden independently via EMBEDDING_PROVIDER (or the `provider` argument).

    Args:
        provider: Optional provider name override. If omitted, uses
                  settings.resolved_embedding_provider (which falls back to ai_provider).

    Returns:
        A BaseAIProvider instance whose get_embeddings() will be used.
    """
    name = (provider or settings.resolved_embedding_provider).strip().lower()

    if name not in _embedding_provider_cache:
        if name == "openai":
            _embedding_provider_cache[name] = OpenAIProvider()
        else:
            _embedding_provider_cache[name] = GeminiProvider()

    return _embedding_provider_cache[name]


def get_collection_name(provider: Optional[str] = None) -> str:
    """
    Return the ChromaDB collection name for the text-generation provider.

    When EMBEDDING_PROVIDER differs from AI_PROVIDER, this still uses the
    text-generation provider's collection name to preserve backward compatibility
    with existing storage. For mixed-provider setups, the embedding provider's
    collection name is appended as a suffix for vector isolation.
    """
    name = (provider or settings.ai_provider or "gemini").strip().lower()
    if name == "openai":
        text_coll = "langchain"
    else:
        import re
        clean_model = re.sub(r'[^a-zA-Z0-9]', '_', settings.gemini_embedding_model)
        text_coll = f"aura_gemini_{clean_model}"

    embedding_name = settings.resolved_embedding_provider
    ai_name = (settings.ai_provider or "gemini").strip().lower()

    if embedding_name != ai_name:
        # Embedding provider differs — use a composite collection name
        if embedding_name == "openai":
            emb_coll = "langchain"
        else:
            import re
            clean_model = re.sub(r'[^a-zA-Z0-9]', '_', settings.gemini_embedding_model)
            emb_coll = f"aura_gemini_{clean_model}"
        return f"{text_coll}__emb__{emb_coll}"

    return text_coll


def get_embeddings(provider: Optional[str] = None):
    """
    Get or create a singleton embeddings instance.

    Uses the resolved embedding provider (which may differ from the text-generation provider).

    Args:
        provider: Optional provider override. If omitted, uses resolved_embedding_provider.

    Returns:
        Embeddings instance conforming to embed_documents / embed_query.
    """
    from ..logic import _embeddings_instance

    if _embeddings_instance is None:
        provider_obj = get_embedding_provider(provider)
        _embeddings_instance = provider_obj.get_embeddings()

    return _embeddings_instance


def reset_provider_cache() -> None:
    """Clear all cached provider instances (for testing or reconfiguration)."""
    global _provider_cache, _embedding_provider_cache
    _provider_cache.clear()
    _embedding_provider_cache.clear()
