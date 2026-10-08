"""
Provider Factory and Singleton Access.
Provides access to the active AI provider (Google Gemini or OpenAI).
"""

import logging
from typing import Optional, Dict
from ..config import settings
from .base import BaseAIProvider

logger = logging.getLogger(__name__)

_active_providers: Dict[str, BaseAIProvider] = {}


def get_provider(provider_name: Optional[str] = None, api_key: Optional[str] = None) -> BaseAIProvider:
    """
    Get or create singleton AI provider based on configuration or explicit parameter.
    
    Args:
        provider_name: Optional provider override ('gemini' or 'openai')
        api_key: Optional API key override
        
    Returns:
        BaseAIProvider instance
    """
    global _active_providers
    name = (provider_name or settings.ai_provider or "gemini").strip().lower()

    cache_key = f"{name}_{api_key[:8] if api_key else 'default'}"
    if cache_key not in _active_providers:
        if name == "gemini":
            from .gemini import GeminiProvider
            logger.info("Initializing Google Gemini provider")
            _active_providers[cache_key] = GeminiProvider(api_key=api_key)
        elif name == "openai":
            from .openai_provider import OpenAIProvider
            logger.info("Initializing OpenAI provider")
            _active_providers[cache_key] = OpenAIProvider(api_key=api_key)
        else:
            raise ValueError(f"Unsupported AI provider '{name}'. Must be 'gemini' or 'openai'.")

    return _active_providers[cache_key]


def reset_providers() -> None:
    """Clear cached provider singletons (useful for testing)."""
    global _active_providers
    _active_providers.clear()
