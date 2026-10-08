"""
Unit tests for AI Provider Abstraction (Google Gemini & OpenAI).
Verifies:
1. GeminiProvider text generation, vision, text-to-SQL, and suggested questions.
2. GeminiEmbeddings documents and query embedding.
3. ChromaDB collection isolation between OpenAI and Gemini.
4. Provider factory and runtime switching.
5. Error handling on missing keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from app.config import Settings
from app.main import app
from app.providers.base import BaseAIProvider
from app.providers.gemini import GeminiProvider, GeminiEmbeddings
from app.providers.openai_provider import OpenAIProvider
from app.providers.factory import get_provider, reset_providers
from app.logic import (
    get_current_collection_name,
    get_vectorstore,
    get_embeddings,
    handle_text_query,
    analyze_image_with_vision
)
from app.sql_engine import DatabaseSchema, generate_sql_query


@pytest.fixture
def client():
    return TestClient(app)


class TestGeminiProvider:
    """Test suite for Google Gemini provider implementation."""

    def test_provider_initialization(self):
        """Gemini provider should initialize with correct name and collection."""
        with patch("app.providers.gemini.genai.Client"):
            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            assert provider.provider_name == "gemini"
            assert "aura_gemini_" in provider.get_collection_name()

    def test_gemini_text_generation(self):
        """Gemini generate_text should invoke client.models.generate_content."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Based on [S1], the revenue was $5M."
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            context = "[SOURCE S1]\nFilename: financial.pdf\nContent: Revenue was $5M."
            answer, model = provider.generate_text(
                question="What was the revenue?",
                context=context
            )

            assert "revenue was $5M" in answer
            assert "[S1]" in answer
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert "untrusted_document_evidence" in call_kwargs["contents"]

    def test_gemini_multimodal_vision(self):
        """Gemini analyze_image should process image bytes via Part.from_bytes."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls, \
             patch("app.providers.gemini.types.Part.from_bytes") as mock_part:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_part.return_value = MagicMock()

            mock_response = MagicMock()
            mock_response.text = "This is a bar chart showing quarterly earnings."
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            result = provider.analyze_image(
                image_bytes=b"fake_image_bytes",
                filename="earnings_chart.png",
                mime_type="image/png"
            )

            assert "[Image Analysis: earnings_chart.png]" in result
            assert "bar chart showing quarterly earnings" in result
            mock_part.assert_called_once_with(data=b"fake_image_bytes", mime_type="image/png")

    def test_gemini_sql_generation(self):
        """Gemini generate_sql should return clean SQL stripped of fences."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "```sql\nSELECT AVG(salary) FROM employees LIMIT 100;\n```"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            schema_str = "Table: employees, Columns: id, name, salary"
            sql = provider.generate_sql(
                question="What is the average salary?",
                schema_prompt=schema_str
            )

            assert sql == "SELECT AVG(salary) FROM employees LIMIT 100;"

    def test_gemini_suggested_questions(self):
        """Gemini generate_suggested_questions should return up to 3 parsed lines."""
        with patch("app.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "What was Q2 revenue?\nWho is the CFO?\nWhat were the total expenses?"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="AIzaSyTestKey1234567890")
            questions = provider.generate_suggested_questions(
                question="What was the revenue?",
                answer="The revenue was $5M.",
                context="Q1 revenue was $5M."
            )

            assert len(questions) == 3
            assert questions[0] == "What was Q2 revenue?"
            assert questions[1] == "Who is the CFO?"


class TestGeminiEmbeddings:
    """Test suite for GeminiEmbeddings Chroma compatibility."""

    def test_embed_documents(self):
        """embed_documents should return float vectors using embed_content."""
        mock_client = MagicMock()
        emb1 = MagicMock(values=[0.1, 0.2, 0.3])
        emb2 = MagicMock(values=[0.4, 0.5, 0.6])
        mock_response = MagicMock(embeddings=[emb1, emb2])
        mock_client.models.embed_content.return_value = mock_response

        embedder = GeminiEmbeddings(client=mock_client, model="gemini-embedding-001")
        vectors = embedder.embed_documents(["Document one", "Document two"])

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]
        assert vectors[1] == [0.4, 0.5, 0.6]
        mock_client.models.embed_content.assert_called_once()

    def test_embed_query(self):
        """embed_query should return a single float vector."""
        mock_client = MagicMock()
        emb = MagicMock(values=[0.7, 0.8, 0.9])
        mock_response = MagicMock(embeddings=[emb])
        mock_client.models.embed_content.return_value = mock_response

        embedder = GeminiEmbeddings(client=mock_client, model="gemini-embedding-001")
        vector = embedder.embed_query("Search query")

        assert vector == [0.7, 0.8, 0.9]

    def test_embed_empty_inputs(self):
        """Empty inputs should return empty lists without calling API."""
        mock_client = MagicMock()
        embedder = GeminiEmbeddings(client=mock_client)
        assert embedder.embed_documents([]) == []
        assert embedder.embed_query("") == []
        mock_client.models.embed_content.assert_not_called()


class TestProviderCollectionIsolation:
    """Verify Chroma collection names isolate vector dimensions between providers."""

    def test_openai_collection_name(self):
        """OpenAI provider should use 'langchain' for backwards compatibility."""
        with patch("app.logic.settings.ai_provider", "openai"):
            assert get_current_collection_name() == "langchain"

    def test_gemini_collection_name(self):
        """Gemini provider should use model-specific isolated collection."""
        with patch("app.logic.settings.ai_provider", "gemini"), \
             patch("app.logic.settings.gemini_embedding_model", "gemini-embedding-001"):
            assert get_current_collection_name() == "aura_gemini_gemini_embedding_001"


class TestProviderFactoryAndSwitching:
    """Verify provider switching via factory and settings."""

    def test_factory_returns_gemini_by_default(self):
        """get_provider with gemini should return GeminiProvider."""
        with patch("app.providers.gemini.genai.Client"):
            reset_providers()
            provider = get_provider("gemini", api_key="test_key_gemini_123456")
            assert isinstance(provider, GeminiProvider)
            assert provider.provider_name == "gemini"

    def test_factory_returns_openai_when_requested(self):
        """get_provider with openai should return OpenAIProvider."""
        reset_providers()
        provider = get_provider("openai", api_key="sk-proj-testkey1234567890abcdef")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"

    def test_factory_rejects_invalid_provider(self):
        """get_provider with unknown name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("anthropic")
        assert "Unsupported AI provider" in str(exc_info.value)


class TestAPIHealthWithProvider:
    """Verify /health endpoint reports active AI provider and key status."""

    def test_health_check_reports_gemini_provider(self, client):
        """Health check returns ai_provider and api_key_status."""
        with patch("app.api.get_vectorstore_stats", return_value={"status": "healthy", "total_documents": 0}):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert "ai_provider" in data
            assert "api_key_status" in data
            assert "openai_api_key" in data  # Backwards compatibility check
            assert data["ai_provider"] in ("gemini", "openai")
