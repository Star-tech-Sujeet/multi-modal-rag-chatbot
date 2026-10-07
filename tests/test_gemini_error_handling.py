"""
Tests for Gemini Model Routing, Logging Inconsistency Fix,
and Robust Error Handling (429 Quota Exhaustion, 503 Bounded Retry).
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.config import settings
from app.logic import resolve_model, handle_text_query
from app.providers.gemini import (
    GeminiProvider,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiServiceUnavailableError,
    GeminiAPIError
)
from app.providers.base import ProviderQuotaExceededError, ProviderUnavailableError
from app.main import app


def test_resolve_model_gemini_defaults():
    with patch.object(settings, 'ai_provider', 'gemini'):
        model = resolve_model(provider='gemini')
        assert model == settings.gemini_model
        assert not model.startswith('gpt-')


def test_resolve_model_gemini_complex_query():
    with patch.object(settings, 'ai_provider', 'gemini'):
        complex_q = 'Please critically evaluate and compare and contrast the two quarterly reports.'
        model = resolve_model(question=complex_q, provider='gemini')
        assert model == settings.gemini_pro_model


def test_resolve_model_gemini_normalizes_legacy_openai_preference():
    with patch.object(settings, 'ai_provider', 'gemini'):
        model = resolve_model(model_preference='gpt-4o-mini', provider='gemini')
        assert model == settings.gemini_model
        assert not model.startswith('gpt-')


def test_resolve_model_gemini_preserves_explicit_gemini_preference():
    with patch.object(settings, 'ai_provider', 'gemini'):
        model = resolve_model(model_preference='gemini-3.1-flash-lite', provider='gemini')
        assert model == 'gemini-3.1-flash-lite'


def test_resolve_model_openai_defaults():
    with patch.object(settings, 'ai_provider', 'openai'):
        model = resolve_model(provider='openai')
        assert model == settings.openai_mini_model


def test_handle_text_query_logs_gemini_model(caplog):
    mock_provider = MagicMock()
    mock_provider.generate_text.return_value = ('Test answer [S1].', settings.gemini_model)

    with patch('app.logic._is_openai_mocked', return_value=False), \
         patch.object(settings, 'ai_provider', 'gemini'), \
         patch('app.logic.get_provider', return_value=mock_provider), \
         caplog.at_level('INFO'):

        answer, used_model = handle_text_query(
            question='What is the policy?',
            context='[SOURCE S1: doc.pdf (Page 1)]\nPolicy details.'
        )

        assert answer == 'Test answer [S1].'
        assert used_model == settings.gemini_model
        assert any(
            f"Processing text query with model: {settings.gemini_model}" in record.message
            for record in caplog.records
        )
        assert not any('model: gpt-4o-mini' in record.message for record in caplog.records)


def test_gemini_quota_exhaustion_detected_no_retry():
    provider = GeminiProvider(api_key='test-key')
    call_count = 0

    class MockClientError(Exception):
        code = 429

    def failing_call():
        nonlocal call_count
        call_count += 1
        raise MockClientError(
            '429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing details. ' 
            '* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.5-flash'
        )

    with pytest.raises(GeminiQuotaExceededError) as exc_info:
        provider._execute_with_retry(failing_call, operation_name='test_quota', target_model='gemini-3.5-flash')

    assert call_count == 1
    assert 'quota has been exhausted' in str(exc_info.value).lower()
    assert 'gemini-3.5-flash' in str(exc_info.value)
    assert exc_info.value.status_code == 429


def test_gemini_transient_rate_limit_handled():
    provider = GeminiProvider(api_key='test-key')

    class MockRateLimitError(Exception):
        code = 429

    def failing_call():
        raise MockRateLimitError('429 Rate limit exceeded. Please retry in 3.5s.')

    with pytest.raises(GeminiRateLimitError) as exc_info:
        provider._execute_with_retry(failing_call, operation_name='test_rate_limit', target_model='gemini-3.5-flash')

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 3.5


def test_gemini_503_recovers_on_retry():
    provider = GeminiProvider(api_key='test-key')
    attempts = 0

    class MockServerError(Exception):
        code = 503

    def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MockServerError('503 The model is currently experiencing high demand.')
        mock_resp = MagicMock()
        mock_resp.text = 'Recovered answer [S1].'
        return mock_resp

    with patch('time.sleep') as mock_sleep:
        res = provider._execute_with_retry(flaky_call, operation_name='test_503_recover')
        assert res.text == 'Recovered answer [S1].'
        assert attempts == 2
        mock_sleep.assert_called_once_with(1.0)


def test_gemini_503_fails_after_max_retries():
    provider = GeminiProvider(api_key='test-key')
    attempts = 0

    class MockServerError(Exception):
        code = 503

    def persistently_failing_call():
        nonlocal attempts
        attempts += 1
        raise MockServerError('503 The model is currently experiencing high demand.')

    with patch('time.sleep'):
        with pytest.raises(GeminiServiceUnavailableError) as exc_info:
            provider._execute_with_retry(persistently_failing_call, operation_name='test_503_exhaust')

        assert attempts == 3
        assert exc_info.value.status_code == 503
        assert 'temporarily unavailable' in str(exc_info.value).lower()


def test_api_query_returns_429_on_quota_exhaustion():
    client = TestClient(app)

    with patch('app.api.perform_rag_query') as mock_rag:
        mock_rag.side_effect = GeminiQuotaExceededError(
            message="Google Gemini API quota has been exhausted for model 'gemini-3.5-flash'. Free-tier limit reached (limit: 20 requests/day).",
            provider='gemini',
            model='gemini-3.5-flash'
        )

        response = client.post(
            '/api/v1/query',
            json={
                'question': 'What is the policy?',
                'file_id': '11111111-1111-1111-1111-111111111111'
            }
        )

        assert response.status_code == 429
        data = response.json()
        assert data.get('error') == 'provider_error'
        assert 'quota has been exhausted' in data.get('message', '').lower()
        assert 'gemini-3.5-flash' in data.get('message', '')


def test_api_query_returns_503_on_service_unavailable():
    client = TestClient(app)

    with patch('app.api.perform_rag_query') as mock_rag:
        mock_rag.side_effect = GeminiServiceUnavailableError(
            message='Google Gemini service is temporarily unavailable due to high demand (HTTP 503).',
            provider='gemini'
        )

        response = client.post(
            '/api/v1/query',
            json={
                'question': 'What is the policy?',
                'file_id': '11111111-1111-1111-1111-111111111111'
            }
        )

        assert response.status_code == 503
        data = response.json()
        assert data.get('error') == 'provider_error'
        assert 'high demand' in data.get('message', '').lower()
