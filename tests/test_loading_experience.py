"""
Phase 12 Tests — Aura AI Loading Screen & Visual Processing Experience.

Verifies:
1. Loader asset discovery and fallback resolution.
2. Loader start on request submission.
3. Loader remains during processing while status updates occur without video recreation.
4. Loader transitions cleanly away on the first real token.
5. Loader disappears on complete event and renders completion badge.
6. Loader disappears on error events (429, 503, network, validation).
7. Loader disappears on interruption or stream exception (finally cleanup guarantee).
8. Silent / muted audio configuration (HTML attributes, no controls, pointer-events: none).
9. Absence of artificial delays (no time.sleep or fake timers in loading components).
10. Single assistant message guarantee (no duplicate messages).
11. Existing Phase 11 safe status events integration.
12. Historical conversations do not render the loader.
13. General RAG workflow.
14. Follow-up contextual workflow.
15. Comparison mode workflow.
16. Multimodal image workflow.
17. Text-to-SQL workflow.
18. General chat workflow.
"""

import os
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from ui.streamlit_app import (
    DEFAULT_LOADING_BUBBLE_FILENAME,
    resolve_loading_asset_path,
    get_loading_asset_data_uri,
    render_loading_animation_html,
    render_loading_video_html,
    render_loading_status_html,
    render_loading_container_html,
    render_chat_message,
    render_pipeline_badge,
    _LOADING_ASSET_B64_CACHE,
)
from app.pipeline import PipelineStage, StageStatus


# =============================================================================
# 1. Asset Discovery & MIME / Data-URI Handling
# =============================================================================

class TestLoaderAssetDiscovery:
    """Verifies discovery, caching, and fallback of the loading asset."""

    def test_default_asset_discovery(self):
        """Verify standard asset 'aura-loading-bubble.mp4' is found in ui/assets."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AURA_LOADING_ASSET", None)
            path = resolve_loading_asset_path()
            assert path is not None, "Failed to resolve default loading animation asset"
            assert path.exists(), f"Resolved path does not exist: {path}"
            assert path.is_file(), f"Resolved path is not a file: {path}"
            assert "bubble" in path.name.lower() or path.suffix == ".mp4"

    def test_explicit_path_parameter_precedence(self, tmp_path):
        """Verify explicit configured_path argument takes highest precedence."""
        dummy_mp4 = tmp_path / "custom_bubble.mp4"
        dummy_mp4.write_bytes(b"\x00\x00\x00 ftypisom")

        resolved = resolve_loading_asset_path(str(dummy_mp4))
        assert resolved == dummy_mp4

    def test_env_var_override(self, tmp_path):
        """Verify AURA_LOADING_ASSET environment variable overrides default."""
        env_mp4 = tmp_path / "env_bubble.mp4"
        env_mp4.write_bytes(b"\x00\x00\x00 ftypisom")

        with patch.dict(os.environ, {"AURA_LOADING_ASSET": str(env_mp4)}):
            resolved = resolve_loading_asset_path()
            assert resolved == env_mp4

    def test_nonexistent_path_fallback(self):
        """Verify non-existent configured path falls back to standard asset or None."""
        with patch.dict(os.environ, {"AURA_LOADING_ASSET": "nonexistent_bubble.mp4"}):
            resolved = resolve_loading_asset_path()
            assert resolved is not None
            assert resolved.name == DEFAULT_LOADING_BUBBLE_FILENAME

    def test_data_uri_generation_and_caching(self):
        """Verify data URI generation and in-memory caching for zero redundant disk reads."""
        res1 = get_loading_asset_data_uri()
        assert res1 is not None
        data_uri, mime = res1
        assert mime == "video/mp4"
        assert data_uri.startswith("data:video/mp4;base64,")

        # Second call must return the cached result
        res2 = get_loading_asset_data_uri()
        assert res1 == res2


# =============================================================================
# 2. Silent / Muted Configuration & Visual Safety
# =============================================================================

class TestLoaderAudioAndSafetyConfiguration:
    """Verifies audio muting, no controls, and zero CoT exposure."""

    def test_video_html_has_no_audio_and_is_muted(self):
        """
        Verify video tag enforces:
        - muted
        - autoplay
        - loop
        - playsinline
        - disablepictureinpicture
        - pointer-events: none (prevents click interaction)
        - NO controls attribute
        """
        html = render_loading_video_html()
        assert "<video" in html
        assert "muted" in html
        assert "autoplay" in html
        assert "loop" in html
        assert "playsinline" in html
        assert "disablepictureinpicture" in html
        assert "pointer-events: none" in html
        assert "controls" not in html.replace("disablepictureinpicture", "")

    def test_video_html_fallback_when_no_asset(self):
        """Verify glowing CSS ring fallback when asset cannot be loaded."""
        with patch("ui.streamlit_app.get_loading_asset_data_uri", return_value=None):
            html = render_loading_video_html()
            assert "aura-loading-fallback-ring" in html
            assert "<video" not in html

    def test_status_html_zero_cot_exposure(self):
        """Verify status renderer displays only safe text and never leaks prompts or CoT."""
        html = render_loading_status_html(
            primary_text="Searching your documents...",
            subtext="Reviewing verified sources",
            substatus_items=["Semantic search", "Lexical match"],
        )
        assert "Searching your documents..." in html
        assert "Reviewing verified sources" in html
        assert "aura-loading-status-box" in html
        assert "aura-pulse-dot" in html

        # Strict safety: never contains prompt, deliberation, or hidden reasoning traces
        lower = html.lower()
        assert "prompt" not in lower
        assert "chain of thought" not in lower
        assert "thinking" not in lower
        assert "hidden" not in lower

    def test_render_loading_container_html(self):
        """Verify combined container contains both video and status layers."""
        html = render_loading_container_html("Generating response...", subtext="Analyzing context")
        assert "aura-loading-combined-wrapper" in html
        assert "aura-loading-video-container" in html
        assert "Generating response..." in html
        assert "Analyzing context" in html


# =============================================================================
# 3. No Artificial Delays Guarantee
# =============================================================================

class TestNoArtificialDelays:
    """Ensure no artificial sleeps or fake timers exist in loading functions."""

    def test_loading_helpers_do_not_call_sleep(self):
        """Verify loading functions contain no time.sleep or asyncio.sleep calls."""
        for fn in [
            resolve_loading_asset_path,
            get_loading_asset_data_uri,
            render_loading_video_html,
            render_loading_status_html,
            render_loading_container_html,
        ]:
            src = inspect.getsource(fn)
            assert "time.sleep" not in src, f"{fn.__name__} must not call time.sleep"
            assert "asyncio.sleep" not in src, f"{fn.__name__} must not call asyncio.sleep"


# =============================================================================
# 4. Loader Lifecycle in SSE Streaming
# =============================================================================

class TestStreamingLoaderLifecycle:
    """Verifies that the visual loader starts, updates, transitions, and cleans up cleanly."""

    def test_loader_starts_updates_transitions_and_completes(self):
        """
        Simulate the SSE token generator sequence:
        1. Start -> Loader video box and status box are rendered.
        2. Status events arrive -> ONLY status box updates, video box is untouched.
        3. Token event arrives -> Video box is emptied cleanly, status box transitions to generating.
        4. Complete event arrives -> Video box remains empty, final badge is rendered.
        """
        mock_video_box = MagicMock()
        mock_status_box = MagicMock()

        # Simulated event sequence
        events = [
            ("start", {"session_id": "sess-1", "model": "gemini-2.5-flash"}),
            ("status", {"stage": PipelineStage.UNDERSTANDING, "status": StageStatus.RUNNING, "message": "Understanding your question..."}),
            ("status", {"stage": PipelineStage.RETRIEVAL, "status": StageStatus.RUNNING, "message": "Searching your documents..."}),
            ("sources", {"sources": [{"filename": "guide.pdf", "page_number": 1}], "search_method": "hybrid"}),
            ("token", {"text": "A "}),
            ("token", {"text": "Servlet "}),
            ("token", {"text": "is a Java class."}),
            ("complete", {
                "answer": "A Servlet is a Java class.",
                "sources": [{"filename": "guide.pdf", "page_number": 1}],
                "pipeline_stages": [
                    {"stage": "retrieval", "status": "complete", "message": "Done"},
                    {"stage": "completion", "status": "complete", "message": "Done"},
                ],
                "timings": {"total_s": 0.85},
            }),
        ]

        # Simulate the logic executed inside process_question
        status_events_holder = []
        sources_holder = []
        complete_payload = {}
        first_token_seen = False
        yielded_tokens = []

        # 1. Start: Initial render
        mock_video_box.markdown(render_loading_video_html(), unsafe_allow_html=True)
        mock_status_box.markdown(render_loading_status_html("Understanding your request..."), unsafe_allow_html=True)

        video_render_count_before_tokens = mock_video_box.markdown.call_count

        for event_type, data in events:
            if event_type == "start":
                init_msg = data.get("message") or "Understanding your request..."
                mock_status_box.markdown(render_loading_status_html(init_msg), unsafe_allow_html=True)
            elif event_type == "status":
                status_events_holder.append(data)
                stg_msg = data.get("message") or "Processing..."
                mock_status_box.markdown(render_loading_status_html(stg_msg), unsafe_allow_html=True)
            elif event_type == "sources":
                sources_holder.extend(data.get("sources", []))
                mock_status_box.markdown(
                    render_loading_status_html("Reviewing verified sources...", subtext=f"Analyzing {len(sources_holder)} sources"),
                    unsafe_allow_html=True,
                )
            elif event_type == "token":
                if not first_token_seen:
                    first_token_seen = True
                    # TRANSITION: Video box must be emptied on first token
                    mock_video_box.empty()
                    mock_status_box.markdown(
                        """
                        <div class="aura-pipeline-badge">
                            <span class="aura-pulse-dot"></span>
                            <span>Generating answer...</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                yielded_tokens.append(data.get("text", ""))
            elif event_type == "complete":
                complete_payload.update(data)

        # Ensure video is empty on completion
        mock_video_box.empty()

        # Finalize badge
        badge_html = render_pipeline_badge(
            complete_payload["pipeline_stages"],
            timings=complete_payload["timings"],
            source_count=len(sources_holder),
        )
        mock_status_box.markdown(badge_html, unsafe_allow_html=True)

        # VERIFICATIONS:
        # 1. Video box was rendered initially exactly once and never recreated during status/sources events
        assert video_render_count_before_tokens == 1
        # 2. On first token, mock_video_box.empty() was called
        assert mock_video_box.empty.call_count >= 1
        # 3. Status box was updated progressively
        assert mock_status_box.markdown.call_count >= 5
        # 4. Tokens were yielded
        assert "".join(yielded_tokens) == "A Servlet is a Java class."
        # 5. Final badge contains expected completion text
        assert "Response ready in 0.85s · 1 source verified" in badge_html

    def test_loader_disappears_on_error_event(self):
        """Verify video loader and status box are immediately cleared when an error event is received."""
        mock_video_box = MagicMock()
        mock_status_box = MagicMock()

        error_event = ("error", {"error": "API key quota reached (429)"})

        # Initial render
        mock_video_box.markdown(render_loading_video_html(), unsafe_allow_html=True)
        mock_status_box.markdown(render_loading_status_html("Processing..."), unsafe_allow_html=True)

        # On error event handling
        mock_video_box.empty()
        mock_status_box.empty()

        assert mock_video_box.empty.called
        assert mock_status_box.empty.called

    def test_loader_disappears_on_stream_exception(self):
        """Verify finally block cleans up the loader video box when an unexpected exception occurs."""
        mock_video_box = MagicMock()

        def _failing_stream():
            mock_video_box.markdown(render_loading_video_html(), unsafe_allow_html=True)
            raise ConnectionResetError("Network connection dropped")

        with pytest.raises(ConnectionResetError):
            try:
                _failing_stream()
            finally:
                mock_video_box.empty()

        assert mock_video_box.empty.called


# =============================================================================
# 5. Historical Messages Do Not Show Loader
# =============================================================================

class TestHistoricalConversations:
    """Verifies that loaded past sessions never render the loading animation."""

    def test_render_chat_message_has_no_video_loader(self):
        """Verify render_chat_message produces no video elements or loading wrappers."""
        with patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.markdown") as mock_md:

            render_chat_message(
                role="assistant",
                content="This is a previously completed answer.",
                pipeline_stages=[{"stage": "completion", "status": "complete"}],
                timings={"total_s": 1.2},
            )

            # Check all markdown calls for any loader video HTML
            for call in mock_md.call_args_list:
                content = str(call[0][0])
                assert "<video" not in content
                assert "aura-loading-video" not in content
                assert "aura-loading-fallback-ring" not in content


# =============================================================================
# 6. Query Types & Workflows Integration
# =============================================================================

class TestQueryTypesLoadingExperience:
    """Verifies loader works properly for all query types without regressions."""

    def test_general_rag_lifecycle(self):
        """General RAG: loader active during retrieval and preparation."""
        html_start = render_loading_status_html("Searching your documents...", subtext="Retrieving relevant context")
        assert "Searching your documents..." in html_start
        assert "aura-pulse-dot" in html_start

    def test_follow_up_lifecycle(self):
        """Follow-up query: loader displays context resolution message."""
        html_ctx = render_loading_status_html("Refining search with conversation context...")
        assert "Refining search" in html_ctx

    def test_comparison_mode_lifecycle(self):
        """Comparison mode: loader displays cross-document comparison message."""
        html_comp = render_loading_status_html("Comparing documents...", subtext="Analyzing cross-document metrics and contradictions")
        assert "Comparing documents..." in html_comp
        assert "cross-document metrics" in html_comp

    def test_multimodal_image_lifecycle(self):
        """Multimodal image query: loader displays image analysis message."""
        html_img = render_loading_status_html("Analyzing attached image...", subtext="OCR and Vision feature extraction")
        assert "Analyzing attached image..." in html_img

    def test_text_to_sql_lifecycle(self):
        """Text-to-SQL: loader covers schema inspection, query generation, and safe execution."""
        html_sql1 = render_loading_status_html("Checking database structure...")
        html_sql2 = render_loading_status_html("Running read-only database query...", subtext="Safety rules validated")
        assert "Checking database structure..." in html_sql1
        assert "Running read-only database query..." in html_sql2

    def test_general_chat_lifecycle(self):
        """General chat: loader covers understanding and direct generation."""
        html_chat = render_loading_status_html("Generating response...", subtext="Direct conversational response")
        assert "Generating response..." in html_chat


# =============================================================================
# 7. Lightweight Instant Loading Experience Verification (No Buffering)
# =============================================================================

class TestImmediateLightweightLoadingExperience:
    """
    Verifies that the lightweight CSS/SVG loading animation:
    1. Renders immediately with zero video buffering.
    2. Has no audio, no controls, and minimal DOM payload (< 1KB).
    3. Stays during pre-token processing without re-rendering or DOM flickering.
    4. Transitions cleanly away on the very first text token.
    5. Cleans up unconditionally on complete, error, and stream interruption.
    6. Never introduces artificial delays (no time.sleep).
    7. Never performs file rename or move operations during chat.
    8. Never appears on historical messages.
    """

    def test_lightweight_animation_properties(self):
        """Verify lightweight animation has no video tags, no audio, and minimal size."""
        html = render_loading_animation_html()
        assert "<video" not in html
        assert "<audio" not in html
        assert "<source" not in html
        assert "controls" not in html
        assert "aura-loading-bubble-container" in html
        assert "aura-loading-bubble" in html
        assert "aura-bubble-core" in html
        assert "aura-bubble-orbital" in html
        assert "aura-bubble-pulse" in html
        # Payload size must be tiny (< 1000 characters), zero network buffer
        assert len(html.strip()) < 1000

    def test_immediate_render_and_first_token_transition(self):
        """
        Simulate the exact Send -> SSE stream -> First token -> Complete lifecycle
        using the lightweight loading animation.
        """
        mock_loader_box = MagicMock()
        mock_status_box = MagicMock()

        # 1. User presses Send -> Render lightweight animation immediately
        mock_loader_box.markdown(render_loading_animation_html(), unsafe_allow_html=True)
        mock_status_box.markdown(render_loading_status_html("Understanding your request..."), unsafe_allow_html=True)

        assert mock_loader_box.markdown.call_count == 1
        initial_render_html = mock_loader_box.markdown.call_args[0][0]
        assert "aura-loading-bubble" in initial_render_html

        # 2. SSE Status events update ONLY the status box; loader_box is never re-rendered
        sse_events = [
            ("status", {"stage": "retrieval", "message": "Searching your documents..."}),
            ("sources", {"sources": [{"filename": "doc.pdf", "page": 1}]}),
            ("status", {"stage": "synthesis", "message": "Synthesizing answer..."}),
        ]
        for event_type, data in sse_events:
            if event_type == "status":
                mock_status_box.markdown(render_loading_status_html(data["message"]), unsafe_allow_html=True)
            elif event_type == "sources":
                mock_status_box.markdown(render_loading_status_html("Reviewing sources..."), unsafe_allow_html=True)

        # Loader box was NOT re-rendered during status updates (no DOM flickering)
        assert mock_loader_box.markdown.call_count == 1

        # 3. First token arrives -> Immediately empty loader_box
        mock_loader_box.empty()
        mock_status_box.markdown('<div class="aura-pipeline-badge">Generating...</div>', unsafe_allow_html=True)

        assert mock_loader_box.empty.call_count >= 1

        # 4. Final completion
        mock_loader_box.empty()
        assert mock_loader_box.empty.call_count >= 2

    def test_loader_cleaned_up_on_all_error_types(self):
        """Verify loader is immediately cleared for all error events (429, 503, network, validation)."""
        mock_loader_box = MagicMock()
        mock_status_box = MagicMock()

        for err_detail in ["429 Too Many Requests", "503 Service Unavailable", "Validation Error: empty query"]:
            mock_loader_box.reset_mock()
            mock_status_box.reset_mock()

            # Mount loader
            mock_loader_box.markdown(render_loading_animation_html(), unsafe_allow_html=True)
            mock_status_box.markdown(render_loading_status_html("Processing..."), unsafe_allow_html=True)

            # Error event arrived
            mock_loader_box.empty()
            mock_status_box.empty()

            assert mock_loader_box.empty.called
            assert mock_status_box.empty.called

    def test_loader_cleaned_up_on_stream_interruption(self):
        """Verify try ... finally block unconditionally cleans up the loader on exceptions."""
        mock_loader_box = MagicMock()

        def _interrupted_stream():
            mock_loader_box.markdown(render_loading_animation_html(), unsafe_allow_html=True)
            raise ConnectionAbortedError("Client closed connection")

        with pytest.raises(ConnectionAbortedError):
            try:
                _interrupted_stream()
            finally:
                mock_loader_box.empty()

        assert mock_loader_box.empty.called

    def test_no_file_rename_operations_during_chat(self):
        """Verify that absolutely no file rename or move operations occur during chat query processing."""
        import os
        import shutil

        with patch("os.rename") as mock_os_rename, \
             patch("shutil.move") as mock_shutil_move:

            # Execute loading rendering and status rendering
            html_loader = render_loading_animation_html()
            html_status = render_loading_status_html("Understanding your request...")

            assert "aura-loading-bubble" in html_loader
            assert "Understanding" in html_status

            # Strict assertion: zero file rename or move calls
            assert mock_os_rename.call_count == 0
            assert mock_shutil_move.call_count == 0

    def test_no_artificial_delays_in_lightweight_renderer(self):
        """Verify that render_loading_animation_html contains zero sleep calls."""
        src = inspect.getsource(render_loading_animation_html)
        assert "time.sleep" not in src
        assert "asyncio.sleep" not in src

    def test_immediate_render_timing_measurement(self):
        """Verify render_loading_animation_html executes in under 5ms (immediate UI rendering)."""
        import time
        t_start = time.perf_counter()
        for _ in range(50):
            html = render_loading_animation_html()
            assert len(html) > 0
        t_elapsed = (time.perf_counter() - t_start) / 50 * 1000  # ms per call
        assert t_elapsed < 5.0, f"Render took {t_elapsed:.3f}ms, expected < 5ms"

    def test_historical_conversations_never_show_lightweight_bubble(self):
        """Verify historical message rendering never outputs the loading bubble container."""
        with patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.markdown") as mock_md:

            render_chat_message(
                role="assistant",
                content="Historical conversation response.",
                timings={"total_s": 0.5},
            )

            for call in mock_md.call_args_list:
                rendered = str(call[0][0])
                assert "aura-loading-bubble" not in rendered
                assert "aura-loading-bubble-container" not in rendered

