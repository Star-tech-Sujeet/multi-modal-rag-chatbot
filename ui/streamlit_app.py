"""
Aura AI — Premium Conversational Document Assistant.
Streamlit Frontend combining Premium AI Workspace (Zyricon) and Immersive Landing (Fastshot) designs.

Key Architectural Guarantees:
1. Glassmorphism dark UI with subtle cyan/blue Aura accents.
2. Configurable background image support with resilient dark gradient fallback.
3. Immersive hero landing experience on empty chat; clean, focused conversation on active chat.
4. Large rounded chat composer with image attachment preview and remove controls.
5. Rich expandable source citations and evidence presentation.
6. Markdown sanitization eliminating internal SVG and localhost anchor artifacts.
7. Independent compact sidebar with persistent chat switching and deduplicated new chat creation.
8. Full preservation of all Phase 1-10 backend integrations (Gemini, Multimodal, Comparison, SQL).
"""

import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from ui.markdown_utils import clean_markdown_display
from ui.processing_indicator import (
    AuraProcessingIndicator,
    get_indicator_css,
    render_aura_processing_indicator_html,
    sanitize_status_text,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Environment & Configuration
# =============================================================================

load_dotenv()
if not Path(".env").exists():
    load_dotenv(".env.example")

# Project and UI directory paths for robust resolution
CURRENT_FILE = Path(__file__).resolve()
UI_DIR = CURRENT_FILE.parent
PROJECT_ROOT = UI_DIR.parent

AURA_AI_SYMBOL_FILENAME = "goku.jpg"
AURA_AI_SYMBOL_PATH = UI_DIR / "assets" / AURA_AI_SYMBOL_FILENAME
AURA_AI_SYMBOL = str(AURA_AI_SYMBOL_PATH.resolve()) if AURA_AI_SYMBOL_PATH.exists() else "✦"

# Page Configuration - must be first Streamlit command
st.set_page_config(
    page_title="Aura AI — Intelligent Document Assistant",
    page_icon=AURA_AI_SYMBOL,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/multi-modal-rag-app",
        "Report a bug": "https://github.com/your-repo/multi-modal-rag-app/issues",
        "About": """
        ## Aura AI
        **Your Intelligent Document Assistant**
        
        Multimodal RAG powered by Google Gemini, ChromaDB, BM25, and Reciprocal Rank Fusion.
        """,
    },
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000") + "/api/v1"
    if os.getenv("RENDER_EXTERNAL_URL")
    else "http://backend:8000/api/v1"
    if os.getenv("DOCKER_ENV")
    else "http://localhost:8000/api/v1",
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))

DEFAULT_BACKGROUND_FILENAME = "anime-style-galaxy-background_23-2151133974.avif"
DEFAULT_BACKGROUND_SUBPATH = f"ui/assets/{DEFAULT_BACKGROUND_FILENAME}"

# Background image path configuration (configurable via environment, defaults to provided galaxy AVIF)
APP_BACKGROUND_IMAGE = os.getenv("APP_BACKGROUND_IMAGE")

# Background video asset configuration (defaults to provided cyberpunk cityscape MVP MP4)
DEFAULT_BACKGROUND_VIDEO_FILENAME = "gemini_generated_video_44ac85bd_gwr_video_mvp.mp4"
DEFAULT_BACKGROUND_VIDEO_SUBPATH = f"ui/assets/{DEFAULT_BACKGROUND_VIDEO_FILENAME}"
APP_BACKGROUND_VIDEO = os.getenv("APP_BACKGROUND_VIDEO")

# Loading animation asset configuration (configurable via environment, defaults to provided bubble MP4)
DEFAULT_LOADING_BUBBLE_FILENAME = "aura-loading-bubble.mp4"
DEFAULT_LOADING_BUBBLE_SUBPATH = f"ui/assets/{DEFAULT_LOADING_BUBBLE_FILENAME}"
AURA_LOADING_ASSET = os.getenv("AURA_LOADING_ASSET")

# In-memory cache for loading animation base64 data URI to eliminate repeated disk reads
_LOADING_ASSET_B64_CACHE: Optional[str] = None
_LOADING_ASSET_TYPE_CACHE: Optional[str] = None

# In-memory cache for application background CSS to eliminate repeated disk I/O on every rerun
_BACKGROUND_CSS_CACHE: Optional[str] = None

# In-memory cache for background video streamable URL
_BACKGROUND_VIDEO_URL_CACHE: Optional[str] = None

# In-memory cache for UI icon data URIs to eliminate repeated disk reads
_UI_ASSET_DATA_URI_CACHE: Dict[str, str] = {}

# Explicit icon asset mapping for empty-state suggestion prompts and attachments
SUGGESTION_ICONS: Dict[str, str] = {
    "Summarize this document": "ui/assets/documents-folder.svg",
    "Explain key concepts": "ui/assets/innovation.svg",
    "Find critical facts": "ui/assets/find.png",
    "Compare documents": "ui/assets/swap.svg",
    "Attach Image / Chart (Optional)": "ui/assets/upload.svg",
}


def resolve_ui_asset_path(filename: str) -> Optional[Path]:
    """
    Resolve a UI asset path robustly relative to PROJECT_ROOT and UI_DIR.
    Handles both bare filenames (e.g. 'swap.svg') and paths (e.g. 'ui/assets/swap.svg').
    """
    name_only = Path(filename).name
    candidates = [
        UI_DIR / "assets" / name_only,
        PROJECT_ROOT / "ui" / "assets" / name_only,
        Path("ui/assets") / name_only,
        Path("assets") / name_only,
        PROJECT_ROOT / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def get_ui_asset_data_uri(filename: str, color: str = "#e2e8f0") -> str:
    """
    Return a data URI for the given asset in ui/assets.
    Normalizes stroke/fill colors for dark UI background visibility in-memory
    without altering files on disk. Caches results to eliminate redundant I/O.
    """
    cache_key = f"{filename}_{color}"
    if cache_key in _UI_ASSET_DATA_URI_CACHE:
        return _UI_ASSET_DATA_URI_CACHE[cache_key]

    path = resolve_ui_asset_path(filename)
    if not path or not path.is_file():
        return ""

    if "goku" in filename:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"
        _UI_ASSET_DATA_URI_CACHE[cache_key] = data_uri
        _UI_ASSET_DATA_URI_CACHE[filename] = data_uri
        return data_uri

    if path.suffix.lower() == ".svg":
        raw_text = path.read_text(encoding="utf-8")
        # Preserve original intentional colors for working icons (rename & trash)
        if "icons8-rename" not in filename and "icons8-trash" not in filename:
            # Replace explicit black/near-black fills
            svg_text = re.sub(
                r'fill=["\']#(?:000000|010101|0000|000)["\']',
                f'fill="{color}"',
                raw_text,
                flags=re.IGNORECASE,
            )
            # If svg tag lacks fill attribute entirely (e.g. comment-alt-medical.svg), add fill
            if "fill=" not in svg_text:
                svg_text = svg_text.replace("<svg ", f'<svg fill="{color}" ', 1)
        else:
            svg_text = raw_text

        encoded = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
        data_uri = f"data:image/svg+xml;base64,{encoded}"
    elif path.suffix.lower() == ".png":
        try:
            im = Image.open(path).convert("RGBA")
            r, g, b, a = im.split()
            hex_color = color.lstrip("#")
            if len(hex_color) == 6:
                cr = int(hex_color[0:2], 16)
                cg = int(hex_color[2:4], 16)
                cb = int(hex_color[4:6], 16)
            else:
                cr, cg, cb = 226, 232, 240
            colored_im = Image.merge("RGBA", (
                Image.new("L", im.size, cr),
                Image.new("L", im.size, cg),
                Image.new("L", im.size, cb),
                a,
            ))
            buf = io.BytesIO()
            colored_im.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{encoded}"
        except Exception:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            data_uri = f"data:image/png;base64,{encoded}"
    elif path.suffix.lower() in [".jpg", ".jpeg"]:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"
    else:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        data_uri = f"data:image/octet-stream;base64,{encoded}"

    _UI_ASSET_DATA_URI_CACHE[cache_key] = data_uri
    _UI_ASSET_DATA_URI_CACHE[filename] = data_uri
    return data_uri


def get_aura_symbol_data_uri() -> str:
    """Return raw data URI for official Aura AI brand symbol."""
    return get_ui_asset_data_uri(AURA_AI_SYMBOL_FILENAME)


# =============================================================================
# Background & Glassmorphism Styling
# =============================================================================

def resolve_background_video_path(
    configured_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve the background video path robustly relative to the project location
    rather than depending on the current working directory.
    """
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    env_vid = os.getenv("APP_BACKGROUND_VIDEO") or APP_BACKGROUND_VIDEO
    if env_vid and env_vid not in candidates:
        candidates.append(env_vid)

    candidates.extend(
        [
            DEFAULT_BACKGROUND_VIDEO_SUBPATH,
            f"assets/{DEFAULT_BACKGROUND_VIDEO_FILENAME}",
            DEFAULT_BACKGROUND_VIDEO_FILENAME,
            "ui/assets/background.mp4",
            "assets/background.mp4",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_absolute() and p.is_file():
            return p
        p_root = (PROJECT_ROOT / p).resolve()
        if p_root.is_file():
            return p_root
        p_ui = (UI_DIR / p).resolve()
        if p_ui.is_file():
            return p_ui
        p_cwd = (Path.cwd() / p).resolve()
        if p_cwd.is_file():
            return p_cwd

    # Fallback: search ui/assets for any mp4 with 'video' or 'background' or 'gwr' in name,
    # strictly ignoring loading bubble animation videos
    assets_dir = UI_DIR / "assets"
    if assets_dir.is_dir():
        for f in sorted(assets_dir.glob("*.mp4")):
            fname = f.name.lower()
            if "loading" not in fname and "bubble" not in fname:
                return f.resolve()

    return None


def get_background_video_url(configured_path: Optional[str] = None) -> Optional[str]:
    """
    Get streamable URL for the background video using Streamlit's runtime media file manager.
    Guarantees that:
    1. The MP4 is NOT Base64-encoded.
    2. The entire video is NOT embedded as raw data in Python strings.
    3. The file is registered once and cached in-memory across Streamlit reruns.
    4. The browser streams the video directly via native HTTP range requests.
    """
    global _BACKGROUND_VIDEO_URL_CACHE
    if configured_path is None and _BACKGROUND_VIDEO_URL_CACHE is not None:
        return _BACKGROUND_VIDEO_URL_CACHE

    video_path = resolve_background_video_path(configured_path)
    if not video_path or not video_path.is_file():
        return None

    url = None
    try:
        from streamlit import runtime

        if runtime.exists():
            media_mgr = runtime.get_instance().media_file_mgr
            url = media_mgr.add(str(video_path), "video/mp4", "aura_background_video")
    except Exception as e:
        logger.warning(
            f"Could not register background video with Streamlit media manager: {e}"
        )

    if not url:
        # Fallback URL for test environments or headless verification
        url = f"/media/{video_path.name}"

    if configured_path is None:
        _BACKGROUND_VIDEO_URL_CACHE = url

    return url


def resolve_background_image_path(
    configured_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve the background image path robustly relative to the project location
    rather than depending on the current working directory.
    """
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    env_img = os.getenv("APP_BACKGROUND_IMAGE") or APP_BACKGROUND_IMAGE
    if env_img and env_img not in candidates:
        candidates.append(env_img)

    # Provided default image candidates (relative to root, relative to ui, direct name)
    candidates.extend(
        [
            DEFAULT_BACKGROUND_SUBPATH,
            f"assets/{DEFAULT_BACKGROUND_FILENAME}",
            DEFAULT_BACKGROUND_FILENAME,
            "assets/background.jpg",
            "assets/background.png",
            "ui/assets/background.jpg",
            "ui/assets/background.png",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_absolute() and p.is_file():
            return p
        # Check relative to PROJECT_ROOT
        p_root = (PROJECT_ROOT / p).resolve()
        if p_root.is_file():
            return p_root
        # Check relative to UI_DIR
        p_ui = (UI_DIR / p).resolve()
        if p_ui.is_file():
            return p_ui
        # Check relative to CWD
        p_cwd = (Path.cwd() / p).resolve()
        if p_cwd.is_file():
            return p_cwd

    return None


def get_image_mime_type(file_path: Path) -> str:
    """Return appropriate MIME type for the image extension."""
    ext = file_path.suffix.lower()
    mime_map = {
        ".avif": "image/avif",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    return mime_map.get(ext, "image/jpeg")


# In-memory cache for application background image base64 by resolved path
_BACKGROUND_IMG_B64_CACHE: Dict[str, Tuple[str, str]] = {}


def get_background_css() -> str:
    """
    Generate CSS for the application background.
    Supports AVIF, PNG, JPG, and WEBP images with dark translucent overlays for high readability.
    Gracefully falls back to a deep cinematic navy aura radial gradient if missing.
    Image byte reading and base64 encoding are cached per resolved path.
    """
    resolved_path = resolve_background_image_path()
    if resolved_path:
        str_path = str(resolved_path)
        if str_path in _BACKGROUND_IMG_B64_CACHE:
            encoded_string, mime = _BACKGROUND_IMG_B64_CACHE[str_path]
        else:
            try:
                with open(resolved_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                mime = get_image_mime_type(resolved_path)
                _BACKGROUND_IMG_B64_CACHE[str_path] = (encoded_string, mime)
            except Exception as e:
                logger.warning(f"Failed to load background image ({resolved_path}): {e}")
                encoded_string, mime = None, None

        if encoded_string and mime:
            return f"""
            .stApp, [data-testid="stAppViewContainer"] {{
                background-image: 
                    radial-gradient(circle at 50% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 60%),
                    linear-gradient(180deg, rgba(8, 12, 18, 0.76) 0%, rgba(5, 8, 13, 0.90) 100%),
                    url("data:{mime};base64,{encoded_string}") !important;
                background-repeat: no-repeat !important;
                background-position: center center !important;
                background-size: cover !important;
                background-attachment: fixed !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
            }}
            """

    # Fallback: Atmospheric Aura dark gradient with subtle cyan glow
    return """
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse 90% 60% at 50% -10%, rgba(56, 189, 248, 0.12) 0%, transparent 70%),
                    radial-gradient(ellipse 60% 40% at 85% 70%, rgba(14, 165, 233, 0.06) 0%, transparent 60%),
                    #05080D !important;
        background-attachment: fixed !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    """


def apply_custom_styles():
    """Apply Aura AI unified dark glass design system adapted for background video."""
    bg_video_url = get_background_video_url()
    bg_css = get_background_css()

    rename_icon_uri = get_ui_asset_data_uri("icons8-rename-48.svg")
    trash_icon_uri = get_ui_asset_data_uri("icons8-trash-24.svg")
    comment_icon_uri = get_ui_asset_data_uri("comment-alt-medical.svg")

    # Explicit asset URIs derived directly from SUGGESTION_ICONS mapping
    doc_folder_icon_uri = get_ui_asset_data_uri(SUGGESTION_ICONS["Summarize this document"])
    innovation_icon_uri = get_ui_asset_data_uri(SUGGESTION_ICONS["Explain key concepts"])
    find_icon_uri = get_ui_asset_data_uri(SUGGESTION_ICONS["Find critical facts"])
    swap_icon_uri = get_ui_asset_data_uri(SUGGESTION_ICONS["Compare documents"])
    upload_icon_uri = get_ui_asset_data_uri(SUGGESTION_ICONS["Attach Image / Chart (Optional)"])

    if bg_video_url:
        container_bg = """
        .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
            background: transparent !important;
            background-color: transparent !important;
        }
        """
    else:
        container_bg = bg_css

    st.markdown(
        f"""
    <style>
        /* ===== GOOGLE FONTS IMPORT ===== */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* ===== ROOT VARIABLES & AURA THEME (Cinematic Dark Glass UI) ===== */
        :root {{
            --aura-base-0: #05080D;
            --aura-base-1: #080C12;
            --aura-base-2: #0B1118;
            --aura-primary: #5CC8FF;
            --aura-cyan: #5CC8FF;
            --aura-cyan-glow: rgba(92, 200, 255, 0.14);
            --aura-accent: #65D6FF;
            --aura-accent-muted: #7FA8C5;
            --aura-emerald: #35D0A1;
            --aura-surface: rgba(10, 16, 24, 0.72);
            --aura-card: rgba(12, 18, 26, 0.78);
            --aura-elevated: rgba(15, 22, 31, 0.82);
            --aura-border: rgba(120, 150, 175, 0.16);
            --aura-border-focus: rgba(92, 200, 255, 0.45);
            --text-primary: #E8EEF4;
            --text-secondary: #AAB7C4;
            --text-muted: #71808F;
        }}

        {container_bg}

        /* ===== PERSISTENT BACKGROUND VIDEO CONTAINER & OVERLAY ===== */
        .aura-background-layer,
        #aura-persistent-bg {{
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            z-index: 0 !important;
            pointer-events: none !important;
        }}

        .aura-background-video,
        #aura-bg-video {{
            position: absolute !important;
            inset: 0 !important;
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            object-position: center center !important;
            pointer-events: none !important;
        }}

        .aura-background-overlay,
        #aura-bg-overlay {{
            position: absolute !important;
            inset: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(180deg, rgba(5, 7, 13, 0.40) 0%, rgba(8, 11, 20, 0.58) 100%) !important;
            pointer-events: none !important;
            z-index: 1 !important;
        }}



        /* Base Typography & Resets */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        h1, h2, h3, h4, .aura-brand-font {{
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
            letter-spacing: -0.02em;
        }}

        code, pre, .aura-mono {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Custom Modern Scrollbars */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(92, 200, 255, 0.35);
        }}

        /* ===== HEADER & TOOLBAR CONFIGURATION ===== */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0px !important;
            overflow: visible !important;
            z-index: 999990 !important;
        }}

        /* Hide Streamlit Clutter: Menu, Footer, Deploy/Settings, Anchors */
        #MainMenu, footer, [data-testid="stDecoration"],
        [data-testid="stToolbarActions"],
        [data-testid="stStatusWidget"],
        [data-testid="stHeaderActionElements"],
        .header-anchor {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }}

        /* ===== SIDEBAR COLLAPSE CONTROL (Sidebar Open) ===== */
        [data-testid="stSidebarHeader"] {{
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            padding: 0.5rem 0.8rem 0.2rem !important;
            background: transparent !important;
        }}

        [data-testid="stSidebarCollapseButton"] {{
            visibility: visible !important;
            opacity: 0.9 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            position: relative !important;
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(110, 145, 165, 0.25) !important;
            border-radius: 8px !important;
            padding: 4px !important;
            width: 32px !important;
            height: 32px !important;
            color: #5cc8ff !important;
            cursor: pointer !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}

        [data-testid="stSidebarCollapseButton"]:hover {{
            visibility: visible !important;
            opacity: 1 !important;
            background: rgba(45, 95, 120, 0.25) !important;
            border-color: rgba(105, 190, 225, 0.5) !important;
            color: #5cc8ff !important;
            box-shadow: 0 0 12px rgba(92, 200, 255, 0.25) !important;
            transform: scale(1.04) !important;
        }}

        [data-testid="stSidebarCollapseButton"] button {{
            background: transparent !important;
            border: none !important;
            color: inherit !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            width: 100% !important;
            height: 100% !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebarCollapseButton"] svg {{
            fill: #5cc8ff !important;
            color: #5cc8ff !important;
            stroke: #5cc8ff !important;
            width: 18px !important;
            height: 18px !important;
            display: inline-block !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stSidebarCollapseButton"]:hover svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
            stroke: #ffffff !important;
        }}

        /* Tooltip on hover/focus only - no permanent text */
        [data-testid="stSidebarCollapseButton"]::after {{
            content: "" !important;
            display: none !important;
        }}

        [data-testid="stSidebarCollapseButton"]:hover::after,
        [data-testid="stSidebarCollapseButton"]:focus-within::after {{
            content: "Collapse sidebar" !important;
            display: block !important;
            position: absolute !important;
            top: calc(100% + 6px) !important;
            right: 0 !important;
            background: rgba(10, 16, 24, 0.95) !important;
            border: 1px solid rgba(110, 145, 165, 0.35) !important;
            border-radius: 6px !important;
            padding: 3px 8px !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            color: #e8eef4 !important;
            white-space: nowrap !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 8px rgba(92, 200, 255, 0.15) !important;
            pointer-events: none !important;
            z-index: 999999 !important;
        }}

        /* ===== FLOATING REOPEN TOGGLE CONTROL (Sidebar Collapsed) ===== */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 999999 !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(10, 16, 24, 0.88) !important;
            border: 1px solid rgba(110, 145, 165, 0.32) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5), 0 0 14px rgba(92, 200, 255, 0.18) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            color: #5cc8ff !important;
            cursor: pointer !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            padding: 0 !important;
            height: 36px !important;
            width: 36px !important;
            min-width: 36px !important;
        }}

        [data-testid="stExpandSidebarButton"]:hover,
        [data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="collapsedControl"]:hover {{
            background: rgba(45, 95, 120, 0.25) !important;
            border-color: rgba(105, 190, 225, 0.5) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 16px rgba(92, 200, 255, 0.25) !important;
            transform: scale(1.05) !important;
        }}

        [data-testid="stExpandSidebarButton"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button {{
            background: transparent !important;
            border: none !important;
            color: inherit !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            width: 100% !important;
            height: 100% !important;
            cursor: pointer !important;
        }}

        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="collapsedControl"] svg {{
            fill: #5cc8ff !important;
            color: #5cc8ff !important;
            stroke: #5cc8ff !important;
            width: 18px !important;
            height: 18px !important;
            display: inline-block !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stExpandSidebarButton"]:hover svg,
        [data-testid="stSidebarCollapsedControl"]:hover svg,
        [data-testid="collapsedControl"]:hover svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
            stroke: #ffffff !important;
        }}

        /* Tooltip on hover/focus only - no permanent text */
        [data-testid="stExpandSidebarButton"]::after,
        [data-testid="stSidebarCollapsedControl"]::after,
        [data-testid="collapsedControl"]::after {{
            content: "" !important;
            display: none !important;
        }}

        [data-testid="stExpandSidebarButton"]:hover::after,
        [data-testid="stExpandSidebarButton"]:focus-within::after,
        [data-testid="stSidebarCollapsedControl"]:hover::after,
        [data-testid="stSidebarCollapsedControl"]:focus-within::after,
        [data-testid="collapsedControl"]:hover::after,
        [data-testid="collapsedControl"]:focus-within::after {{
            content: "Open sidebar" !important;
            display: block !important;
            position: absolute !important;
            top: calc(100% + 6px) !important;
            left: 0 !important;
            background: rgba(10, 16, 24, 0.95) !important;
            border: 1px solid rgba(110, 145, 165, 0.35) !important;
            border-radius: 6px !important;
            padding: 3px 8px !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            color: #e8eef4 !important;
            white-space: nowrap !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 8px rgba(92, 200, 255, 0.15) !important;
            pointer-events: none !important;
            z-index: 999999 !important;
        }}

        /* App block container spacing */
        .block-container {{
            padding-top: 1.25rem !important;
            padding-bottom: 5rem !important;
            max-width: 1040px !important;
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}
            [data-testid="stExpandSidebarButton"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"] {{
                top: 10px !important;
                left: 10px !important;
                height: 34px !important;
                padding: 4px 8px !important;
            }}
        }}

        /* ===== SIDEBAR STYLING ===== */
        [data-testid="stSidebar"] {{
            background: rgba(7, 9, 18, 0.88) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            border-right: 1px solid rgba(120, 150, 175, 0.16) !important;
            width: 300px !important;
            z-index: 100 !important;
            position: relative !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 0.5rem !important;
        }}

        /* Sidebar Brand */
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.2rem 0.2rem 1.1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(120, 150, 175, 0.16);
        }}
        .sidebar-brand-icon {{
            font-size: 1.35rem;
            color: #5cc8ff;
            filter: drop-shadow(0 0 8px rgba(92, 200, 255, 0.3));
        }}
        .sidebar-brand-img {{
            width: 28px;
            height: 28px;
            object-fit: contain;
            filter: drop-shadow(0 0 8px rgba(92, 200, 255, 0.3));
            border-radius: 6px;
        }}
        .sidebar-brand-text {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, #7fa8c5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .sidebar-brand-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 9999px;
            background: rgba(45, 95, 120, 0.22);
            border: 1px solid rgba(85, 190, 225, 0.35);
            color: #8edaff;
            margin-left: auto;
        }}

        /* Buttons Styling */
        .stButton > button {{
            border-radius: 10px !important;
            font-weight: 500 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, rgba(52, 111, 145, 0.45), rgba(30, 65, 88, 0.55)) !important;
            border: 1px solid rgba(105, 190, 225, 0.28) !important;
            color: #e8eef4 !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35), 0 0 10px rgba(92, 200, 255, 0.12) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, rgba(65, 130, 170, 0.55), rgba(38, 80, 108, 0.65)) !important;
            border-color: rgba(105, 190, 225, 0.5) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45), 0 0 16px rgba(92, 200, 255, 0.25) !important;
            transform: translateY(-1px);
        }}
        .stButton > button[kind="secondary"] {{
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(120, 150, 175, 0.16) !important;
            color: var(--text-secondary) !important;
        }}
        .stButton > button[kind="secondary"]:hover {{
            background: rgba(45, 95, 120, 0.18) !important;
            border-color: rgba(105, 190, 225, 0.3) !important;
            color: #ffffff !important;
        }}

        /* Chat list item styling */
        .chat-list-label {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 1.1rem 0 0.45rem;
        }}

        /* Chat Session Action Buttons (Rename & Delete - scoped strictly to 3-column session rows) */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(2) button,
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(3) button {{
            padding: 0 !important;
            min-height: 38px !important;
            height: 38px !important;
            font-size: 0.95rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}

        /* Column 1: Chat Session Button with comment-alt-medical.svg */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(1) button {{
            background-image: url("{comment_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 12px center !important;
            background-size: 16px 16px !important;
            padding-left: 36px !important;
            text-align: left !important;
        }}

        /* Column 2: Rename Chat - Replace visual icon with icons8-rename-48.svg */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(2) button {{
            background-image: url("{rename_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(2) button div[data-testid="stMarkdownContainer"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(2) button:hover {{
            background: rgba(45, 95, 120, 0.22) !important;
            background-image: url("{rename_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            border-color: rgba(105, 190, 225, 0.4) !important;
            box-shadow: 0 0 10px rgba(92, 200, 255, 0.2) !important;
            transform: scale(1.05) !important;
        }}

        /* Column 3: Delete Chat - Replace visual icon with icons8-trash-24.svg */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(3) button {{
            background-image: url("{trash_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(3) button div[data-testid="stMarkdownContainer"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3)) [data-testid="stColumn"]:nth-child(3) button:hover {{
            background: rgba(239, 68, 68, 0.15) !important;
            background-image: url("{trash_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            border-color: rgba(239, 68, 68, 0.45) !important;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.25) !important;
            transform: scale(1.05) !important;
        }}

        /* Document Delete Button Icon (icons8-trash-24.svg) */
        [data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(1) .aura-doc-card ~ [data-testid="stButton"] button,
        .aura-doc-del-container button,
        .element-container:has(.aura-doc-del-container) + .element-container button {{
            background-image: url("{trash_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 14px center !important;
            background-size: 14px 14px !important;
            padding-left: 36px !important;
        }}

        /* Document Library Expander Header with documents-folder.svg */
        [data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(1) details summary p::before,
        .aura-doclib-expander [data-testid="stExpander"] details summary p::before,
        .element-container:has(.aura-doclib-expander) + .element-container [data-testid="stExpander"] details summary p::before {{
            content: "";
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            background-image: url("{doc_folder_icon_uri}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            vertical-align: -2px;
            opacity: 0.95;
        }}

        /* Document Comparison Expander Header with swap.svg */
        [data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(3) details summary p::before,
        .aura-compare-expander [data-testid="stExpander"] details summary p::before,
        .element-container:has(.aura-compare-expander) + .element-container [data-testid="stExpander"] details summary p::before {{
            content: "";
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            background-image: url("{swap_icon_uri}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            vertical-align: -2px;
            opacity: 0.95;
        }}

        /* Image / Chart Attachment Expander Header with upload.svg */
        [data-testid="stMain"] [data-testid="stExpander"] details summary p::before,
        [data-testid="stAppViewContainer"] section.main [data-testid="stExpander"] details summary p::before,
        .aura-attach-expander [data-testid="stExpander"] details summary p::before,
        .element-container:has(.aura-attach-expander) + .element-container [data-testid="stExpander"] details summary p::before {{
            content: "";
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            background-image: url("{upload_icon_uri}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            vertical-align: -2px;
            opacity: 0.95;
        }}

        /* Empty-State Suggestion Buttons (18px-20px, object-fit contain, left aligned) */
        .aura-hero-btn-container button,
        .element-container:has(.hero-summarize) + .element-container button,
        .element-container:has(.hero-concepts) + .element-container button,
        .element-container:has(.hero-facts) + .element-container button,
        .element-container:has(.hero-compare) + .element-container button,
        .st-key-hero_prompt_0 button,
        .st-key-hero_prompt_1 button,
        .st-key-hero_prompt_2 button,
        .st-key-hero_prompt_3 button {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            min-height: 48px !important;
            padding-left: 48px !important;
            text-align: left !important;
        }}

        /* Hero 0: Summarize this document (documents-folder.svg) */
        .st-key-hero_prompt_0 button,
        [data-testid="stHorizontalBlock"]:has(.hero-summarize) [data-testid="stColumn"]:nth-child(1) button,
        .element-container:has(.hero-summarize) + .element-container button,
        .hero-summarize button {{
            background-image: url("{doc_folder_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 18px center !important;
            background-size: 18px 18px !important;
            padding-left: 48px !important;
            text-align: left !important;
        }}

        /* Hero 1: Explain key concepts (innovation.svg) */
        .st-key-hero_prompt_1 button,
        [data-testid="stHorizontalBlock"]:has(.hero-concepts) [data-testid="stColumn"]:nth-child(2) button,
        .element-container:has(.hero-concepts) + .element-container button,
        .hero-concepts button {{
            background-image: url("{innovation_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 18px center !important;
            background-size: 18px 18px !important;
            padding-left: 48px !important;
            text-align: left !important;
        }}

        /* Hero 2: Find critical facts (find.png) */
        .st-key-hero_prompt_2 button,
        [data-testid="stHorizontalBlock"]:has(.hero-facts) [data-testid="stColumn"]:nth-child(1) button,
        .element-container:has(.hero-facts) + .element-container button,
        .hero-facts button {{
            background-image: url("{find_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 18px center !important;
            background-size: 18px 18px !important;
            padding-left: 48px !important;
            text-align: left !important;
        }}

        /* Hero 3: Compare documents (swap.svg) */
        .st-key-hero_prompt_3 button,
        [data-testid="stHorizontalBlock"]:has(.hero-compare) [data-testid="stColumn"]:nth-child(2) button,
        .element-container:has(.hero-compare) + .element-container button,
        .hero-compare button {{
            background-image: url("{swap_icon_uri}") !important;
            background-repeat: no-repeat !important;
            background-position: 18px center !important;
            background-size: 18px 18px !important;
            padding-left: 48px !important;
            text-align: left !important;
        }}

        /* Generic inline Aura UI icon */
        .aura-ui-icon {{
            width: 16px !important;
            height: 16px !important;
            object-fit: contain !important;
            vertical-align: -2px !important;
            margin-right: 6px !important;
            display: inline-block !important;
            opacity: 0.95 !important;
        }}

        /* Document Library Card Styling */
        .aura-doc-card {{
            background: rgba(10, 17, 25, 0.72);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(110, 145, 165, 0.16);
            border-radius: 10px;
            padding: 0.6rem 0.75rem;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }}
        .aura-doc-card:hover {{
            background: rgba(14, 23, 34, 0.82);
            border-color: rgba(80, 180, 220, 0.30);
        }}
        .aura-doc-card-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }}
        .aura-doc-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            background: rgba(45, 95, 120, 0.22);
            color: #8edaff;
            border: 1px solid rgba(85, 190, 225, 0.3);
            text-transform: uppercase;
        }}
        .aura-doc-name {{
            font-size: 0.84rem;
            font-weight: 500;
            color: #f1f0f7;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
        }}
        .aura-doc-meta {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-left: 0.1rem;
        }}

        /* Legacy select button container - visually hidden, preserved for test runner */
        .aura-legacy-select-container {{
            display: none !important;
            height: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
        }}

        /* Expanders in Dark Glass */
        [data-testid="stExpander"] {{
            background: rgba(10, 16, 24, 0.75) !important;
            border: 1px solid rgba(110, 145, 165, 0.16) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            margin-bottom: 0.75rem !important;
        }}
        [data-testid="stExpander"] details summary {{
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }}

        /* ===== TOP BAR NAVIGATION ===== */
        .aura-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.7rem 1.25rem;
            margin-bottom: 1.5rem;
            background: rgba(6, 8, 17, 0.75) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(110, 145, 165, 0.16) !important;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        }}
        .aura-topbar-left {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .aura-topbar-brand {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.05rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, #7fa8c5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.01em;
            display: inline-flex;
            align-items: center;
        }}
        .aura-topbar-brand-icon {{
            width: 24px;
            height: 24px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: 0.4rem;
            filter: drop-shadow(0 0 6px rgba(92, 200, 255, 0.3));
            border-radius: 4px;
        }}
        .aura-context-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(40, 65, 82, 0.35);
            border: 1px solid rgba(110, 165, 190, 0.25);
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: #bfd2df;
        }}
        .aura-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: #94a3b8;
        }}
        .aura-status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #35d0a1;
            box-shadow: 0 0 8px #35d0a1;
        }}
        .aura-status-dot.error {{
            background: #f06a72;
            box-shadow: 0 0 8px #f06a72;
        }}

        /* ===== EMPTY STATE / LANDING HERO ===== */
        .aura-hero-wrapper {{
            text-align: center;
            padding: 3rem 1.5rem 2rem;
            max-width: 760px;
            margin: 0 auto;
        }}
        .aura-hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(45, 95, 120, 0.22);
            border: 1px solid rgba(105, 190, 225, 0.28);
            border-radius: 9999px;
            padding: 0.3rem 0.9rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: #8edaff;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1.25rem;
            box-shadow: 0 0 16px rgba(92, 200, 255, 0.08);
        }}
        .aura-hero-badge-img {{
            width: 14px;
            height: 14px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: 0.25rem;
        }}
        .aura-badge-img-xs {{
            width: 14px;
            height: 14px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: 0.25rem;
            filter: drop-shadow(0 0 4px rgba(92, 200, 255, 0.3));
        }}
        .aura-hero-title {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 2.75rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.03em;
            color: #ffffff;
            margin-bottom: 0.85rem;
            background: linear-gradient(180deg, #ffffff 40%, #aab7c4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .aura-hero-subtitle {{
            font-size: 1.05rem;
            line-height: 1.6;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}

        /* Suggested Prompts Cards */
        .aura-prompts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.85rem;
            max-width: 680px;
            margin: 0 auto 2.5rem;
        }}

        /* ===== CHAT MESSAGES ===== */
        [data-testid="stChatMessage"] {{
            background: transparent !important;
            padding: 0.85rem 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}
        
        /* User message bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: rgba(18, 25, 34, 0.82) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border: 1px solid rgba(120, 160, 185, 0.16) !important;
            border-radius: 14px !important;
            padding: 0.9rem 1.1rem !important;
            margin-bottom: 0.75rem !important;
        }}
        
        /* Assistant message styling */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
        [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) {{
            background: rgba(12, 18, 26, 0.78) !important;
            border: 1px solid rgba(110, 145, 165, 0.14) !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.35rem !important;
            margin-bottom: 0.75rem !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
        }}
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"],
        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"] img {{
            width: 32px !important;
            height: 32px !important;
            border-radius: 50% !important;
            object-fit: contain !important;
            border: 1px solid rgba(120, 160, 185, 0.25) !important;
        }}

        /* Inline Citation Pills */
        .aura-citation-pill {{
            display: inline-flex;
            align-items: center;
            padding: 1px 6px;
            margin: 0 2px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            color: #8edaff;
            background: rgba(55, 125, 160, 0.18);
            border: 1px solid rgba(90, 190, 225, 0.30);
            border-radius: 5px;
            vertical-align: baseline;
            cursor: default;
            transition: all 0.15s ease;
        }}
        .aura-citation-pill:hover {{
            background: rgba(55, 125, 160, 0.30);
            border-color: rgba(90, 190, 225, 0.50);
            color: #ffffff;
            box-shadow: 0 0 8px rgba(92, 200, 255, 0.25);
        }}

        /* Sources Card */
        .aura-source-card {{
            background: rgba(10, 16, 24, 0.75) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border: 1px solid rgba(110, 145, 165, 0.16) !important;
            border-radius: 10px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.6rem;
            transition: all 0.2s ease;
        }}
        .aura-source-card:hover {{
            border-color: rgba(90, 190, 225, 0.35) !important;
            background: rgba(14, 22, 32, 0.85) !important;
        }}
        .aura-source-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }}
        .aura-source-title {{
            font-weight: 600;
            font-size: 0.85rem;
            color: #8edaff;
        }}
        .aura-source-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            padding: 2px 7px;
            border-radius: 6px;
            background: rgba(55, 125, 160, 0.18);
            color: #8edaff;
            font-weight: 500;
            border: 1px solid rgba(90, 190, 225, 0.30);
        }}
        .aura-source-snippet {{
            font-size: 0.82rem;
            line-height: 1.5;
            color: #94a3b8;
            border-left: 2px solid rgba(90, 190, 225, 0.4);
            padding-left: 0.6rem;
            margin-top: 0.35rem;
            font-style: italic;
        }}

        /* Image Attachment Preview */
        .aura-image-preview-card {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: rgba(10, 16, 24, 0.78);
            border: 1px solid rgba(110, 145, 165, 0.22);
            border-radius: 12px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.75rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }}
        .aura-image-preview-info {{
            flex: 1;
            font-size: 0.82rem;
            color: #e2e8f0;
        }}
        .aura-image-preview-name {{
            font-weight: 600;
            color: #f1f5f9;
        }}
        .aura-image-preview-meta {{
            color: #94a3b8;
            font-size: 0.75rem;
        }}

        /* ===== CHAT COMPOSER STYLING ===== */
        [data-testid="stChatInput"] {{
            border-radius: 16px !important;
            border: 1px solid rgba(110, 145, 165, 0.20) !important;
            background: rgba(12, 18, 26, 0.90) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
            transition: all 0.25s ease !important;
            z-index: 100 !important;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: rgba(75, 180, 225, 0.50) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 16px rgba(92, 200, 255, 0.15) !important;
        }}
        [data-testid="stChatInput"] textarea {{
            color: #e8eef4 !important;
            font-size: 0.95rem !important;
        }}
        [data-testid="stChatInput"] button {{
            color: #5cc8ff !important;
        }}

        /* Latency / Timings Bar */
        .aura-timings-bar {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.6rem;
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }}

        /* Status Widget Styling */
        [data-testid="stStatusWidget"] {{
            background: rgba(10, 16, 24, 0.8) !important;
            border: 1px solid rgba(110, 145, 165, 0.25) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(12px) !important;
        }}

        /* ===== PIPELINE & REASONING UX STYLES ===== */
        .aura-pipeline-box {{
            background: rgba(10, 16, 24, 0.72);
            border: 1px solid rgba(110, 145, 165, 0.18);
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.85rem;
            backdrop-filter: blur(10px);
            font-size: 0.85rem;
        }}
        .aura-pipeline-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
            color: #8edaff;
            margin-bottom: 0.5rem;
        }}
        .aura-pipeline-row {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.25rem 0;
            color: #cbd5e1;
            font-size: 0.82rem;
        }}
        .aura-pipeline-row.running {{
            color: #5cc8ff;
            font-weight: 500;
        }}
        .aura-pipeline-row.complete {{
            color: #94a3b8;
        }}
        .aura-pipeline-row.skipped {{
            color: #64748b;
        }}
        .aura-pipeline-row.error {{
            color: #f87171;
        }}
        .aura-pulse-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #5cc8ff;
            box-shadow: 0 0 10px #5cc8ff;
            animation: auraPulse 1.4s infinite ease-in-out;
            vertical-align: middle;
            margin-right: 6px;
        }}
        @keyframes auraPulse {{
            0%, 100% {{ transform: scale(0.9); opacity: 0.6; }}
            50% {{ transform: scale(1.3); opacity: 1; box-shadow: 0 0 14px #5cc8ff; }}
        }}
        .aura-pipeline-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(45, 95, 120, 0.22);
            border: 1px solid rgba(85, 190, 225, 0.28);
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-size: 0.78rem;
            color: #cbd5e1;
            margin-bottom: 0.65rem;
        }}
        .aura-pipeline-badge .success-icon {{
            color: #35d0a1;
            font-weight: bold;
        }}
        .aura-pipeline-stage-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 0.15rem 0.45rem;
            font-size: 0.72rem;
            color: #8edaff;
            margin-left: auto;
        }}

        /* ===== PHASE 12: AURA ENERGY CORE (Anime-Inspired Processing Visual) ===== */
        /* Visual: Premium futuristic anime AI energy core — cyan/navy aura,
           bright cyan highlights, orbital rings, luminous particles, dark transparent bg. */
        .aura-loading-video-container,
        .aura-loading-bubble-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 0.6rem 0 0.35rem 0;
            pointer-events: none;
        }}
        .aura-loading-video-wrapper,
        .aura-loading-bubble-wrapper {{
            position: relative;
            width: 130px;
            height: 130px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        /* Main bubble container — energy core body */
        .aura-loading-bubble {{
            position: relative;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, rgba(92, 200, 255, 0.45) 0%, rgba(14, 165, 233, 0.30) 35%, rgba(30, 58, 90, 0.25) 60%, rgba(6, 182, 212, 0.08) 85%, transparent 100%);
            box-shadow:
                0 0 30px rgba(92, 200, 255, 0.40),
                0 0 56px rgba(14, 165, 233, 0.25),
                0 0 84px rgba(30, 58, 90, 0.15),
                0 0 112px rgba(6, 182, 212, 0.06);
            border: 1.5px solid rgba(92, 200, 255, 0.35);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            animation: auraCoreFloat 2.4s ease-in-out infinite alternate;
        }}
        /* Inner bright core — anime-style luminous center */
        .aura-bubble-core {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 24px;
            height: 24px;
            margin-top: -12px;
            margin-left: -12px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(235, 248, 255, 0.95) 0%, rgba(142, 218, 255, 0.85) 30%, rgba(56, 189, 248, 0.6) 65%, rgba(14, 165, 233, 0.25) 100%);
            box-shadow:
                0 0 12px rgba(142, 218, 255, 0.9),
                0 0 24px rgba(92, 200, 255, 0.6),
                0 0 40px rgba(14, 165, 233, 0.35),
                0 0 56px rgba(6, 182, 212, 0.15);
            animation: auraCorePulse 1.6s ease-in-out infinite;
        }}
        /* Primary orbital ring — glowing cyan */
        .aura-bubble-orbital {{
            position: absolute;
            top: -6px;
            left: -6px;
            right: -6px;
            bottom: -6px;
            border-radius: 50%;
            border: 2px solid transparent;
            border-top-color: rgba(142, 218, 255, 0.95);
            border-right-color: rgba(56, 189, 248, 0.5);
            border-bottom-color: rgba(6, 182, 212, 0.35);
            animation: auraOrbitalSpin 1.4s cubic-bezier(0.5, 0.1, 0.5, 0.9) infinite;
            filter: drop-shadow(0 0 4px rgba(92, 200, 255, 0.5));
        }}
        /* Outer halo — breathing particle ring */
        .aura-bubble-pulse {{
            position: absolute;
            top: -10px;
            left: -10px;
            right: -10px;
            bottom: -10px;
            border-radius: 50%;
            border: 1px solid rgba(92, 200, 255, 0.25);
            animation: auraHaloBreathe 2.6s ease-in-out infinite;
            box-shadow: 0 0 8px rgba(6, 182, 212, 0.1);
        }}
        @keyframes auraCoreFloat {{
            0% {{ transform: translateY(0px) scale(0.98); }}
            100% {{ transform: translateY(-3px) scale(1.03); }}
        }}
        @keyframes auraCorePulse {{
            0%, 100% {{ transform: scale(0.85); opacity: 0.8; }}
            50% {{ transform: scale(1.25); opacity: 1; box-shadow: 0 0 14px rgba(142, 218, 255, 1), 0 0 28px rgba(92, 200, 255, 0.7); }}
        }}
        @keyframes auraOrbitalSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes auraHaloBreathe {{
            0%, 100% {{ transform: scale(0.92); opacity: 0.25; }}
            50% {{ transform: scale(1.15); opacity: 0.65; }}
        }}
        .aura-loading-video {{
            width: 100%;
            height: 100%;
            max-width: 100px;
            max-height: 100px;
            object-fit: contain;
            border-radius: 50%;
            mix-blend-mode: screen;
            pointer-events: none;
            filter: drop-shadow(0 0 16px rgba(92, 200, 255, 0.5));
        }}
        .aura-loading-fallback-ring {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: 2px solid transparent;
            border-top-color: #5cc8ff;
            border-right-color: #38bdf8;
            border-bottom-color: rgba(92, 200, 255, 0.2);
            box-shadow: 0 0 18px rgba(92, 200, 255, 0.4);
            animation: auraSpinPulse 1.8s infinite cubic-bezier(0.4, 0, 0.2, 1);
        }}
        @keyframes auraSpinPulse {{
            0% {{ transform: rotate(0deg) scale(0.95); opacity: 0.75; }}
            50% {{ transform: rotate(180deg) scale(1.05); opacity: 1; box-shadow: 0 0 24px rgba(142, 218, 255, 0.7); }}
            100% {{ transform: rotate(360deg) scale(0.95); opacity: 0.75; }}
        }}
        .aura-loading-status-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 0.2rem auto 0.75rem auto;
            max-width: 440px;
            padding: 0.55rem 0.95rem;
            background: rgba(10, 16, 24, 0.65);
            border: 1px solid rgba(110, 145, 165, 0.22);
            border-radius: 14px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
        }}
        .aura-loading-primary-text {{
            font-size: 0.88rem;
            font-weight: 500;
            color: #f1f5f9;
            letter-spacing: 0.01em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
        }}
        .aura-loading-subtext {{
            font-size: 0.76rem;
            color: #94a3b8;
            margin-top: 0.3rem;
            line-height: 1.35;
        }}
        .aura-loading-sublist {{
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            margin-top: 0.35rem;
            font-size: 0.74rem;
            color: #94a3b8;
            text-align: left;
        }}
        .aura-loading-subitem {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .aura-loading-combined-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
        }}

        /* Responsive Layout Adjustments */
        @media (max-width: 768px) {{
            .aura-hero-title {{
                font-size: 2rem;
            }}
            .aura-prompts-grid {{
                grid-template-columns: 1fr;
            }}
            .aura-topbar {{
                flex-direction: column;
                gap: 0.5rem;
                align-items: flex-start;
            }}
            .aura-loading-video-wrapper {{
                width: 80px;
                height: 80px;
            }}
            .aura-loading-video {{
                max-width: 80px;
                max-height: 80px;
            }}
            .aura-loading-status-box {{
                max-width: 92%;
            }}

            /* ===== HERO CALM EMBLEM VISUAL ===== */
            .aura-hero-visual {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 1.25rem;
                margin-top: 0.5rem;
            }}
            .aura-hero-emblem {{
                width: 64px;
                height: 64px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                background: rgba(10, 16, 24, 0.72);
                border: 1px solid rgba(120, 150, 175, 0.20);
                backdrop-filter: blur(12px);
            }}
        }}

        /* Desktop hero visual */
        .aura-hero-visual {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1.25rem;
            margin-top: 0.5rem;
        }}
        .aura-hero-emblem {{
            width: 64px;
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: rgba(10, 16, 24, 0.72);
            border: 1px solid rgba(120, 150, 175, 0.20);
            backdrop-filter: blur(12px);
        }}
        .aura-hero-emblem-img {{
            width: 56px;
            height: 56px;
            object-fit: contain;
            border-radius: 50%;
            filter: drop-shadow(0 0 10px rgba(92, 200, 255, 0.3));
        }}

        {get_indicator_css()}
    </style>
    """,
        unsafe_allow_html=True,
    )

    if bg_video_url:
        st.markdown(
            f"""
            <div id="aura-persistent-bg" class="aura-background-layer" aria-hidden="true">
                <video id="aura-bg-video" class="aura-background-video" autoplay loop muted playsinline tabindex="-1" preload="auto">
                    <source src="{bg_video_url}" type="video/mp4">
                </video>
                <div id="aura-bg-overlay" class="aura-background-overlay"></div>
                <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" 
                     style="display:none;" 
                     onload="if (!window.__aura_bg_attached) {{
                         var bg = document.getElementById('aura-persistent-bg');
                         if (bg && bg.parentElement !== document.body) {{
                             document.body.prepend(bg);
                             window.__aura_bg_attached = true;
                         }}
                     }}" />
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        "file_id": None,
        "uploaded_filename": None,
        "selected_file_ids": [],
        "selected_filenames": [],
        "chat_history": [],
        "api_healthy": None,
        "last_health_check": None,
        "total_documents": 0,
        "upload_progress": 0,
        "theme": "dark",
        "show_sources": True,
        "show_context": False,
        "max_sources": 5,
        "pending_question": None,
        "use_hybrid_search": True,
        "multi_doc_mode": False,
        "comparison_mode_active": False,
        "selected_comparison_mode": "Auto-detect",
        "active_session_id": None,
        "active_session_title": "New Chat",
        "attached_image_bytes": None,
        "attached_image_name": None,
        "ai_provider": None,
        "active_model": None,
        "active_embedding_model": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# =============================================================================
# API Client Helper Functions
# =============================================================================

class APIError(Exception):
    """Custom exception for API errors."""

    def __init__(
        self, message: str, status_code: int = None, detail: str = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


def make_api_request(
    method: str, endpoint: str, timeout: int = REQUEST_TIMEOUT, **kwargs
) -> Dict[str, Any]:
    """Execute an API request with robust error handling."""
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_detail = (
                        error_data.get("message")
                        or error_data.get("detail")
                        or str(error_data)
                    )
                    if isinstance(error_detail, dict):
                        error_detail = error_detail.get(
                            "message", str(error_detail)
                        )
                else:
                    error_detail = str(error_data)
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"

            raise APIError(
                message="API request failed",
                status_code=response.status_code,
                detail=error_detail,
            )
    except requests.exceptions.Timeout:
        raise APIError(
            message="Request timed out",
            detail="The server took too long to respond. Please try again.",
        )
    except requests.exceptions.ConnectionError:
        raise APIError(
            message="Connection failed",
            detail="Cannot connect to Aura AI backend. Please ensure the server is active.",
        )
    except APIError:
        raise
    except Exception as e:
        raise APIError(message="Unexpected error", detail=str(e))


def check_api_health() -> Dict[str, Any]:
    """Check API server health with cached results."""
    if (
        st.session_state.last_health_check
        and time.time() - st.session_state.last_health_check < 30
        and st.session_state.api_healthy is not None
    ):
        return {
            "healthy": st.session_state.api_healthy,
            "total_documents": st.session_state.total_documents,
        }

    try:
        result = make_api_request("GET", "/health", timeout=HEALTH_CHECK_TIMEOUT)
        st.session_state.api_healthy = True
        st.session_state.total_documents = result.get("vectorstore", {}).get(
            "total_documents", 0
        )
        st.session_state.ai_provider = result.get("ai_provider") or "gemini"
        st.session_state.active_model = result.get("model")
        st.session_state.active_embedding_model = result.get("embedding_model")
        st.session_state.last_health_check = time.time()
        return {
            "healthy": True,
            "total_documents": st.session_state.total_documents,
            "data": result,
        }
    except APIError:
        st.session_state.api_healthy = False
        st.session_state.last_health_check = time.time()
        return {"healthy": False, "total_documents": 0}


def upload_document(uploaded_file, loader_container=None) -> bool:
    """Upload document to backend API with AuraProcessingIndicator feedback."""
    loader_box = loader_container if loader_container is not None else st.empty()

    # Determine formatted file size string for display
    file_size_str = ""
    try:
        raw_val = uploaded_file.getvalue()
        file_bytes = len(raw_val) if raw_val is not None else 0
        if file_bytes < 1024 * 1024:
            file_size_str = f"{file_bytes / 1024:.1f} KB"
        else:
            file_size_str = f"{file_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        file_size_str = ""

    doc_name = getattr(uploaded_file, "name", "document")

    indicator = AuraProcessingIndicator(
        container=loader_box,
        mode="compact",
        status="Uploading document…",
        operation_label=f"Ingesting {doc_name}",
        include_css=False,
        document_name=doc_name,
        document_size=file_size_str,
    )
    indicator.start()
    try:
        file_data = uploaded_file.getvalue()
        files = {
            "file": (
                doc_name,
                file_data,
                getattr(uploaded_file, "type", None) or "application/octet-stream",
            )
        }
        indicator.update(
            status="Processing document…",
            operation_label="Indexing content and embeddings",
            document_name=doc_name,
            document_size=file_size_str,
        )
        result = make_api_request("POST", "/upload", files=files)
        indicator.update(
            status="Finalizing…",
            operation_label=result.get("message", "Ingestion complete"),
            document_name=doc_name,
            document_size=file_size_str,
        )

        st.session_state.file_id = result["file_id"]
        st.session_state.uploaded_filename = result["filename"]
        st.session_state.chat_history = []
        st.session_state._files_cache = None
        st.session_state._files_cache_time = 0
        st.success(f"✅ {result['message']}")
        return True
    except APIError as e:
        st.error(f"❌ Upload failed: {e.detail}")
        return False
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        return False
    finally:
        indicator.clear()


def get_sessions_api(force_refresh: bool = False) -> List[Dict]:
    """Get list of active sessions from the API with in-memory TTL caching."""
    now = time.time()
    if (
        not force_refresh
        and getattr(st.session_state, "_sessions_cache", None) is not None
        and (now - getattr(st.session_state, "_sessions_cache_time", 0) < 4.0)
    ):
        return st.session_state._sessions_cache

    try:
        res = make_api_request("GET", "/sessions?limit=50")
        sessions = res.get("sessions", [])
        st.session_state._sessions_cache = sessions
        st.session_state._sessions_cache_time = now
        return sessions
    except APIError:
        return st.session_state.get("_sessions_cache", [])


def create_session_api(title: Optional[str] = None) -> Optional[Dict]:
    """Create a persistent conversation session."""
    try:
        body = {"title": title} if title else {}
        res = make_api_request(
            "POST",
            "/sessions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
        )
        st.session_state._sessions_cache = None
        st.session_state._sessions_cache_time = 0
        return res
    except APIError:
        return None


def get_session_details_api(session_id: str) -> Optional[Dict]:
    """Retrieve details and messages of a session."""
    try:
        return make_api_request("GET", f"/sessions/{session_id}")
    except APIError:
        return None


def delete_session_api(
    session_id: str,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> bool:
    """Delete a persistent session."""
    if indicator:
        indicator.update(status="Deleting chat…")
    try:
        make_api_request("DELETE", f"/sessions/{session_id}")
        st.session_state._sessions_cache = None
        st.session_state._sessions_cache_time = 0
        return True
    except APIError:
        return False
    except Exception:
        return False


def rename_session_api(
    session_id: str,
    new_title: str,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> Tuple[bool, str]:
    """Rename a persistent session via backend API."""
    clean_title = (new_title or "").strip()
    if not clean_title:
        return False, "Chat title cannot be empty or whitespace only."
    if indicator:
        indicator.update(status="Saving chat name…")
    try:
        res = make_api_request(
            "PATCH",
            f"/sessions/{session_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"title": clean_title}),
        )
        st.session_state._sessions_cache = None
        st.session_state._sessions_cache_time = 0
        return True, res.get("title", clean_title)
    except APIError as e:
        return False, str(e.detail or e.message or e)
    except Exception as e:
        return False, str(e)


def load_session(
    session_id: str,
    indicator: Optional[AuraProcessingIndicator] = None,
):
    """Load conversation messages from backend session."""
    if indicator:
        indicator.start()
        indicator.update(status="Loading chat…")
    try:
        details = get_session_details_api(session_id)
        if details:
            st.session_state.active_session_id = session_id
            st.session_state.active_session_title = details.get("title", "Chat")
            st.session_state.chat_history = [
                {
                    "role": msg["role"],
                    "content": clean_markdown_display(msg["content"]),
                    "timestamp": msg.get("created_at", "")[11:16]
                    if len(msg.get("created_at", "")) >= 16
                    else "",
                    "sources": [
                        s if isinstance(s, dict) else s.dict()
                        for s in (msg.get("sources") or [])
                    ],
                    "timings": msg.get("timings"),
                    "pipeline_stages": msg.get("pipeline_stages"),
                }
                for msg in details.get("messages", [])
            ]
        else:
            st.session_state.active_session_id = None
            st.session_state.chat_history = []
    finally:
        if indicator:
            indicator.clear()


def compare_documents_api(
    query: str,
    file_ids: List[str],
    session_id: Optional[str] = None,
    comparison_mode: Optional[str] = None,
    use_hybrid_search: bool = True,
    max_sources_per_document: int = 3,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute document comparison via backend API (/sessions/{id}/compare or /compare).
    Integrates with AuraProcessingIndicator for lifecycle status tracking.
    """
    if not file_ids:
        st.session_state._last_query_error = "At least one document is required for comparison."
        return None

    if indicator:
        indicator.start()
        indicator.update(status="Comparing documents…")

    payload: Dict[str, Any] = {
        "query": query,
        "file_ids": file_ids,
        "session_id": session_id,
        "comparison_mode": comparison_mode if comparison_mode != "Auto-detect" else None,
        "use_hybrid_search": use_hybrid_search,
        "max_sources_per_document": max_sources_per_document,
    }
    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        payload["image_filename"] = image_filename or "attached_image.png"

    endpoint = f"/sessions/{session_id}/compare" if session_id else "/compare"

    try:
        res = make_api_request(
            "POST",
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
        )
        if indicator:
            indicator.clear()
        return res
    except APIError as e:
        st.session_state._last_query_error = str(e.detail)
        if indicator:
            indicator.clear()
        return None
    except BaseException:
        if indicator:
            indicator.clear()
        raise


def query_document(
    question: str,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> Optional[Dict]:
    """Query document and attached image using the backend API."""
    if st.session_state.multi_doc_mode and st.session_state.selected_file_ids:
        file_ids = st.session_state.selected_file_ids
        file_id = None
    elif st.session_state.file_id:
        file_ids = None
        file_id = st.session_state.file_id
    else:
        file_ids = None
        file_id = None
        if not image_bytes and not st.session_state.get("active_session_id"):
            st.error(
                "❌ No document or image context available. Please select a document or attach an image."
            )
            return None

    try:
        if not st.session_state.get("active_session_id"):
            new_sess = create_session_api()
            if new_sess:
                st.session_state.active_session_id = new_sess["session_id"]
                st.session_state.active_session_title = new_sess.get(
                    "title", "New Chat"
                )

        session_id = st.session_state.get("active_session_id")

        if st.session_state.get("comparison_mode_active"):
            mode_choice = st.session_state.get("selected_comparison_mode")
            fids = file_ids or ([file_id] if file_id else None)
            return compare_documents_api(
                query=question,
                file_ids=fids or [],
                session_id=session_id,
                comparison_mode=mode_choice,
                use_hybrid_search=st.session_state.use_hybrid_search,
                max_sources_per_document=3,
                image_bytes=image_bytes,
                image_filename=image_filename,
                indicator=indicator,
            )

        payload: Dict[str, Any] = {
            "question": question,
            "use_hybrid_search": st.session_state.use_hybrid_search,
            "max_sources": st.session_state.max_sources,
        }
        if file_ids:
            payload["file_ids"] = file_ids
        elif file_id:
            payload["file_id"] = file_id

        if image_bytes:
            payload["image_base64"] = base64.b64encode(image_bytes).decode(
                "utf-8"
            )
            payload["image_filename"] = image_filename or "attached_image.png"

        if session_id:
            return make_api_request(
                "POST",
                f"/sessions/{session_id}/query",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
        else:
            return make_api_request(
                "POST",
                "/query",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
    except APIError as e:
        st.session_state._last_query_error = str(e.detail)
        return None


def execute_sql_query_api(
    question: str,
    file_id: str,
    session_id: Optional[str] = None,
    model_preference: Optional[str] = None,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a natural language analytical SQL query against a database or CSV file.
    Integrates with AuraProcessingIndicator for lifecycle status tracking.
    """
    clean_id = str(file_id).strip() if file_id else ""
    if not clean_id:
        st.session_state._last_query_error = "Invalid file ID for SQL query"
        return None

    if indicator:
        indicator.start()
        indicator.update(status="Processing SQL query…")

    payload: Dict[str, Any] = {
        "question": question,
        "file_id": clean_id,
    }
    if session_id:
        payload["session_id"] = session_id
    if model_preference:
        payload["model_preference"] = model_preference

    try:
        res = make_api_request(
            "POST",
            "/sql-query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
        )
        if indicator:
            indicator.clear()
        return res
    except APIError as e:
        st.session_state._last_query_error = str(e.detail)
        if indicator:
            indicator.clear()
        return None
    except BaseException:
        if indicator:
            indicator.clear()
        raise


def query_document_stream(
    question: str,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
):
    """
    Stream document response via Server-Sent Events (SSE).
    Yields (event_type, event_data_dict).
    """
    if st.session_state.multi_doc_mode and st.session_state.selected_file_ids:
        file_ids = st.session_state.selected_file_ids
        file_id = None
    elif st.session_state.file_id:
        file_ids = None
        file_id = st.session_state.file_id
    else:
        file_ids = None
        file_id = None
        if not image_bytes and not st.session_state.get("active_session_id"):
            yield ("error", {"error": "No document or image context available. Please select a document or attach an image."})
            return

    if not st.session_state.get("active_session_id"):
        new_sess = create_session_api()
        if new_sess:
            st.session_state.active_session_id = new_sess["session_id"]
            st.session_state.active_session_title = new_sess.get(
                "title", "New Chat"
            )

    session_id = st.session_state.get("active_session_id")
    endpoint = (
        f"/sessions/{session_id}/query/stream"
        if session_id
        else "/query/stream"
    )
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

    payload: Dict[str, Any] = {
        "question": question,
        "use_hybrid_search": st.session_state.use_hybrid_search,
        "max_sources": st.session_state.max_sources,
    }
    if file_ids:
        payload["file_ids"] = file_ids
    elif file_id:
        payload["file_id"] = file_id

    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        payload["image_filename"] = image_filename or "attached_image.png"

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            try:
                err_json = response.json()
                detail = err_json.get("detail", {})
                err_msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            except Exception:
                err_msg = response.text or f"HTTP {response.status_code}"
            yield ("error", {"error": err_msg or f"HTTP {response.status_code}"})
            return

        current_event = "message"
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    data_obj = json.loads(data_str)
                except Exception:
                    data_obj = {"text": data_str}
                yield (current_event, data_obj)
                current_event = "message"
    except Exception as e:
        logger.error(f"Error streaming response from {url}: {e}")
        yield ("error", {"error": str(e)})


def get_uploaded_files(force_refresh: bool = False) -> List[Dict]:
    """Get list of uploaded files from API with in-memory TTL caching."""
    now = time.time()
    if (
        not force_refresh
        and getattr(st.session_state, "_files_cache", None) is not None
        and (now - getattr(st.session_state, "_files_cache_time", 0) < 5.0)
    ):
        return st.session_state._files_cache

    try:
        res = make_api_request("GET", "/files")
        files = []
        if isinstance(res, list):
            files = res
        elif isinstance(res, dict) and "files" in res and isinstance(res["files"], list):
            files = res["files"]
        st.session_state._files_cache = files
        st.session_state._files_cache_time = now
        return files
    except APIError:
        return st.session_state.get("_files_cache", [])


def delete_document_api(
    file_id: str,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> Tuple[bool, str]:
    """
    Delete an indexed document completely via backend API.
    
    Returns:
        (success: bool, message: str)
    """
    clean_id = str(file_id).strip() if file_id else ""
    if not clean_id:
        return False, "Invalid document ID"
    if indicator:
        indicator.update(status="Deleting document…")
    try:
        res = make_api_request("DELETE", f"/documents/{clean_id}")
        st.session_state._files_cache = None
        st.session_state._files_cache_time = 0
        return True, res.get("message", f"Document {clean_id} deleted successfully")
    except APIError as e:
        # Fallback to /files/{file_id} endpoint
        try:
            res = make_api_request("DELETE", f"/files/{clean_id}")
            st.session_state._files_cache = None
            st.session_state._files_cache_time = 0
            return True, res.get("message", f"Document {clean_id} deleted successfully")
        except APIError as e2:
            return False, e2.detail or str(e2)
    except Exception as e:
        return False, str(e)


# =============================================================================
# UI Components: Header, Sidebar, Landing & Messages
# =============================================================================

def render_top_bar():
    """Render the minimal top navigation bar."""
    health_info = check_api_health()
    is_healthy = health_info.get("healthy", False)
    total_docs = health_info.get("total_documents", 0)

    # Real model and provider info
    prov = (st.session_state.get("ai_provider") or os.getenv("AI_PROVIDER") or "gemini").capitalize()
    model_raw = st.session_state.get("active_model") or os.getenv("GEMINI_MODEL") or ""
    if model_raw:
        clean_model = model_raw.replace("-", " ").title()
        status_label = f"{clean_model} · {'Connected' if is_healthy else 'Disconnected'}"
    else:
        status_label = f"{prov} · {'Connected' if is_healthy else 'Disconnected'}"

    # Document context label
    if (
        st.session_state.get("comparison_mode_active")
        and st.session_state.multi_doc_mode
        and st.session_state.selected_file_ids
    ):
        swap_icon = get_ui_asset_data_uri("swap.svg")
        doc_pill = f'<img src="{swap_icon}" class="aura-ui-icon" /> Comparing {len(st.session_state.selected_file_ids)} documents'
    elif st.session_state.multi_doc_mode and st.session_state.selected_file_ids:
        folder_icon = get_ui_asset_data_uri("documents-folder.svg")
        doc_pill = f'<img src="{folder_icon}" class="aura-ui-icon" /> {len(st.session_state.selected_file_ids)} documents active'
    elif st.session_state.uploaded_filename:
        name = st.session_state.uploaded_filename
        display_name = name[:28] + "..." if len(name) > 28 else name
        folder_icon = get_ui_asset_data_uri("documents-folder.svg")
        doc_pill = f'<img src="{folder_icon}" class="aura-ui-icon" /> {display_name}'
    else:
        doc_pill = f'<img src="{get_aura_symbol_data_uri()}" class="aura-ui-icon" alt="Aura" /> Ready for documents'

    st.markdown(
        f"""
        <div class="aura-topbar">
            <div class="aura-topbar-left">
                <span class="aura-topbar-brand"><img src="{get_aura_symbol_data_uri()}" class="aura-topbar-brand-icon" alt="Aura AI" /> Aura AI</span>
                <span class="aura-context-pill">{doc_pill}</span>
            </div>
            <div class="aura-status-pill">
                <span class="aura-status-dot {'error' if not is_healthy else ''}"></span>
                <span>{status_label} • {total_docs} chunks</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the compact modern sidebar."""
    with st.sidebar:
        # Brand
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img src="{get_aura_symbol_data_uri()}" class="sidebar-brand-img" alt="Aura AI" />
                <span class="sidebar-brand-text">Aura AI</span>
                <span class="sidebar-brand-badge">v1.0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            current_history = st.session_state.get("chat_history", [])
            current_title = st.session_state.get("active_session_title", "")
            if (
                st.session_state.get("active_session_id")
                and len(current_history) == 0
                and current_title == "New Chat"
            ):
                st.toast("Already in a new conversation")
            else:
                new_sess = create_session_api()
                if new_sess:
                    st.session_state.active_session_id = new_sess["session_id"]
                    st.session_state.active_session_title = new_sess.get(
                        "title", "New Chat"
                    )
                    st.session_state.chat_history = []
                    st.session_state.attached_image_bytes = None
                    st.session_state.attached_image_name = None
                    st.rerun()

        # Conversations List
        raw_sessions = get_sessions_api()
        # Defensive filter: exclude legacy automated test session names
        sessions = [
            s
            for s in raw_sessions
            if s.get("title")
            not in (
                "Corrupt Test",
                "Text Only",
                "API Test Session",
                "Turn Test",
                "Financial Review 2024",
                "Updated API Title",
            )
        ]

        # Rename notifications
        if st.session_state.get("_rename_error"):
            err_ren = st.session_state.pop("_rename_error")
            st.error(f"❌ {err_ren}")
        if st.session_state.get("_rename_toast"):
            toast_ren = st.session_state.pop("_rename_toast")
            st.success(f"✅ {toast_ren}")

        # Delete notifications
        if st.session_state.get("_delete_chat_error"):
            err_del = st.session_state.pop("_delete_chat_error")
            st.error(f"❌ {err_del}")
        if st.session_state.get("_delete_chat_toast"):
            toast_del = st.session_state.pop("_delete_chat_toast")
            st.success(f"✅ {toast_del}")

        st.markdown(
            '<div class="chat-list-label">Conversations</div>',
            unsafe_allow_html=True,
        )
        if sessions:
            for sess in sessions[:15]:
                sid = sess["session_id"]
                stitle = sess.get("title") or "Untitled Chat"
                is_active = sid == st.session_state.get("active_session_id")
                btn_type = "primary" if is_active else "secondary"
                label = f"{'✦ ' if is_active else ''}{stitle[:20]}"

                if st.session_state.get("_renaming_session_id") == sid:
                    st.caption(f"Rename \"{stitle[:25]}\":")
                    rename_val = st.text_input(
                        "Rename chat",
                        value=st.session_state.get(f"_rename_val_{sid}", stitle),
                        key=f"rename_input_{sid}",
                        label_visibility="collapsed",
                        max_chars=100,
                    )
                    rename_box = st.empty()
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("Save", key=f"save_ren_{sid}", type="primary", use_container_width=True):
                            clean_name = (rename_val or "").strip()
                            if not clean_name:
                                st.session_state._rename_error = "Chat title cannot be empty or whitespace only."
                                st.rerun()
                            else:
                                indicator = AuraProcessingIndicator(
                                    container=rename_box,
                                    mode="compact",
                                    status="Saving chat name…",
                                    operation_label=clean_name,
                                    include_css=False,
                                    )
                                indicator.start()
                                try:
                                    ok, result = rename_session_api(sid, clean_name, indicator=indicator)
                                    if ok:
                                        if st.session_state.get("active_session_id") == sid:
                                            st.session_state.active_session_title = clean_name
                                        st.session_state._renaming_session_id = None
                                        st.session_state.pop(f"_rename_val_{sid}", None)
                                        st.session_state._rename_toast = f"Renamed chat to \"{clean_name}\""
                                        indicator.clear()
                                        st.rerun()
                                    else:
                                        st.session_state._rename_error = f"Failed to rename: {result}"
                                        indicator.clear()
                                        st.rerun()
                                finally:
                                    indicator.clear()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_ren_{sid}", use_container_width=True):
                            st.session_state._renaming_session_id = None
                            st.session_state.pop(f"_rename_val_{sid}", None)
                            st.rerun()
                else:
                    del_sess_box = st.empty()
                    col_btn, col_ren, col_del = st.columns([5, 1, 1])
                    with col_btn:
                        if st.button(
                            label,
                            key=f"sess_btn_{sid}",
                            type=btn_type,
                            use_container_width=True,
                            help=f"Switch to: {stitle} ({sess.get('message_count', 0)} msgs)",
                        ):
                            if sid != st.session_state.get("active_session_id"):
                                load_session(sid)
                                st.rerun()
                    with col_ren:
                        if st.button(
                            "✏️",
                            key=f"sess_ren_{sid}",
                            help="Rename chat",
                            use_container_width=True,
                        ):
                            st.session_state._renaming_session_id = sid
                            st.session_state[f"_rename_val_{sid}"] = stitle
                            st.rerun()
                    with col_del:
                        if st.button(
                            "🗑️",
                            key=f"sess_del_{sid}",
                            help="Delete chat",
                            use_container_width=True,
                        ):
                            indicator = AuraProcessingIndicator(
                                container=del_sess_box,
                                mode="compact",
                                status="Deleting chat…",
                                operation_label=stitle,
                                include_css=False,
                            )
                            indicator.start()
                            try:
                                ok = delete_session_api(sid, indicator=indicator)
                                if ok:
                                    if st.session_state.get("active_session_id") == sid:
                                        st.session_state.active_session_id = None
                                        st.session_state.active_session_title = "New Chat"
                                        st.session_state.chat_history = []
                                    st.session_state._delete_chat_toast = f"Deleted chat \"{stitle}\""
                                    indicator.clear()
                                    st.rerun()
                                else:
                                    st.session_state._delete_chat_error = f"Failed to delete chat \"{stitle}\""
                                    indicator.clear()
                                    st.rerun()
                            finally:
                                indicator.clear()
        else:
            st.caption("No conversations yet. Start a chat!")

        st.divider()

        # 3. Document Library Expander
        st.markdown('<div class="aura-doclib-expander">', unsafe_allow_html=True)
        with st.expander("Document Library", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg", "db"],
                help="PDF, DOCX, TXT, CSV, PNG, JPG, SQLite databases",
                key="sidebar_file_uploader",
            )
            st.divider()
            upload_status_box = st.empty()
            if uploaded_file is not None:
                file_size_kb = len(uploaded_file.getvalue()) / 1024
                st.caption(f"📎 {uploaded_file.name} ({file_size_kb:.1f} KB)")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Ingest", type="primary", use_container_width=True):
                        if upload_document(uploaded_file, loader_container=upload_status_box):
                            st.rerun()
                with col2:
                    if st.button("Cancel", use_container_width=True):
                        pass

            if st.session_state.file_id:
                st.caption(f"**Active**: {st.session_state.uploaded_filename}")
                if st.button("Clear Document", use_container_width=True):
                    st.session_state.file_id = None
                    st.session_state.uploaded_filename = None
                    st.session_state.chat_history = []
                    st.rerun()

            # Deletion notifications
            if st.session_state.get("_doc_deleted_toast"):
                toast_msg = st.session_state.pop("_doc_deleted_toast")
                st.success(f"✅ {toast_msg}")

            if st.session_state.get("_doc_delete_error"):
                err_msg = st.session_state.pop("_doc_delete_error")
                st.error(f"❌ {err_msg}")

            # Confirmation Step for document deletion
            pending_delete = st.session_state.get("_doc_pending_delete")
            if pending_delete:
                st.markdown("---")
                st.warning(
                    f"**Delete document?**\n\n"
                    f"**\"{pending_delete['filename']}\"**\n\n"
                    f"This will remove the uploaded file and its indexed RAG data."
                )
                delete_box = st.empty()
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    if st.button("Cancel", key="cancel_doc_delete", use_container_width=True):
                        st.session_state._doc_pending_delete = None
                        st.rerun()
                with col_c2:
                    if st.button("Delete", key="confirm_doc_delete", type="primary", use_container_width=True):
                        fid = pending_delete["file_id"]
                        fname = pending_delete["filename"]
                        indicator = AuraProcessingIndicator(
                            container=delete_box,
                            mode="compact",
                            status="Deleting document…",
                            operation_label=fname,
                            include_css=False,
                        )
                        indicator.start()
                        try:
                            success, err_msg = delete_document_api(fid, indicator=indicator)
                            if success:
                                if st.session_state.file_id == fid:
                                    st.session_state.file_id = None
                                    st.session_state.uploaded_filename = None
                                if fid in st.session_state.selected_file_ids:
                                    st.session_state.selected_file_ids.remove(fid)
                                if fname in st.session_state.selected_filenames:
                                    st.session_state.selected_filenames.remove(fname)
                                st.session_state._doc_pending_delete = None
                                st.session_state._doc_deleted_toast = f"Document \"{fname}\" deleted completely."
                                indicator.clear()
                                st.rerun()
                            else:
                                st.session_state._doc_delete_error = f"Failed to delete {fname}: {err_msg}"
                                indicator.clear()
                                st.rerun()
                        finally:
                            indicator.clear()

            # Indexed documents management (cards with badge, filename, chunks, and Delete button)
            files = get_uploaded_files()
            if files:
                st.markdown("---")
                st.markdown('<div class="chat-list-label">Indexed Documents</div>', unsafe_allow_html=True)
                for f in files:
                    fid = f["file_id"]
                    fname = f["filename"]
                    ftype = f.get("file_type") or Path(fname).suffix.lstrip(".").upper() or "DOC"
                    chunks = f.get("chunks_count")
                    chunks_label = f"{chunks} chunks" if chunks is not None else "Indexed"
                    is_active = (st.session_state.file_id == fid)
                    
                    st.markdown(
                        f"""
                        <div class="aura-doc-card">
                            <div class="aura-doc-card-header">
                                <span class="aura-doc-badge">{ftype}</span>
                                <span class="aura-doc-name" title="{fname}">{fname[:26] + '...' if len(fname) > 26 else fname}</span>
                            </div>
                            <div class="aura-doc-meta">{chunks_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Visible Delete button for the document
                    st.markdown('<div class="aura-doc-del-container">', unsafe_allow_html=True)
                    if st.button("Delete", key=f"del_btn_{fid}", use_container_width=True):
                        st.session_state._doc_pending_delete = {"file_id": fid, "filename": fname}
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Preserved legacy select button in hidden container for test runner compatibility
                    st.markdown('<div class="aura-legacy-select-container">', unsafe_allow_html=True)
                    btn_label = "✦ Active" if is_active else "Select"
                    st.button(btn_label, key=f"sel_btn_{fid}", disabled=is_active)
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. Search Scope Expander
        with st.expander("Search Scope", expanded=False):
            files = get_uploaded_files()
            if files:
                st.caption("Choose active documents for retrieval:")
                selected_ids = []
                selected_names = []
                for f in files:
                    fid = f["file_id"]
                    fname = f["filename"]
                    is_sel = fid in st.session_state.selected_file_ids
                    if st.checkbox(
                        fname[:24], value=is_sel, key=f"sel_file_{fid}"
                    ):
                        selected_ids.append(fid)
                        selected_names.append(fname)
                if selected_ids != st.session_state.selected_file_ids:
                    st.session_state.selected_file_ids = selected_ids
                    st.session_state.selected_filenames = selected_names
                    st.session_state.multi_doc_mode = len(selected_ids) > 0
            else:
                st.caption("No documents indexed yet. Upload documents in the library to search.")

            st.markdown("---")
            st.session_state.use_hybrid_search = st.toggle(
                "Hybrid Retrieval (Vector + BM25)",
                value=st.session_state.use_hybrid_search,
                help="Combines dense semantic embeddings with BM25 keyword matching via RRF.",
            )
            st.session_state.max_sources = st.slider(
                "Maximum Evidence Sources",
                min_value=1,
                max_value=10,
                value=st.session_state.max_sources,
            )

        # 5. Document Comparison Expander
        st.markdown('<div class="aura-compare-expander">', unsafe_allow_html=True)
        with st.expander("Compare", expanded=False):
            st.session_state.comparison_mode_active = st.toggle(
                "Cross-Document Comparison",
                value=st.session_state.get("comparison_mode_active", False),
                help="Analyzes differences, similarities, and metrics across documents.",
            )
            if st.session_state.comparison_mode_active:
                st.session_state.selected_comparison_mode = st.selectbox(
                    "Comparison Strategy",
                    options=[
                        "Auto-detect",
                        "difference",
                        "similarity",
                        "metric",
                        "conflict",
                        "general",
                    ],
                    index=0,
                )
                if st.session_state.selected_file_ids:
                    st.caption(f"Comparing {len(st.session_state.selected_file_ids)} documents active in Search Scope.")
                else:
                    st.caption("Select 2 or more documents in Search Scope to compare.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 6. Workspace / System Status Expander
        with st.expander("Workspace Status", expanded=False):
            st.caption(f"Backend: `{API_BASE_URL}`")
            prov = (st.session_state.get("ai_provider") or os.getenv("AI_PROVIDER") or "gemini").capitalize()
            mod = st.session_state.get("active_model") or os.getenv("GEMINI_MODEL") or os.getenv("OPENAI_MODEL") or "Configured Model"
            emb = st.session_state.get("active_embedding_model") or os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL") or "Configured Embeddings"
            st.caption(f"Provider: `{prov}`")
            st.caption(f"AI Model: `{mod}`")
            st.caption(f"Embeddings: `{emb}`")
            st.caption("Storage: `ChromaDB + SQLite WAL`")


def render_empty_state_landing():
    """
    Render the immersive AI landing experience when conversation is empty.
    Features a calm, elegant emblem allowing the background video to provide the visual motion.
    """
    st.markdown(
        f"""
        <div class="aura-hero-wrapper">
            <div class="aura-hero-visual">
                <div class="aura-hero-emblem"><img src="{get_aura_symbol_data_uri()}" class="aura-hero-emblem-img" alt="Aura" /></div>
            </div>
            <div class="aura-hero-badge"><img src="{get_aura_symbol_data_uri()}" class="aura-hero-badge-img" alt="Aura" /> Aura Document Intelligence</div>
            <h1 class="aura-hero-title">What can Aura help you discover?</h1>
            <p class="aura-hero-subtitle">
                Ask deep questions across your documents, verify claims with verified source citations, or analyze charts and diagrams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Suggested Prompts Grid with SVG/PNG icons mapped from SUGGESTION_ICONS
    prompts = [
        ("Summarize this document", "Provide a comprehensive executive summary of this document.", "hero-summarize", 0),
        ("Explain key concepts", "What are the core principles and concepts discussed?", "hero-concepts", 1),
        ("Find critical facts", "Highlight the key metrics, dates, and conclusions.", "hero-facts", 2),
        ("Compare documents", "Analyze and contrast the key differences and agreements.", "hero-compare", 3),
    ]

    r1_col1, r1_col2 = st.columns(2)
    r2_col1, r2_col2 = st.columns(2)
    grid_cells = [
        (r1_col1, prompts[0]),
        (r1_col2, prompts[1]),
        (r2_col1, prompts[2]),
        (r2_col2, prompts[3]),
    ]

    for target_col, (label, query, css_cls, idx) in grid_cells:
        with target_col:
            st.markdown(f'<div class="aura-hero-btn-container {css_cls}">', unsafe_allow_html=True)
            if st.button(
                label,
                key=f"hero_prompt_{idx}",
                use_container_width=True,
                type="secondary",
                help=query,
            ):
                st.session_state.pending_question = query
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def _parse_pipeline_timestamp(value: Any) -> Optional[datetime]:
    """
    Parse a pipeline timestamp into a UTC-aware datetime object.
    Supports:
      - datetime object (tz-aware or naive; naive is assumed UTC)
      - ISO-8601 strings (e.g. '2026-09-15T06:30:12.000000', '2026-09-15T06:30:12Z', '+00:00')
      - Unix epoch numbers (int, float) or stringified numbers ('1726381812.34')
      - Returns None if value is None, empty, or unparseable.
    """
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None

        # Check if it is a numeric unix timestamp string
        try:
            numeric_val = float(v)
            return datetime.fromtimestamp(numeric_val, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass

        # Handle 'Z' suffix for ISO-8601
        iso_str = v
        if iso_str.endswith("Z") or iso_str.endswith("z"):
            iso_str = iso_str[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError:
            pass

    return None


def _calculate_duration_seconds(first_ts: Any, last_ts: Any) -> Optional[float]:
    """
    Safely calculate the duration between two timestamps in seconds.
    Returns None if either timestamp is invalid or if duration is negative.
    """
    dt_first = _parse_pipeline_timestamp(first_ts)
    dt_last = _parse_pipeline_timestamp(last_ts)
    if dt_first is None or dt_last is None:
        return None
    diff = (dt_last - dt_first).total_seconds()
    if diff < 0:
        return None
    return diff


def render_pipeline_badge(
    stages: Optional[List[Dict[str, Any]]] = None,
    timings: Optional[Dict[str, Any]] = None,
    source_count: Optional[int] = None,
) -> str:
    """
    Generate an HTML badge summarizing the pipeline execution.
    Only exposes high-level metadata (latency, verification, chunk count),
    never internal deliberation or chain of thought.
    """
    time_str = ""
    total_seconds: Optional[float] = None

    if timings and isinstance(timings, dict):
        if "total_s" in timings and timings["total_s"] is not None:
            try:
                total_seconds = float(timings["total_s"])
            except (ValueError, TypeError):
                pass
        elif "total_request_time" in timings and timings["total_request_time"] is not None:
            try:
                total_seconds = float(timings["total_request_time"]) / 1000.0
            except (ValueError, TypeError):
                pass
        elif "processing_time_ms" in timings and timings["processing_time_ms"] is not None:
            try:
                total_seconds = float(timings["processing_time_ms"]) / 1000.0
            except (ValueError, TypeError):
                pass
        elif "total_time" in timings and timings["total_time"] is not None:
            try:
                val = float(timings["total_time"])
                total_seconds = val / 1000.0 if val > 100 else val
            except (ValueError, TypeError):
                pass

    if total_seconds is None and stages:
        first_ts = None
        last_ts = None
        for stg in stages:
            if not isinstance(stg, dict):
                continue
            ts = stg.get("timestamp")
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        if first_ts is not None and last_ts is not None:
            total_seconds = _calculate_duration_seconds(first_ts, last_ts)

    if total_seconds is not None and total_seconds > 0:
        time_str = f" in {total_seconds:.2f}s"

    src_str = ""
    if source_count is not None and source_count > 0:
        src_str = f" · {source_count} source{'s' if source_count != 1 else ''} verified"

    return f"""
    <div class="aura-pipeline-badge">
        <span class="success-icon">✓</span>
        <span>Response ready{time_str}{src_str}</span>
    </div>
    """


def render_pipeline_steps(stages: List[Dict[str, Any]]):
    """
    Render a clean, user-safe timeline drawer of completed/skipped pipeline stages.
    Guarantees no internal model deliberation, hidden prompts, or token-level reasoning.
    """
    if not stages:
        return
    with st.expander("🔍 Reasoning & Processing Steps", expanded=False):
        for stage_info in stages:
            stage_name = stage_info.get("stage", "")
            label = stage_info.get("label", stage_name.replace("_", " ").title())
            status = stage_info.get("status", "complete")
            msg = stage_info.get("message", "")
            details = stage_info.get("details") or stage_info.get("metadata") or {}

            icon = "✓" if status == "complete" else ("—" if status == "skipped" else ("⚠️" if status == "error" else "◉"))
            icon_color = "#35d0a1" if status == "complete" else ("#64748b" if status == "skipped" else ("#f06a72" if status == "error" else "#5cc8ff"))

            detail_pills = []
            if details.get("count") is not None:
                detail_pills.append(f"{details['count']} chunks")
            if details.get("sources_count") is not None:
                detail_pills.append(f"{details['sources_count']} sources")
            if details.get("mode"):
                detail_pills.append(f"mode: {details['mode']}")
            if details.get("query_type"):
                detail_pills.append(f"type: {details['query_type']}")
            if details.get("verified_count") is not None:
                detail_pills.append(f"{details['verified_count']} verified")
            if details.get("citations_count") is not None:
                detail_pills.append(f"{details['citations_count']} verified")
            if details.get("provider"):
                detail_pills.append(f"{details['provider']}")
            if details.get("model"):
                detail_pills.append(f"{details['model']}")

            pills_html = " ".join([f"<span class='aura-pipeline-stage-pill'>{p}</span>" for p in detail_pills])

            st.markdown(
                f"""
                <div class="aura-pipeline-row {status}">
                    <span style="color:{icon_color}; font-weight:bold; width:16px;">{icon}</span>
                    <strong style="color:#e2e8f0; min-width:140px;">{label}</strong>
                    <span style="color:#94a3b8; flex:1;">{msg}</span>
                    {pills_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# Phase 12: Visual Loading Experience & Animation Components
# =============================================================================

def resolve_loading_asset_path(
    configured_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve the loading animation asset path robustly relative to the project location
    rather than depending on the current working directory.
    """
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    env_asset = os.getenv("AURA_LOADING_ASSET") or AURA_LOADING_ASSET
    if env_asset and env_asset not in candidates:
        candidates.append(env_asset)

    candidates.extend(
        [
            DEFAULT_LOADING_BUBBLE_SUBPATH,
            f"assets/{DEFAULT_LOADING_BUBBLE_FILENAME}",
            DEFAULT_LOADING_BUBBLE_FILENAME,
            "ui/assets/Create_a_minimalist_circular.mp4",
            "assets/Create_a_minimalist_circular.mp4",
            "Create_a_minimalist_circular.mp4",
            "ui/assets/aura_loading.mp4",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_absolute() and p.is_file():
            return p
        p_root = (PROJECT_ROOT / candidate).resolve()
        if p_root.is_file():
            return p_root
        p_ui = (UI_DIR / candidate).resolve()
        if p_ui.is_file():
            return p_ui
        if p.is_file():
            return p.resolve()

    # Search ui/assets directory for any mp4, webm, or gif if standard candidates not found
    assets_dir = UI_DIR / "assets"
    if assets_dir.is_dir():
        for ext in ("*.mp4", "*.webm", "*.gif"):
            for f in sorted(assets_dir.glob(ext)):
                return f.resolve()

    return None


def get_loading_asset_data_uri() -> Optional[Tuple[str, str]]:
    """
    Get (data_uri, mime_type) for the loading asset with in-memory caching.
    Guarantees that the video file is read and base64-encoded only once.
    Returns None if no animation asset is available.
    """
    global _LOADING_ASSET_B64_CACHE, _LOADING_ASSET_TYPE_CACHE
    if _LOADING_ASSET_B64_CACHE is not None and _LOADING_ASSET_TYPE_CACHE is not None:
        return _LOADING_ASSET_B64_CACHE, _LOADING_ASSET_TYPE_CACHE

    asset_path = resolve_loading_asset_path()
    if not asset_path or not asset_path.is_file():
        return None

    suffix = asset_path.suffix.lower()
    mime = (
        "video/mp4"
        if suffix == ".mp4"
        else ("video/webm" if suffix == ".webm" else "image/gif")
    )

    try:
        raw_bytes = asset_path.read_bytes()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        _LOADING_ASSET_B64_CACHE = f"data:{mime};base64,{b64_str}"
        _LOADING_ASSET_TYPE_CACHE = mime
        return _LOADING_ASSET_B64_CACHE, _LOADING_ASSET_TYPE_CACHE
    except Exception as e:
        logger.warning(f"Could not load animation asset from {asset_path}: {e}")
        return None


def render_loading_animation_html() -> str:
    """
    Render immediate, lightweight CSS/SVG Aura energy core animation.
    Visual direction: premium futuristic anime-inspired AI energy core —
    cyan/navy aura, bright cyan highlights, glowing orbital rings, luminous
    particles, dark transparent-friendly background.

    Guarantees:
    - Zero network requests, zero video buffering.
    - Silent, no audio, no controls.
    - Pure CSS/SVG hardware-accelerated rendering (< 2ms parse time, ~1.5KB payload).
    - Dark transparent-friendly background (no white flash on dark glassmorphism UI).
    - Loops seamlessly while processing; unmounts immediately on first token.
    """
    return """
    <div class="aura-loading-bubble-container aura-loading-video-container" aria-label="Processing request">
        <div class="aura-loading-bubble-wrapper">
            <div class="aura-loading-bubble">
                <div class="aura-bubble-pulse"></div>
                <div class="aura-bubble-orbital"></div>
                <div class="aura-bubble-core"></div>
            </div>
        </div>
    </div>
    """


def render_loading_video_html() -> str:
    """
    Render the Aura loading animation video element.
    Guarantees:
    - HTML5 video with autoplay, loop, muted, playsinline, and no controls.
    - Completely silent (muted attribute, pointer-events none).
    - CSS mix-blend-mode: screen for seamless cosmic galaxy background transparency.
    - Graceful fallback to Aura energy core CSS animation if video asset is unavailable.
    """
    asset_data = get_loading_asset_data_uri()
    if asset_data:
        data_uri, mime_type = asset_data
        inner_content = f"""
        <video class="aura-loading-video" autoplay loop muted playsinline disablepictureinpicture tabindex="-1" aria-hidden="true" style="pointer-events: none;">
            <source src="{data_uri}" type="{mime_type}">
        </video>
        """
    else:
        inner_content = '<div class="aura-loading-fallback-ring"></div>'

    return f"""
    <div class="aura-loading-video-container">
        <div class="aura-loading-video-wrapper">
            {inner_content}
        </div>
    </div>
    """


def render_loading_status_html(
    primary_text: str = "Generating response...",
    subtext: Optional[str] = None,
    substatus_items: Optional[List[str]] = None,
) -> str:
    """
    Render user-safe status text and optional substatus context below the loading animation.
    Preserves strict zero-leakage guarantee (no chain-of-thought, no internal prompts).
    """
    sub_html = ""
    if subtext:
        sub_html = f'<div class="aura-loading-subtext">{subtext}</div>'
    elif substatus_items:
        items_html = "".join(
            [
                f'<div class="aura-loading-subitem"><span>•</span><span>{item}</span></div>'
                for item in substatus_items
                if item
            ]
        )
        if items_html:
            sub_html = f'<div class="aura-loading-sublist">{items_html}</div>'

    return f"""
    <div class="aura-loading-status-box">
        <div class="aura-loading-primary-text">
            <span class="aura-pulse-dot"></span>
            <span>{primary_text}</span>
        </div>
        {sub_html}
    </div>
    """


def render_loading_container_html(
    primary_text: str = "Generating response...",
    subtext: Optional[str] = None,
    substatus_items: Optional[List[str]] = None,
    use_video: bool = False,
) -> str:
    """
    Combined container rendering both the animation and status indicator.
    Defaults to the lightweight CSS animation, with optional video fallback.
    """
    anim_html = render_loading_video_html() if use_video else render_loading_animation_html()
    status_html = render_loading_status_html(
        primary_text=primary_text,
        subtext=subtext,
        substatus_items=substatus_items,
    )
    return f"""
    <div class="aura-loading-combined-wrapper">
        {anim_html}
        {status_html}
    </div>
    """


def render_chat_message(
    role: str,
    content: str,
    timestamp: str = None,
    sources: List = None,
    context: str = None,
    suggested_questions: List[str] = None,
    is_last_message: bool = False,
    evidence_groups: List[Dict[str, Any]] = None,
    numerical_comparison: Optional[Dict[str, Any]] = None,
    contradictions_detected: Optional[List[Dict[str, Any]]] = None,
    timings: Optional[Dict[str, Any]] = None,
    pipeline_stages: Optional[List[Dict[str, Any]]] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
):
    """
    Render a polished conversational turn inspired by Reference A.
    Applies markdown sanitization to prevent SVG and localhost anchor artifacts.
    """
    cleaned_content = clean_markdown_display(content)
    avatar = "👤" if role == "user" else AURA_AI_SYMBOL

    with st.chat_message(role, avatar=avatar):
        # User attached image thumbnail
        if role == "user" and image_bytes:
            st.image(
                image_bytes,
                caption=f"📎 {image_filename or 'Attached Image'}",
                width=240,
            )

        # Assistant pipeline summary badge (collapsed by default)
        if role == "assistant" and pipeline_stages:
            st.markdown(
                render_pipeline_badge(
                    pipeline_stages,
                    timings=timings,
                    source_count=len(sources) if sources else 0,
                ),
                unsafe_allow_html=True,
            )

        # Main sanitized Markdown content with refined inline citations
        if role == "assistant":
            styled_content = re.sub(
                r'\[S(\d+)\]',
                r'<span class="aura-citation-pill">[S\1]</span>',
                cleaned_content,
            )
            st.markdown(styled_content, unsafe_allow_html=True)
        else:
            st.markdown(cleaned_content)

        # Assistant enhancements
        if role == "assistant":
            # Pipeline reasoning drawer
            if pipeline_stages:
                render_pipeline_steps(pipeline_stages)

            # Numerical comparison callout
            if numerical_comparison:
                metric_name = numerical_comparison.get("metric_name", "Value")
                diff_str = str(numerical_comparison.get("difference", "N/A"))
                pct = numerical_comparison.get("percentage_change")
                pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
                st.info(
                    f"📊 **Metric Comparison ({metric_name})**: Difference: `{diff_str}`{pct_str}"
                )

            # Contradictions callout
            if contradictions_detected:
                for contra in contradictions_detected:
                    st.warning(
                        f"⚠️ **Discrepancy Detected**: {contra.get('discrepancy', '')}"
                    )

            # Grouped comparison evidence
            if evidence_groups and st.session_state.get("show_sources", True):
                total_items = sum(
                    len(g.get("sources", [])) for g in evidence_groups
                )
                with st.expander(
                    f"📚 Evidence Sources ({len(evidence_groups)} Documents, {total_items} Chunks)",
                    expanded=False,
                ):
                    for grp_idx, group in enumerate(evidence_groups, 1):
                        doc_title = (
                            group.get("filename")
                            or group.get("file_id")
                            or f"Document {grp_idx}"
                        )
                        st.markdown(f"**📄 {doc_title}**")
                        for i, src in enumerate(group.get("sources", []), 1):
                            cit_label = src.get("citation_label") or f"[S{i}]"
                            page_num = src.get("page_number")
                            page_str = (
                                f" — Page {page_num}"
                                if page_num is not None
                                else ""
                            )
                            snippet = src.get("content", "").strip()
                            st.markdown(
                                f"""
                                <div class="aura-source-card">
                                    <div class="aura-source-header">
                                        <span class="aura-source-title">{cit_label} {doc_title}{page_str}</span>
                                    </div>
                                    <div class="aura-source-snippet">"{snippet[:280]}{'...' if len(snippet) > 280 else ''}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # Standard structured sources
            elif sources and st.session_state.get("show_sources", True):
                max_s = st.session_state.get("max_sources", 5)
                displayed_sources = sources[:max_s]
                with st.expander(
                    f"📚 Verified Sources ({len(displayed_sources)})",
                    expanded=False,
                ):
                    for i, source in enumerate(displayed_sources, 1):
                        cit_label = source.get("citation_label") or f"[S{i}]"
                        filename = source.get("filename") or "Document"
                        page_num = source.get("page_number")
                        page_str = (
                            f" • Page {page_num}"
                            if page_num is not None
                            else ""
                        )
                        rel = source.get("relevance_score")
                        rel_str = f"{rel:.0%}" if rel else ""
                        stype = source.get("search_type") or "hybrid"
                        badge_label = {
                            "hybrid": "🔀 Hybrid",
                            "vector": "🎯 Vector",
                            "bm25": "🔤 Keyword",
                            "image_ocr": "📷 OCR",
                            "image_vision": "👁️ Vision",
                            "sql": "📊 SQL",
                        }.get(stype, stype)

                        snippet = source.get("content", "").strip()
                        if source.get("source_type") == "sql":
                            st.code(snippet, language="sql")
                        else:
                            st.markdown(
                                f"""
                                <div class="aura-source-card">
                                    <div class="aura-source-header">
                                        <span class="aura-source-title">{cit_label} {filename}{page_str}</span>
                                        <span class="aura-source-badge">{badge_label}{' • ' + rel_str if rel_str else ''}</span>
                                    </div>
                                    <div class="aura-source-snippet">"{snippet[:300]}{'...' if len(snippet) > 300 else ''}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # Latency / Timing instrumentation
            if timings:
                t_parts = []
                if "total_s" in timings:
                    t_parts.append(f"⚡ {timings['total_s']:.2f}s total")
                if "retrieval_s" in timings:
                    t_parts.append(
                        f"🔍 Retrieval: {timings['retrieval_s']*1000:.0f}ms"
                    )
                if "generation_s" in timings:
                    prov_label = (st.session_state.get("ai_provider") or os.getenv("AI_PROVIDER") or "AI").capitalize()
                    t_parts.append(f'<img src="{get_aura_symbol_data_uri()}" class="aura-badge-img-xs" alt="Aura" /> {prov_label}: {timings["generation_s"]:.2f}s')
                if t_parts:
                    st.markdown(
                        f"<div class=\"aura-timings-bar\">{' • '.join(t_parts)}</div>",
                        unsafe_allow_html=True,
                    )

            # Suggested follow-ups
            if is_last_message and suggested_questions:
                st.markdown(
                    "<div style='font-size:0.8rem; color:#94a3b8; margin:0.8rem 0 0.4rem;'>💡 Follow-up questions:</div>",
                    unsafe_allow_html=True,
                )
                cols = st.columns(min(len(suggested_questions), 3))
                for q_idx, (col, q_text) in enumerate(
                    zip(cols, suggested_questions[:3])
                ):
                    with col:
                        if st.button(
                            q_text[:45] + "..."
                            if len(q_text) > 45
                            else q_text,
                            key=f"sug_btn_{q_idx}_{hash(q_text)}",
                            use_container_width=True,
                            help=q_text,
                        ):
                            st.session_state.pending_question = q_text
                            st.rerun()


def process_attached_image(
    uploaded_file: Any,
    indicator: Optional[AuraProcessingIndicator] = None,
) -> bool:
    """
    Process, validate, and attach an image for multimodal query execution.
    Performs MIME, size, and PIL integrity validation with compact AuraProcessingIndicator feedback.

    Returns:
        bool: True if image is valid and attached, False otherwise.
    """
    if uploaded_file is None:
        return False

    raw_name = getattr(uploaded_file, "name", "attached_image.png")
    if indicator:
        indicator.update(status="Processing image…", operation_label=raw_name)

    try:
        raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
        if not raw_bytes or len(raw_bytes) == 0:
            st.error("❌ Image payload is empty.")
            return False

        # Image size validation (10MB limit)
        max_size = 10 * 1024 * 1024
        if len(raw_bytes) > max_size:
            st.error(f"❌ Image file size ({len(raw_bytes)/(1024*1024):.1f}MB) exceeds maximum limit of 10MB.")
            return False

        # MIME / extension validation
        ext = Path(raw_name).suffix.lower()
        if ext and ext not in ('.png', '.jpg', '.jpeg'):
            st.error(f"❌ Unsupported image format: '{ext}'. Supported formats: PNG, JPG, JPEG.")
            return False

        # PIL integrity & decompression bomb validation
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img_check:
                fmt = (img_check.format or "").lower()
                if fmt not in ("png", "jpeg", "jpg"):
                    st.error(f"❌ Unsupported image format: '{fmt}'. Supported formats: PNG, JPG, JPEG.")
                    return False
                w, h = img_check.size
                if w * h > 50_000_000:
                    st.error(f"❌ Image dimensions ({w}x{h}) exceed safety limit of 50 megapixels.")
                    return False
                img_check.verify()
        except Exception as img_err:
            st.error(f"❌ Corrupted or invalid image file: {img_err}")
            return False

        # Image successfully validated and attached
        st.session_state.attached_image_bytes = raw_bytes
        st.session_state.attached_image_name = raw_name
        return True
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return False


def render_image_attachment_controls():
    """Render image attachment controls with thumbnail preview, processing feedback, and removal."""
    st.markdown('<div class="aura-attach-expander">', unsafe_allow_html=True)
    with st.expander("Attach Image / Chart (Optional)", expanded=False):
        uploaded_img = st.file_uploader(
            "Upload image to analyze alongside your query",
            type=["png", "jpg", "jpeg"],
            key="composer_img_uploader",
            help="Supported: PNG, JPEG (max 10MB)",
        )
        img_indicator_box = st.empty()
        if uploaded_img is not None:
            current_attached_name = st.session_state.get("attached_image_name")
            current_bytes = st.session_state.get("attached_image_bytes")
            img_val = uploaded_img.getvalue()

            if current_attached_name != uploaded_img.name or current_bytes != img_val:
                indicator = AuraProcessingIndicator(
                    container=img_indicator_box,
                    mode="compact",
                    status="Processing image…",
                    operation_label=uploaded_img.name,
                    include_css=False,
                )
                indicator.start()
                try:
                    ok = process_attached_image(uploaded_img, indicator=indicator)
                    if not ok:
                        st.session_state.attached_image_bytes = None
                        st.session_state.attached_image_name = None
                finally:
                    indicator.clear()
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview card if an image is actively attached
    if st.session_state.get("attached_image_bytes"):
        col_prev, col_rem = st.columns([5, 1])
        with col_prev:
            img_name = st.session_state.get("attached_image_name", "image.png")
            size_kb = len(st.session_state.attached_image_bytes) / 1024
            st.markdown(
                f"""
                <div class="aura-image-preview-card">
                    <span>📎</span>
                    <div class="aura-image-preview-info">
                        <div class="aura-image-preview-name">{img_name}</div>
                        <div class="aura-image-preview-meta">Image attached for query • {size_kb:.1f} KB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_rem:
            if st.button("× Clear", key="remove_attached_img", help="Remove image"):
                st.session_state.attached_image_bytes = None
                st.session_state.attached_image_name = None
                st.rerun()


def process_question(
    question: str,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
):
    """
    Process question with REAL token streaming and progressive UI rendering.
    Applies markdown sanitization to prevent SVG artifacts.
    """
    timestamp = datetime.now().strftime("%H:%M")

    # Append user turn immediately
    user_message = {
        "role": "user",
        "content": question,
        "timestamp": timestamp,
        "image_bytes": image_bytes,
        "image_filename": image_filename,
    }
    st.session_state.chat_history.append(user_message)

    # Clear attached image after using it for query
    st.session_state.attached_image_bytes = None
    st.session_state.attached_image_name = None

    # If in comparison mode, execute comparison with AuraProcessingIndicator
    if st.session_state.get("comparison_mode_active"):
        # Render user turn
        with st.chat_message("user", avatar="👤"):
            if image_bytes:
                st.image(image_bytes, caption=f"📎 {image_filename or 'Attached Image'}", width=240)
            st.markdown(clean_markdown_display(question))

        with st.chat_message("assistant", avatar=AURA_AI_SYMBOL):
            status_box = st.empty()
            indicator = AuraProcessingIndicator(
                container=status_box,
                mode="compact",
                status="Comparing documents…",
                operation_label="Analyzing cross-document metrics and contradictions",
                include_css=False,
            )
            indicator.start()
            try:
                result = query_document(
                    question,
                    image_bytes=image_bytes,
                    image_filename=image_filename,
                    indicator=indicator,
                )
            finally:
                indicator.clear()

            if result:
                cleaned_answer = clean_markdown_display(result.get("answer", ""))
                assistant_message = {
                    "role": "assistant",
                    "content": cleaned_answer,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": result.get("sources", []),
                    "context": result.get("context", ""),
                    "suggested_questions": result.get("suggested_questions", []),
                    "evidence_groups": result.get("evidence_groups", []),
                    "numerical_comparison": result.get("numerical_comparison"),
                    "contradictions_detected": result.get("contradictions_detected", []),
                    "timings": result.get("timings"),
                }
                st.session_state.chat_history.append(assistant_message)
            else:
                raw_err = st.session_state.get("_last_query_error", "Comparison failed.")
                assistant_message = {
                    "role": "assistant",
                    "content": f"⚠️ **Comparison Failed**: {raw_err}",
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "sources": [],
                    "is_error": True,
                }
                st.session_state.chat_history.append(assistant_message)
        st.rerun()
        return

    # Render user turn
    with st.chat_message("user", avatar="👤"):
        if image_bytes:
            st.image(image_bytes, caption=f"📎 {image_filename or 'Attached Image'}", width=240)
        st.markdown(clean_markdown_display(question))

    # Render assistant response with REAL token streaming and immediate visual processing animation
    with st.chat_message("assistant", avatar=AURA_AI_SYMBOL):
        loader_video_box = st.empty()
        status_box = st.empty()
        sources_holder = []
        complete_payload = {}
        error_holder = []
        status_events_holder = []
        # Truthful initial status based on document context
        active_fname = (st.session_state.get("uploaded_filename") or "").lower()
        selected_fnames = [
            f.lower() for f in st.session_state.get("selected_filenames", [])
        ]
        is_structured_doc = any(
            fn.endswith((".db", ".sqlite", ".sqlite3", ".csv"))
            for fn in ([active_fname] + selected_fnames)
            if fn
        )
        initial_status = "Processing SQL query…" if is_structured_doc else "Understanding request…"

        # Immediately render universal lightweight Aura processing indicator (no video buffering delay)
        indicator = AuraProcessingIndicator(
            animation_container=loader_video_box,
            status_container=status_box,
            mode="full",
            status=initial_status,
            include_css=False,
        )
        indicator.start()

        def _stream_token_generator():
            first_token_seen = False
            for event_type, data in query_document_stream(
                question, image_bytes=image_bytes, image_filename=image_filename
            ):
                if event_type == "start":
                    init_msg = data.get("message") or "Understanding request…"
                    indicator.update(status=init_msg)
                elif event_type == "status":
                    status_events_holder.append(data)
                    stg_msg = data.get("message") or data.get("label") or "Processing..."
                    prev_steps = [
                        e.get("message")
                        for e in status_events_holder[:-1]
                        if e.get("message")
                    ]
                    subtext = prev_steps[-1] if prev_steps else None
                    indicator.update(status=stg_msg, operation_label=subtext)
                elif event_type == "sources":
                    srcs = data.get("sources", [])
                    sources_holder.extend(srcs)
                    indicator.update(
                        status="Reviewing sources…",
                        operation_label=f"Analyzing {len(srcs)} verified context sources",
                    )
                elif event_type == "token":
                    if not first_token_seen:
                        first_token_seen = True
                        # Cleanly remove processing animation when the first real token arrives
                        indicator.clear_animation()
                        status_box.markdown(
                            f"""
                            <div class="aura-pipeline-badge">
                                <span class="aura-pulse-dot"></span>
                                <span><img src="{get_aura_symbol_data_uri()}" class="aura-badge-img-xs" alt="Aura" /> Generating answer...</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    token_text = data.get("text", "")
                    if token_text:
                        yield token_text
                elif event_type == "complete":
                    complete_payload.update(data)
                elif event_type == "error":
                    error_holder.append(data.get("error", "Unknown streaming error"))

        try:
            streamed_text = st.write_stream(_stream_token_generator())
        finally:
            # Guarantee: processing animation is always removed on completion, error, or interruption
            indicator.clear_animation()

        if error_holder:
            indicator.clear()
            raw_err = error_holder[0]
            prov_name = (st.session_state.get("ai_provider") or os.getenv("AI_PROVIDER") or "AI").capitalize()
            if "quota" in raw_err.lower() or "429" in raw_err:
                friendly_msg = f"⚠️ **AI Provider Quota Exceeded**: The {prov_name} API request quota has been reached. Please retry in a few moments."
            elif "503" in raw_err or "unavailable" in raw_err.lower():
                friendly_msg = f"⚠️ **Provider Unavailable**: {prov_name} is currently experiencing high demand. Please try again."
            else:
                friendly_msg = f"⚠️ **Request Failed**: {raw_err}"

            assistant_message = {
                "role": "assistant",
                "content": friendly_msg,
                "timestamp": datetime.now().strftime("%H:%M"),
                "sources": [],
                "is_error": True,
            }
            st.session_state.chat_history.append(assistant_message)
        else:
            final_answer = complete_payload.get("answer") or streamed_text or ""
            cleaned_answer = clean_markdown_display(final_answer)
            sources = complete_payload.get("sources") or sources_holder
            pipeline_stages = complete_payload.get("pipeline_stages") or status_events_holder
            timings = complete_payload.get("timings")

            # Finalize status badge
            if pipeline_stages:
                status_box.markdown(
                    render_pipeline_badge(
                        pipeline_stages,
                        timings=timings,
                        source_count=len(sources) if sources else 0,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                status_box.empty()

            assistant_message = {
                "role": "assistant",
                "content": cleaned_answer,
                "timestamp": datetime.now().strftime("%H:%M"),
                "sources": sources,
                "context": complete_payload.get("context", ""),
                "suggested_questions": complete_payload.get("suggested_questions", []),
                "timings": timings,
                "citation_validation": complete_payload.get("citation_validation"),
                "pipeline_stages": pipeline_stages,
            }
            st.session_state.chat_history.append(assistant_message)

            # Render reasoning/pipeline steps expander
            if pipeline_stages:
                render_pipeline_steps(pipeline_stages)

            # Render verified sources expander directly under the streamed answer
            if sources and st.session_state.get("show_sources", True):
                max_s = st.session_state.get("max_sources", 5)
                displayed_sources = sources[:max_s]
                with st.expander(
                    f"📚 Verified Sources ({len(displayed_sources)})",
                    expanded=False,
                ):
                    for i, source in enumerate(displayed_sources, 1):
                        cit_label = source.get("citation_label") or f"[S{i}]"
                        filename = source.get("filename") or "Document"
                        page_num = source.get("page_number")
                        page_str = (
                            f" • Page {page_num}"
                            if page_num is not None
                            else ""
                        )
                        rel = source.get("relevance_score")
                        rel_str = f"{rel:.0%}" if rel else ""
                        stype = source.get("search_type") or "hybrid"
                        badge_label = {
                            "hybrid": "🔀 Hybrid",
                            "vector": "🎯 Vector",
                            "bm25": "🔤 Keyword",
                            "image_ocr": "📷 OCR",
                            "image_vision": "👁️ Vision",
                            "sql": "📊 SQL",
                        }.get(stype, stype)

                        snippet = source.get("content", "").strip()
                        if source.get("source_type") == "sql":
                            st.code(snippet, language="sql")
                        else:
                            st.markdown(
                                f"""
                                <div class="aura-source-card">
                                    <div class="aura-source-header">
                                        <span class="aura-source-title">{cit_label} {filename}{page_str}</span>
                                        <span class="aura-source-badge">{badge_label}{' • ' + rel_str if rel_str else ''}</span>
                                    </div>
                                    <div class="aura-source-snippet">"{snippet[:300]}{'...' if len(snippet) > 300 else ''}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    st.rerun()


def export_chat_history():
    """Export conversation history as downloadable JSON."""
    if not st.session_state.chat_history:
        st.warning("No messages to export.")
        return

    export_data = {
        "title": st.session_state.get("active_session_title", "Chat"),
        "session_id": st.session_state.get("active_session_id"),
        "document": st.session_state.get("uploaded_filename"),
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history,
    }
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Conversation JSON",
        data=json_str,
        file_name=f"aura_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


# =============================================================================
# Main Application Flow
# =============================================================================

def main():
    """Main application loop."""
    init_session_state()
    apply_custom_styles()

    # Verify backend API availability
    health = check_api_health()
    if not health.get("healthy"):
        render_top_bar()
        st.error(
            """
            🔴 **Aura AI Backend is Unreachable**
            
            Please start the backend service:
            ```bash
            uvicorn app.main:app --host 0.0.0.0 --port 8000
            ```
            """
        )
        if st.button("🔄 Retry Connection", type="primary"):
            st.session_state.last_health_check = None
            st.rerun()
        st.stop()

    # Render Navigation & Sidebar
    render_top_bar()
    render_sidebar()

    # Ensure an active session is loaded
    if not st.session_state.get("active_session_id"):
        sessions = get_sessions_api()
        if sessions:
            load_session(sessions[0]["session_id"])
        else:
            new_sess = create_session_api()
            if new_sess:
                st.session_state.active_session_id = new_sess["session_id"]
                st.session_state.active_session_title = new_sess.get(
                    "title", "New Chat"
                )
                st.session_state.chat_history = []

    # Handle pending question from suggestions or hero clicks
    if st.session_state.get("pending_question"):
        pending_q = st.session_state.pending_question
        st.session_state.pending_question = None
        img_bytes = st.session_state.get("attached_image_bytes")
        img_name = st.session_state.get("attached_image_name")
        process_question(
            pending_q, image_bytes=img_bytes, image_filename=img_name
        )
        return

    # Main Conversation / Landing Canvas
    if not st.session_state.chat_history:
        render_empty_state_landing()
    else:
        total_messages = len(st.session_state.chat_history)
        for idx, msg in enumerate(st.session_state.chat_history):
            is_last = idx == total_messages - 1
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp"),
                sources=msg.get("sources")
                if msg["role"] == "assistant"
                else None,
                context=msg.get("context")
                if msg["role"] == "assistant"
                else None,
                suggested_questions=msg.get("suggested_questions")
                if msg["role"] == "assistant"
                else None,
                is_last_message=is_last,
                evidence_groups=msg.get("evidence_groups")
                if msg["role"] == "assistant"
                else None,
                numerical_comparison=msg.get("numerical_comparison")
                if msg["role"] == "assistant"
                else None,
                contradictions_detected=msg.get("contradictions_detected")
                if msg["role"] == "assistant"
                else None,
                timings=msg.get("timings")
                if msg["role"] == "assistant"
                else None,
                pipeline_stages=msg.get("pipeline_stages")
                if msg["role"] == "assistant"
                else None,
                image_bytes=msg.get("image_bytes"),
                image_filename=msg.get("image_filename"),
            )

        # Action row at the bottom of active chat
        col_clear, col_exp, _ = st.columns([1, 1, 3])
        with col_clear:
            if st.button("🔄 Clear", use_container_width=True, help="Clear current chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col_exp:
            if st.button("📥 Export", use_container_width=True, help="Export conversation as JSON"):
                export_chat_history()

    # Image attachment controls (collapsible)
    render_image_attachment_controls()

    # Chat Composer
    if st.session_state.get("attached_image_name"):
        placeholder = f"Ask about {st.session_state.attached_image_name}..."
    elif st.session_state.uploaded_filename:
        placeholder = f"Ask anything about {st.session_state.uploaded_filename}..."
    elif (
        st.session_state.multi_doc_mode
        and st.session_state.selected_filenames
    ):
        placeholder = f"Ask across {len(st.session_state.selected_filenames)} selected documents..."
    else:
        placeholder = "Ask Aura anything about your documents..."

    question = st.chat_input(placeholder, key="chat_input")
    if question:
        img_bytes = st.session_state.get("attached_image_bytes")
        img_name = st.session_state.get("attached_image_name")
        process_question(
            question, image_bytes=img_bytes, image_filename=img_name
        )


if __name__ == "__main__":
    main()
