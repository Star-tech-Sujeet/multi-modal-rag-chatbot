"""
Regression tests for Aura AI UI Sidebar Collapse/Expand Toggle Controls.

Verifies:
1. Open sidebar collapse icon is present with SVG styling and NO permanent "Collapse" text.
2. Collapsed sidebar expand icon is present with SVG styling and NO permanent "Sidebar" text.
3. Hover/focus tooltips "Collapse sidebar" and "Open sidebar" are configured.
4. Header configuration and responsive behavior styling for mobile/tablet viewports.
5. Sidebar action buttons, collapse/expand capability, and chat composer remain active and unobstructed.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest


def test_sidebar_toggle_css_rules_injected():
    """Verify that sidebar toggle and collapse CSS rules are properly injected into the app with icon-only controls."""
    app_path = (Path(__file__).parent.parent / "ui" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Find the injected stylesheet markdown block
    style_blocks = [m.value for m in at.markdown if "stSidebarCollapseButton" in m.value]
    assert len(style_blocks) > 0, "Sidebar style block was not found in rendered app"
    css = style_blocks[0]

    # 1. Sidebar open: collapse button rules & icon presence
    assert "data-testid=\"stSidebarCollapseButton\"" in css
    assert "visibility: visible !important;" in css
    assert "display: inline-flex !important;" in css
    assert "width: 32px !important;" in css
    assert "height: 32px !important;" in css
    
    # 2. Permanent "Collapse" text must NOT be rendered
    assert "content: \"Collapse\";" not in css
    assert "content: \"Collapse Sidebar\";" not in css

    # 3. Sidebar collapsed: expand toggle button rules & icon presence
    assert "data-testid=\"stExpandSidebarButton\"" in css
    assert "data-testid=\"stSidebarCollapsedControl\"" in css
    assert "position: fixed !important;" in css
    assert "z-index: 999999 !important;" in css
    assert "backdrop-filter: blur(16px) !important;" in css

    # 4. Permanent "Sidebar" text must NOT be rendered
    assert "content: \"Sidebar\";" not in css
    assert "content: \"Open Sidebar\";" not in css
    assert "content: \"Expand Sidebar\";" not in css

    # 5. Tooltips on hover/focus
    assert "content: \"Collapse sidebar\" !important;" in css
    assert "content: \"Open sidebar\" !important;" in css

    # 6. Header configuration: transparent without global visibility: hidden on header element
    assert "header[data-testid=\"stHeader\"]" in css
    assert "overflow: visible !important;" in css
    assert "data-testid=\"stToolbarActions\"" in css

    # 7. Responsive styling
    assert "@media (max-width: 768px)" in css


def test_sidebar_interactions_and_composer():
    """Verify that sidebar action buttons and chat composer are active and accessible."""
    app_path = (Path(__file__).parent.parent / "ui" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.session_state["api_healthy"] = True
    at.session_state["last_health_check"] = 9999999999
    at.session_state["active_session_id"] = "test-session"
    at.session_state["active_session_title"] = "Test Chat"

    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"session_id": "new-session", "title": "New Chat", "sessions": []}
        mock_req.return_value = mock_resp

        at.run()

        # Primary New Chat button
        new_chat_btns = [b for b in at.sidebar.button if b.label == "➕ New Chat"]
        assert len(new_chat_btns) == 1
        new_chat_btns[0].click().run()

        # Composer is present and unobstructed
        assert len(at.chat_input) == 1
        assert "Ask" in at.chat_input[0].placeholder


def test_chat_session_action_icons_and_isolation():
    """Verify chat session action buttons use pencil & trash icons with accessible tooltips,

    and verify that document/file library controls remain completely untouched."""
    app_path = (Path(__file__).parent.parent / "ui" / "streamlit_app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.session_state["api_healthy"] = True
    at.session_state["last_health_check"] = 9999999999
    at.session_state["active_session_id"] = "session-1"
    at.session_state["active_session_title"] = "Research Notes"

    sample_sessions = [
        {"session_id": "session-1", "title": "Research Notes", "message_count": 5},
        {"session_id": "session-2", "title": "Project Roadmap", "message_count": 2},
    ]
    sample_files = [
        {"file_id": "doc-uuid-1", "filename": "financial_report.pdf", "file_type": "PDF", "chunks_count": 12},
    ]

    with patch("requests.request") as mock_req:
        def fake_req(method, url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "/files" in url:
                resp.json.return_value = sample_files
            elif "/health" in url:
                resp.json.return_value = {"status": "healthy", "total_documents": 1}
            elif "/sessions/session-1" in url and method == "PATCH":
                resp.json.return_value = {"session_id": "session-1", "title": "Updated Research"}
            elif "/sessions/session-1" in url and method == "DELETE":
                sample_sessions.pop(0)
                resp.json.return_value = {"success": True}
            elif "/sessions" in url:
                resp.json.return_value = {"sessions": sample_sessions}
            else:
                resp.json.return_value = {}
            return resp

        mock_req.side_effect = fake_req

        # Initial render
        at.run()

        # 1. Verify CSS rules target 3-column chat action controls
        style_blocks = [m.value for m in at.markdown if "stHorizontalBlock" in m.value]
        assert len(style_blocks) > 0, "Chat action button CSS rules not found"
        css = style_blocks[0]
        assert '[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))' in css

        # 2. Verify Session 1 has independent Pencil (Rename) and Trash (Delete) controls
        ren1 = [b for b in at.sidebar.button if b.key == "sess_ren_session-1"]
        del1 = [b for b in at.sidebar.button if b.key == "sess_del_session-1"]
        assert len(ren1) == 1, "Session 1 rename button not rendered"
        assert len(del1) == 1, "Session 1 delete button not rendered"
        assert ren1[0].label == "✏️", f"Expected pencil icon ✏️, got {ren1[0].label}"
        assert ren1[0].help == "Rename chat", f"Expected tooltip 'Rename chat', got {ren1[0].help}"
        assert del1[0].label == "🗑️", f"Expected trash icon 🗑️, got {del1[0].label}"
        assert del1[0].help == "Delete chat", f"Expected tooltip 'Delete chat', got {del1[0].help}"

        # 3. Verify Session 2 has independent Pencil (Rename) and Trash (Delete) controls
        ren2 = [b for b in at.sidebar.button if b.key == "sess_ren_session-2"]
        del2 = [b for b in at.sidebar.button if b.key == "sess_del_session-2"]
        assert len(ren2) == 1, "Session 2 rename button not rendered"
        assert len(del2) == 1, "Session 2 delete button not rendered"
        assert ren2[0].label == "✏️", f"Expected pencil icon ✏️, got {ren2[0].label}"
        assert ren2[0].help == "Rename chat", f"Expected tooltip 'Rename chat', got {ren2[0].help}"
        assert del2[0].label == "🗑️", f"Expected trash icon 🗑️, got {del2[0].label}"
        assert del2[0].help == "Delete chat", f"Expected tooltip 'Delete chat', got {del2[0].help}"

        # 4. Verify Document Library controls remain completely untouched
        doc_select_btns = [b for b in at.sidebar.button if b.key == "sel_btn_doc-uuid-1"]
        doc_del_btns = [b for b in at.sidebar.button if b.key == "del_btn_doc-uuid-1"]
        assert len(doc_select_btns) == 1, "Document select button missing"
        assert len(doc_del_btns) == 1, "Document delete button missing"
        assert doc_select_btns[0].label == "Select", "Document select button label altered"
        assert doc_del_btns[0].label == "Delete", "Document delete button label altered"

        # 5. Verify Rename Flow: click pencil -> opens rename UI
        ren1[0].click().run()
        assert at.session_state["_renaming_session_id"] == "session-1"
        save_btns = [b for b in at.sidebar.button if b.key == "save_ren_session-1"]
        cancel_btns = [b for b in at.sidebar.button if b.key == "cancel_ren_session-1"]
        assert len(save_btns) == 1, "Save button not rendered in rename mode"
        assert len(cancel_btns) == 1, "Cancel button not rendered in rename mode"

        # Click Cancel -> clears rename mode
        cancel_btns[0].click().run()
        assert at.session_state["_renaming_session_id"] is None

        # 6. Verify Delete Flow: click trash -> triggers deletion of session-1
        del_btn = [b for b in at.sidebar.button if b.key == "sess_del_session-1"][0]
        del_btn.click().run()
        # Session 1 is deleted, session 2 is loaded
        assert at.session_state["active_session_id"] == "session-2"
        remaining_del = [b for b in at.sidebar.button if b.key == "sess_del_session-1"]
        assert len(remaining_del) == 0, "Deleted session-1 still has delete button"

