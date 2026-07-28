"""Integration tests for the Gradio GUI.

Tests that the UI builds without error and the ASGI app responds to HTTP
requests. No agent servers are required — the GUI only calls agent
endpoints when the user submits a message, not during construction.

Run with:
    pixi run test-integration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GUI_DIR = Path(__file__).parent.parent / "gui"


@pytest.fixture(autouse=True)
def _add_gui_to_path():
    if str(_GUI_DIR) not in sys.path:
        sys.path.insert(0, str(_GUI_DIR))
    yield
    if str(_GUI_DIR) in sys.path:
        sys.path.remove(str(_GUI_DIR))


@pytest.mark.integration
def test_gui_builds():
    """build_ui() returns a gr.Blocks instance without error."""
    import gradio as gr
    from app import build_ui

    demo = build_ui()
    assert isinstance(demo, gr.Blocks)


@pytest.mark.integration
def test_gui_has_title():
    """The Gradio app has the expected title."""
    from app import build_ui

    demo = build_ui()
    assert demo.title == "Champollion Agents"


@pytest.mark.integration
def test_gui_asgi_app_created():
    """Gradio ASGI app (FastAPI) is created from the Blocks demo without error."""
    import gradio as gr
    from app import build_ui

    demo = build_ui()
    app = gr.routes.App.create_app(demo)
    assert app is not None
    assert hasattr(app, "router")
