"""Smoke tests: every page renders under every predefined scenario without exceptions."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.state import PRESETS

APP = Path(__file__).resolve().parent.parent / "app.py"
PAGES = ["views/overview.py", "views/forecasting.py", "views/scenarios.py", "views/decisions.py"]


def _app():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert any("synthetic data" in m.value.lower() for m in at.markdown)


@pytest.mark.parametrize("preset", PRESETS)
def test_overview_under_each_preset(preset):
    at = _app()
    at.button(key=f"preset_{preset}").click().run()
    assert not at.exception, at.exception
    assert at.session_state["preset"] == preset
    assert at.title[0].value == "Executive Overview"


def test_reset_to_baseline():
    at = _app()
    at.button(key="preset_High Growth").click().run()
    assert at.session_state["pg_on"]
    at.button(key="reset").click().run()
    assert at.session_state["preset"] == "Baseline"
    assert not at.session_state["pg_on"]


def test_invalid_thresholds_show_clear_error():
    at = _app()
    at.slider(key="warn_pct").set_value(95).run()
    at.slider(key="crit_pct").set_value(90).run()
    assert not at.exception
    assert any("can't be modeled" in e.value for e in at.error)
