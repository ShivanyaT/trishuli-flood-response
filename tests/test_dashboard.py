"""Headless run of the Streamlit app (streamlit.testing AppTest) against the synthetic town."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("streamlit_folium")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.test_day5 import demo, full_demo  # noqa: E402,F401  (fixtures)


def _app(monkeypatch, out, lang="en"):
    monkeypatch.setenv("FLOOD_OUT", str(out))
    monkeypatch.setenv("FLOOD_CONFIG", "config/aoi_trishuli.yaml")
    monkeypatch.delenv("COPILOT_LLM", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py"), default_timeout=90)
    at.run()
    return at


def test_dashboard_renders_metrics_banner_and_report(full_demo, monkeypatch):
    out, _, _ = full_demo
    at = _app(monkeypatch, out)
    assert not at.exception, at.exception
    assert any("SYNTHETIC DEMO DATA" in e.value for e in at.error)
    vals = {m.label: m.value for m in at.metric}
    assert vals["Buildings in hazard"] == "6" and vals["Settlements cut off"] == "1" and vals["Bridges exposed"] == "1/1"
    assert any("Village 5" in m.value for m in at.markdown) or any("Village 5" in t.value for t in at.text)


def test_dashboard_copilot_and_nepali(full_demo, monkeypatch):
    out, _, _ = full_demo
    at = _app(monkeypatch, out)
    at.chat_input[0].set_value("Which villages are cut off?").run()
    assert not at.exception, at.exception
    txt = " ".join(m.value for m in at.markdown)
    assert "Village 5" in txt and "verified True" in " ".join(c.value for c in at.caption)
    at.radio[0].set_value("ne").run()
    assert not at.exception, at.exception
    assert "बस्ती" in " ".join(m.value for m in at.markdown)


def test_dashboard_missing_outputs_is_friendly(tmp_path, monkeypatch):
    at = _app(monkeypatch, tmp_path)
    assert not at.exception, at.exception
    assert any("Outputs missing" in w.value for w in at.warning)
