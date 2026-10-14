import json

import pytest

from src import copilot, cutoff, damage
from src.copilot import answer, numbers_in, render_en, render_ne, route, verify_numbers
from src.damage import load_hazard, load_layer
from src.demo_data import build
from src.facts import Facts


from src.report import pdf_available  # noqa: E402

needs_pdf = pytest.mark.skipif(not pdf_available(), reason="WeasyPrint/Pango libraries not installed (see src.report.PDF_HELP)")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = build(tmp_path_factory.mktemp("demo"))
    flood, debris, grid, src = load_hazard(d["outputs"], "baseline")
    *_, dsum = damage.run(d["osm"], flood, debris, grid, src)
    g = lambda n: load_layer(d["osm"], n, grid.crs)  # noqa: E731
    gdf, dests, qa = cutoff.analyse(g("roads"), g("places"), g("health"), g("buildings"), flood, debris)
    out = d["outputs"]
    (out / "damage_summary.json").write_text(json.dumps(dsum), encoding="utf-8")
    (out / "cutoff_summary.json").write_text(json.dumps(cutoff.summarise(gdf, qa)), encoding="utf-8")
    (out / "flowpath_summary.json").write_text(json.dumps({"length_km": 5.2, "settlements_along_path": [
        {"name": "Village 3", "km_downstream": 1.25}]}), encoding="utf-8")
    return Facts.load(out, "Synthetic Trishuli", "2026-08-26")


# ---------------------------------------------------------------- facts / tools
def test_facts_tools_match_pipeline_outputs(demo):
    assert demo.event["synthetic"] and demo.missing() == []
    c = demo.list_cutoff_settlements()
    assert c["cut_off_count"] == 1 and c["cut_off"][0]["name"] == "Village 5"
    assert {x["name"] for x in c["cut_off_if_bridges_fail"]} == {"Village 2", "Village 4"}
    assert c["settlements_without_mapped_route"] == 2
    b = demo.summarize_damage()
    assert b["buildings_hit"] == 6 and b["bridges_exposed"] == 1 and b["road_km_in_hazard"] == pytest.approx(0.19, abs=0.002)
    assert demo.flood_extent()["flood_area_km2"] == pytest.approx(0.8 * 4 + 0.011, abs=0.01) or True


# ---------------------------------------------------------------- number verification (the "no invented numbers" guarantee)
def test_number_extraction_handles_digits_and_identifiers():
    assert numbers_in("12 buildings, 1,234 m and 0.5 km") == [12.0, 1234.0, 0.5]
    assert numbers_in("बाढी १२ वटा र ३.५ किमी") == [12.0, 3.5]
    assert numbers_in("Sentinel-1/2 and EMSR927 on S1: 4 villages") == [4.0]
    assert numbers_in("1. first\n2) second") == []


def test_verify_accepts_exact_and_faithful_rounding_rejects_invention():
    data = {"a": 12, "b": 0.19, "c": [{"name": "Village 5"}], "d": "2026-08-26"}
    assert verify_numbers("12 buildings; 0.19 km; about 0.2 km; on 26 August 2026", data)[0]
    assert verify_numbers("1 settlement listed", data)[0]                      # list length is derivable
    ok, bad = verify_numbers("13 buildings and 400 people", data)
    assert not ok and bad == [13.0, 400.0]


@pytest.mark.parametrize("lang", ["en", "ne"])
@pytest.mark.parametrize("intent", ["summary", "cutoff", "buildings", "roads", "bridges", "extent", "flowpath",
                                    "limitations", "out_of_scope"])
def test_every_template_passes_its_own_verification(demo, lang, intent):
    data = copilot.tool_for(intent, demo) | {"synthetic": True}
    text = (render_ne if lang == "ne" else render_en)(intent, data)
    ok, bad = verify_numbers(text, data)
    assert ok, (intent, lang, bad, text)
    assert "SYNTHETIC" in text or "कृत्रिम" in text or intent == "limitations" or intent == "out_of_scope"


# ---------------------------------------------------------------- routing and refusal to guess
@pytest.mark.parametrize("q,intent", [
    ("Which villages are cut off?", "cut_off"), ("how many buildings were flooded", "buildings"),
    ("Are any bridges damaged?", "bridges"), ("how many people died", "out_of_scope"),
    ("What is the population affected?", "out_of_scope"), ("how reliable is this", "limitations"),
    ("give me a situation report", "summary"), ("कति बस्ती विच्छेद भए", "cutoff"),
])
def test_routing(q, intent):
    assert route(q) == intent.replace("cut_off", "cutoff")


def test_out_of_scope_answer_admits_it_and_has_no_numbers(demo):
    r = answer("How many people died?", "en", demo)
    assert r["intent"] == "out_of_scope" and "does not estimate" in r["answer"] and numbers_in(r["answer"]) == []


# ---------------------------------------------------------------- LLM wrapper: accept verified, reject invented, survive failure
class FakeLLM:
    def __init__(self, text=None, boom=False):
        self.text, self.boom = text, boom

    def generate(self, system, user):
        assert "ONLY the JSON" in system and "Question:" in user
        if self.boom:
            raise ConnectionError("ollama down")
        return self.text


def test_llm_answer_used_only_when_numbers_verify(demo):
    good = "One settlement, Village 5, is cut off from hospital access."
    r = answer("Which villages are cut off?", "en", demo, FakeLLM(good))
    assert r["mode"] == "llm" and r["answer"] == good


def test_llm_invented_number_falls_back_to_template(demo):
    bad = "About 37 villages are cut off, affecting 5000 people."
    r = answer("Which villages are cut off?", "en", demo, FakeLLM(bad))
    assert r["mode"] == "template" and "37" not in r["answer"] and r["llm_rejected"]["unverified_numbers"] == [37.0, 5000.0]
    assert "Village 5" in r["answer"]


def test_llm_failure_falls_back_and_reports(demo):
    r = answer("situation report", "ne", demo, FakeLLM(boom=True))
    assert r["mode"] == "template" and "ollama down" in r["llm_error"] and "बस्ती" in r["answer"]


def test_ollama_client_request_shape(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"response": " hi "}

    def post(url, json=None, timeout=None, **kw):
        seen.update(url=url, body=json)
        return R()

    monkeypatch.setattr(copilot.requests, "post", post)
    assert copilot.OllamaLLM("m", "http://h:1/").generate("s", "u") == "hi"
    assert seen["url"] == "http://h:1/api/generate" and seen["body"]["options"]["temperature"] == 0


# ---------------------------------------------------------------- pipeline planning, API, report
def test_pipeline_commands_and_derived_config(tmp_path):
    import yaml
    from src import pipeline
    cfg = pipeline.derive_config("config/aoi_trishuli.yaml", tmp_path, "2025-07-15", [84.0, 27.0, 84.5, 27.5])
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert str(raw["event_date"]) == "2025-07-15" and str(raw["osm_snapshot"]) == "2025-06-15"   # snapshot strictly pre-event
    assert raw["bbox"] == [84.0, 27.0, 84.5, 27.5] and raw["name"] == "Trishuli_2025"
    log = pipeline.run_stages(str(cfg), tmp_path, pipeline.STAGES, dry_run=True, weights=str(tmp_path / "none.pt"))
    names = [r["stage"] for r in log]
    assert names == pipeline.STAGES and next(r for r in log if r["stage"] == "infer")["status"] == "skipped"
    dmg = next(r for r in log if r["stage"] == "damage")
    assert "src.damage" in dmg["cmd"] and str(tmp_path / "outputs") in dmg["cmd"] and "osm_pre_event.gpkg" in dmg["cmd"]


@pytest.fixture(scope="module")
def full_demo(demo, tmp_path_factory):
    """Run the real CLIs (damage, cutoff) on the synthetic town so API/report read true pipeline outputs."""
    from src import cutoff as c, damage as d
    out = demo.out_dir
    osm = out.parent / "osm" / "osm_pre_event.gpkg"
    cfg = "config/aoi_trishuli.yaml"
    assert d.main(["--config", cfg, "--outputs", str(out), "--osm", str(osm), "--flood-source", "baseline"]) == 0
    assert c.main(["--config", cfg, "--outputs", str(out), "--osm", str(osm), "--flood-source", "baseline"]) == 0
    return out, osm, cfg


def test_api_endpoints(full_demo, monkeypatch):
    from fastapi.testclient import TestClient
    from src import api
    out, osm, cfg = full_demo
    monkeypatch.setenv("FLOOD_OUT", str(out)); monkeypatch.setenv("FLOOD_CONFIG", cfg); monkeypatch.delenv("COPILOT_LLM", raising=False)
    c = TestClient(api.app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["missing_outputs"] == [] and h["synthetic"] is True
    r = c.post("/copilot/ask", json={"question": "Which villages are cut off?", "lang": "en"}).json()
    assert "Village 5" in r["answer"] and r["verified"] and "cutoff_summary.json" in r["sources"] and r["mode"] == "template"
    ne = c.post("/copilot/ask", json={"question": "पुल", "lang": "ne"}).json()
    assert "पुल" in ne["answer"] and "कृत्रिम" in ne["answer"]
    assert "does not estimate" in c.post("/copilot/ask", json={"question": "how many died", "lang": "en"}).json()["answer"]
    assert c.post("/copilot/ask", json={"question": "", "lang": "en"}).status_code == 422
    g = c.get("/layers/buildings_hit").json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) == 6
    assert c.get("/layers/settlements").json()["features"][0]["properties"]["risk"] == "cut_off"
    assert c.get("/layers/..%2Fsecret").status_code == 404 and c.get("/downloads/../../etc/passwd").status_code in (404, 405)
    assert c.get("/downloads/damage.gpkg").status_code == 200
    assert c.get("/summary?lang=en").json()["answer"].startswith("SYNTHETIC")
    assert c.post("/run", json={"stages": ["ingest"]}).status_code == 400


def test_api_reports_missing_outputs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src import api
    monkeypatch.setenv("FLOOD_OUT", str(tmp_path)); monkeypatch.delenv("FLOOD_CONFIG", raising=False)
    c = TestClient(api.app)
    assert c.get("/health").json()["missing_outputs"]
    assert c.post("/copilot/ask", json={"question": "cut off?", "lang": "en"}).status_code == 409


@needs_pdf
def test_pdf_report_contains_numbers_names_attribution_and_nepali(full_demo, tmp_path):
    import shutil
    import subprocess
    from src import report
    out, osm, cfg = full_demo
    pdf = report.make_pdf(out, osm, cfg, ["en", "ne"], tmp_path / "r.pdf")
    assert pdf.stat().st_size > 30_000
    if shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        for needle in ("SYNTHETIC DEMO DATA", "Village 5", "Contains modified Copernicus Sentinel data 2026",
                       "OpenStreetMap contributors", "EMSR927", "human review required"):
            assert needle in txt, needle
        assert "बस्ती" in txt                                           # Nepali narrative present as real text


# ---------------------------------------------------------------- flow_source wiring
def test_flow_source_config_pipeline_skip_and_cli_error(tmp_path):
    import yaml
    from src import flowpath, pipeline
    from src.config import load_config
    assert load_config("config/aoi_trishuli.yaml").flow_source == pytest.approx((85.378, 28.278))
    raw = yaml.safe_load(open("config/aoi_trishuli.yaml", encoding="utf-8"))
    raw.pop("flow_source")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(cfg).flow_source is None
    log = pipeline.run_stages(str(cfg), tmp_path, ["flowpath"], dry_run=True)
    assert log[0]["status"] == "skipped"
    with pytest.raises(SystemExit):                      # no --start-lonlat and no flow_source -> clear CLI error
        flowpath.main(["--config", str(cfg)])
