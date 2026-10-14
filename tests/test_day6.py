"""Day 6: EMSR927 validation module, tested on an EMSR-like reference built from the synthetic town (known answers)."""
import json
import re
from pathlib import Path

import geopandas as gpd
import pytest
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union

from src import validate
from src.damage import load_layer
from tests.test_day5 import demo, full_demo, needs_pdf  # noqa: F401  (fixtures)

E0, N0, CRS = 330000.0, 3110000.0, "EPSG:32645"


def _bx(x0, x1, y0, y1):
    return box(E0 + x0, N0 + y0, E0 + x1, N0 + y1)


def make_emsr(folder: Path, osm: Path, shift_m: float = 0.0, with_transport: bool = True) -> Path:
    """Write EMSR-style vector files. Reference flood = exactly our synthetic flood + debris (optionally shifted)."""
    folder.mkdir(parents=True, exist_ok=True)
    flood = unary_union([_bx(2960, 3040, 0, 4000), _bx(1450, 1550, 2790, 2900), _bx(2900, 2960, 1000, 1100)])
    flood = affinity.translate(flood, xoff=shift_m)
    aoi = _bx(2500, 3500, 800, 2400)
    out = lambda g, name: g.to_crs(4326).to_file(folder / name, driver="GeoJSON")  # noqa: E731
    out(gpd.GeoDataFrame({"obj_desc": ["Flooded area"], "geometry": [flood]}, crs=CRS), "EMSR927_AOI01_DEL_observedEventA.json")
    out(gpd.GeoDataFrame({"name": ["AOI01"], "geometry": [aoi]}, crs=CRS), "EMSR927_AOI01_areaOfInterestA.json")
    b = load_layer(osm, "buildings", CRS)
    b = b[b.intersects(flood.buffer(1))].copy()
    b = b[b.intersects(aoi)]
    b["damage_gra"] = ["Destroyed", "Damaged", "Possibly damaged", "No visible damage", "Damaged", "Damaged"][:len(b)] if len(b) <= 6 \
        else ["Damaged"] * len(b)
    out(b[["damage_gra", "geometry"]].reset_index(drop=True), "EMSR927_AOI01_GRA_builtUpP.json")
    if with_transport:
        roads = load_layer(osm, "roads", CRS)
        br = roads[roads["bridge"] == "yes"].copy()
        br["obj_type"], br["damage_gra"] = "Bridge", "Damaged"
        hw =gpd.GeoDataFrame({"obj_type": ["Road"], "damage_gra": ["Destroyed"],
                               "geometry": [roads[roads["@osmId"] == "way/S5"].geometry.iloc[0].intersection(
                                   _bx(1450, 1550, 2790, 2900))]}, crs=CRS)
        out(gpd.GeoDataFrame(__import__("pandas").concat([br, hw], ignore_index=True), crs=CRS),
            "EMSR927_AOI01_GRA_transportationL.json")
    return folder


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    return tmp_path_factory.mktemp("reports")


def _run(full_demo, emsr, reports, **kw):  # noqa: F811
    out, osm, cfg = full_demo
    return validate.run(cfg, emsr, out, osm, reports, source="baseline", **kw)


# ---------------------------------------------------------------- pure helpers
def test_prf_and_grade_normalisation():
    m = validate.prf(8, 2, 2, 88)
    assert m["iou"] == pytest.approx(8 / 12) and m["f1"] == pytest.approx(0.8) and m["precision"] == 0.8
    assert validate.prf(0, 0, 0)["iou"] is None
    n = validate.normalise_grade
    assert n("Destroyed") == "destroyed" and n("Damaged") == "damaged" and n("Possibly damaged") == "possibly damaged"
    assert n("No visible damage") == "none" and n("") is None and n(None) is None


def test_raster_scores_tolerance_forgives_one_pixel_shift():
    import numpy as np
    a = np.zeros((50, 50), bool); a[10:40, 20:28] = True
    b = np.roll(a, 1, axis=1)
    mask = np.ones_like(a)
    s = validate.raster_scores(a, b, mask, tol_px=1)
    assert s["strict"]["iou"] < 0.9
    assert s["tolerant"]["precision"] == 1.0 and s["tolerant"]["recall"] == 1.0
    s2 = validate.raster_scores(a, np.roll(a, 3, axis=1), mask, tol_px=1)
    assert s2["tolerant"]["f1"] < 1.0


# ---------------------------------------------------------------- end to end
def test_identical_reference_scores_one(full_demo, tmp_path, reports):  # noqa: F811
    out, osm, cfg = full_demo
    emsr = make_emsr(tmp_path / "emsr", osm)
    r = _run(full_demo, emsr, reports, emsr_image_date="2026-08-27")
    f = r["flood"]["flood_or_debris"]
    assert f["strict"]["iou"] == pytest.approx(1.0) and f["tolerant"]["f1"] == pytest.approx(1.0)
    assert f["eval_px"] > 0 and r["flood"]["per_aoi"][0]["iou"] == pytest.approx(1.0)
    # flood mask alone misses the debris strip that the reference contains
    assert r["flood"]["flood_only"]["strict"]["recall"] < 1.0
    assert (reports / "validation.json").exists() and (reports / "validation.md").exists()
    assert (reports / "fig_val_confusion.png").exists() and (reports / "fig_val_flood_aoi1.png").exists()
    assert "EMSR is an analyst product" in (reports / "validation.md").read_text(encoding="utf-8")


def test_shifted_reference_lowers_strict_but_tolerant_recovers(full_demo, tmp_path, reports):  # noqa: F811
    out, osm, cfg = full_demo
    emsr = make_emsr(tmp_path / "emsr_shift", osm, shift_m=10.0)   # exactly 1 pixel
    f = _run(full_demo, emsr, reports)["flood"]["flood_or_debris"]
    assert f["strict"]["iou"] < 0.95
    assert f["tolerant"]["precision"] == pytest.approx(1.0) and f["tolerant"]["recall"] == pytest.approx(1.0)


def test_buildings_roads_bridges_cutoff(full_demo, tmp_path, reports):  # noqa: F811
    out, osm, cfg = full_demo
    r = _run(full_demo, make_emsr(tmp_path / "emsr_b", osm), reports)
    b = r["buildings"]
    assert b["emsr_graded"] >= 1 and b["osm_match_rate_affected"] == pytest.approx(1.0)
    assert b["recall_on_matched_affected"] == pytest.approx(1.0)
    assert b["precision_vs_emsr_affected_lower_bound"] is not None
    # the bridge is exposed in our damage layer and damaged in the reference
    br = r["bridges"]
    assert br["emsr_bridges_damaged"] == 1 and br["ours_exposed_in_aoi"] == 1 and br["recall"] == 1.0
    assert r["roads"]["emsr_km"] > 0
    c = r["cutoff"]["cut_off_or_if_bridges_fail"]
    assert c["jaccard"] is not None and 0.0 <= c["jaccard"] <= 1.0
    assert r["cutoff"]["settlements_compared"] >= 1


def test_missing_layers_degrade_gracefully(full_demo, tmp_path, reports):  # noqa: F811
    out, osm, cfg = full_demo
    r = _run(full_demo, make_emsr(tmp_path / "emsr_nt", osm, with_transport=False), reports)
    assert "bridges" not in r and "cutoff" not in r
    assert any("transportation" in n for n in r["notes"])


def test_no_flood_file_is_an_error(full_demo, tmp_path, reports):  # noqa: F811
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="observedEvent"):
        _run(full_demo, tmp_path / "empty", reports)


def test_cli_and_inspect(full_demo, tmp_path, capsys):  # noqa: F811
    out, osm, cfg = full_demo
    emsr = make_emsr(tmp_path / "emsr_cli", osm)
    assert validate.main(["inspect", "--emsr-dir", str(emsr)]) == 0
    assert "observedEvent" in capsys.readouterr().out
    rep = tmp_path / "rep"
    assert validate.main(["run", "--config", cfg, "--emsr-dir", str(emsr), "--outputs", str(out), "--osm", str(osm),
                          "--reports", str(rep), "--flood-source", "baseline"]) == 0
    assert json.loads((rep / "validation.json").read_text(encoding="utf-8"))["flood"]["flood_or_debris"]["strict"]["iou"] == pytest.approx(1.0)


# ---------------------------------------------------------------- the leakage guard
def test_nothing_else_imports_validate():
    for p in Path("src").rglob("*.py"):
        if p.name == "validate.py":
            continue
        txt = p.read_text(encoding="utf-8")
        assert not re.search(r"(from\s+\.\s*import\s+[^\n]*\bvalidate\b|from\s+\.validate|import\s+src\.validate|from\s+src\.validate|from\s+src\s+import\s+[^\n]*\bvalidate\b)", txt), p


# ---------------------------------------------------------------- final report
@needs_pdf
def test_final_report_pending_then_filled(full_demo, tmp_path, reports):  # noqa: F811
    import shutil
    from src import final_report as fr
    out, osm, cfg = full_demo
    rep = tmp_path / "rep"; rep.mkdir()
    shutil.copy("reports/limitations.md", rep)
    # no validation / eval yet: real-result numbers must be PENDING, never invented
    assert fr.main(["--outputs", str(out), "--reports", str(rep), "--repo-url", "https://example.org/r"]) == 0
    t = (rep / "final_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in t and "PENDING" in t and "https://example.org/r" in t
    for h in ("## Abstract", "## Architecture", "## Methods", "## Results", "## Limitations",
              "## Attributions and citations", "## Reproducibility"):
        assert h in t
    assert (rep / "final_report.pdf").stat().st_size > 5000 and (rep / "fig_architecture.png").exists()
    # with validation results the numbers come from validation.json
    emsr = make_emsr(tmp_path / "emsr_fr", osm)
    _run(full_demo, emsr, rep)
    assert fr.main(["--outputs", str(out), "--reports", str(rep)]) == 0
    t2 = (rep / "final_report.md").read_text(encoding="utf-8")
    assert "Strict IoU 100.0%" in t2


def test_abstract_is_at_most_150_words():
    from src import final_report as fr
    a = fr.abstract({"iou": "PENDING", "f1": "PENDING", "f1t": "PENDING", "cut": "PENDING"})
    assert 100 <= len(a.split()) <= 150


@needs_pdf
def test_final_report_page_limit(tmp_path):
    from src import final_report as fr
    long = "# T\n\n" + "\n\n".join(["Lorem ipsum dolor sit amet " * 60] * 40)
    assert fr.make_pdf(long, tmp_path, tmp_path / "x.pdf") > fr.MAX_PAGES


# ---------------------------------------------------------------- a second AOI (pipeline is not tied to Trishuli)
def test_second_aoi_derives_config_and_utm_zone(tmp_path):
    from src.config import load_config
    from src.pipeline import commands, derive_config
    k = load_config("config/aoi_koshi_example.yaml")
    assert k.utm_epsg == 32645 and k.flow_source is None
    # a western-Nepal box falls in a different UTM zone: nothing may be hard-coded to 32645
    wd = tmp_path / "w"
    cfg = derive_config("config/aoi_koshi_example.yaml", wd, "2025-07-10", [81.0, 28.5, 81.5, 29.0])
    w = load_config(cfg)
    assert w.utm_epsg == 32644 and w.event_date.isoformat() == "2025-07-10" and w.osm_snapshot < w.event_date
    names = [n for n, _ in commands(cfg, wd, ["ingest", "s1", "damage"])]
    assert names == ["ingest", "s1", "damage"]


def test_second_aoi_pipeline_dry_run_and_tail(full_demo, tmp_path):  # noqa: F811
    from src.pipeline import main
    out, osm, cfg = full_demo
    wd = tmp_path / "k"
    assert main(["--aoi", "config/aoi_koshi_example.yaml", "--workdir", str(wd), "--dry-run"]) == 0
    log = json.loads((wd / "pipeline_log.json").read_text(encoding="utf-8"))
    assert [r["stage"] for r in log][:2] == ["ingest", "s1"] and all(r["status"] in ("dry-run", "skipped") for r in log)
    assert any("flow_source" in r.get("why", "") for r in log)   # no flow source for this AOI: stage skipped, not failed


# ---------------------------------------------------------------- Day 7: reproducibility files (static checks; no Docker daemon needed)
def test_dockerfile_matches_the_plan():
    t = Path("Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM mambaorg/micromamba:1\.5", t, re.M)
    assert "COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/" in t
    assert "micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a" in t
    assert "WORKDIR /app" in t and '"src.api:app"' in t and '"0.0.0.0"' in t and '"8000"' in t


def test_compose_makefile_env_and_readme():
    import yaml
    c = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert {"init", "api", "dashboard", "pipeline"} <= set(c["services"])
    assert c["services"]["api"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert c["services"]["pipeline"]["profiles"] == ["pipeline"]
    mk = Path("Makefile").read_text(encoding="utf-8")
    for tgt in ("ingest", "train", "infer", "dashboard", "report"):
        assert re.search(rf"^{tgt}:", mk, re.M), tgt
    assert Path(".env.example").exists(), "missing .env.example (recreate it from the README)"
    assert "CDSE_USER" in Path(".env.example").read_text(encoding="utf-8")
    rd = Path("README.md").read_text(encoding="utf-8")
    assert "docker compose up" in rd and "python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26" in rd
    assert "Expected outputs" in rd and ".env" in Path(".dockerignore").read_text(encoding="utf-8")
    assert Path("docker/init.sh").stat().st_mode & 0o111
