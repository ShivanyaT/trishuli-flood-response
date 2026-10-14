
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.flood_map import FloodParams, cleanup, run_baseline, s1_water
from src.grid import make_grid, utm_epsg, warp_raster, write_raster
from src.s1_preprocess import GRAPH, coregister, estimate_shift, run_gpt, to_db
from src.s2_preprocess import baseline_offset, to_reflectance, valid_from_scl
from src.terrain import corridor_mask, relative_elevation, slope_deg
from tests.synth import RES, make_scene


def f1(pred, truth):
    tp = (pred & truth).sum()
    return 2 * tp / max(1, pred.sum() + truth.sum())


# ---------------------------------------------------------------- grid / IO
def test_utm_zone_and_grid_alignment():
    assert utm_epsg(85.2, 28.0) == 32645
    assert utm_epsg(-70.0, -33.0) == 32719
    g = make_grid((84.85, 27.80, 85.55, 28.35), 10.0)
    assert g.crs == "EPSG:32645" and g.transform.c % 10 == 0 and g.transform.f % 10 == 0
    assert 6000 < g.width < 8000 and 5000 < g.height < 7000


def test_warp_roundtrip_constant_field(tmp_path):
    src = tmp_path / "a.tif"
    arr = np.full((1, 200, 200), 7.0, np.float32)
    with rasterio.open(src, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(85.0, 28.1, 0.001, 0.001)) as d:
        d.write(arr)
    g = make_grid((85.02, 27.93, 85.18, 28.07), 30.0)
    out = warp_raster(src, g, [1])[0]
    assert np.isfinite(out).mean() > 0.95 and np.allclose(out[np.isfinite(out)], 7.0, atol=1e-4)
    p = write_raster(tmp_path / "o.tif", np.ones(g.shape, np.uint8), g)
    with rasterio.open(p) as d:
        assert d.crs.to_epsg() == 32645 and d.dtypes[0] == "uint8"


# ---------------------------------------------------------------- terrain
def test_slope_and_corridor():
    s = make_scene()
    sl = slope_deg(s["dem"], RES)
    assert 28 < np.median(sl[:, 5][:40]) < 34          # side wall ~ 31 deg
    cor = corridor_mask(s["dem"], RES)
    assert cor[150, 150] and not cor[10, 150] and not cor[290, 150]
    rel = relative_elevation(s["dem"], RES)
    assert np.nanmax(rel[145:155, 150]) < 15 and np.nanmax(rel[:, 150]) > 40


# ---------------------------------------------------------------- S1
def test_to_db_and_nodata():
    out = to_db(np.array([1.0, 0.1, 0.0, -1.0, np.nan], np.float32))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(-10.0)
    assert np.isnan(out[2:]).all()


def test_coregistration_recovers_known_shift():
    rng = np.random.default_rng(1)
    from scipy import ndimage as ndi
    base = ndi.gaussian_filter(rng.normal(size=(400, 400)), 3).astype(np.float32)
    shifted = ndi.shift(base, (1.0, -1.5), order=3, mode="nearest")
    dr, dc = estimate_shift(base, shifted)
    assert abs(dr - (-1.0)) < 0.3 and abs(dc - 1.5) < 0.3
    post = np.stack([shifted, shifted - 7])
    fixed, _ = coregister(base, post)
    c = slice(20, -20)
    assert np.abs(fixed[0][c, c] - base[c, c]).mean() < np.abs(shifted[c, c] - base[c, c]).mean() / 3


def test_graph_has_required_operators_and_gpt_missing_is_clear(tmp_path):
    for op in ("Apply-Orbit-File", "ThermalNoiseRemoval", "Remove-GRD-Border-Noise", "Calibration",
               "Speckle-Filter", "Terrain-Correction", "External DEM"):
        assert op in GRAPH
    with pytest.raises(RuntimeError, match="SNAP"):
        run_gpt(tmp_path / "x.zip", tmp_path / "o.tif", tmp_path / "d.tif",
                make_grid((85, 28, 85.1, 28.1)), (85, 28, 85.1, 28.1), gpt="definitely-not-installed")


# ---------------------------------------------------------------- S2
def test_s2_reflectance_offset_and_scl():
    assert baseline_offset("S2B_MSIL2A_20260827T050000_N0511_R119_T45RUL_x") == -1000
    assert baseline_offset("S2A_MSIL2A_20190827T050000_N0213_R119_T45RUL_x") == 0
    r = to_reflectance(np.array([0, 1000, 3000, 11000], np.float32), -1000.0)
    assert np.isnan(r[0]) and r[1] == 0.0 and r[2] == pytest.approx(0.2)
    scl = np.full((20, 20), 4.0)
    scl[10, 10] = 9  # one cloudy pixel is dilated
    v = valid_from_scl(scl, dilate_px=2)
    assert not v[10, 10] and not v[12, 12] and v[0, 0]


# ---------------------------------------------------------------- baseline
@pytest.fixture(scope="module")
def result():
    s = make_scene()
    p = FloodParams(min_component_px=10)
    return s, run_baseline(s["s1_pre"], s["s1_post"], s["dem"], RES, p, s["s2_pre"], s["s2_post"])


def test_new_flood_found_permanent_river_and_shadow_not(result):
    s, r = result
    flood, t = r["flood"].astype(bool), s["truth"]
    assert f1(flood, t["flood_new"]) > 0.9
    assert (flood & t["shadow"]).sum() == 0              # same-track change cancels radar shadow
    assert (flood & t["river"]).mean() < 0.05            # already-dark river is not "new flood"


def test_debris_detected_by_both_sensors_and_not_in_flood(result):
    s, r = result
    deb = r["debris"]
    z = s["truth"]["debris"]
    # S1 brightening and S2 vegetation loss both fire on the debris strip; opening trims the 3-px-wide edges
    assert (deb[z] & 1).mean() > 0.4 and (deb[z] & 2).mean() > 0.4
    assert (deb[r["flood"].astype(bool)] > 0).sum() == 0
    assert (deb > 0)[~s["truth"]["debris"]].mean() < 0.01


def test_s1_alone_still_works_without_s2_and_reports_observability():
    s = make_scene(seed=3)
    r = run_baseline(s["s1_pre"], s["s1_post"], s["dem"], RES, FloodParams())
    assert f1(r["flood"].astype(bool), s["truth"]["flood_new"]) > 0.85
    assert r["summary"]["s2_clear_fraction_both_dates"] == 0.0
    assert 0 < r["summary"]["corridor_fraction"] < 1


def test_corridor_blocks_hillside_false_positives():
    s = make_scene(seed=4)
    post = s["s1_post"].copy()
    post[0][10:30, 200:260] = -22.0                       # wet-snow-like drop on a steep hillside
    r = run_baseline(s["s1_pre"], post, s["dem"], RES, FloodParams())
    assert r["flood"][10:30, 200:260].sum() == 0


def test_unobservable_pixels_excluded():
    s = make_scene(seed=5)
    unobs = np.zeros(s["dem"].shape, bool)
    unobs[140:160, 100:175] = True
    r = run_baseline(s["s1_pre"], s["s1_post"], s["dem"], RES, FloodParams(), s1_unobservable=unobs)
    assert r["flood"][140:160, 100:175].sum() == 0 and r["flood"][140:160, 180:240].sum() > 0
    assert r["summary"]["s1_unobservable_fraction"] > 0


def test_cleanup_removes_specks_and_keeps_blobs():
    m = np.zeros((50, 50), bool)
    m[5, 5] = True
    m[20:30, 20:30] = True
    out = cleanup(m, 10)
    assert not out[5, 5] and out[25, 25]


def test_threshold_semantics():
    p = FloodParams()
    assert s1_water(np.array([-20.0]), np.array([-10.0]), p)[0]
    assert not s1_water(np.array([-20.0]), np.array([-19.0]), p)[0]   # dark but was already dark
    assert not s1_water(np.array([-10.0]), np.array([-5.0]), p)[0]    # changed but not dark enough
    assert not s1_water(np.array([np.nan]), np.array([-5.0]), p)[0]


def test_preview_figure_renders(tmp_path, result):
    from src.preview import plot_flood_overview
    s, r = result
    out = tmp_path / "p.png"
    plot_flood_overview(s["s1_pre"][0], s["s1_post"][0], r["flood"], r["debris"], out, step=2)
    assert out.exists() and out.stat().st_size > 10_000


def test_config_flood_block_overrides_and_rejects_unknown_keys(tmp_path):
    from src.flood_map import load_thresholds
    p = load_thresholds(None, {"dvv_water_db": -2.0, "vv_water_db": -15.0, "min_component_px": 7})
    assert (p.vv_drop_db, p.vv_water_db, p.min_component_px) == (-2.0, -15.0, 7)
    with pytest.raises(ValueError):
        load_thresholds(None, {"nonsense": 1})
    f = tmp_path / "t.json"
    f.write_text('{"vv_water_db": -18, "vv_drop_db": -4}', encoding="utf-8")
    q = load_thresholds(f, {"vv_water_db": -15.0})          # explicit config beats the tuned file
    assert (q.vv_water_db, q.vv_drop_db) == (-15.0, -4)


def test_pre_water_margin_blocks_pixels_that_were_already_dark():
    from src.flood_map import FloodParams, s1_water
    p = FloodParams(vv_drop_db=-1.0)
    assert not s1_water(np.array([-20.0]), np.array([-15.0]), p)[0]   # pre -15 is within margin of the water threshold
    assert s1_water(np.array([-20.0]), np.array([-10.0]), p)[0]
