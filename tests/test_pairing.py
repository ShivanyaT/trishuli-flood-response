from datetime import date, datetime, timezone

import pytest
from shapely.geometry import box

from src.cdse import Product, build_filter
from src.config import AOIConfig, validate
from src.dem import tile_names
from src.osm import tiles
from src.pairing import diagnose_s1, pick_s2, rank_s1_pairs, tile_of

CFG = AOIConfig(name="t", bbox=(85.0, 28.0, 85.4, 28.4), event_date=date(2026, 8, 26),
                osm_snapshot=date(2026, 7, 27))
FULL = box(84.9, 27.9, 85.5, 28.5)       # covers AOI
HALF = box(84.9, 27.9, 85.2, 28.5)       # covers ~50% of AOI


def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def s1(day, hh, ro, direction, geom=FULL, sat="S1A", uid=None):
    name = f"{sat}_IW_GRDH_1SDV_{day.replace('-', '')}T{hh}0000_x"
    return Product(uid or name, name, dt(f"{day}T{hh}:00:00"), geom,
                   {"relativeOrbitNumber": ro, "orbitDirection": direction})


def test_same_track_pair_chosen_and_cross_track_rejected():
    prods = [
        s1("2026-08-20", "12", 41, "ASCENDING"),
        s1("2026-08-28", "12", 41, "ASCENDING"),      # same track as 08-20 -> valid pair
        s1("2026-08-24", "12", 143, "DESCENDING"),    # pre on another track
        s1("2026-08-27", "12", 12, "ASCENDING"),      # post on yet another track
    ]
    pairs, _ = rank_s1_pairs(prods, CFG)
    assert len(pairs) == 1
    assert pairs[0].track == (41, "ASCENDING")
    assert (pairs[0].pre.day, pairs[0].post.day) == ("2026-08-20", "2026-08-28")


def test_same_relative_orbit_but_opposite_direction_is_not_a_pair():
    prods = [s1("2026-08-20", "12", 41, "ASCENDING"), s1("2026-08-28", "12", 41, "DESCENDING")]
    pairs, acqs = rank_s1_pairs(prods, CFG)
    assert pairs == []
    assert "No same-track" in diagnose_s1(acqs, CFG)


def test_same_day_pass_is_not_post_event_unless_event_time_given():
    prods = [s1("2026-08-20", "12", 41, "ASCENDING"), s1("2026-08-26", "12", 41, "ASCENDING")]
    assert rank_s1_pairs(prods, CFG)[0] == []
    cfg2 = AOIConfig(**{**CFG.__dict__, "event_time_utc": dt("2026-08-26T05:00:00")})
    assert len(rank_s1_pairs(prods, cfg2)[0]) == 1


def test_frames_of_one_pass_are_merged_for_coverage():
    a, b = box(84.9, 27.9, 85.2, 28.5), box(85.2, 27.9, 85.5, 28.5)  # each ~50% of AOI
    prods = [
        s1("2026-08-20", "12", 41, "ASCENDING", a, uid="a1"), s1("2026-08-20", "12", 41, "ASCENDING", b, uid="a2"),
        s1("2026-08-28", "12", 41, "ASCENDING", a, uid="b1"), s1("2026-08-28", "12", 41, "ASCENDING", b, uid="b2"),
    ]
    prods[1].name += "_f2"; prods[3].name += "_f2"
    pairs, _ = rank_s1_pairs(prods, CFG)
    assert len(pairs) == 1 and len(pairs[0].products) == 4
    assert pairs[0].pre.coverage == pytest.approx(1.0)


def test_insufficient_coverage_rejected():
    prods = [s1("2026-08-20", "12", 41, "ASCENDING", HALF), s1("2026-08-28", "12", 41, "ASCENDING", HALF)]
    assert rank_s1_pairs(prods, CFG)[0] == []


def test_faster_post_preferred_and_single_pol_ignored():
    prods = [
        s1("2026-08-20", "12", 41, "ASCENDING"), s1("2026-08-31", "12", 41, "ASCENDING"),
        s1("2026-08-25", "00", 143, "DESCENDING"), s1("2026-08-27", "00", 143, "DESCENDING"),
    ]
    single = s1("2026-08-26", "23", 99, "ASCENDING")
    single.name = single.name.replace("1SDV", "1SSV")
    pairs, _ = rank_s1_pairs(prods + [single], CFG)
    assert [p.track for p in pairs] == [(143, "DESCENDING"), (41, "ASCENDING")]


def s2(day, tile, cloud, ver="N0511"):
    name = f"S2B_MSIL2A_{day.replace('-', '')}T050000_{ver}_R119_T{tile}_2026"
    return Product(name, name, dt(f"{day}T05:00:00"), FULL, {"cloudCover": cloud})


def test_s2_prefers_clear_scene_and_flags_cloudy_fallback():
    prods = [s2("2026-08-12", "45RUL", 5), s2("2026-08-20", "45RUL", 70),
             s2("2026-08-27", "45RUL", 90), s2("2026-09-01", "45RUL", 60)]
    ch = {c.role: c for c in pick_s2(prods, CFG)}
    assert ch["pre"].product.start.day == 12 and not ch["pre"].cloudy   # clear beats closer-but-cloudy
    assert ch["post"].product.start.day == 1 and ch["post"].cloudy      # fallback = lowest cloud
    assert tile_of(ch["pre"].product.name) == "45RUL"


def test_s2_reprocessing_duplicates_collapsed():
    prods = [s2("2026-08-12", "45RUL", 5, "N0510"), s2("2026-08-12", "45RUL", 5, "N0511")]
    assert len(pick_s2(prods, CFG)) == 1


def test_config_rejects_post_event_osm_snapshot():
    bad = AOIConfig(**{**CFG.__dict__, "osm_snapshot": date(2026, 8, 30)})
    with pytest.raises(ValueError):
        validate(bad)


def test_dem_tile_names_and_osm_tiling():
    assert tile_names((85.1, 27.8, 85.5, 28.35)) == [
        "Copernicus_DSM_COG_10_N27_00_E085_00_DEM", "Copernicus_DSM_COG_10_N28_00_E085_00_DEM"]
    ts = list(tiles((85.0, 28.0, 85.5, 28.4), 0.25))
    assert len(ts) == 4 and ts[0][0] == 85.0 and max(t[2] for t in ts) == 85.5


def test_odata_filter_contains_required_clauses():
    f = build_filter("SENTINEL-1", CFG.aoi_wkt, CFG.search_start, CFG.search_end, "IW_GRDH_1S")
    assert "Collection/Name eq 'SENTINEL-1'" in f and "OData.CSC.Intersects" in f and "IW_GRDH_1S" in f
