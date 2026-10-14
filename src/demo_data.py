"""Synthetic town with KNOWN answers: used by tests and by `make demo` so the dashboard/copilot/report can be
tried before any real data exists. Everything it writes is marked synthetic; never present it as a real result.

Layout (UTM metres, local origin): a west-east highway with the hospital/town at the west end, a north-south river
(flooded) at x~3000 crossed by a bridge, and spurs to villages:
  V1 (1000,2600) fine            V2 (5000,2300) east of the bridge -> cut off only if the bridge fails
  V3 (2500,1500) fine            V4 (5500,1500) east of the bridge -> cut off only if the bridge fails
  V5 (1500,3500) its spur is flooded for ~100 m -> cut off in both scenarios
  V7 (4500,3800) no road within 400 m   V8 (4000,200) on a road island with no destination (unreachable_pre)
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from .grid import Grid, write_raster

E0, N0 = 330000.0, 3110000.0     # arbitrary UTM 45N origin (lon ~85.2, lat ~28.1)
CRS = "EPSG:32645"
RES = 10.0
SIZE = (700, 700)                # 7 km x 7 km


def L(*pts):
    return LineString([(E0 + x, N0 + y) for x, y in pts])


def P(x, y):
    return Point(E0 + x, N0 + y)


def build(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "osm").mkdir(parents=True, exist_ok=True)
    roads = [
        ("A", L(*[(x, 2000) for x in range(0, 2801, 100)]), "primary", "no"),
        ("B", L((2800, 2000), (3000, 2000), (3200, 2000)), "primary", "yes"),
        ("C", L(*[(x, 2000) for x in range(3200, 6001, 100)]), "primary", "no"),
        ("S1", L((1000, 2000), (1000, 2300), (1000, 2600)), "track", "no"),
        ("S2", L((5000, 2000), (5000, 2300)), "track", "no"),
        ("S3", L((2500, 2000), (2500, 1500)), "track", "no"),
        ("S4", L((5500, 2000), (5500, 1500)), "track", "no"),
        ("S5", L((1500, 2000), (1500, 2700), (1500, 3500)), "track", "no"),
        ("I1", L((4000, 200), (4300, 200), (4300, 400)), "track", "no"),   # island, no destination
    ]
    r = gpd.GeoDataFrame({"@osmId": [f"way/{n}" for n, *_ in roads], "highway": [h for *_, h, _ in roads],
                          "bridge": [b for *_, b in roads], "name": [None] * len(roads),
                          "geometry": [g for _, g, *_ in roads]}, crs=CRS)
    r.loc[r["@osmId"] == "way/B", "name"] = "Trishuli Bridge (synthetic)"
    bridges = r[r["bridge"] == "yes"].copy()

    pl = [("Town West", "town", 30, 1980), ("Village 1", "village", 1000, 2600), ("Village 2", "village", 5000, 2300),
          ("Village 3", "village", 2500, 1500), ("Village 4", "village", 5500, 1500),
          ("Village 5", "village", 1500, 3500), ("Village 7", "village", 4500, 3800),
          ("Village 8", "village", 4300, 400)]
    places = gpd.GeoDataFrame({"name": [p[0] for p in pl], "place": [p[1] for p in pl],
                               "geometry": [P(p[2], p[3]) for p in pl]}, crs=CRS)
    health = gpd.GeoDataFrame({"name": ["District Hospital (synthetic)"], "amenity": ["hospital"],
                               "geometry": [P(60, 2060)]}, crs=CRS)

    rng = np.random.default_rng(1)
    bl = []
    for _, _, x, y in pl[1:]:
        for _ in range(12):
            bx, by = x + rng.uniform(-100, 100), y + rng.uniform(-100, 100)
            bl.append(box(E0 + bx, N0 + by, E0 + bx + 8, N0 + by + 8))
    river = [box(E0 + 2975 + 12 * i, N0 + 1000 + 10 * i, E0 + 2975 + 12 * i + 8, N0 + 1000 + 10 * i + 8) for i in range(5)]
    bl += river                                           # 5 buildings squarely inside the flooded river corridor
    bl.append(box(E0 + 2954, N0 + 1500, E0 + 2962, N0 + 1508))  # straddles the corridor edge: 25% covered
    buildings = gpd.GeoDataFrame({"building": ["yes"] * len(bl), "geometry": bl}, crs=CRS)

    h, w = SIZE
    grid = Grid(CRS, from_origin(E0, N0 + h * RES, RES, RES), w, h, RES)
    flood = np.zeros((h, w), np.uint8)

    def paint(m, x0, x1, y0, y1, v=1):
        # raster row 0 is the north edge
        m[h - int(y1 / RES):h - int(y0 / RES), int(x0 / RES):int(x1 / RES)] = v

    paint(flood, 2960, 3040, 0, 4000)        # the river corridor (new flood water), crosses the highway at the bridge
    paint(flood, 1450, 1550, 2790, 2900)     # flooded stretch of the V5 spur (a ford, not a bridge)
    debris = np.zeros_like(flood)
    paint(debris, 2900, 2960, 1000, 1100, 1)  # S1 debris next to the river
    for name, g in (("buildings", buildings), ("roads", r), ("bridges", bridges), ("places", places),
                    ("health", health)):
        g.to_crs("EPSG:4326").to_file(out_dir / "osm" / "osm_pre_event.gpkg", layer=name, driver="GPKG")
    outp = out_dir / "outputs"
    write_raster(outp / "flood_mask.tif", flood, grid, descriptions=["flood"])
    write_raster(outp / "debris_mask.tif", debris, grid, descriptions=["debris"])
    (outp / "flood_report.json").write_text(json.dumps({
        "synthetic": True, "flood_area_km2": float(flood.sum() * RES * RES / 1e6),
        "debris_area_km2_s1": float(debris.sum() * RES * RES / 1e6), "debris_area_km2_s2": 0.0,
        "s1_valid_fraction": 1.0, "s1_unobservable_fraction": 0.0}, indent=2), encoding="utf-8")
    (out_dir / "SYNTHETIC_DEMO.txt").write_text("Synthetic data for testing and demos. Not a real event.\n", encoding="utf-8")
    return {"grid": grid, "osm": out_dir / "osm" / "osm_pre_event.gpkg", "outputs": outp}


if __name__ == "__main__":
    import argparse

    from .pipeline import run_stages

    ap = argparse.ArgumentParser(description="Write a synthetic demo and run damage -> cut-off -> report on it")
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--config", default="config/aoi_trishuli.yaml")
    a = ap.parse_args()
    d = build(Path(a.out))
    for r in run_stages(a.config, Path(a.out), ["damage", "cutoff", "report"], outputs=d["outputs"], osm=d["osm"]):
        print(f"{r['stage']:8s} {r['status']}", r.get("stderr_tail", ""))
    print("synthetic demo ready:", d["outputs"], "(open it with: FLOOD_OUT=%s streamlit run dashboard/app.py)" % d["outputs"])
