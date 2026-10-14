"""Damage overlay: flood + debris masks x PRE-event OSM (buildings, roads, bridges).

Inputs : data/outputs/flood_mask*.tif, debris_mask.tif, data/raw/osm/osm_pre_event.gpkg (from src.ingest)
Outputs: data/outputs/damage.gpkg (layers buildings_hit, roads_hit, bridges_exposed) and damage_summary.json

What "hit" means here (and does not mean):
  * Buildings: `centroid_hit` is the plan's rule (centroid inside the hazard). We also record the share of the
    footprint covered and a coarse severity. OSM footprints are ~30-100 m^2, i.e. 3-10 pixels, so the share is
    noisy; treat severity as a ranking aid, not a damage grade. Radar cannot see structural collapse.
  * Roads: the flooded stretch is exported as its own geometry.
  * Bridges: a bridge over a river always crosses the river's water, so we only count NEW flood/debris within
    `bridge_buffer_m`. "Exposed" means exposed to hazard, NOT collapsed.

    python -m src.damage --config config/aoi_trishuli.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import features
from shapely.geometry import shape
from shapely.ops import unary_union

from .config import load_config
from .grid import Grid, grid_from_raster, read_raster

log = logging.getLogger(__name__)

SEVERITY_BINS = [(0.5, "high"), (0.1, "moderate"), (0.0, "low")]
ROAD_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
             "service", "track"]


# ----------------------------------------------------------------------------- hazard
def mask_to_polygons(mask: np.ndarray, grid: Grid):
    """Binary mask (H,W) -> shapely geometry (union of polygons) in the grid CRS. Empty mask -> empty geometry."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return shapely.Polygon()
    geoms = [shape(g) for g, v in features.shapes(m, mask=m > 0, transform=grid.transform) if v == 1]
    return unary_union(geoms)


def load_hazard(outputs: Path, flood_source: str = "auto", reports: Path = Path("reports")):
    """Returns (flood_geom, debris_geom, grid, source_used)."""
    src = choose_flood_source(outputs, reports) if flood_source == "auto" else flood_source
    fm = {"baseline": outputs / "flood_mask.tif", "ml": outputs / "flood_mask_ml.tif"}
    if src == "union":
        a, b = read_raster(fm["baseline"])[0][0] > 0, read_raster(fm["ml"])[0][0] > 0
        flood, grid = (a | b).astype(np.uint8), grid_from_raster(fm["baseline"])
    else:
        flood, grid = read_raster(fm[src])[0][0], grid_from_raster(fm[src])
    dp = outputs / "debris_mask.tif"
    debris = read_raster(dp)[0][0] if dp.exists() else np.zeros_like(flood)
    return mask_to_polygons(flood, grid), mask_to_polygons(debris, grid), grid, src


def choose_flood_source(outputs: Path, reports: Path) -> str:
    """Use the ML mask only if it exists AND beat the threshold baseline on held-out Kuro Siwo test patches.

    Kuro Siwo F1 is not Trishuli F1, so this is a defensible default, not proof; override with --flood-source."""
    ml, ev = outputs / "flood_mask_ml.tif", reports / "eval_metrics.json"
    if ml.exists() and ev.exists():
        m = json.loads(ev.read_text(encoding="utf-8"))
        if m["model"]["f1"] > m["baseline"]["f1"]:
            return "ml"
    return "baseline"


# ----------------------------------------------------------------------------- OSM
def load_layer(gpkg: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        g = gpd.read_file(gpkg, layer=layer)
    except Exception:  # layer absent (e.g. no bridges mapped)
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.to_crs(crs) if len(g) else gpd.GeoDataFrame(g, crs=crs)


def _col(g: gpd.GeoDataFrame, name: str, default=None):
    return g[name] if name in g.columns else default


# ----------------------------------------------------------------------------- analyses
def damaged_buildings(buildings: gpd.GeoDataFrame, flood, debris) -> gpd.GeoDataFrame:
    hazard = unary_union([flood, debris])
    if buildings.empty or hazard.is_empty:
        return buildings.iloc[0:0].assign(hazard=[], overlap_frac=[], severity=[], centroid_hit=[])
    shapely.prepare(hazard)
    cand = buildings[shapely.intersects(buildings.geometry.values, hazard)].copy()
    if cand.empty:
        return cand.assign(hazard=[], overlap_frac=[], severity=[], centroid_hit=[])
    geoms = cand.geometry.values
    area = np.maximum(shapely.area(geoms), 1e-6)
    f_frac = shapely.area(shapely.intersection(geoms, flood)) / area if not flood.is_empty else np.zeros(len(cand))
    d_frac = shapely.area(shapely.intersection(geoms, debris)) / area if not debris.is_empty else np.zeros(len(cand))
    tot = shapely.area(shapely.intersection(geoms, hazard)) / area
    cand["overlap_frac"] = np.clip(tot, 0, 1).round(3)
    cand["hazard"] = np.where((f_frac > 0) & (d_frac > 0), "both", np.where(f_frac > 0, "flood", "debris"))
    cen = shapely.centroid(geoms)
    cand["centroid_hit"] = shapely.contains_xy(hazard, shapely.get_x(cen), shapely.get_y(cen))

    def sev(x):
        return next(lbl for thr, lbl in SEVERITY_BINS if x > thr or thr == 0.0)

    cand["severity"] = cand["overlap_frac"].map(sev)
    return cand


def roads_in_hazard(roads: gpd.GeoDataFrame, hazard) -> gpd.GeoDataFrame:
    if roads.empty or hazard.is_empty:
        return roads.iloc[0:0].assign(length_hit_m=[])
    shapely.prepare(hazard)
    cand = roads[shapely.intersects(roads.geometry.values, hazard)].copy()
    if cand.empty:
        return cand.assign(length_hit_m=[])
    inter = shapely.intersection(cand.geometry.values, hazard)
    cand["length_hit_m"] = shapely.length(inter).round(1)
    cand["length_m"] = shapely.length(cand.geometry.values).round(1)
    cand = cand.set_geometry(gpd.GeoSeries(inter, index=cand.index, crs=roads.crs))
    return cand[cand["length_hit_m"] > 0]


def bridges_exposed(bridges: gpd.GeoDataFrame, flood, debris, buffer_m: float = 10.0) -> gpd.GeoDataFrame:
    hazard = unary_union([flood, debris])
    if bridges.empty or hazard.is_empty:
        return bridges.iloc[0:0].assign(hazard=[])
    shapely.prepare(hazard)
    near = shapely.buffer(bridges.geometry.values, buffer_m)
    hit = shapely.intersects(near, hazard)
    out = bridges[hit].copy()
    f = shapely.intersects(near[hit], flood) if not flood.is_empty else np.zeros(hit.sum(), bool)
    d = shapely.intersects(near[hit], debris) if not debris.is_empty else np.zeros(hit.sum(), bool)
    out["hazard"] = np.where(f & d, "both", np.where(f, "flood", "debris"))
    return out


def summarise(bld_all, bld_hit, roads_all, roads_hit, br_all, br_hit, src: str, grid: Grid) -> dict:
    by_sev = bld_hit["severity"].value_counts().to_dict() if len(bld_hit) else {}
    hw = _col(roads_hit, "highway")
    km_by_class = (roads_hit.assign(km=roads_hit["length_hit_m"] / 1000).groupby(hw)["km"].sum().round(3).to_dict()
                   if len(roads_hit) and hw is not None else {})
    return {
        "flood_source": src,
        "buildings_total_osm": int(len(bld_all)),
        "buildings_hit": int(len(bld_hit)),
        "buildings_centroid_hit": int(bld_hit["centroid_hit"].sum()) if len(bld_hit) else 0,
        "buildings_by_severity": {k: int(v) for k, v in by_sev.items()},
        "buildings_by_hazard": ({k: int(v) for k, v in bld_hit["hazard"].value_counts().items()}
                                if len(bld_hit) else {}),
        "road_km_total_osm": round(float(roads_all.length.sum() / 1000), 2) if len(roads_all) else 0.0,
        "road_km_in_hazard": round(float(roads_hit["length_hit_m"].sum() / 1000), 3) if len(roads_hit) else 0.0,
        "road_km_in_hazard_by_class": km_by_class,
        "bridges_total_osm": int(len(br_all)),
        "bridges_exposed": int(len(br_hit)),
        "bridges_exposed_names": [str(n) for n in _col(br_hit, "name", []) if isinstance(n, str)] if len(br_hit) else [],
        "caveat": ("OSM completeness in rural Nepal is partial: counts are lower bounds on exposure. 'Exposed' is not "
                   "'collapsed'; SAR/optical masks at 10 m cannot grade structural damage."),
        "grid_crs": grid.crs,
    }


def run(osm: Path, flood, debris, grid: Grid, src: str, bridge_buffer_m: float = 10.0):
    bld = load_layer(osm, "buildings", grid.crs)
    roads = load_layer(osm, "roads", grid.crs)
    br = load_layer(osm, "bridges", grid.crs)
    if "@osmId" in roads.columns and "@osmId" in br.columns:
        br = br.drop_duplicates("@osmId")
    # roads tagged as bridges are bridges too (ohsome may return them only under `highway`)
    if "bridge" in roads.columns:
        extra = roads[roads["bridge"].fillna("no").astype(str).isin(["yes", "viaduct", "movable", "cantilever"])]
        both = pd.concat([br, extra], ignore_index=True)
        if "@osmId" in both.columns:
            both = both.drop_duplicates(subset=["@osmId"])
        br = gpd.GeoDataFrame(both, geometry="geometry", crs=grid.crs)
    hazard = unary_union([flood, debris])
    bld_hit = damaged_buildings(bld, flood, debris)
    roads_hit = roads_in_hazard(roads, hazard)
    br_hit = bridges_exposed(br, flood, debris, bridge_buffer_m)
    return bld_hit, roads_hit, br_hit, summarise(bld, bld_hit, roads, roads_hit, br, br_hit, src, grid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--osm", default="data/raw/osm/osm_pre_event.gpkg")
    ap.add_argument("--flood-source", default="auto", choices=["auto", "baseline", "ml", "union"])
    ap.add_argument("--bridge-buffer-m", type=float, default=10.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_config(a.config)
    out = Path(a.outputs)
    flood, debris, grid, src = load_hazard(out, a.flood_source)
    bld_hit, roads_hit, br_hit, summ = run(Path(a.osm), flood, debris, grid, src, a.bridge_buffer_m)
    gpkg = out / "damage.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for name, g in (("buildings_hit", bld_hit), ("roads_hit", roads_hit), ("bridges_exposed", br_hit)):
        if len(g):
            g.to_file(gpkg, layer=name, driver="GPKG")
    (out / "damage_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log.info("buildings hit %d | road km in hazard %.2f | bridges exposed %d (flood source: %s)",
             summ["buildings_hit"], summ["road_km_in_hazard"], summ["bridges_exposed"], src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
