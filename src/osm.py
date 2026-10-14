"""Pre-event OSM snapshot via the ohsome API (historical OSM, so no post-event edits can leak in).

The AOI is tiled so each request stays small, results are de-duplicated on @osmId
(features crossing tile borders are returned by both tiles), and written to one GeoPackage.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

log = logging.getLogger(__name__)
OHSOME = "https://api.ohsome.org/v1/elements/geometry"

LAYERS = {
    "buildings": "building=* and geometry:polygon",
    "roads": "highway=* and geometry:line",
    "bridges": "(bridge=yes or man_made=bridge) and geometry:line or (man_made=bridge and geometry:polygon)",
    "places": "place in (city,town,village,hamlet,suburb,neighbourhood,locality) and geometry:point",
    "health": "(amenity in (hospital,clinic,doctors,health_post) or healthcare=*) and (geometry:point or geometry:polygon)",
}

ATTRIBUTION = "© OpenStreetMap contributors (ODbL). Extracted via the ohsome API (HeiGIT)."


def tiles(bbox, step: float):
    w, s, e, n = bbox
    lon = w
    while lon < e:
        lat = s
        while lat < n:
            yield (round(lon, 6), round(lat, 6), round(min(lon + step, e), 6), round(min(lat + step, n), 6))
            lat += step
        lon += step


def _fetch(bbox, snapshot: date, flt: str, tries: int = 4) -> gpd.GeoDataFrame:
    data = {
        "bboxes": ",".join(map(str, bbox)),
        "time": snapshot.isoformat(),
        "filter": flt,
        "properties": "tags",
    }
    for i in range(tries):
        r = requests.post(OHSOME, data=data, timeout=300)
        if r.status_code == 200:
            feats = r.json().get("features", [])
            if not feats:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        if r.status_code in (429, 502, 503, 504):
            time.sleep(5 * (i + 1))
            continue
        raise RuntimeError(f"ohsome {r.status_code}: {r.text[:300]}")
    raise RuntimeError("ohsome: retries exhausted")


def download_osm(bbox, snapshot: date, step: float, out_gpkg: Path) -> dict[str, int]:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        out_gpkg.unlink()
    counts = {}
    for layer, flt in LAYERS.items():
        parts = [_fetch(t, snapshot, flt) for t in tiles(bbox, step)]
        parts = [p for p in parts if len(p)]
        if not parts:
            counts[layer] = 0
            continue
        gdf = pd.concat(parts, ignore_index=True)
        if "@osmId" in gdf:
            gdf = gdf.drop_duplicates("@osmId")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
        counts[layer] = len(gdf)
        log.info("OSM %-9s %d features", layer, len(gdf))
    (out_gpkg.parent / "OSM_ATTRIBUTION.txt").write_text(
        f"{ATTRIBUTION}\nSnapshot date: {snapshot.isoformat()}\n", encoding="utf-8")
    return counts
