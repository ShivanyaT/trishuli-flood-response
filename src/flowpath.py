"""Bonus: trace a flood path downstream from an upstream point on the Copernicus DEM, list settlements along it.

Pure numpy + heapq (no richdem/whitebox): those need compiling, and judges must be able to run this.
Depressions are handled with a Priority-Flood fill (Barnes et al. 2014) on a block-averaged copy of the DEM
(default 30 m); flow follows D8 steepest descent on the filled surface.

Honest limits: GLO-30 vertical error (~4 m) can misroute flow across flat valley floors; a debris flow can
leave the thalweg and an ice-rock avalanche is not water-in-a-channel physics. This gives the likely corridor,
not an inundation model.

    python -m src.flowpath --config config/aoi_trishuli.yaml --start-lonlat 85.38 28.28
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from .config import load_config
from .damage import load_layer
from .grid import Grid, grid_from_raster, read_raster

log = logging.getLogger(__name__)
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a.astype(np.float64)
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return np.nanmean(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))


def priority_flood(dem: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Fill depressions so every cell drains to the border (or to NaN no-data). O(N log N) in pure Python:
    ~1 min per 4 M cells."""
    h, w = dem.shape
    filled = dem.copy()
    closed = np.zeros(dem.shape, bool)
    heap: list[tuple[float, int, int]] = []

    def push(r, c):
        closed[r, c] = True
        heapq.heappush(heap, (filled[r, c], r, c))

    for r in range(h):
        for c in (0, w - 1):
            if not closed[r, c]:
                push(r, c)
    for c in range(w):
        for r in (0, h - 1):
            if not closed[r, c]:
                push(r, c)
    nan = np.isnan(dem)
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                if nan[rr, cc]:
                    closed[rr, cc] = True
                    continue
                if filled[rr, cc] <= z:
                    filled[rr, cc] = z + eps
                closed[rr, cc] = True
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def trace_downstream(filled: np.ndarray, start: tuple[int, int], max_steps: int = 200_000) -> list[tuple[int, int]]:
    """D8 steepest descent from (row, col) until the border / a no-data cell / a loop."""
    h, w = filled.shape
    r, c = start
    path, seen = [(r, c)], {(r, c)}
    for _ in range(max_steps):
        best, bd = None, 0.0
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not np.isnan(filled[rr, cc]):
                d = (filled[r, c] - filled[rr, cc]) / (1.41421356 if dr and dc else 1.0)
                if d > bd:
                    best, bd = (rr, cc), d
        if best is None or best in seen:
            break
        r, c = best
        path.append(best)
        seen.add(best)
        if r in (0, h - 1) or c in (0, w - 1):
            break
    return path


def snap_to_channel(acc_dem: np.ndarray, start: tuple[int, int], radius: int = 5) -> tuple[int, int]:
    """A user-supplied start rarely lands on the thalweg: move to the lowest cell within `radius` cells."""
    r, c = start
    r0, r1, c0, c1 = max(0, r - radius), min(acc_dem.shape[0], r + radius + 1), max(0, c - radius), min(acc_dem.shape[1], c + radius + 1)
    win = acc_dem[r0:r1, c0:c1]
    i = np.nanargmin(win)
    return r0 + i // win.shape[1], c0 + i % win.shape[1]


def flow_path(dem: np.ndarray, grid: Grid, start_xy: tuple[float, float], coarse_m: float = 30.0):
    """start_xy in grid CRS. Returns LineString in grid CRS."""
    k = max(1, int(round(coarse_m / grid.res)))
    coarse = block_mean(dem, k)
    cres = grid.res * k
    x0, y0 = grid.transform.c, grid.transform.f
    col, row = int((start_xy[0] - x0) / cres), int((y0 - start_xy[1]) / cres)
    if not (0 <= row < coarse.shape[0] and 0 <= col < coarse.shape[1]):
        raise ValueError("start point is outside the DEM grid")
    filled = priority_flood(coarse)
    r, c = snap_to_channel(coarse, (row, col))
    cells = trace_downstream(filled, (r, c))
    pts = [(x0 + (cc + 0.5) * cres, y0 - (rr + 0.5) * cres) for rr, cc in cells]
    return LineString(pts) if len(pts) > 1 else LineString([pts[0], pts[0]])


def settlements_along(path: LineString, places: gpd.GeoDataFrame, corridor_m: float = 300.0) -> gpd.GeoDataFrame:
    """Settlements within `corridor_m` of the path, ordered by distance downstream from the start."""
    if places.empty or path.is_empty:
        return places.iloc[0:0]
    sel = places[places["place"].isin({"village", "hamlet", "town", "city", "suburb", "neighbourhood"})] \
        if "place" in places.columns else places
    sel = sel[sel.geometry.distance(path) <= corridor_m].copy()
    sel["km_downstream"] = [round(path.project(g) / 1000, 2) for g in sel.geometry]
    sel["offset_m"] = [round(g.distance(path), 0) for g in sel.geometry]
    return sel.sort_values("km_downstream")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start-lonlat", nargs=2, type=float, default=None, metavar=("LON", "LAT"),
                    help="default: flow_source from the AOI config")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--osm", default="data/raw/osm/osm_pre_event.gpkg")
    ap.add_argument("--corridor-m", type=float, default=300.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    start = a.start_lonlat or cfg.flow_source
    if not start:
        ap.error("no start point: pass --start-lonlat or set flow_source in the AOI config")
    a.start_lonlat = list(start)
    proc, out = Path(a.proc), Path(a.outputs)
    grid = grid_from_raster(proc / "dem.tif")
    dem = read_raster(proc / "dem.tif")[0][0]
    sx, sy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(*a.start_lonlat)
    path = flow_path(dem, grid, (sx, sy))
    places = load_layer(Path(a.osm), "places", grid.crs)
    along = settlements_along(path, places, a.corridor_m)
    gp = out / "flowpath.gpkg"
    if gp.exists():
        gp.unlink()
    gpd.GeoDataFrame({"length_km": [round(path.length / 1000, 2)]}, geometry=[path], crs=grid.crs).to_file(
        gp, layer="path", driver="GPKG")
    if len(along):
        along.to_file(gp, layer="settlements_along_path", driver="GPKG")
    (out / "flowpath_summary.json").write_text(json.dumps({
        "start_lonlat": a.start_lonlat, "length_km": round(path.length / 1000, 2), "corridor_m": a.corridor_m,
        "settlements_along_path": [{"name": r.get("name"), "km_downstream": r["km_downstream"]}
                                   for _, r in along.iterrows()]}, indent=2, default=str), encoding="utf-8")
    log.info("flow path %.1f km, %d settlements within %.0f m", path.length / 1000, len(along), a.corridor_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
