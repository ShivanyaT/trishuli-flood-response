"""DEM-derived layers: slope and a HAND-like 'relative elevation' used to confine the flood corridor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy import ndimage as ndi

from .grid import Grid, warp_raster


def merge_dem_wgs84(tile_paths: list[Path], out_path: Path, bbox, pad_deg: float = 0.05) -> Path:
    """Mosaic Copernicus GLO-30 tiles into one GeoTIFF (EPSG:4326). This is the file SNAP uses as external DEM."""
    w, s, e, n = bbox
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        arr, tr = merge(srcs, bounds=(w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg))
        prof = srcs[0].profile | {"height": arr.shape[1], "width": arr.shape[2], "transform": tr,
                                  "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    finally:
        for d in srcs:
            d.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr)
    return out_path


def dem_on_grid(dem_wgs84: Path, grid: Grid) -> np.ndarray:
    return warp_raster(dem_wgs84, grid, [1])[0]


def slope_deg(dem: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(np.float32), res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def relative_elevation(dem: np.ndarray, res: float, window_m: float = 1000.0) -> np.ndarray:
    """Height above the lowest point within ~window_m (a cheap HAND proxy).

    Valley floors and low terraces get small values; ridges and hillsides get large ones. Avoids needing a
    river network (hard to derive reliably in a narrow Himalayan gorge). NaNs stay NaN.
    """
    size = max(3, int(round(window_m / res)) | 1)
    d = dem.astype(np.float32)
    nan = np.isnan(d)
    filled = np.where(nan, np.float32(np.nanmax(d)) if (~nan).any() else 0.0, d)
    low = ndi.minimum_filter(filled, size=size, mode="nearest")
    rel = d - low
    rel[nan] = np.nan
    return rel


def corridor_mask(dem: np.ndarray, res: float, max_rel_m: float = 40.0, max_slope_deg: float = 25.0,
                  window_m: float = 1000.0) -> np.ndarray:
    """True where flood/debris is physically plausible: low above local valley floor and not too steep."""
    rel = relative_elevation(dem, res, window_m)
    slope = slope_deg(dem, res)
    return (rel <= max_rel_m) & (slope <= max_slope_deg)
