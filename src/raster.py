"""Common analysis grid + small raster I/O helpers.

Every product (S1 pre/post, S2 pre/post, DEM, masks) is resampled onto ONE grid derived from the AOI
(UTM zone chosen automatically, 10 m by default). Downstream code can then do plain array maths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from .config import AOIConfig


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def res(self) -> float:
        return float(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)


def make_grid(cfg: AOIConfig) -> Grid:
    crs = f"EPSG:{cfg.utm_epsg}"
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *cfg.bbox, densify_pts=21)
    res = cfg.grid_res
    left, top = math.floor(left / res) * res, math.ceil(top / res) * res
    width, height = math.ceil((right - left) / res), math.ceil((top - bottom) / res)
    return Grid(crs, Affine(res, 0, left, 0, -res, top), width, height)


def grid_of(path: str | Path) -> Grid:
    with rasterio.open(path) as s:
        return Grid(s.crs.to_string(), s.transform, s.width, s.height)


def write_raster(path: str | Path, arr: np.ndarray, grid: Grid, nodata=None,
                 descriptions: list[str] | None = None) -> Path:
    """arr: (h, w) or (bands, h, w)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=a.shape[0],
                       dtype=a.dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(a)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    return path


def read_raster(path: str | Path) -> tuple[np.ndarray, Grid]:
    """Returns (bands, h, w) array (float32 stays float32) and its Grid."""
    with rasterio.open(path) as s:
        return s.read(), Grid(s.crs.to_string(), s.transform, s.width, s.height)


def warp_to_grid(src: np.ndarray, grid: Grid, *, src_crs, src_transform=None, gcps=None,
                 resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan, **warp_opts) -> np.ndarray:
    """Resample a single 2-D array onto `grid`. Use `gcps` (list of GroundControlPoint) for SAR GRD data."""
    dst = np.full(grid.shape, dst_nodata, dtype=src.dtype)
    kw = dict(source=src, destination=dst, src_crs=src_crs, dst_transform=grid.transform, dst_crs=grid.crs,
              resampling=resampling, src_nodata=src_nodata, dst_nodata=dst_nodata)
    if gcps:
        kw["gcps"] = gcps
    else:
        kw["src_transform"] = src_transform
    reproject(**kw, **warp_opts)
    return dst


def to_db(lin: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(lin)
    db = np.where(np.isfinite(db), np.maximum(db, floor_db), np.nan)
    return db.astype(np.float32)
