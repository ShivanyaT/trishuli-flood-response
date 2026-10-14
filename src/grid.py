"""Common processing grid and raster I/O helpers.

Every raster in the pipeline (S1, S2, DEM, masks) is warped onto ONE grid derived from the AOI,
so any AOI works: the UTM zone is chosen from the bbox centre instead of being hard-coded to 32645.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)

    @property
    def pixel_area_km2(self) -> float:
        return (self.res / 1000.0) ** 2


def make_grid(bbox: tuple[float, float, float, float], res: float = 10.0, crs: str | None = None) -> Grid:
    w, s, e, n = bbox
    crs = crs or f"EPSG:{utm_epsg((w + e) / 2, (s + n) / 2)}"
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, w, s, e, n, densify_pts=21)
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
    return Grid(crs, from_origin(left, top, res, res), width, height, res)


def warp_raster(src: str | Path, grid: Grid, bands: list[int] | None = None,
                resampling: Resampling = Resampling.bilinear, src_nodata: float | None = None) -> np.ndarray:
    """Warp bands of a raster onto `grid`. Returns float32 (bands, H, W) with NaN where there is no data."""
    with rasterio.open(src) as ds:
        bands = bands or list(range(1, ds.count + 1))
        nodata = src_nodata if src_nodata is not None else ds.nodata
        out = np.full((len(bands), grid.height, grid.width), np.nan, dtype=np.float32)
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(ds, b), destination=out[i],
                src_nodata=nodata, dst_nodata=np.nan,
                dst_transform=grid.transform, dst_crs=grid.crs, resampling=resampling,
                init_dest_nodata=True,
            )
    return out


def fill_nan(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Fill NaN pixels of `base` from `other` (used to merge overlapping frames/tiles)."""
    m = np.isnan(base)
    base[m] = other[m]
    return base


def write_raster(path: str | Path, arr: np.ndarray, grid: Grid, nodata: float | None = None,
                 descriptions: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.height, width=grid.width, count=arr.shape[0],
        dtype=arr.dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    return path


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read()
        meta = {"crs": ds.crs, "transform": ds.transform, "descriptions": ds.descriptions, "nodata": ds.nodata}
    return arr, meta


def grid_from_raster(path: str | Path) -> Grid:
    with rasterio.open(path) as ds:
        return Grid(str(ds.crs), ds.transform, ds.width, ds.height, abs(ds.transform.a))
