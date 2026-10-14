"""Sentinel-2 L2A preprocessing: bands -> common 10 m grid, reflectance scaling, SCL-based cloud/shadow mask.

Cloud masking uses the L2A Scene Classification Layer (SCL) rather than s2cloudless: s2cloudless needs L1C
bands, but the pipeline downloads L2A, and SCL is already shipped with it.

    python -m src.s2_preprocess --config config/aoi_trishuli.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from rasterio.warp import Resampling
from scipy import ndimage as ndi

from .config import load_config
from .grid import Grid, fill_nan, make_grid, warp_raster, write_raster

log = logging.getLogger(__name__)

BANDS = {"B02": "R10m", "B03": "R10m", "B04": "R10m", "B08": "R10m", "B11": "R20m"}
SCL_BAD = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud medium/high prob, thin cirrus
OUT_BANDS = ["B02", "B03", "B04", "B08", "B11", "valid"]


def baseline_offset(name: str) -> float:
    """L2A processing baseline >= 04.00 (N0400+) adds BOA_ADD_OFFSET = -1000 to the DN."""
    m = re.search(r"_N(\d{4})_", name)
    return -1000.0 if (m and int(m.group(1)) >= 400) else 0.0


def to_reflectance(dn: np.ndarray, offset: float) -> np.ndarray:
    """DN -> surface reflectance in [0, ~1.5]. DN == 0 is no data."""
    out = (dn.astype(np.float32) + offset) / 10000.0
    out[~np.isfinite(dn) | (dn == 0)] = np.nan
    return np.clip(out, 0.0, 1.5)


def valid_from_scl(scl: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    """True for usable pixels. Bad classes are dilated to also drop cloud edges."""
    bad = np.isin(np.nan_to_num(scl, nan=0).astype(np.int16), SCL_BAD)
    if dilate_px > 0:
        bad = ndi.binary_dilation(bad, structure=np.ones((3, 3), bool), iterations=dilate_px)
    return ~bad


def _find(zf: zipfile.ZipFile, band: str, res: str) -> str:
    pat = re.compile(rf".*/{res}/.*_{band}_{res[1:]}\.jp2$")
    hits = [n for n in zf.namelist() if pat.match(n)]
    if not hits:
        raise FileNotFoundError(f"{band} ({res}) not found in {zf.filename}")
    return hits[0]


def read_tile(zip_path: Path, grid: Grid) -> np.ndarray:
    """One L2A zip -> (6, H, W): B02 B03 B04 B08 B11 reflectance + valid (1/0). NaN outside the tile."""
    off = baseline_offset(zip_path.name)
    out = np.full((6, grid.height, grid.width), np.nan, dtype=np.float32)
    with zipfile.ZipFile(zip_path) as zf:
        for i, (b, res) in enumerate(BANDS.items()):
            src = f"/vsizip/{zip_path}/{_find(zf, b, res)}"
            dn = warp_raster(src, grid, [1], Resampling.bilinear, src_nodata=0)[0]
            out[i] = to_reflectance(dn, off)
        scl = warp_raster(f"/vsizip/{zip_path}/{_find(zf, 'SCL', 'R20m')}", grid, [1], Resampling.nearest)[0]
    covered = np.isfinite(scl)
    out[5] = np.where(covered, valid_from_scl(scl).astype(np.float32), np.nan)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--res", type=float, default=10.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    raw, out = Path(args.raw), Path(args.out)
    man = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    grid = make_grid(cfg.bbox, args.res)

    stats = {}
    for role in ("pre", "post"):
        items = [s for s in man.get("s2", []) if s["role"] == role]
        if not items:
            log.warning("no S2 %s scenes in manifest", role)
            continue
        stack = None
        for it in items:
            z = raw / "s2" / f"{it['name']}.zip"
            if not z.exists():
                log.warning("missing %s", z.name)
                continue
            tile = read_tile(z, grid)
            if stack is None:
                stack = tile
            else:
                for k in range(6):
                    fill_nan(stack[k], tile[k])
        if stack is None:
            continue
        valid = np.nan_to_num(stack[5], nan=0.0)
        stats[role] = {"clear_fraction": float(valid.mean()), "tiles": [i["tile"] for i in items],
                       "cloudy_flag": any(i["cloudy"] for i in items)}
        stack[5] = valid
        write_raster(out / f"s2_{role}.tif", stack, grid, nodata=np.nan, descriptions=OUT_BANDS)
        log.info("S2 %s: %.0f%% clear over the AOI grid", role, 100 * stats[role]["clear_fraction"])
    (out / "s2_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
