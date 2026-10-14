"""Classical flood + debris baseline from same-track Sentinel-1 change detection, refined by Sentinel-2.

Design notes (what differs from the bare thresholds in the plan, and why):
  * Same-track pre/post means radar shadow and layover are identical in both images, so a *drop* in VV
    (post - pre) cancels them. The absolute VV test is only a second condition; a pixel that was already dark
    before the event (permanent water, shadow) cannot satisfy both.
  * A flood-corridor filter (height above local valley floor + slope, from the Copernicus DEM) removes the
    classic Himalayan false positives: wet snow and wet slopes on hillsides.
  * The event was an ice-rock debris flow, not just water, so a separate debris mask is produced
    (S1 backscatter increase / cross-pol change, and S2 vegetation loss).
  * Every threshold lives in FloodParams; none are tuned on Trishuli (there is no ground truth we may use as
    input). `src.evaluate --tune` tunes the water thresholds on Kuro Siwo training patches.

    python -m src.flood_map --config config/aoi_trishuli.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import load_config
from .grid import grid_from_raster, read_raster, write_raster
from .terrain import corridor_mask

log = logging.getLogger(__name__)


@dataclass
class FloodParams:
    vv_water_db: float = -16.0      # post VV below this ...
    vv_drop_db: float = -3.0        # ... and dropped by at least this much vs pre
    pre_water_margin_db: float = 1.5  # pre VV must exceed vv_water_db + margin (was not already water/shadow)
    mndwi_water: float = 0.3        # S2 post MNDWI above this ...
    mndwi_pre_max: float = 0.0      # ... and pre MNDWI below this (new water only)
    debris_up_db: float = 3.0       # S1: VV rose by at least this much
    debris_abs_db: float = 2.0      # S1: |dVV| at least this much AND cross-pol ratio change below
    debris_ratio_db: float = 2.0
    debris_sigma_k: float = 3.0     # change must also exceed k robust-sigma of the AOI-wide dVV
    ndvi_pre_min: float = 0.4       # S2 debris: vegetated before ...
    ndvi_post_max: float = 0.2      # ... bare after ...
    ndvi_drop: float = -0.25        # ... with a clear drop
    max_rel_elev_m: float = 40.0    # corridor: height above local valley floor
    max_slope_deg: float = 25.0     # corridor: slope
    min_component_px: int = 10      # drop connected components smaller than this (10 px = 1000 m^2 at 10 m)


# ----------------------------------------------------------------------------- S1
def s1_water(vv_post, vv_pre, p: FloodParams) -> np.ndarray:
    """New open water: dark now AND clearly darker than before. NaN compares False."""
    return ((vv_post < p.vv_water_db) & ((vv_post - vv_pre) < p.vv_drop_db)
            & (vv_pre > p.vv_water_db + p.pre_water_margin_db))


def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def s1_debris(vv_post, vh_post, vv_pre, vh_pre, p: FloodParams) -> np.ndarray:
    """Backscatter change that is not 'new water': brighter (rough, wet debris) or a cross-pol shift."""
    dvv = vv_post - vv_pre
    dratio = (vh_post - vv_post) - (vh_pre - vv_pre)
    sigma = robust_sigma(dvv)
    up = max(p.debris_up_db, p.debris_sigma_k * sigma)
    ab = max(p.debris_abs_db, p.debris_sigma_k * sigma)
    return (dvv >= up) | ((np.abs(dvv) >= ab) & (dratio >= p.debris_ratio_db))


# ----------------------------------------------------------------------------- S2
def mndwi(b3, b11) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (b3 - b11) / (b3 + b11)


def ndvi(b4, b8) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (b8 - b4) / (b8 + b4)


def s2_water(s2_pre, s2_post, p: FloodParams) -> np.ndarray:
    """New water from optical data. Inputs are (6,H,W) stacks [B02 B03 B04 B08 B11 valid]; clear in BOTH dates."""
    ok = (s2_pre[5] > 0) & (s2_post[5] > 0)
    return ok & (mndwi(s2_post[1], s2_post[4]) > p.mndwi_water) & (mndwi(s2_pre[1], s2_pre[4]) < p.mndwi_pre_max)


def s2_debris(s2_pre, s2_post, p: FloodParams) -> np.ndarray:
    """Vegetation stripped to bare/sediment surface. Needs corridor + 'not water' gating by the caller."""
    ok = (s2_pre[5] > 0) & (s2_post[5] > 0)
    n_pre, n_post = ndvi(s2_pre[2], s2_pre[3]), ndvi(s2_post[2], s2_post[3])
    return ok & (n_pre > p.ndvi_pre_min) & (n_post < p.ndvi_post_max) & ((n_post - n_pre) < p.ndvi_drop)


# ----------------------------------------------------------------------------- common
def cleanup(mask: np.ndarray, min_px: int) -> np.ndarray:
    m = ndi.binary_opening(mask, structure=np.ones((3, 3), bool))
    lab, n = ndi.label(m)
    if n == 0:
        return m
    keep = np.bincount(lab.ravel()) >= min_px
    keep[0] = False
    return keep[lab]


def run_baseline(s1_pre, s1_post, dem, res, p: FloodParams, s2_pre=None, s2_post=None, s1_unobservable=None):
    """s1_* are (2,H,W) [VV_db, VH_db]. Returns dict of uint8 masks plus an honest 'observability' summary."""
    corridor = corridor_mask(dem, res, p.max_rel_elev_m, p.max_slope_deg)
    unobs = np.zeros(corridor.shape, bool) if s1_unobservable is None else s1_unobservable.astype(bool)

    w1 = s1_water(s1_post[0], s1_pre[0], p) & ~unobs
    have_s2 = s2_pre is not None and s2_post is not None
    w2 = s2_water(s2_pre, s2_post, p) if have_s2 else np.zeros_like(w1)
    flood = cleanup((w1 | w2) & corridor, p.min_component_px)

    d1 = s1_debris(s1_post[0], s1_post[1], s1_pre[0], s1_pre[1], p) & ~unobs
    d2 = s2_debris(s2_pre, s2_post, p) if have_s2 else np.zeros_like(d1)
    keep = corridor & ~flood
    debris1 = cleanup(d1 & keep, p.min_component_px)
    debris2 = cleanup(d2 & keep, p.min_component_px)
    debris = debris1.astype(np.uint8) | (debris2.astype(np.uint8) << 1)  # bit0 = S1, bit1 = S2

    s1_valid = np.isfinite(s1_pre[0]) & np.isfinite(s1_post[0])
    summary = {
        "s1_valid_fraction": float(s1_valid.mean()),
        "s1_unobservable_fraction": float(unobs.mean()),
        "s2_clear_fraction_both_dates": float(((s2_pre[5] > 0) & (s2_post[5] > 0)).mean()) if have_s2 else 0.0,
        "corridor_fraction": float(corridor.mean()),
        "flood_from_s1_only_px": int((flood & w1 & ~w2).sum()),
        "flood_from_s2_only_px": int((flood & w2 & ~w1).sum()),
    }
    return {"flood": flood.astype(np.uint8), "debris": debris.astype(np.uint8), "summary": summary}


# config keys that differ from the FloodParams field names
CONFIG_ALIASES = {"dvv_water_db": "vv_drop_db"}


def load_thresholds(path: Path | None, overrides: dict | None = None) -> FloodParams:
    """Precedence: defaults < tuned thresholds file (Kuro Siwo train) < explicit `flood:` block in the AOI config."""
    p = FloodParams()
    if path and Path(path).exists():
        for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items():
            if hasattr(p, k):
                setattr(p, k, v)
        log.info("loaded thresholds from %s", path)
    for k, v in (overrides or {}).items():
        k = CONFIG_ALIASES.get(k, k)
        if not hasattr(p, k):
            raise ValueError(f"unknown key in config `flood:` block: {k!r}")
        setattr(p, k, type(getattr(p, k))(v))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data/outputs")
    ap.add_argument("--thresholds", default="models/baseline_thresholds.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    proc, out = Path(args.proc), Path(args.out)

    pre, _ = read_raster(proc / "s1_pre.tif")
    post, _ = read_raster(proc / "s1_post.tif")
    dem, _ = read_raster(proc / "dem.tif")
    grid = grid_from_raster(proc / "s1_post.tif")
    s2p, s2q = (proc / "s2_pre.tif"), (proc / "s2_post.tif")
    s2_pre = read_raster(s2p)[0] if s2p.exists() else None
    s2_post = read_raster(s2q)[0] if s2q.exists() else None
    unobs = read_raster(proc / "s1_unobservable.tif")[0][0] if (proc / "s1_unobservable.tif").exists() else None

    p = load_thresholds(Path(args.thresholds), cfg.flood)
    res = run_baseline(pre, post, dem[0], grid.res, p, s2_pre, s2_post, unobs)
    write_raster(out / "flood_mask.tif", res["flood"], grid, descriptions=["flood"])
    write_raster(out / "debris_mask.tif", res["debris"], grid, descriptions=["debris_bit0_S1_bit1_S2"])
    km2 = grid.pixel_area_km2
    report = {
        "params": asdict(p), **res["summary"],
        "flood_area_km2": float(res["flood"].sum() * km2),
        "debris_area_km2_s1": float(((res["debris"] & 1) > 0).sum() * km2),
        "debris_area_km2_s2": float(((res["debris"] & 2) > 0).sum() * km2),
    }
    (out / "flood_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("flood %.2f km2 | debris S1 %.2f km2, S2 %.2f km2 | S1 valid %.0f%%",
             report["flood_area_km2"], report["debris_area_km2_s1"], report["debris_area_km2_s2"],
             100 * report["s1_valid_fraction"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
