"""Sentinel-1 GRD preprocessing: SNAP `gpt` graph -> common grid -> dB -> co-registration check.

Why SNAP instead of the plan's pure-Python route: a defensible terrain-corrected product in the Himalaya
needs precise orbits, thermal/border-noise removal, calibration and range-Doppler terrain correction
with a DEM. Re-implementing that by hand is the riskiest part of the week; `gpt` does it reliably.

The graph mirrors the Kuro Siwo GRD pipeline (sigma0, Lee Sigma 3x3, 10 m terrain correction) so that the
model trained on Kuro Siwo sees inputs processed the same way. Differences: Copernicus DEM instead of SRTM,
the AOI's UTM zone instead of EPSG:3857, and no land/sea mask (the AOI is inland).

    python -m src.s1_preprocess --config config/aoi_trishuli.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi

from .config import load_config
from .dem import tile_names
from .grid import Grid, fill_nan, make_grid, warp_raster, write_raster
from .terrain import dem_on_grid, merge_dem_wgs84, slope_deg

log = logging.getLogger(__name__)

GRAPH = """<graph id="S1_GRD_sigma0_UTM">
  <version>1.0</version>
  <node id="Read"><operator>Read</operator><sources/>
    <parameters><file>${input}</file></parameters></node>
  <node id="Apply-Orbit-File"><operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters><orbitType>Sentinel Precise (Auto Download)</orbitType><polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail></parameters></node>
  <node id="ThermalNoiseRemoval"><operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters><removeThermalNoise>true</removeThermalNoise></parameters></node>
  <node id="Remove-GRD-Border-Noise"><operator>Remove-GRD-Border-Noise</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters><borderLimit>500</borderLimit><trimThreshold>50.0</trimThreshold></parameters></node>
  <node id="Calibration"><operator>Calibration</operator>
    <sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources>
    <parameters><outputSigmaBand>true</outputSigmaBand><outputImageScaleInDb>false</outputImageScaleInDb>
      <selectedPolarisations>VV,VH</selectedPolarisations></parameters></node>
  <node id="Subset"><operator>Subset</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters><geoRegion>${region}</geoRegion><subSamplingX>1</subSamplingX><subSamplingY>1</subSamplingY>
      <copyMetadata>true</copyMetadata></parameters></node>
  <node id="Speckle-Filter"><operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Subset"/></sources>
    <parameters><filter>Lee Sigma</filter><filterSizeX>3</filterSizeX><filterSizeY>3</filterSizeY>
      <dampingFactor>2</dampingFactor><estimateENL>true</estimateENL><enl>1.0</enl>
      <numLooksStr>1</numLooksStr><windowSize>7x7</windowSize><targetWindowSize>3x3</targetWindowSize>
      <sigmaStr>0.9</sigmaStr><anSize>50</anSize></parameters></node>
  <node id="Terrain-Correction"><operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters><demName>External DEM</demName><externalDEMFile>${dem}</externalDEMFile>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue><externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>${res}</pixelSpacingInMeter><mapProjection>${crs}</mapProjection>
      <nodataValueAtSea>false</nodataValueAtSea><saveSelectedSourceBand>true</saveSelectedSourceBand>
      <saveLayoverShadowMask>true</saveLayoverShadowMask></parameters></node>
  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters><file>${output}</file><formatName>GeoTIFF</formatName></parameters></node>
</graph>
"""


def run_gpt(safe_zip: Path, out_tif: Path, dem_wgs84: Path, grid: Grid, bbox, gpt: str = "gpt",
            pad_deg: float = 0.02) -> Path:
    """Run the SNAP graph on one GRD product, clipped to the (padded) AOI. Skips if the output exists."""
    if out_tif.exists():
        return out_tif
    if shutil.which(gpt) is None:
        raise RuntimeError(
            f"SNAP '{gpt}' not found. Install ESA SNAP (the Sentinel-1 toolbox is enough) and put its bin/ on PATH, "
            "or pass --gpt /path/to/gpt. Everything downstream of this step works from the GeoTIFFs it writes.")
    w, s, e, n = bbox
    region = box(w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg).wkt
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        graph = Path(td) / "graph.xml"
        graph.write_text(GRAPH, encoding="utf-8")
        tmp_out = Path(td) / out_tif.name
        cmd = [gpt, str(graph), f"-Pinput={safe_zip}", f"-Poutput={tmp_out}", f"-Pdem={dem_wgs84}",
               f"-Pregion={region}", f"-Pres={grid.res}", f"-Pcrs={grid.crs}", "-q", "2"]
        log.info("gpt: %s", safe_zip.name)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp_out.exists():
            raise RuntimeError(f"gpt failed for {safe_zip.name}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        shutil.move(str(tmp_out), out_tif)
    return out_tif


def to_db(linear: np.ndarray) -> np.ndarray:
    """Linear sigma0 -> dB. Non-positive values become NaN (no data)."""
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(linear) & (linear > 0)
    out[ok] = 10.0 * np.log10(np.maximum(linear[ok], 1e-6))
    return out


def load_gpt_tif(tif: Path, grid: Grid) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a SNAP output GeoTIFF onto the grid. Returns (VV, VH) in dB and the layover/shadow band if present.

    Band order written by SNAP is the selected source bands (Sigma0_VV, Sigma0_VH) then the mask.
    """
    with rasterio.open(tif) as ds:
        n = ds.count
    arr = warp_raster(tif, grid, list(range(1, n + 1)), src_nodata=0.0)
    ls = arr[2] if n >= 3 else None
    return np.stack([to_db(arr[0]), to_db(arr[1])]), ls


def estimate_shift(pre_vv: np.ndarray, post_vv: np.ndarray, max_px: int = 2048) -> tuple[float, float]:
    """Sub-pixel (row, col) shift of post relative to pre by phase correlation on a central, valid window."""
    valid = np.isfinite(pre_vv) & np.isfinite(post_vv)
    if valid.mean() < 0.2:
        return 0.0, 0.0
    h, w = pre_vv.shape
    r0, c0 = max(0, h // 2 - max_px // 2), max(0, w // 2 - max_px // 2)
    sl = (slice(r0, r0 + max_px), slice(c0, c0 + max_px))

    def prep(a):
        a = a[sl].copy()
        a[~np.isfinite(a)] = np.nanmean(a)
        return ndi.gaussian_filter(a - a.mean(), 1.0)

    shift, _, _ = phase_cross_correlation(prep(pre_vv), prep(post_vv), upsample_factor=10, normalization=None)
    return float(shift[0]), float(shift[1])


def coregister(pre_vv: np.ndarray, post: np.ndarray, min_px: float = 0.5, max_px: float = 3.0):
    """Align both post bands to pre. Same-track + same DEM terrain correction should leave ~0 shift, so this is
    mostly a check; it only corrects shifts between `min_px` and `max_px` and warns if outside that range."""
    dr, dc = estimate_shift(pre_vv, post[0])
    mag = float(np.hypot(dr, dc))
    if mag < min_px:
        return post, (dr, dc)
    if mag > max_px:
        log.warning("pre/post shift %.1f px exceeds %.1f px; NOT applying (check orbit files / DEM)", mag, max_px)
        return post, (dr, dc)
    log.info("applying pre/post co-registration shift (%.2f, %.2f) px", dr, dc)
    return np.stack([ndi.shift(b, (dr, dc), order=1, mode="constant", cval=np.nan) for b in post]), (dr, dc)


def mosaic_role(zips: list[Path], dem_wgs84: Path, grid: Grid, bbox, tmp_dir: Path, gpt: str):
    vv_vh, ls = None, None
    for z in zips:
        tif = run_gpt(z, tmp_dir / f"{z.stem}_sigma0.tif", dem_wgs84, grid, bbox, gpt)
        a, m = load_gpt_tif(tif, grid)
        if vv_vh is None:
            vv_vh, ls = a, m
        else:
            for i in range(2):
                fill_nan(vv_vh[i], a[i])
            if ls is not None and m is not None:
                fill_nan(ls, m)
    return vv_vh, ls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--gpt", default="gpt")
    ap.add_argument("--res", type=float, default=10.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    raw, out = Path(args.raw), Path(args.out)
    man = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    if "products" not in man.get("s1", {}):
        log.error("manifest has no S1 pair; run `make plan` / `make ingest` first")
        return 2

    grid = make_grid(cfg.bbox, args.res)
    log.info("grid %s  %d x %d px  (%.1f M px)", grid.crs, grid.width, grid.height, grid.width * grid.height / 1e6)

    dem_files = [raw / "dem" / f"{n}.tif" for n in tile_names(cfg.bbox)]
    dem_files = [p for p in dem_files if p.exists()]
    if not dem_files:
        log.error("no DEM tiles in %s; run ingest", raw / "dem")
        return 2
    dem_wgs = merge_dem_wgs84(dem_files, out / "dem_wgs84.tif", cfg.bbox)
    dem = dem_on_grid(dem_wgs, grid)
    write_raster(out / "dem.tif", dem, grid, nodata=np.nan, descriptions=["elevation_m"])
    write_raster(out / "slope.tif", slope_deg(dem, grid.res), grid, nodata=np.nan, descriptions=["slope_deg"])

    results = {}
    for role in ("pre", "post"):
        zips = [raw / "s1" / f"{p['name']}.zip" for p in man["s1"]["products"] if p["role"] == role]
        missing = [z.name for z in zips if not z.exists()]
        if missing:
            log.error("missing downloads for %s: %s", role, missing)
            return 2
        results[role] = mosaic_role(zips, dem_wgs, grid, cfg.bbox, out / "gpt", args.gpt)

    (pre, ls_pre), (post, ls_post) = results["pre"], results["post"]
    post, shift = coregister(pre[0], post)

    # Layover/shadow: SNAP's band coding is not documented here, so treat any non-zero as unobservable but
    # refuse an implausible result (the whole scene flagged means the coding is different from assumed).
    ls = None
    for m in (ls_pre, ls_post):
        if m is not None:
            f = np.nan_to_num(m, nan=0.0) != 0
            ls = f if ls is None else (ls | f)
    if ls is not None and ls.mean() > 0.5:
        log.warning("layover/shadow band flags %.0f%% of pixels; ignoring it (check band coding)", 100 * ls.mean())
        ls = None

    names = ["VV_db", "VH_db"]
    write_raster(out / "s1_pre.tif", pre, grid, nodata=np.nan, descriptions=names)
    write_raster(out / "s1_post.tif", post, grid, nodata=np.nan, descriptions=names)
    if ls is not None:
        write_raster(out / "s1_unobservable.tif", ls.astype(np.uint8), grid, nodata=None,
                     descriptions=["layover_or_shadow"])
    meta = {"shift_px": shift, "layover_shadow_fraction": None if ls is None else float(ls.mean()),
            "valid_fraction_pre": float(np.isfinite(pre[0]).mean()),
            "valid_fraction_post": float(np.isfinite(post[0]).mean())}
    (out / "s1_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("S1 done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
