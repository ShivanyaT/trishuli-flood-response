"""Run the trained model over the processed Trishuli scene (replaces the Day 2 threshold when it is better).

    python -m src.infer_flood --config config/aoi_trishuli.yaml --weights models/flood_unet.pt

Writes data/outputs/flood_prob.tif (uint8, prob*255), flood_mask_ml.tif and ml_report.json.
The same corridor / layover-shadow / small-component filters as the baseline are applied, so the two masks
are comparable and differ only in the water classifier.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .features import build_features
from .flood_map import cleanup, load_thresholds, s2_water
from .grid import grid_from_raster, read_raster, write_raster
from .model import load_checkpoint
from .terrain import corridor_mask

log = logging.getLogger(__name__)


def _starts(n: int, tile: int, stride: int) -> list[int]:
    s = list(range(0, max(1, n - tile + 1), stride))
    if s[-1] != max(0, n - tile):
        s.append(max(0, n - tile))
    return s


@torch.no_grad()
def predict_raster(model, feats: np.ndarray, valid: np.ndarray, device="cpu", tile=256, overlap=64, batch=8):
    """Sliding-window probabilities with Hann blending. feats (C,H,W); valid (H,W) bool skips empty tiles."""
    C, H, W = feats.shape
    Hp, Wp = max(H, tile), max(W, tile)
    f = np.zeros((C, Hp, Wp), np.float32); f[:, :H, :W] = feats
    v = np.zeros((Hp, Wp), bool); v[:H, :W] = valid
    h = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win = np.outer(h, h)
    acc, wsum = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    coords = [(r, c) for r in _starts(Hp, tile, tile - overlap) for c in _starts(Wp, tile, tile - overlap)
              if v[r:r + tile, c:c + tile].any()]
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([f[:, r:r + tile, c:c + tile] for r, c in cb])).to(device)
        pb = torch.sigmoid(model(xb)[:, 0]).cpu().numpy()
        for (r, c), p in zip(cb, pb):
            acc[r:r + tile, c:c + tile] += p * win
            wsum[r:r + tile, c:c + tile] += win
    prob = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
    return prob[:H, :W]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="models/flood_unet.pt")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data/outputs")
    ap.add_argument("--thresholds", default="models/baseline_thresholds.json")
    ap.add_argument("--threshold", type=float, default=None, help="override the validation-selected threshold")
    ap.add_argument("--no-s2", action="store_true", help="do not OR in Sentinel-2 water")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    proc, out = Path(a.proc), Path(a.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(a.weights, device)
    thr = a.threshold if a.threshold is not None else ck["threshold"]

    pre, _ = read_raster(proc / "s1_pre.tif")
    post, _ = read_raster(proc / "s1_post.tif")
    dem, _ = read_raster(proc / "dem.tif")
    grid = grid_from_raster(proc / "s1_post.tif")
    valid = np.isfinite(pre[0]) & np.isfinite(post[0])
    feats = build_features(post[0], post[1], pre[0], pre[1])

    prob = predict_raster(model, feats, valid, device)
    p = load_thresholds(Path(a.thresholds), cfg.flood)
    corridor = corridor_mask(dem[0], grid.res, p.max_rel_elev_m, p.max_slope_deg)
    unobs_p = proc / "s1_unobservable.tif"
    unobs = read_raster(unobs_p)[0][0].astype(bool) if unobs_p.exists() else np.zeros_like(valid)

    water = (prob >= thr) & valid & ~unobs
    s2p, s2q = proc / "s2_pre.tif", proc / "s2_post.tif"
    if s2p.exists() and s2q.exists() and not a.no_s2:
        water |= s2_water(read_raster(s2p)[0], read_raster(s2q)[0], p)
    mask = cleanup(water & corridor, p.min_component_px).astype(np.uint8)

    write_raster(out / "flood_prob.tif", (prob * 255).astype(np.uint8), grid, descriptions=["flood_prob_x255"])
    write_raster(out / "flood_mask_ml.tif", mask, grid, descriptions=["flood_ml"])
    rep = {"weights": a.weights, "threshold": thr, "flood_area_km2": float(mask.sum() * grid.pixel_area_km2)}
    (out / "ml_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    base = out / "flood_mask.tif"
    if base.exists():
        b = read_raster(base)[0][0].astype(bool)
        m = mask.astype(bool)
        rep["agreement_iou_vs_baseline"] = float((b & m).sum() / max(1, (b | m).sum()))
        (out / "ml_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log.info("ML flood %.2f km2 (thr %.2f)%s", rep["flood_area_km2"], thr,
             f" | IoU vs baseline {rep['agreement_iou_vs_baseline']:.2f}" if "agreement_iou_vs_baseline" in rep else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
