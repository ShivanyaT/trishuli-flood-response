"""Copernicus DEM GLO-30 download from the AWS Open Data bucket (anonymous HTTPS, no auth)."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import requests

log = logging.getLogger(__name__)
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """1x1 degree tile names covering bbox, e.g. Copernicus_DSM_COG_10_N28_00_E085_00_DEM."""
    w, s, e, n = bbox
    names = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            names.append(f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM")
    return names


def download_dem(bbox, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in tile_names(bbox):
        dest = out_dir / f"{name}.tif"
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        url = f"{BASE}/{name}/{name}.tif"
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 404:  # ocean tiles do not exist
            log.warning("no DEM tile %s (ocean?)", name)
            continue
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        log.info("DEM %s", dest.name)
        paths.append(dest)
    return paths
