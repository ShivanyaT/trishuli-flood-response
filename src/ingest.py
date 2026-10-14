"""Day-1 ingest: select and download S1 / S2 / DEM / pre-event OSM for an AOI config.

    python -m src.ingest --config config/aoi_trishuli.yaml --dry-run   # plan only (no credentials needed)
    python -m src.ingest --config config/aoi_trishuli.yaml             # download everything
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .cdse import CDSEClient
from .config import load_config
from .dem import download_dem, tile_names
from .osm import download_osm
from .pairing import diagnose_s1, pick_s2, rank_s1_pairs

log = logging.getLogger("ingest")


def _human(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.2f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--dry-run", action="store_true", help="select scenes and write the manifest; download nothing")
    for k in ("s1", "s2", "dem", "osm"):
        ap.add_argument(f"--skip-{k}", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = load_config(args.config)
    out = Path(args.out)
    client = CDSEClient(os.getenv("CDSE_USER"), os.getenv("CDSE_PASS"))
    manifest: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aoi": cfg.name, "bbox": cfg.bbox, "event_date": cfg.event_date.isoformat(),
        "event_ts_used": cfg.event_ts.isoformat(), "osm_snapshot": cfg.osm_snapshot.isoformat(),
    }
    s1_products, s2_choices = [], []

    # ---------------- Sentinel-1 -------------------------------------------------
    if not args.skip_s1:
        found = client.search("SENTINEL-1", cfg.aoi_wkt, cfg.search_start, cfg.search_end, product_type="IW_GRDH_1S")
        pairs, acqs = rank_s1_pairs(found, cfg)
        if not pairs:
            log.error(diagnose_s1(acqs, cfg))
            manifest["s1"] = {"error": "no same-track pair", "acquisitions": [
                {"track": a.track, "day": a.day, "coverage": round(a.coverage, 3)} for a in acqs]}
        else:
            best = pairs[0]
            s1_products = best.products
            log.info("S1 pair: track %s %s | pre %s | post %s | %d alternative pair(s)",
                     best.track[0], best.track[1], best.pre.day, best.post.day, len(pairs) - 1)
            manifest["s1"] = {
                "relative_orbit": best.track[0], "direction": best.track[1],
                "pre_date": best.pre.day, "post_date": best.post.day,
                "pre_coverage": round(best.pre.coverage, 3), "post_coverage": round(best.post.coverage, 3),
                "products": [{"role": r, "id": p.id, "name": p.name, "size": p.size}
                             for r, a in (("pre", best.pre), ("post", best.post)) for p in a.products],
                "alternatives": [{"track": p.track, "pre": p.pre.day, "post": p.post.day} for p in pairs[1:]],
            }

    # ---------------- Sentinel-2 -------------------------------------------------
    if not args.skip_s2:
        found = client.search("SENTINEL-2", cfg.aoi_wkt, cfg.search_start, cfg.search_end, product_type="S2MSI2A")
        s2_choices = pick_s2(found, cfg)
        for c in s2_choices:
            log.info("S2 %s %-4s %s cloud=%.0f%% %s", c.tile, c.role, c.product.start.date(), c.cloud,
                     "(CLOUDY)" if c.cloudy else "")
        manifest["s2"] = [{"tile": c.tile, "role": c.role, "id": c.product.id, "name": c.product.name,
                           "cloud": c.cloud, "cloudy": c.cloudy, "size": c.product.size} for c in s2_choices]

    # ---------------- DEM --------------------------------------------------------
    if not args.skip_dem:
        manifest["dem_tiles"] = tile_names(cfg.bbox)
        log.info("DEM tiles: %d", len(manifest["dem_tiles"]))

    total = sum(p.size or 0 for p in s1_products) + sum(c.product.size or 0 for c in s2_choices)
    log.info("Planned CDSE download: %s", _human(total))

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if args.dry_run:
        log.info("dry run: wrote %s, downloaded nothing", out / "manifest.json")
        return 0 if (args.skip_s1 or s1_products) else 2

    # ---------------- downloads --------------------------------------------------
    if not args.skip_dem:
        manifest["dem_files"] = [p.name for p in download_dem(cfg.bbox, out / "dem")]
    if not args.skip_osm:
        manifest["osm_counts"] = download_osm(cfg.bbox, cfg.osm_snapshot, cfg.osm_tile_deg,
                                              out / "osm" / "osm_pre_event.gpkg")
    for p in s1_products:
        client.download(p, out / "s1")
    for c in s2_choices:
        client.download(c.product, out / "s2")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("ingest complete -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
