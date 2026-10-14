"""End-to-end orchestration for any AOI/date: ingest -> preprocess -> flood map -> (ML) -> damage -> cut-off -> report.

    python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26
    python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26 --bbox 84.9 27.9 85.5 28.3 --dry-run
    python -m src.pipeline --aoi config/aoi_trishuli.yaml --from-stage damage          # re-run the fast tail

Each run gets its own work directory (data/runs/<AOI>_<date>) with a derived config, so AOIs never overwrite
each other. Stages are the same CLIs you can run by hand; this only chains them and records what happened.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

from .config import load_config

STAGES = ["ingest", "s1", "s2", "baseline", "infer", "damage", "cutoff", "flowpath", "report"]


def derive_config(aoi_cfg: str, workdir: Path, event_date: str | None, bbox: list[float] | None) -> Path:
    raw = yaml.safe_load(Path(aoi_cfg).read_text(encoding="utf-8"))
    if event_date:
        d = date.fromisoformat(event_date)
        raw["event_date"] = d.isoformat()
        raw["osm_snapshot"] = (d - timedelta(days=30)).isoformat()   # strictly pre-event
        raw["name"] = f"{str(raw['name']).rsplit('_', 1)[0] if raw['name'][-4:].isdigit() else raw['name']}_{d.year}"
    if bbox:
        raw["bbox"] = [float(x) for x in bbox]
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / "config.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def commands(cfg: Path, wd: Path, stages: list[str], outputs: Path | None = None, osm: Path | None = None,
             weights: str = "models/flood_unet.pt", gpt: str = "gpt") -> list[tuple[str, list[str]]]:
    raw, proc = wd / "raw", wd / "processed"
    out = outputs or wd / "outputs"
    osm = osm or raw / "osm" / "osm_pre_event.gpkg"
    py = [sys.executable, "-m"]
    c = str(cfg)
    table = {
        "ingest": py + ["src.ingest", "--config", c, "--out", str(raw)],
        "s1": py + ["src.s1_preprocess", "--config", c, "--raw", str(raw), "--out", str(proc), "--gpt", gpt],
        "s2": py + ["src.s2_preprocess", "--config", c, "--raw", str(raw), "--out", str(proc)],
        "baseline": py + ["src.flood_map", "--config", c, "--proc", str(proc), "--out", str(out)],
        "infer": py + ["src.infer_flood", "--config", c, "--weights", weights, "--proc", str(proc), "--out", str(out)],
        "damage": py + ["src.damage", "--config", c, "--outputs", str(out), "--osm", str(osm)],
        "cutoff": py + ["src.cutoff", "--config", c, "--outputs", str(out), "--osm", str(osm)],
        "flowpath": py + ["src.flowpath", "--config", c, "--proc", str(proc), "--outputs", str(out), "--osm", str(osm)],
        "report": py + ["src.report", "--config", c, "--outputs", str(out), "--osm", str(osm)],
    }
    return [(s, table[s]) for s in stages]


def run_stages(aoi_cfg: str, wd: Path, stages: list[str], outputs: Path | None = None, osm: Path | None = None,
               dry_run: bool = False, **kw) -> list[dict]:
    cfg = Path(aoi_cfg)
    cmds = commands(cfg, Path(wd), stages, outputs, osm, **kw)
    log = []
    for name, cmd in cmds:
        if name == "infer" and not Path(kw.get("weights", "models/flood_unet.pt")).exists():
            log.append({"stage": name, "status": "skipped", "why": "no trained weights; baseline mask is used"})
            continue
        if name == "flowpath" and load_config(cfg).flow_source is None:
            log.append({"stage": name, "status": "skipped", "why": "no flow_source in the AOI config"})
            continue
        rec = {"stage": name, "cmd": " ".join(cmd)}
        if dry_run:
            rec["status"] = "dry-run"
            log.append(rec)
            continue
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONUTF8": "1"})
        rec |= {"status": "ok" if r.returncode == 0 else "failed", "seconds": round(time.time() - t0, 1)}
        if r.returncode != 0:
            rec["stderr_tail"] = r.stderr[-1200:]
            log.append(rec)
            break
        log.append(rec)
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--date", default=None, help="event date YYYY-MM-DD (overrides the config)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("W", "S", "E", "N"))
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--from-stage", default="ingest", choices=STAGES)
    ap.add_argument("--to-stage", default="report", choices=STAGES)
    ap.add_argument("--gpt", default="gpt")
    ap.add_argument("--weights", default="models/flood_unet.pt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    base = yaml.safe_load(Path(a.aoi).read_text(encoding="utf-8"))
    ev = a.date or str(base["event_date"])
    wd = Path(a.workdir or f"data/runs/{base['name']}_{ev}")
    cfg = derive_config(a.aoi, wd, a.date, a.bbox)
    i, j = STAGES.index(a.from_stage), STAGES.index(a.to_stage)
    log = run_stages(str(cfg), wd, STAGES[i:j + 1], dry_run=a.dry_run, weights=a.weights, gpt=a.gpt)
    (wd / "pipeline_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    for r in log:
        print(f"{r['stage']:9s} {r['status']:9s} {r.get('seconds', '')}")
    return 0 if all(r["status"] in ("ok", "skipped", "dry-run") for r in log) else 1


if __name__ == "__main__":
    sys.exit(main())
