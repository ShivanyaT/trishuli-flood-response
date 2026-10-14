"""FastAPI backend: copilot, layers for the map, downloads, and a re-run of the analysis stages.

    FLOOD_OUT=data/outputs FLOOD_CONFIG=config/aoi_trishuli.yaml uvicorn src.api:app --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import load_config
from .copilot import answer, get_llm, situation_report
from .facts import Facts

app = FastAPI(title="Flood response copilot", version="0.5")

LAYERS = {  # public name -> (gpkg file, layer)
    "buildings_hit": ("damage.gpkg", "buildings_hit"), "roads_hit": ("damage.gpkg", "roads_hit"),
    "bridges_exposed": ("damage.gpkg", "bridges_exposed"), "settlements": ("cutoff_settlements.gpkg", "settlements"),
    "destinations": ("cutoff_settlements.gpkg", "destinations"), "flowpath": ("flowpath.gpkg", "path"),
}
DOWNLOADS = {"damage.gpkg": "application/geopackage+sqlite3", "cutoff_settlements.gpkg": "application/geopackage+sqlite3",
             "cutoff_settlements.csv": "text/csv", "situation_report.pdf": "application/pdf"}


def out_dir() -> Path:
    return Path(os.getenv("FLOOD_OUT", "data/outputs"))


def load_facts() -> Facts:
    cfg_path = os.getenv("FLOOD_CONFIG")
    if cfg_path and Path(cfg_path).exists():
        c = load_config(cfg_path)
        return Facts.load(out_dir(), c.name.replace("_", " "), c.event_date.isoformat(), list(c.bbox))
    return Facts.load(out_dir())


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    lang: str = "en"


@app.get("/health")
def health():
    f = load_facts()
    return {"status": "ok", "missing_outputs": f.missing(), "synthetic": f.event["synthetic"],
            "llm": os.getenv("COPILOT_LLM", "none")}


@app.get("/summary")
def summary(lang: str = "en"):
    f = load_facts()
    if f.missing():
        raise HTTPException(409, f"outputs missing: {f.missing()}; run the pipeline first")
    r = situation_report(f, lang, get_llm())
    return {k: r[k] for k in ("answer", "lang", "sources", "mode", "verified")}


@app.post("/copilot/ask")
def ask(body: Ask):
    f = load_facts()
    if f.missing():
        raise HTTPException(409, f"outputs missing: {f.missing()}; run the pipeline first")
    r = answer(body.question, body.lang, f, get_llm())
    return {"answer": r["answer"], "sources": r["sources"], "intent": r["intent"], "lang": r["lang"],
            "mode": r["mode"], "verified": r["verified"]}


@app.get("/layers/{name}")
def layer(name: str, limit: int = 20000):
    if name not in LAYERS:
        raise HTTPException(404, f"unknown layer; choose from {sorted(LAYERS)}")
    fn, lyr = LAYERS[name]
    p = out_dir() / fn
    if not p.exists():
        raise HTTPException(404, f"{fn} not produced yet")
    try:
        g = gpd.read_file(p, layer=lyr)
    except Exception:
        return JSONResponse({"type": "FeatureCollection", "features": []})
    return JSONResponse(content=json.loads(g.head(limit).to_crs(4326).to_json()))


@app.get("/downloads/{name}")
def download(name: str):
    if name not in DOWNLOADS:
        raise HTTPException(404, "unknown file")
    p = out_dir() / name
    if not p.exists():
        raise HTTPException(404, f"{name} not produced yet")
    return FileResponse(p, media_type=DOWNLOADS[name], filename=name)


class Run(BaseModel):
    stages: list[str] = ["damage", "cutoff", "report"]


@app.post("/run")
def run(body: Run):
    """Re-run the fast analysis stages on existing flood masks. Full ingest->preprocess needs `python -m src.pipeline`."""
    from .pipeline import run_stages
    cfg = os.getenv("FLOOD_CONFIG", "config/aoi_trishuli.yaml")
    allowed = {"damage", "cutoff", "report"}
    bad = [s for s in body.stages if s not in allowed]
    if bad:
        raise HTTPException(400, f"only {sorted(allowed)} can be run from the API, not {bad}")
    wd = out_dir().parent
    results = run_stages(cfg, wd, body.stages, outputs=out_dir(), osm=wd / "raw" / "osm" / "osm_pre_event.gpkg")
    return {"results": results}
