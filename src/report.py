"""One-page situation report (PDF): map, key numbers, cut-off list, narrative (EN and/or NE), attribution.

HTML -> PDF with WeasyPrint, because Nepali (Devanagari) needs real text shaping that ReportLab does not do.
The page uses whatever Devanagari-capable font the system has (Noto Sans Devanagari, Nirmala UI, Mangal, FreeSerif...).
The map is drawn from the PRE-event OSM snapshot and the masks (no web basemap, so it also works offline).

    python -m src.report --config config/aoi_trishuli.yaml --lang en ne
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import load_config  # noqa: E402
from .copilot import get_llm, situation_report  # noqa: E402
from .damage import load_hazard, load_layer  # noqa: E402
from .facts import Facts  # noqa: E402

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 &copy; DLR e.V. 2010-2014 and "
    "&copy; Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights "
    "reserved. &copy; OpenStreetMap contributors (ODbL). Training data: Kuro Siwo (Bountos et al., NeurIPS 2024, MIT "
    "License); Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020, CC BY 4.0). Evaluation reference: European Union, "
    "Copernicus Emergency Management Service data (EMSR927), used for validation only, not as model input.")

RISK_COLOR = {"cut_off": "#c0392b", "cut_off_if_bridges_fail": "#e67e22"}
RISK_LABEL = {"cut_off": "Cut off", "cut_off_if_bridges_fail": "Cut off if bridges fail", "degraded": "Degraded",
              "ok": "OK", "unreachable_pre": "No mapped route", "no_road_access": "No road within 400 m"}


def _geoms(path: Path, layer: str, crs):
    try:
        g = gpd.read_file(path, layer=layer)
        return g.to_crs(crs) if len(g) else None
    except Exception:
        return None


def render_map_png(out: Path, osm: Path, flood, debris, grid, dpi: int = 130) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=dpi)
    roads = load_layer(osm, "roads", grid.crs)
    if len(roads):
        roads.plot(ax=ax, color="#b9bcc2", linewidth=0.4, zorder=1)
    for geom, color, label in ((flood, "#2f6fdb", "New flood water"), (debris, "#e08a1e", "Debris / scour")):
        if not geom.is_empty:
            gpd.GeoSeries([geom], crs=grid.crs).plot(ax=ax, color=color, alpha=0.75, zorder=2)
            ax.plot([], [], "s", color=color, label=label)
    dm = out / "damage.gpkg"
    rh, bh, br = (_geoms(dm, n, grid.crs) for n in ("roads_hit", "buildings_hit", "bridges_exposed"))
    if rh is not None:
        rh.plot(ax=ax, color="#8e1b1b", linewidth=1.6, zorder=3)
        ax.plot([], [], "-", color="#8e1b1b", label="Road in hazard")
    if bh is not None:
        c = bh.geometry.centroid
        ax.scatter(c.x, c.y, s=5, color="#d62728", zorder=4, label="Building in hazard")
    if br is not None:
        c = br.geometry.centroid
        ax.scatter(c.x, c.y, s=45, marker="D", color="#7b3fa0", edgecolor="white", zorder=5, label="Bridge exposed")
    st = _geoms(out / "cutoff_settlements.gpkg", "settlements", grid.crs)
    if st is not None:
        for risk, col in RISK_COLOR.items():
            s = st[st["risk"] == risk]
            if len(s):
                ax.scatter(s.geometry.x, s.geometry.y, s=70, facecolor="none", edgecolor=col, linewidth=2, zorder=6,
                           label=RISK_LABEL[risk])
                for _, r in s.head(12).iterrows():
                    ax.annotate(str(r["name"])[:22], (r.geometry.x, r.geometry.y), xytext=(5, 5),
                                textcoords="offset points", fontsize=6.5, color=col, zorder=7)
    left, b, r_, t = grid.bounds
    ax.set_xlim(left, r_); ax.set_ylim(b, t); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=6.5, framealpha=0.9)
    ax.set_title("Hazard footprint over pre-event OpenStreetMap (no web basemap)", fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _table(rows, head) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


CSS = """
@page { size: A4; margin: 12mm 13mm; @bottom-center { content: "Automated estimate - human review required  |  page " counter(page);
        font: 7pt sans-serif; color: #666; } }
body { font: 9pt/1.35 'Noto Sans','Helvetica Neue',Arial,sans-serif; color: #1c2430; }
h1 { font-size: 15pt; margin: 0 0 1mm; } h2 { font-size: 10pt; margin: 3mm 0 1mm; color: #24456e; border-bottom: .3pt solid #c9d3e0; }
.meta { color: #55606e; font-size: 8pt; margin-bottom: 2mm; }
.banner { background: #fdecea; border: .6pt solid #c0392b; color: #8e1b1b; padding: 1.5mm 3mm; font-weight: 600; margin-bottom: 2mm; }
.row { display: flex; gap: 4mm; } .col { flex: 1; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; } th { text-align: left; background: #eef2f8; }
th, td { padding: .8mm 1.6mm; border-bottom: .3pt solid #dbe2ec; }
img.map { width: 100%; } p.narr { margin: 1mm 0 2mm; text-align: left; }
.ne { font-family: 'Noto Sans Devanagari','Nirmala UI','Mangal','Lohit Devanagari','FreeSerif',serif; font-size: 9.5pt; line-height: 1.5; }
.attr { font-size: 6.5pt; color: #55606e; border-top: .3pt solid #c9d3e0; margin-top: 3mm; padding-top: 1mm; }
"""


def build_html(facts: Facts, cfg_name: str, map_png: bytes, cutoff_csv: Path | None, langs: list[str], llm=None) -> str:
    ex, dm, co = facts.flood_extent(), facts.summarize_damage(), facts.list_cutoff_settlements()
    banner = ('<div class="banner">SYNTHETIC DEMO DATA - not a real event</div>' if facts.event["synthetic"] else "")
    stats = _table([
        ("New flood water", f"{ex['flood_area_km2']} km²"), ("Debris / scour", f"{ex['debris_area_km2']} km²"),
        ("Buildings in hazard zone", dm["buildings_hit"]), ("  of which high overlap", dm["buildings_high_overlap"]),
        ("Road in hazard", f"{dm['road_km_in_hazard']} km"),
        ("Bridges exposed", f"{dm['bridges_exposed']} of {dm['bridges_total']}"),
        ("Settlements cut off", co["cut_off_count"]), ("... if exposed bridges fail", co["cut_off_if_bridges_fail_count"]),
        ("Radar could not observe", f"{ex['radar_unobservable_pct']}%")], ["Key figure", "Value"])
    rows = []
    if cutoff_csv and cutoff_csv.exists():
        df = pd.read_csv(cutoff_csv)
        df = df[df["risk"].isin(["cut_off", "cut_off_if_bridges_fail", "degraded"])].head(10)
        for _, r in df.iterrows():
            gap = r.get("to_intact_road_m_optimistic")
            rows.append((r["name"], RISK_LABEL.get(r["risk"], r["risk"]), int(r["buildings_300m"]),
                         "" if pd.isna(gap) else f"{int(gap)} m"))
    cut_tbl = (_table(rows, ["Settlement", "Status", "Buildings within 300 m", "To nearest intact road"])
               if rows else "<p>No settlements flagged.</p>")
    narr = ""
    for lang in langs:
        r = situation_report(facts, lang, llm)
        cls = "ne" if lang == "ne" else ""
        tag = {"en": "Situation report (English)", "ne": "स्थिति प्रतिवेदन (नेपाली)"}[lang]
        narr += (f'<h2>{tag}</h2><p class="narr {cls}" lang="{lang}">{html.escape(r["answer"])}</p>')
    img = base64.b64encode(map_png).decode()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
{banner}<h1>Flood situation report: {html.escape(facts.event['name'])}</h1>
<div class="meta">Event date {facts.event['date']} | AOI {html.escape(cfg_name)} | generated {now} | Sentinel-1/2, Copernicus DEM, pre-event OSM</div>
<img class="map" src="data:image/png;base64,{img}">
<div class="row"><div class="col"><h2>Key figures</h2>{stats}</div><div class="col"><h2>Settlements at risk</h2>{cut_tbl}</div></div>
{narr}
<h2>Read before use</h2><p style="font-size:8pt">Lower-bound estimates from 10 m radar/optical masks. Exposed means inside the hazard footprint,
not collapsed. OSM is incomplete in rural Nepal; settlements with no mapped route are not counted as cut off. Road access only.
Numbers in the narrative come from system outputs; the text has not been checked by a person.</p>
<div class="attr">{ATTRIBUTION}</div></body></html>"""


PDF_HELP = ("WeasyPrint cannot load its GTK/Pango libraries. Windows: in your environment run "
            "`micromamba install -n flood -c conda-forge pango glib` and set WEASYPRINT_DLL_DIRECTORIES to "
            "%CONDA_PREFIX%\\Library\\bin (PowerShell: $env:WEASYPRINT_DLL_DIRECTORIES=\"$env:CONDA_PREFIX\\Library\\bin\"). "
            "Linux: apt install libpango-1.0-0 libpangoft2-1.0-0. Or use Docker. "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html")


def weasyprint_html():
    """Import WeasyPrint's HTML, turning the cryptic missing-DLL OSError into an actionable message."""
    try:
        from weasyprint import HTML
    except OSError as e:
        raise RuntimeError(f"{PDF_HELP}\n(original error: {e})") from e
    return HTML


def pdf_available() -> bool:
    try:
        weasyprint_html()
        return True
    except RuntimeError:
        return False


def make_pdf(out_dir: Path, osm: Path, cfg_path: str, langs: list[str], pdf_path: Path | None = None) -> Path:
    HTML = weasyprint_html()  # lazy: needs pango (conda-forge pango, or apt libpango)

    cfg = load_config(cfg_path)
    facts = Facts.load(out_dir, cfg.name.replace("_", " "), cfg.event_date.isoformat(), list(cfg.bbox))
    if facts.missing():
        raise RuntimeError(f"outputs missing: {facts.missing()}; run damage and cutoff first")
    flood, debris, grid, _ = load_hazard(out_dir, "auto")
    png = render_map_png(out_dir, osm, flood, debris, grid)
    csv = out_dir / "cutoff_settlements.csv"
    doc = build_html(facts, cfg.name, png, csv, langs, get_llm())
    pdf = pdf_path or out_dir / "situation_report.pdf"
    HTML(string=doc).write_pdf(pdf)
    return pdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--osm", default="data/raw/osm/osm_pre_event.gpkg")
    ap.add_argument("--lang", nargs="+", default=["en", "ne"], choices=["en", "ne"])
    ap.add_argument("--pdf", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = make_pdf(Path(a.outputs), Path(a.osm), a.config, a.lang, Path(a.pdf) if a.pdf else None)
    log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
