"""Streamlit dashboard: map of flood/debris/damage/cut-off settlements, situation report (EN/NE), copilot chat.

    FLOOD_OUT=data/outputs FLOOD_CONFIG=config/aoi_trishuli.yaml streamlit run dashboard/app.py
    python -m src.demo_data && FLOOD_OUT=data/demo/outputs streamlit run dashboard/app.py     # synthetic demo

Reads the same files as the API (src/api.py). The copilot runs in-process via src.copilot, so every number in an
answer comes from src.facts and is checked before it is shown. OSM tiles are only a visual basemap here, never an input.
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.config import load_config  # noqa: E402
from src.copilot import answer, get_llm, situation_report  # noqa: E402
from src.damage import load_hazard  # noqa: E402
from src.facts import Facts  # noqa: E402
from src.pipeline import run_stages  # noqa: E402
from src.report import ATTRIBUTION, RISK_LABEL, make_pdf, render_map_png  # noqa: E402

st.set_page_config(page_title="Flood response copilot", layout="wide")

PULSE_CSS = """<style>
.pulse{width:16px;height:16px;border-radius:50%;background:var(--c);border:2px solid #fff;box-shadow:0 0 0 0 var(--c);
animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(192,57,43,.7)}70%{box-shadow:0 0 0 16px rgba(192,57,43,0)}100%{box-shadow:0 0 0 0 rgba(192,57,43,0)}}
</style>"""
EXAMPLES = ["Which villages are cut off?", "How many buildings are in the hazard zone?", "Which bridges are exposed?",
            "How reliable is this?"]


# ----------------------------------------------------------------------------- cached data access
def _stamp(out: Path, *names: str) -> float:
    return sum(p.stat().st_mtime for n in names if (p := out / n).exists())


@st.cache_data(show_spinner="Reading flood masks...")
def hazard_geojson(out: str, stamp: float) -> dict:
    flood, debris, grid, src = load_hazard(Path(out), "auto", reports=ROOT / "reports")

    def gj(g):
        if g.is_empty:
            return None
        return json.loads(gpd.GeoSeries([g], crs=grid.crs).to_crs(4326).simplify(2e-5).to_json())

    w, s, e, n = gpd.GeoSeries([box(*grid.bounds)], crs=grid.crs).to_crs(4326).total_bounds
    return {"flood": gj(flood), "debris": gj(debris), "source": src, "bounds": [w, s, e, n]}


@st.cache_data(show_spinner=False)
def layer_geojson(out: str, fname: str, layer: str, stamp: float, limit: int = 5000):
    p = Path(out) / fname
    if not p.exists():
        return None
    try:
        g = gpd.read_file(p, layer=layer)
    except Exception:
        return None
    return json.loads(g.head(limit).to_crs(4326).to_json()) if len(g) else None


@st.cache_data(show_spinner="Drawing static map...")
def static_map(out: str, osm: str, stamp: float) -> bytes:
    flood, debris, grid, _ = load_hazard(Path(out), "auto", reports=ROOT / "reports")
    return render_map_png(Path(out), Path(osm), flood, debris, grid)


def find_osm(out: Path) -> Path:
    for c in (out.parent / "raw" / "osm" / "osm_pre_event.gpkg", out.parent / "osm" / "osm_pre_event.gpkg",
              ROOT / "data" / "raw" / "osm" / "osm_pre_event.gpkg"):
        if c.exists():
            return c
    return out.parent / "raw" / "osm" / "osm_pre_event.gpkg"


# ----------------------------------------------------------------------------- map
def build_map(out: Path, show: dict[str, bool]) -> folium.Map:
    stamp = _stamp(out, "flood_mask.tif", "flood_mask_ml.tif", "debris_mask.tif")
    hz = hazard_geojson(str(out), stamp)
    w, s, e, n = hz["bounds"]
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[s, w], [n, e]])
    m.get_root().html.add_child(folium.Element(PULSE_CSS))

    def add(name, data, style, **kw):
        if data and show.get(name, True):
            g = folium.FeatureGroup(name=name)
            folium.GeoJson(data, style_function=lambda _f, st_=style: st_, **kw).add_to(g)
            g.add_to(m)

    add("Flood water", hz["flood"], {"color": "#1f5fd1", "fillColor": "#2f6fdb", "weight": 0.5, "fillOpacity": 0.65})
    add("Debris / scour", hz["debris"], {"color": "#c46b00", "fillColor": "#e08a1e", "weight": 0.5, "fillOpacity": 0.65})
    t = lambda f: _stamp(out, f)  # noqa: E731
    add("Flow path", layer_geojson(str(out), "flowpath.gpkg", "path", _stamp(out, "flowpath.gpkg")),
        {"color": "#0b7a75", "weight": 3, "dashArray": "6 6"})
    add("Roads in hazard", layer_geojson(str(out), "damage.gpkg", "roads_hit", t("damage.gpkg")),
        {"color": "#e67e22", "weight": 4})
    bh = layer_geojson(str(out), "damage.gpkg", "buildings_hit", t("damage.gpkg"))
    if bh and show.get("Buildings in hazard", True):
        g = folium.FeatureGroup(name="Buildings in hazard")
        fields = [c for c in ("severity", "hazard") if c in bh["features"][0]["properties"]]
        folium.GeoJson(bh, marker=folium.CircleMarker(radius=3, color="#d62728", fill=True, fill_opacity=0.9),
                       tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None).add_to(g)
        g.add_to(m)
    br = layer_geojson(str(out), "damage.gpkg", "bridges_exposed", t("damage.gpkg"))
    if br and show.get("Bridges exposed", True):
        g = folium.FeatureGroup(name="Bridges exposed")
        for f in br["features"]:
            pt = shape(f["geometry"]).representative_point()
            folium.Marker([pt.y, pt.x], icon=folium.Icon(color="purple", icon="road", prefix="fa"),
                          tooltip=str(f["properties"].get("name") or "bridge (exposed)")).add_to(g)
        g.add_to(m)
    st_ = layer_geojson(str(out), "cutoff_settlements.gpkg", "settlements", t("cutoff_settlements.gpkg"))
    if st_ and show.get("Settlements at risk", True):
        g = folium.FeatureGroup(name="Settlements at risk")
        colour = {"cut_off": "#c0392b", "cut_off_if_bridges_fail": "#e67e22", "degraded": "#d4ac0d"}
        for f in st_["features"]:
            p = f["properties"]
            if p.get("risk") not in colour:
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            icon = folium.DivIcon(html=f'<div class="pulse" style="--c:{colour[p["risk"]]}"></div>',
                                  icon_size=(16, 16), icon_anchor=(8, 8))
            folium.Marker([lat, lon], icon=icon, tooltip=f'{p["name"]}: {RISK_LABEL.get(p["risk"], p["risk"])} '
                          f'({p.get("buildings_300m", "?")} buildings within 300 m)').add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ----------------------------------------------------------------------------- page
def main():
    cfgs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").glob("aoi_*.yaml"))
    default_cfg = os.getenv("FLOOD_CONFIG", cfgs[0] if cfgs else "")
    sb = st.sidebar
    sb.header("Analysis")
    cfg_path = sb.selectbox("Area of interest", cfgs, index=cfgs.index(default_cfg) if default_cfg in cfgs else 0)
    out = Path(sb.text_input("Outputs folder", os.getenv("FLOOD_OUT", "data/outputs")))
    lang = sb.radio("Language", ["en", "ne"], format_func=lambda x: {"en": "English", "ne": "नेपाली"}[x], horizontal=True)
    cfg = load_config(ROOT / cfg_path if not Path(cfg_path).is_absolute() else cfg_path)
    facts = Facts.load(out, cfg.name.replace("_", " "), cfg.event_date.isoformat(), list(cfg.bbox))
    llm = get_llm()
    sb.caption(f"Copilot wording: {'template only (deterministic)' if llm is None else type(llm).__name__}; every number "
               "is verified against system outputs before it is shown.")

    if sb.button("Re-run damage → cut-off → report", disabled=bool(facts.missing() and not (out / 'flood_mask.tif').exists())):
        with st.spinner("Running..."):
            res = run_stages(str(ROOT / cfg_path), out.parent, ["damage", "cutoff", "report"], outputs=out, osm=find_osm(out))
        st.cache_data.clear()
        failed = [r for r in res if r["status"] == "failed"]
        (st.sidebar.error if failed else st.sidebar.success)(
            "; ".join(f"{r['stage']}: {r['status']}" for r in res) + (f"\n{failed[0].get('stderr_tail', '')[-400:]}" if failed else ""))
        facts = Facts.load(out, cfg.name.replace("_", " "), cfg.event_date.isoformat(), list(cfg.bbox))
    with sb.expander("Full run from raw satellite data"):
        d = st.date_input("Event date", value=cfg.event_date)
        st.code(f"python -m src.pipeline --aoi {cfg_path} --date {d.isoformat()}", language="bash")
        st.caption("Downloads Sentinel data (many GB) and needs ESA SNAP, so it runs from a terminal, not a button.")

    st.title("Flood situation: " + cfg.name.replace("_", " "))
    if facts.event["synthetic"]:
        st.error("SYNTHETIC DEMO DATA: not a real event. Generated by src/demo_data.py for testing the interface.")
    if facts.missing():
        st.warning(f"Outputs missing in `{out}`: {', '.join(facts.missing())}. Run the pipeline "
                   f"(`python -m src.pipeline --aoi {cfg_path}`) or try the synthetic demo: `python -m src.demo_data`.")
        st.caption(html.unescape(ATTRIBUTION))
        return

    ex, dm, co = facts.flood_extent(), facts.summarize_damage(), facts.list_cutoff_settlements()
    cols = st.columns(6)
    for c, (label, val) in zip(cols, [("Flood water", f"{ex['flood_area_km2']} km²"), ("Debris / scour", f"{ex['debris_area_km2']} km²"),
                                       ("Buildings in hazard", dm["buildings_hit"]), ("Road in hazard", f"{dm['road_km_in_hazard']} km"),
                                       ("Bridges exposed", f"{dm['bridges_exposed']}/{dm['bridges_total']}"),
                                       ("Settlements cut off", co["cut_off_count"])]):
        c.metric(label, val)
    st.caption(f"Radar could not observe {ex['radar_unobservable_pct']}% of the area; OSM is incomplete. "
               "Figures are lower bounds. Exposed ≠ collapsed.")

    left, right = st.columns([3, 2])
    with left:
        names = ["Flood water", "Debris / scour", "Flow path", "Roads in hazard", "Buildings in hazard", "Bridges exposed", "Settlements at risk"]
        show = {n: st.session_state.get(f"cb_{n}", True) for n in names}
        t_map, t_static = st.tabs(["Interactive map", "Static map (works offline)"])
        with t_map:
            st_folium(build_map(out, show), height=620, use_container_width=True, returned_objects=[])
            st.caption("The interactive map loads its map library from a CDN and needs internet; use the static tab offline.")
        with t_static:
            osm = find_osm(out)
            if osm.exists():
                st.image(static_map(str(out), str(osm), _stamp(out, "flood_mask.tif", "flood_mask_ml.tif", "damage.gpkg",
                                                               "cutoff_settlements.gpkg")), width="stretch")
            else:
                st.info(f"Static map needs the pre-event OSM snapshot ({osm}).")
        with st.expander("Map layers"):
            for n in names:
                st.checkbox(n, value=True, key=f"cb_{n}")
        st.caption("Basemap © OpenStreetMap contributors (display only; the analysis uses a pre-event OSM snapshot).")
    with right:
        t_rep, t_chat, t_cut, t_dl = st.tabs(["Situation report", "Copilot", "Cut-off list", "Downloads"])
        with t_rep:
            r = situation_report(facts, lang, llm)
            st.write(r["answer"])
            st.caption(f"mode: {r['mode']} · numbers verified: {r['verified']} · sources: {', '.join(r['sources'])}")
            if "llm_rejected" in r:
                st.warning("The language model's wording was rejected (it contained numbers not in the data); showing the template.")
        with t_chat:
            hist = st.session_state.setdefault("chat", [])
            qcols = st.columns(2)
            pending = None
            for i, q in enumerate(EXAMPLES):
                if qcols[i % 2].button(q, key=f"ex{i}"):
                    pending = q
            typed = st.chat_input("Ask about damage, cut-off villages, bridges, limitations...")
            q = typed or pending
            if q:
                res = answer(q, lang, facts, llm)
                hist.append({"q": q, "a": res["answer"], "meta": f"{res['intent']} · {res['mode']} · verified {res['verified']} · "
                             f"{', '.join(res['sources']) or 'no data used'}"})
            for h in hist[-6:]:
                st.chat_message("user").write(h["q"])
                with st.chat_message("assistant"):
                    st.write(h["a"])
                    st.caption(h["meta"])
        with t_cut:
            csv = out / "cutoff_settlements.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["risk"] = df["risk"].map(lambda x: RISK_LABEL.get(x, x))
                keep = [c for c in ("name", "kind", "risk", "buildings_300m", "pre_dist_km", "post_dist_km_optimistic",
                                    "to_intact_road_m_optimistic") if c in df.columns]
                st.dataframe(df[keep], width="stretch", hide_index=True)
            else:
                st.info("No cut-off table yet.")
        with t_dl:
            for fn, mime in (("damage.gpkg", "application/geopackage+sqlite3"), ("cutoff_settlements.gpkg", "application/geopackage+sqlite3"),
                             ("cutoff_settlements.csv", "text/csv"), ("situation_report.pdf", "application/pdf")):
                p = out / fn
                if p.exists():
                    st.download_button(f"Download {fn}", p.read_bytes(), file_name=fn, mime=mime, key=f"dl_{fn}")
            if not (out / "situation_report.pdf").exists() and st.button("Generate PDF report (EN + NE)"):
                try:
                    make_pdf(out, find_osm(out), str(ROOT / cfg_path), ["en", "ne"])
                    st.rerun()
                except Exception as e:  # WeasyPrint needs pango; say so instead of crashing
                    st.error(f"PDF generation failed: {e}")
    st.divider()
    st.caption(html.unescape(ATTRIBUTION))


main()
