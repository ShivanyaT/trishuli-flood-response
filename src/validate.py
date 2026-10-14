"""Validation against Copernicus EMS EMSR927: EVALUATION ONLY, never an input to the pipeline.

Nothing else in `src/` imports this module (a test enforces it), so reference data cannot leak into the analysis.

What is compared (and the caveats that go with each number; they are also written into the output):
  * Flood extent : our flood mask vs EMSR "observed event" polygons, inside the EMSR areas of interest only
                   (outside them EMSR says nothing, so it is neither a false positive nor a true negative).
                   Strict pixel IoU/F1 and a tolerant score (1 px = 10 m): EMSR is delineated on <=1 m imagery, we work at 10 m.
  * Buildings    : our "in hazard" OSM footprints vs EMSR graded buildings. Exposure is not damage, so precision is a lower
                   bound; the OSM match rate is itself a measure of OSM completeness.
  * Roads/bridges: length agreement within a tolerance; EMSR damaged bridges vs our exposed bridges.
  * Cut-off      : EMSR has no cut-off product. We rerun the SAME cut-off analysis with an EMSR-derived hazard (observed
                   event + damaged transport) and compare the two lists. That reference is itself incomplete outside
                   the four EMSR areas, so it is a cross-check, not truth.

File and column names of the EMSR download are matched loosely (observedEvent, areaOfInterest, builtUp, transportation;
a damage-grade column such as damage_gra). Run `inspect` first and check them against your download.

    python -m src.validate inspect --emsr-dir data/emsr927
    python -m src.validate run --config config/aoi_trishuli.yaml --emsr-dir data/emsr927 --emsr-image-date 2026-08-27
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from rasterio import features
from scipy import ndimage as ndi
from shapely.ops import unary_union

from .config import load_config
from .cutoff import CutoffParams, analyse
from .damage import choose_flood_source, load_hazard, load_layer
from .grid import Grid, grid_from_raster, read_raster

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

EXT = (".shp", ".geojson", ".gpkg", ".json")
KINDS = {"flood": "observedevent", "aoi": "areaofinterest", "buildings": "builtup", "transport": "transportation"}
GRADE_COLS = ("damage_gra", "damage_grade", "damage_grad", "damage", "grade")
GRADE_ORDER = ["destroyed", "damaged", "possibly damaged", "none"]


# ============================================================================= loading EMSR
def discover(folder: Path, work: Path) -> dict[str, list[Path]]:
    """Find EMSR vector files by kind. Zips are extracted into `work` first."""
    for z in folder.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work / z.stem)
    files = [p for root in (folder, work) for p in root.rglob("*") if p.suffix.lower() in EXT]
    return {k: sorted(p for p in files if key in p.name.lower()) for k, key in KINDS.items()}


def read_kind(paths: list[Path], crs) -> gpd.GeoDataFrame | None:
    parts = []
    for p in paths:
        g = gpd.read_file(p)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs(crs)
        g["_src"] = p.stem
        parts.append(g)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs) if parts else None


def normalise_grade(v) -> str | None:
    s = str(v).strip().lower()
    if s in ("", "none", "nan", "null"):
        return None
    if "possibl" in s:
        return "possibly damaged"
    if "destroy" in s:
        return "destroyed"
    if "damag" in s:
        return "none" if re.search(r"no\s+visible|not\s+damag|undamag", s) else "damaged"
    if re.search(r"no\s+visible|not\s+affected|undamag|no\s+damage", s):
        return "none"
    return None


def add_grade(g: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    if g is None:
        return None
    col = next((c for c in g.columns if c.lower() in GRADE_COLS), None)
    g = g.copy()
    g["grade"] = g[col].map(normalise_grade) if col else None
    return g


def is_bridge(g: gpd.GeoDataFrame) -> np.ndarray:
    rows = g.drop(columns="geometry").astype(object).where(lambda d: d.notna(), "").astype(str).to_numpy()
    return np.array([any("bridge" in c.lower() for c in row) for row in rows], bool)


# ============================================================================= metrics
def prf(tp: int, fp: int, fn: int, tn: int = 0) -> dict:
    d = lambda a, b: float(a / b) if b else None  # noqa: E731
    return {"iou": d(tp, tp + fp + fn), "f1": d(2 * tp, 2 * tp + fp + fn), "precision": d(tp, tp + fp),
            "recall": d(tp, tp + fn), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def raster_scores(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, tol_px: int = 1) -> dict:
    p, r = pred & mask, ref & mask
    strict = prf(int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum()), int((~p & ~r & mask).sum()))
    if tol_px > 0:
        near_r = ndi.binary_dilation(ref, iterations=tol_px) & mask
        near_p = ndi.binary_dilation(pred, iterations=tol_px) & mask
        pt = (p & near_r).sum() / p.sum() if p.sum() else None
        rt = (r & near_p).sum() / r.sum() if r.sum() else None
        f1t = (2 * pt * rt / (pt + rt)) if pt and rt else None
        tol = {"precision": pt and float(pt), "recall": rt and float(rt), "f1": f1t and float(f1t), "tol_px": tol_px}
    else:
        tol = None
    return {"strict": strict, "tolerant": tol, "pred_px": int(p.sum()), "ref_px": int(r.sum()), "eval_px": int(mask.sum())}


def rasterize(gdf, grid: Grid) -> np.ndarray:
    if gdf is None or not len(gdf):
        return np.zeros(grid.shape, bool)
    return features.rasterize([(g, 1) for g in gdf.geometry], out_shape=grid.shape, transform=grid.transform,
                              fill=0, dtype="uint8").astype(bool)


def load_masks(outputs: Path, source: str, reports: Path):
    src = choose_flood_source(outputs, reports) if source == "auto" else source
    if src == "union":
        flood = (read_raster(outputs / "flood_mask.tif")[0][0] > 0) | (read_raster(outputs / "flood_mask_ml.tif")[0][0] > 0)
        grid = grid_from_raster(outputs / "flood_mask.tif")
    else:
        p = outputs / ("flood_mask_ml.tif" if src == "ml" else "flood_mask.tif")
        flood, grid = read_raster(p)[0][0] > 0, grid_from_raster(p)
    dp = outputs / "debris_mask.tif"
    debris = read_raster(dp)[0][0] > 0 if dp.exists() else np.zeros_like(flood)
    return flood, debris, grid, src


def flood_validation(flood, debris, ref_flood, aoi, grid: Grid, tol_px: int) -> dict:
    ref = rasterize(ref_flood, grid)
    mask = rasterize(aoi, grid)
    out = {"flood_only": raster_scores(flood, ref, mask, tol_px),
           "flood_or_debris": raster_scores(flood | debris, ref, mask, tol_px), "per_aoi": []}
    for _, row in aoi.iterrows():
        m = rasterize(gpd.GeoDataFrame(geometry=[row.geometry], crs=aoi.crs), grid)
        s = raster_scores(flood | debris, ref, m, tol_px)
        out["per_aoi"].append({"aoi": str(row.get("_src", "aoi")), "area_km2": round(float(row.geometry.area / 1e6), 2),
                               "iou": s["strict"]["iou"], "f1_tolerant": s["tolerant"] and s["tolerant"]["f1"]})
    return out, ref, mask


def building_validation(emsr_b, osm_b: gpd.GeoDataFrame, hit: gpd.GeoDataFrame, aoi_geom, match_m: float = 15.0) -> dict:
    emsr_b = emsr_b[emsr_b["grade"].notna()].reset_index(drop=True)
    osm_b = osm_b.reset_index(drop=True)
    tree = shapely.STRtree(osm_b.geometry.values)
    # which OSM footprints did we flag? (centroid of each hit footprint falls inside a footprint)
    flagged = np.zeros(len(osm_b), bool)
    if len(hit):
        _, oi = tree.query(shapely.centroid(hit.geometry.values), predicate="within")
        flagged[np.unique(oi)] = True
    inside = shapely.intersects(osm_b.geometry.values, aoi_geom)
    # EMSR feature -> nearest OSM footprint within match_m
    match = np.full(len(emsr_b), -1)
    for i, g in enumerate(emsr_b.geometry.values):
        cand = tree.query(g, predicate="dwithin", distance=match_m)
        if len(cand):
            match[i] = cand[np.argmin(shapely.distance(osm_b.geometry.values[cand], g))]
    grade = emsr_b["grade"].to_numpy()
    aff, aff_len = np.isin(grade, ["destroyed", "damaged"]), np.isin(grade, ["destroyed", "damaged", "possibly damaged"])
    m_aff, m_len = aff & (match >= 0), aff_len & (match >= 0)
    aff_osm = set(match[m_aff].tolist())
    n_flag_aoi = int((flagged & inside).sum())
    cross = {}
    if len(hit) and "severity" in hit.columns:
        sev = dict(zip(*[tree.query(shapely.centroid(hit.geometry.values), predicate="within")[1],
                         hit["severity"].values[tree.query(shapely.centroid(hit.geometry.values), predicate="within")[0]]]))
        for i in np.where(match >= 0)[0]:
            s = sev.get(match[i])
            if s:
                cross.setdefault(s, {}).setdefault(grade[i], 0)
                cross[s][grade[i]] += 1
    d = lambda a, b: float(a / b) if b else None  # noqa: E731
    return {
        "emsr_graded": int(len(emsr_b)), "emsr_by_grade": {k: int((grade == k).sum()) for k in GRADE_ORDER},
        "emsr_affected": int(aff.sum()), "emsr_affected_matched_to_osm": int(m_aff.sum()),
        "osm_match_rate_affected": d(int(m_aff.sum()), int(aff.sum())),
        "recall_on_matched_affected": d(int(flagged[match[m_aff]].sum()), int(m_aff.sum())),
        "recall_on_matched_incl_possibly": d(int(flagged[match[m_len]].sum()), int(m_len.sum())),
        "ours_flagged_in_aoi": n_flag_aoi,
        "precision_vs_emsr_affected_lower_bound": d(len(aff_osm & set(np.where(flagged & inside)[0].tolist())), n_flag_aoi),
        "severity_vs_emsr_grade": cross,
        "note": "Exposure is not damage: footprints inside the hazard but graded 'no visible damage' count against precision.",
    }


def line_agreement(ours: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, aoi_geom, tol_m: float = 15.0) -> dict:
    o = shapely.intersection(ours.geometry.values, aoi_geom) if len(ours) else np.array([])
    o_len = float(shapely.length(o).sum()) if len(o) else 0.0
    r_len = float(ref.geometry.length.sum()) if len(ref) else 0.0
    if not (o_len and r_len):
        return {"ours_km": round(o_len / 1000, 3), "emsr_km": round(r_len / 1000, 3), "precision": None, "recall": None}
    near_r, near_o = unary_union(ref.geometry.buffer(tol_m)), unary_union(shapely.buffer(o, tol_m))
    p = float(shapely.length(shapely.intersection(o, near_r)).sum() / o_len)
    r = float(ref.geometry.intersection(near_o).length.sum() / r_len)
    return {"ours_km": round(o_len / 1000, 3), "emsr_km": round(r_len / 1000, 3), "precision": p, "recall": r, "tol_m": tol_m}


def bridge_validation(emsr_t, ours_br: gpd.GeoDataFrame, aoi_geom, tol_m: float = 30.0) -> dict:
    eb = emsr_t[is_bridge(emsr_t)] if emsr_t is not None and len(emsr_t) else None
    dmg = eb[eb["grade"].isin(["destroyed", "damaged"])] if eb is not None else None
    ours = ours_br[shapely.intersects(ours_br.geometry.values, aoi_geom)] if len(ours_br) else ours_br
    out = {"emsr_bridges": 0 if eb is None else int(len(eb)), "emsr_bridges_damaged": 0 if dmg is None else int(len(dmg)),
           "ours_exposed_in_aoi": int(len(ours))}
    if dmg is not None and len(dmg) and len(ours):
        near = unary_union(ours.geometry.buffer(tol_m))
        hit_d = int(sum(near.intersects(g) for g in dmg.geometry))
        near_d = unary_union(dmg.geometry.buffer(tol_m))
        hit_o = int(sum(near_d.intersects(g) for g in ours.geometry))
        out |= {"emsr_damaged_found_by_us": hit_d, "ours_confirmed_by_emsr": hit_o,
                "recall": hit_d / len(dmg), "precision": hit_o / len(ours), "tol_m": tol_m}
    return out


def cutoff_comparison(roads, places, health, bld, our_flood, our_debris, ref_hazard, aoi_geom) -> dict:
    """Both lists come from the SAME cut-off analysis; only the hazard differs. Our hazard is clipped to the EMSR areas so
    that both sides have the same spatial coverage (EMSR is silent outside them), and every settlement is compared."""
    empty = shapely.Polygon()
    ours, _, _ = analyse(roads, places, health, bld, our_flood.intersection(aoi_geom), our_debris.intersection(aoi_geom),
                         CutoffParams())
    ref, _, _ = analyse(roads, places, health, bld, ref_hazard, empty, CutoffParams())
    key = lambda g: [(str(n), round(p.x), round(p.y)) for n, p in zip(g["name"], g.geometry)]  # noqa: E731
    ours = ours.assign(_k=key(ours)); ref = ref.assign(_k=key(ref))
    res = {}
    for label, risks in (("cut_off", {"cut_off"}), ("cut_off_or_if_bridges_fail", {"cut_off", "cut_off_if_bridges_fail"})):
        a, b = set(ours.loc[ours["risk"].isin(risks), "_k"]), set(ref.loc[ref["risk"].isin(risks), "_k"])
        res[label] = {"ours": len(a), "emsr_derived": len(b), "both": len(a & b),
                      "jaccard": (len(a & b) / len(a | b)) if (a | b) else None,
                      "only_ours": sorted(k[0] for k in a - b), "only_emsr_derived": sorted(k[0] for k in b - a)}
    res["settlements_compared"] = int(len(ours))
    res["note"] = ("Both lists use the same cut-off analysis; our hazard is clipped to the EMSR areas of interest so the two "
                   "have equal coverage. Blockages outside those areas are in neither list, so this is a cross-check, "
                   "not an estimate of the true cut-off count (see the main outputs for that).")
    return res


# ============================================================================= figures
def figures(flood, debris, ref, aoi_gdf, grid: Grid, scores: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pred = flood | debris
    for i, (_, row) in enumerate(aoi_gdf.head(4).iterrows()):
        x0, b, r, t = row.geometry.bounds
        c0, r0 = ~grid.transform @ (x0, t)
        c1, r1 = ~grid.transform @ (r, b)
        sl = (slice(max(0, int(r0)), min(grid.height, int(r1) + 1)), slice(max(0, int(c0)), min(grid.width, int(c1) + 1)))
        if sl[0].stop - sl[0].start < 2 or sl[1].stop - sl[1].start < 2:
            continue
        p, q = pred[sl], ref[sl]
        diff = np.zeros(p.shape + (3,))
        diff[p & q] = (0.15, 0.55, 0.25); diff[p & ~q] = (0.85, 0.2, 0.2); diff[~p & q] = (0.2, 0.35, 0.85)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
        ax[0].imshow(p, cmap="Blues"); ax[0].set_title("This system (flood + debris)")
        ax[1].imshow(q, cmap="Oranges"); ax[1].set_title("EMSR927 observed event (reference)")
        ax[2].imshow(diff); ax[2].set_title("green agree | red ours only | blue EMSR only")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{row.get('_src', 'AOI')}", fontsize=9)
        pth = out_dir / f"fig_val_flood_aoi{i + 1}.png"
        fig.savefig(pth, dpi=120); plt.close(fig); paths.append(pth)
    s = scores["flood_or_debris"]["strict"]
    cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]], float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.4), constrained_layout=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v):,}\n{cmn[i, j]:.0%}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["no flood", "flood"]); ax.set_yticks([0, 1], ["no flood", "flood"])
    ax.set_xlabel("ours"); ax.set_ylabel("EMSR927"); ax.set_title("Pixel confusion (EMSR areas only)", fontsize=9)
    pth = out_dir / "fig_val_confusion.png"
    fig.savefig(pth, dpi=130); plt.close(fig); paths.append(pth)
    return paths


# ============================================================================= driver
def run(cfg_path: str, emsr_dir: Path, outputs: Path, osm: Path, reports: Path, source: str = "auto",
        tol_px: int = 1, emsr_image_date: str | None = None, flood_filter: str | None = None, proc: Path | None = None) -> dict:
    load_config(cfg_path)
    flood, debris, grid, src = load_masks(outputs, source, reports)
    with tempfile.TemporaryDirectory() as td:
        found = discover(emsr_dir, Path(td))
        data = {k: read_kind(v, grid.crs) for k, v in found.items()}
    if data["flood"] is None:
        raise RuntimeError(f"no 'observedEvent' vector file under {emsr_dir}; run `inspect` to see what was found")
    if flood_filter:
        col, _, val = flood_filter.partition("=")
        data["flood"] = data["flood"][data["flood"][col].astype(str).str.lower() == val.lower()]
    emsr_b, emsr_t = add_grade(data["buildings"]), add_grade(data["transport"])
    aoi = data["aoi"]
    note_aoi = None
    if aoi is None:
        hull = unary_union([g.geometry.union_all() for g in (data["flood"], emsr_b, emsr_t) if g is not None and len(g)]).convex_hull
        aoi = gpd.GeoDataFrame({"_src": ["derived_from_emsr_extent"]}, geometry=[hull.buffer(200)], crs=grid.crs)
        note_aoi = "no areaOfInterest layer found: AOI derived as the hull of EMSR features + 200 m"
    aoi_geom = unary_union(aoi.geometry)

    res = {"flood_source": src, "emsr_files": {k: [p.name for p in v] for k, v in found.items()}, "notes": []}
    if note_aoi:
        res["notes"].append(note_aoi)
    fv, ref, _ = flood_validation(flood, debris, data["flood"], aoi, grid, tol_px)
    res["flood"] = fv

    osm_b = load_layer(osm, "buildings", grid.crs)
    dm = outputs / "damage.gpkg"

    def lay(name):
        try:
            return gpd.read_file(dm, layer=name).to_crs(grid.crs)
        except Exception:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=grid.crs)

    if emsr_b is not None and "grade" in emsr_b and emsr_b["grade"].notna().any():
        res["buildings"] = building_validation(emsr_b, osm_b, lay("buildings_hit"), aoi_geom)
    else:
        res["notes"].append("no graded EMSR buildings found (looked for a damage_gra-like column)")
    if emsr_t is not None and emsr_t["grade"].notna().any():
        dmg_t = emsr_t[emsr_t["grade"].isin(["destroyed", "damaged"]) & ~is_bridge(emsr_t)]
        res["roads"] = line_agreement(lay("roads_hit"), dmg_t, aoi_geom)
        res["bridges"] = bridge_validation(emsr_t, lay("bridges_exposed"), aoi_geom)
        ref_haz = unary_union([unary_union(data["flood"].geometry), unary_union(emsr_t[emsr_t["grade"].isin(
            ["destroyed", "damaged"])].geometry.buffer(15))])
        cs = outputs / "cutoff_settlements.gpkg"
        if cs.exists():
            g = lambda n: load_layer(osm, n, grid.crs)  # noqa: E731
            our_flood, our_debris, _, _ = load_hazard(outputs, src)
            res["cutoff"] = cutoff_comparison(g("roads"), g("places"), g("health"), g("buildings"), our_flood, our_debris,
                                              ref_haz, aoi_geom)
    else:
        res["notes"].append("no graded EMSR transportation layer found: road, bridge and cut-off comparisons skipped")

    # context a reader needs to interpret disagreement
    ctx = {}
    man = (proc or outputs.parent / "raw").parent / "raw" / "manifest.json" if proc is None else proc.parent / "raw" / "manifest.json"
    for cand in (outputs.parent / "raw" / "manifest.json", man):
        if cand.exists():
            s1 = json.loads(cand.read_text(encoding="utf-8")).get("s1", {})
            ctx["s1_post_date"], ctx["s1_pre_date"] = s1.get("post_date"), s1.get("pre_date")
            break
    if emsr_image_date:
        ctx["emsr_image_date"] = emsr_image_date
        if ctx.get("s1_post_date"):
            ctx["days_between_s1_post_and_emsr_image"] = (date.fromisoformat(ctx["s1_post_date"]) -
                                                          date.fromisoformat(emsr_image_date)).days
    fr = outputs / "flood_report.json"
    if fr.exists():
        j = json.loads(fr.read_text(encoding="utf-8"))
        ctx["s1_unobservable_fraction"] = j.get("s1_unobservable_fraction")
        ctx["s2_clear_fraction_both_dates"] = j.get("s2_clear_fraction_both_dates")
    res["context"] = ctx
    res["notes"] += ["EMSR is an analyst product from very-high-resolution imagery, not ground truth.",
                     "Resolution mismatch: EMSR <=1 m vs 10 m here; see the tolerant scores.",
                     "Timing mismatch: acquisitions differ; floods and debris evolve between them."]

    reports.mkdir(parents=True, exist_ok=True)
    res["figures"] = [p.name for p in figures(flood, debris, ref, aoi, grid, fv, reports)]
    (reports / "validation.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (reports / "validation.md").write_text(to_markdown(res), encoding="utf-8")
    return res


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(r: dict) -> str:
    f = r["flood"]["flood_or_debris"]
    s, t = f["strict"], f["tolerant"] or {}
    L = ["# Validation against EMSR927 (evaluation only)", "",
         f"Flood source used: `{r['flood_source']}`. EMSR areas of interest only; {f['eval_px']:,} pixels evaluated.", "",
         "| Flood extent (ours = flood + debris) | IoU | F1 | Precision | Recall |", "|---|---|---|---|---|",
         f"| Strict, pixel level | {_pct(s['iou'])} | {_pct(s['f1'])} | {_pct(s['precision'])} | {_pct(s['recall'])} |",
         f"| Tolerant (+-{t.get('tol_px', 0)} px) | n/a | {_pct(t.get('f1'))} | {_pct(t.get('precision'))} | {_pct(t.get('recall'))} |", ""]
    fo = r["flood"]["flood_only"]["strict"]
    L.append(f"Flood mask alone (debris excluded): IoU {_pct(fo['iou'])}, F1 {_pct(fo['f1'])}.\n")
    if "buildings" in r:
        b = r["buildings"]
        L += ["## Buildings", f"- EMSR graded {b['emsr_graded']} buildings; affected (destroyed+damaged): {b['emsr_affected']}, of which "
              f"{b['emsr_affected_matched_to_osm']} match a pre-event OSM footprint (match rate {_pct(b['osm_match_rate_affected'])}, "
              "an OSM completeness measure).",
              f"- Recall on matched affected buildings: {_pct(b['recall_on_matched_affected'])}.",
              f"- Precision against EMSR-affected (lower bound; exposure is not damage): {_pct(b['precision_vs_emsr_affected_lower_bound'])} "
              f"of {b['ours_flagged_in_aoi']} flagged footprints.", ""]
    if "roads" in r:
        ro, br = r["roads"], r.get("bridges", {})
        L += ["## Roads and bridges", f"- Damaged road length: ours {ro['ours_km']} km vs EMSR {ro['emsr_km']} km; precision {_pct(ro['precision'])}, "
              f"recall {_pct(ro['recall'])}.",
              f"- Bridges: EMSR damaged {br.get('emsr_bridges_damaged')}, ours exposed in the areas {br.get('ours_exposed_in_aoi')}; "
              f"recall {_pct(br.get('recall'))}, precision {_pct(br.get('precision'))}.", ""]
    if "cutoff" in r:
        c = r["cutoff"]["cut_off_or_if_bridges_fail"]
        L += ["## Cut-off settlements (EMSR-derived reference)",
              f"- Settlements compared: {r['cutoff']['settlements_compared']}. Ours {c['ours']}, reference {c['emsr_derived']}, both {c['both']} "
              f"(Jaccard {_pct(c['jaccard'])}).",
              f"- Only ours: {', '.join(c['only_ours']) or 'none'}. Only reference: {', '.join(c['only_emsr_derived']) or 'none'}.", ""]
    L += ["## Read this before quoting the numbers"] + [f"- {n}" for n in r["notes"]]
    if r["context"]:
        L += ["", "Context: " + json.dumps(r["context"])]
    return "\n".join(L) + "\n"


def inspect(emsr_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        found = discover(emsr_dir, Path(td))
        for kind, paths in found.items():
            print(f"\n[{kind}] {len(paths)} file(s)")
            for p in paths[:6]:
                g = gpd.read_file(p)
                print(f"  {p.name}: {len(g)} features, {set(g.geom_type)}, crs={g.crs}")
                for c in g.columns:
                    if c != "geometry" and g[c].nunique() <= 12:
                        print(f"     {c}: {g[c].value_counts(dropna=False).to_dict()}")
        if not any(found.values()):
            print("nothing recognised: expected names containing observedEvent, areaOfInterest, builtUp, transportation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "run"])
    ap.add_argument("--config")
    ap.add_argument("--emsr-dir", required=True)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--osm", default="data/raw/osm/osm_pre_event.gpkg")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--flood-source", default="auto", choices=["auto", "baseline", "ml", "union"])
    ap.add_argument("--tol-px", type=int, default=1)
    ap.add_argument("--emsr-image-date", default=None, help="YYYY-MM-DD of the EMSR source image (for the timing note)")
    ap.add_argument("--flood-filter", default=None, help="COLUMN=VALUE to keep only some observedEvent rows")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "inspect":
        inspect(Path(a.emsr_dir))
        return 0
    if not a.config:
        ap.error("--config is required for `run`")
    r = run(a.config, Path(a.emsr_dir), Path(a.outputs), Path(a.osm), Path(a.reports), a.flood_source, a.tol_px,
            a.emsr_image_date, a.flood_filter)
    s = r["flood"]["flood_or_debris"]["strict"]
    log.info("flood IoU %.3f F1 %.3f (strict)", s["iou"] or 0, s["f1"] or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
