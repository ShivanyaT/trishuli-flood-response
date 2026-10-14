"""Final report (Markdown + PDF, max 6 pages). Every number is read from a JSON file the pipeline wrote; a missing
number is printed as PENDING, never filled in or estimated. The text around the numbers is fixed.

    python -m src.final_report --outputs data/outputs --reports reports --repo-url https://github.com/<you>/<repo>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import markdown as md
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .facts import Facts  # noqa: E402
from .report import ATTRIBUTION  # noqa: E402

log = logging.getLogger(__name__)
PENDING = "PENDING"
MAX_PAGES = 6


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x) -> str:
    return PENDING if x is None else f"{100 * x:.1f}%"


def num(x, nd=2, unit="") -> str:
    return PENDING if x is None else f"{x:,.{nd}f}{unit}"


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ------------------------------------------------------------------ diagram
def architecture_png(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.1), dpi=140)
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    boxes = [
        (1, 20, "Inputs", "Sentinel-1 GRD\n(same-orbit pair)\nSentinel-2 L2A + SCL\nCopernicus DEM\nOSM pre-event", "#e8eef9"),
        (22, 20, "Preprocess", "SNAP graph\n(sigma0, Lee, TC)\nco-register to\n10 m UTM grid", "#e8f3ec"),
        (43, 20, "Flood + debris", "change baseline\n+ 5-ch U-Net\ncorridor filter\n(elev, slope)", "#fdf3e1"),
        (64, 20, "Impact", "damage (OSM overlap)\ncut-off graph\n2 bridge scenarios\nflow path", "#fbe9e7"),
        (85, 20, "Delivery", "FastAPI copilot\nStreamlit map\nPDF sitrep\n(EN / NE)", "#efe8f7"),
    ]
    for x, w, t, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 8), 14.5, 22, boxstyle="round,pad=0.3", fc=col, ec="#556", lw=0.8))
        ax.text(x + 7.25, 27.5, t, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + 7.25, 17.5, body, ha="center", va="center", fontsize=6.6, linespacing=1.35)
    for x in (15.8, 36.8, 57.8, 78.8):
        ax.annotate("", xy=(x + 5.6, 19), xytext=(x, 19), arrowprops=dict(arrowstyle="-|>", color="#334"))
    ax.add_patch(FancyBboxPatch((1, 0.5), 98, 4.5, boxstyle="round,pad=0.2", fc="#f7f7f7", ec="#999", lw=0.6, ls="--"))
    ax.text(50, 2.75, "Facts layer (facts.py): all copilot numbers come from JSON outputs; wording with an unknown number is "
            "rejected.   Validation vs EMSR927 is separate and never an input.", ha="center", va="center", fontsize=6.6)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ------------------------------------------------------------------ content
def abstract(v: dict) -> str:
    text = (
        "We present an end-to-end, reproducible system that turns Sentinel-1 and Sentinel-2 imagery, a DEM and pre-event "
        "OpenStreetMap into a flood damage and access assessment for any area and date. A same-orbit radar change "
        "baseline and a change-aware U-Net map flood water; a separate mask captures debris; overlaying roads and "
        "buildings yields exposure, and a pre-event road graph identifies settlements newly cut off under two bridge "
        "scenarios. A copilot answers in English and Nepali using only numbers from the outputs. For the 26 August 2026 "
        f"Bhote Koshi/Trishuli flood, the flood mask reaches an IoU of {v['iou']} against Copernicus EMS EMSR927 "
        f"(F1 {v['f1']}; tolerant F1 {v['f1t']}), and {v['cut']} settlements are flagged as cut off. These are "
        "lower-bound, exposure-based estimates; EMSR is an analyst product, not ground truth.")
    n = len(text.split())
    if n > 150:
        raise ValueError(f"abstract has {n} words (limit 150)")
    return text


def build_markdown(outputs: Path, reports: Path, repo_url: str, fig: str) -> tuple[str, dict]:
    facts = Facts.load(outputs)
    flood = facts.flood_extent() if facts.flood else {}
    dmg = facts.summarize_damage() if facts.damage else {}
    cut = facts.list_cutoff_settlements() if facts.cutoff else {}
    ev, val = _j(reports / "eval_metrics.json"), _j(reports / "validation.json")
    fv = get(val, "flood", "flood_or_debris") or {}
    s, t = fv.get("strict") or {}, fv.get("tolerant") or {}
    synthetic = bool(get(facts.flood, "synthetic"))
    v = {"iou": pct(s.get("iou")), "f1": pct(s.get("f1")), "f1t": pct(t.get("f1")),
         "cut": PENDING if not cut else cut.get("cut_off_count")}
    nums = {"synthetic": synthetic, "pending": []}

    def track(label, txt):
        if PENDING in str(txt):
            nums["pending"].append(label)
        return txt

    banner = ("> **These numbers come from the SYNTHETIC demo town, not from the Trishuli event.**\n\n" if synthetic else "")
    b, m = (get(ev, "baseline") or {}), (get(ev, "model") or {})
    ml_rows = (f"| Threshold baseline | {pct(b.get('iou'))} | {pct(b.get('f1'))} | {pct(b.get('precision'))} | {pct(b.get('recall'))} |\n"
               f"| Change-aware U-Net | {pct(m.get('iou'))} | {pct(m.get('f1'))} | {pct(m.get('precision'))} | {pct(m.get('recall'))} |")
    bv, rv, cv = (get(val, "buildings") or {}), (get(val, "roads") or {}), (get(val, "cutoff", "cut_off_or_if_bridges_fail") or {})
    brv = get(val, "bridges") or {}
    ctx = get(val, "context") or {}

    lim_md = (reports / "limitations.md").read_text(encoding="utf-8") if (reports / "limitations.md").exists() else ""
    key_lim = [ln.strip() for ln in lim_md.splitlines() if ln[:2].strip(".").isdigit() or ln[:3].strip(". ").isdigit()]
    key_lim = [ln for ln in key_lim if any(k in ln for k in (
        "Exposure is not damage", "no road route", "Two bridge scenarios", "No Himalayan", "Wet snow", "analyst product",
        "Timing mismatch", "Resolution", "Clouds", "OSM completeness", "Nepali text", "Layover"))]

    doc = f"""# Satellite flood damage assessment: Bhote Koshi / Trishuli, 26 August 2026

{banner}## Abstract

{abstract(v)}

## Architecture

![System architecture]({fig})

Every stage is a command-line module that reads and writes files; `python -m src.pipeline --aoi <config> --date <date>`
chains them for any area and date. UTM zone, windows and thresholds come from the AOI config, nothing is hard-coded.

## Methods

**Pairing.** Sentinel-1 GRD scenes are searched through the Copernicus Data Space catalogue. The pre-event scene must share
the post-event scene's relative orbit and cover at least 90% of the AOI, so layover and shadow cancel in the difference.
Sentinel-2 L2A scenes are chosen by cloud cover and masked with the scene classification layer.

**Flood and debris.** After SNAP preprocessing (orbit file, noise removal, sigma0, Lee Sigma 3x3, terrain correction at
10 m with the Copernicus DEM), the baseline flags pixels that became dark and were not water before, restricted to
low slopes and low height above the channel to suppress wet snow and shadow. A 5-channel U-Net (post VV/VH, pre VV/VH, VV
change) trained on Kuro Siwo, validated on held-out events, is used only where it beats the baseline. Debris is a separate
mask from radar backscatter increase and optical vegetation loss.

**Damage and access.** Buildings, roads and bridges from a pre-event OSM snapshot are overlaid on the hazard footprint.
A road graph built from the same snapshot finds settlements that could reach a hospital or district centre before the
event and cannot after it. Results are given for bridges intact and for exposed bridges failing. A priority-flood D8 path
traces where water goes downstream of the source.

**Copilot.** A tool layer (`facts.py`) returns numbers from the JSON outputs. Any generated sentence containing a number
that is not in those files is rejected and replaced by an English or Nepali template.

**Validation.** EMSR927 is used only for evaluation, in a module nothing else imports. Flood extent is scored inside the EMSR
areas of interest, strictly and with a 1 pixel tolerance (EMSR is drawn on imagery of about 1 m, this system works at 10 m).

## Results

{"**Synthetic demo shown; real-event numbers are marked PENDING.**" if synthetic else ""}

*Event outputs.* New flood water {track("flood area", num(flood.get('flood_area_km2'), 2, ' km²'))}; debris
{track("debris area", num(flood.get('debris_area_km2'), 2, ' km²'))}; radar could not observe
{track("unobservable", num(flood.get('radar_unobservable_pct'), 1, '%'))} of the area. Buildings in the hazard zone:
{track("buildings", dmg.get('buildings_hit', PENDING))} (high overlap {dmg.get('buildings_high_overlap', PENDING)}); road in hazard
{track("road km", num(dmg.get('road_km_in_hazard'), 2, ' km'))}; bridges exposed {dmg.get('bridges_exposed', PENDING)} of {dmg.get('bridges_total', PENDING)}.
Settlements cut off with bridges intact: {cut.get('cut_off_count', PENDING)}; additionally cut off if exposed bridges fail:
{cut.get('cut_off_if_bridges_fail_count', PENDING)}.

*Model on held-out Kuro Siwo test events (flood class, pixels).*

| Method | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
{ml_rows}

*Against EMSR927 (flood + debris, inside EMSR areas only).* Strict IoU {track("IoU", pct(s.get('iou')))}, F1 {pct(s.get('f1'))},
precision {pct(s.get('precision'))}, recall {pct(s.get('recall'))}. With 1 pixel tolerance: F1 {pct(t.get('f1'))}
(precision {pct(t.get('precision'))}, recall {pct(t.get('recall'))}). Flood mask alone, without debris: IoU
{pct(get(val, 'flood', 'flood_only', 'strict', 'iou'))}.

*Buildings (EMSR-graded vs our "in hazard").* EMSR-affected buildings matched to an OSM footprint:
{pct(bv.get('osm_match_rate_affected'))} (this is a measure of OSM completeness). Recall on matched affected buildings
{pct(bv.get('recall_on_matched_affected'))}; precision {pct(bv.get('precision_vs_emsr_affected_lower_bound'))}, a lower bound because
exposure is not damage.

*Roads, bridges, cut-off.* Damaged road length ours {rv.get('ours_km', PENDING)} km vs EMSR {rv.get('emsr_km', PENDING)} km
(precision {pct(rv.get('precision'))}, recall {pct(rv.get('recall'))}). Bridges: EMSR damaged {brv.get('emsr_bridges_damaged', PENDING)},
recall {pct(brv.get('recall'))}. Cut-off lists, ours vs EMSR-derived: Jaccard {pct(cv.get('jaccard'))}
(only ours: {', '.join(cv.get('only_ours', [])) or 'none'}; only reference: {', '.join(cv.get('only_emsr_derived', [])) or 'none'}).

*Where and why we disagree.* Image timing: {ctx.get('days_between_s1_post_and_emsr_image', PENDING)} days between the Sentinel-1
post scene and the EMSR source image. Radar unobservable share {pct(ctx.get('s1_unobservable_fraction'))}; Sentinel-2 clear on both dates
{pct(ctx.get('s2_clear_fraction_both_dates'))}. Resolution and timing differences are the first suspects for any boundary disagreement;
see the confusion matrix and side-by-side maps in `reports/fig_val_*.png` and notebook 06.

## Limitations

{chr(10).join('- ' + ln.split('. ', 1)[1] if '. ' in ln[:5] else '- ' + ln for ln in key_lim) or '- See reports/limitations.md'}

The full list ({len([ln for ln in lim_md.splitlines() if ln[:1].isdigit()])} items) is in `reports/limitations.md`.
No casualty or population estimates are made. All outputs are automated estimates that need human review.

## Attributions and citations

{ATTRIBUTION.replace('&copy;', '©')}

Bountos, N. I. et al., *Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood
mapping*, NeurIPS Datasets and Benchmarks 2024. Bonafilia, D. et al., *Sen1Floods11*, CVPR Workshops 2020. Iakubovskii, P.,
*Segmentation Models PyTorch*. Pre-event OSM through the ohsome API (HeiGIT). Weights and code: MIT unless stated.

## Reproducibility

Code: {repo_url}. One command: `docker compose up` (synthetic demo, API on :8000, dashboard on :8501). Real event:
`python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26` with CDSE credentials in `.env`, then
`python -m src.validate run ...` for the EMSR927 comparison and `python -m src.final_report` for this document. All numbers
here are read from `data/outputs/*.json`, `reports/eval_metrics.json` and `reports/validation.json`; `PENDING` means that step has
not been run. Tests: `pytest` (synthetic town with known answers).
"""
    return doc, nums


CSS = """@page { size: A4; margin: 14mm 15mm; @bottom-center { content: counter(page) " / " counter(pages); font: 7pt sans-serif; color:#666; } }
body { font: 8.8pt/1.38 'Noto Sans','Helvetica Neue',Arial,sans-serif; color:#1c2430; }
h1 { font-size: 15pt; margin: 0 0 3mm; } h2 { font-size: 10.5pt; margin: 4mm 0 1mm; color:#24456e; border-bottom:.3pt solid #c9d3e0; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; } th { background:#eef2f8; text-align:left; }
th, td { padding:.8mm 1.6mm; border-bottom:.3pt solid #dbe2ec; } img { width: 100%; } p { margin: 1mm 0 2mm; text-align: left; }
blockquote { background:#fdecea; border-left:1mm solid #c0392b; margin:2mm 0; padding:1mm 3mm; } code { font-size: 7.8pt; }
ul { margin: 1mm 0 2mm; padding-left: 5mm; }"""


def make_pdf(md_text: str, base: Path, pdf: Path) -> int:
    from .report import weasyprint_html
    HTML = weasyprint_html()
    html = md.markdown(md_text, extensions=["tables"])
    doc = HTML(string=f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{html}", base_url=str(base)).render()
    doc.write_pdf(pdf)
    return len(doc.pages)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="data/outputs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--repo-url", default="<REPO URL: add before submitting>")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rep = Path(a.reports)
    rep.mkdir(parents=True, exist_ok=True)
    architecture_png(rep / "fig_architecture.png")
    text, nums = build_markdown(Path(a.outputs), rep, a.repo_url, "fig_architecture.png")
    (rep / "final_report.md").write_text(text, encoding="utf-8")
    pages = make_pdf(text, rep, rep / "final_report.pdf")
    log.info("final_report.pdf: %d page(s); PENDING numbers: %s", pages, nums["pending"] or "none")
    if pages > MAX_PAGES:
        log.error("report is %d pages, limit is %d", pages, MAX_PAGES)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
