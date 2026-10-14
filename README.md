# trishuli-flood-response

Satellite-based flood damage assessment: Sentinel-1/2 + Copernicus DEM + **pre-event** OSM,
config-driven so it runs for any AOI/date.

Case study: Bhote Koshi → Trishuli flash flood (ice-rock avalanche / GLOF), 26 Aug 2026, Rasuwa/Nuwakot/Dhading, Nepal.
Evaluation-only reference: Copernicus EMS EMSR927 (never used as input).

## Run it: one command

```bash
git clone <repo-url> && cd trishuli-flood-response
docker compose up
```

Then open the dashboard at <http://localhost:8501> and the API docs at <http://localhost:8000/docs>.
With no data and no credentials, the first start fills `data/outputs/` with a **synthetic demo town** (labelled
"SYNTHETIC DEMO DATA" in every output) so you can try the map, the copilot (English/Nepali) and the PDF immediately.
It never overwrites real results. Needs Docker with Compose >= 2.24; the first build is large (conda GDAL + PyTorch).

> The Docker files were checked with `docker compose config` and static tests only. The image itself could not be built
> in the environment where this was written (no Docker daemon), so the first `docker compose build` is its first real test.

### Credentials setup

```bash
cp .env.example .env     # then edit it
```

| Variable | Needed for |
|---|---|
| `CDSE_USER`, `CDSE_PASS` | downloading Sentinel-1/2 from the Copernicus Data Space (free account: <https://dataspace.copernicus.eu>). Searching is public. |
| `COPILOT_LLM`, `OLLAMA_*`, `OPENAI_*` | optional nicer copilot wording; default is deterministic templates |
| `HF_TOKEN` | optional, only if the Kuro Siwo download asks for it |
| `FLOOD_OUT`, `AUTO_DEMO` | which results the API/dashboard show; `AUTO_DEMO=0` disables the synthetic demo |

`.env` is git-ignored and read by Compose automatically (it is optional, so the demo starts without it).

### Run the real case study

```bash
# in Docker
docker compose run --rm pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26
# or in a local environment (micromamba create -n flood -f environment.yml && micromamba activate flood)
python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26
```

Same-orbit S1 pair selection, SNAP preprocessing, flood and debris masks, damage, cut-off, flow path and PDF run in order.
Add `--dry-run` to see the commands, `--from-stage damage` to re-run only the fast tail, `--bbox W S E N` for another area.
To view a real run: `FLOOD_OUT=data/runs/Trishuli_2026_2026-08-26/outputs docker compose up`.

**Sentinel-1 preprocessing needs ESA SNAP (`gpt`).** The image does not include it by default. Either build with
`--build-arg SNAP_INSTALLER_URL=<SNAP Linux installer url>` (optional, untested here) or run the pipeline locally with SNAP
installed (`make pipeline GPT=/path/to/gpt`). Everything after preprocessing, the API, the dashboard and the demo need no SNAP.
There is no GPU in the image; training (`make kuro-data tune train`) is meant for a GPU machine (see "Training on Kaggle").

### Expected outputs

For a run in `data/runs/<AOI>_<date>/` (or `data/outputs/` for the demo):

| File | Content |
|---|---|
| `raw/manifest.json` | the chosen S1 pair (same relative orbit), S2 scenes, DEM and OSM snapshot |
| `outputs/flood_mask.tif`, `flood_mask_ml.tif` | new flood water (baseline; trained model if weights exist) |
| `outputs/debris_mask.tif`, `flood_report.json` | debris/scour mask; areas, radar-unobservable and cloud fractions |
| `outputs/damage.gpkg`, `damage_summary.json` | buildings, roads and bridges inside the hazard (exposure, not damage) |
| `outputs/cutoff_settlements.{gpkg,csv}`, `cutoff_summary.json` | newly cut-off settlements under two bridge scenarios |
| `outputs/flowpath.gpkg` | downstream flow path from `flow_source` |
| `outputs/situation_report.pdf` | one-page report, English and Nepali |
| `reports/validation.{json,md}`, `fig_val_*.png` | EMSR927 comparison (`make validate`; needs the EMSR vectors in `data/emsr927/`) |
| `reports/final_report.{md,pdf}` | six-page report; any number not yet produced prints as `PENDING` |
| `reports/limitations.md` | what the numbers do and do not mean |

### Make targets

`make ingest`, `make train`, `make infer`, `make dashboard`, `make report` are the main steps; `make help`-style comments are
in the Makefile (also `plan`, `preprocess`, `baseline`, `damage`, `cutoff`, `flowpath`, `validate`, `final-report`, `pipeline`,
`demo`, `api`, `test`, `lint`, `docker-build`, `docker-up`).

### Troubleshooting

- *"env_file ... required" error:* upgrade Docker Compose to >= 2.24, or `touch .env`.
- *Permission denied writing `data/`:* the container user is uid 1000; `chmod -R a+rwX data models reports` or run as that uid.
- *Nepali text shows boxes in the PDF:* a Devanagari font is missing (the image installs Noto; locally install Noto Sans Devanagari).
- *Dashboard map is blank offline:* the interactive map loads Leaflet from a CDN; use the static map tab.

### Case study validation (Day 6)

```bash
python -m src.validate inspect --emsr-dir data/emsr927          # check file/column names first
make validate EMSR_DATE=2026-08-27                             # strict + tolerant IoU/F1, buildings, bridges, cut-off
make final-report REPO_URL=https://github.com/<you>/<repo>
```

`notebooks/06_case_study.ipynb` walks through it. `src/validate.py` is evaluation only; no other module imports it (tested).
`config/aoi_koshi_example.yaml` is a second AOI to show nothing is hard-coded to Trishuli (its date is a placeholder).

---

## Step-by-step (local environment)

```bash
micromamba create -n flood -f environment.yml && micromamba activate flood
cp .env.example .env          # add free CDSE credentials (https://dataspace.copernicus.eu)
make test
make plan                     # picks scenes, prints sizes, writes data/raw/manifest.json (no credentials needed)
make ingest                   # downloads S1, S2, DEM, OSM into data/raw/
```

`make plan` is the important first step: it shows which same-orbit-track S1 pair and which S2 tiles
will be used, and the total download size, before you spend bandwidth.

## Days 2-3

```bash
make preprocess      # needs ESA SNAP's `gpt` on PATH (or GPT=/path/to/gpt); also does Sentinel-2
make baseline        # data/outputs/flood_mask.tif, debris_mask.tif, flood_report.json
# AI component (run on a GPU machine, see "Training on Kaggle")
make kuro-data && make tune && make train && make evaluate
make infer           # data/outputs/flood_mask_ml.tif
```

## Days 4-5

```bash
make damage cutoff flowpath     # data/outputs/damage.gpkg, cutoff_settlements.{gpkg,csv}, flowpath.gpkg, *_summary.json
make report                     # data/outputs/situation_report.pdf (EN + NE)
make dashboard                  # http://localhost:8501   (map, situation report, copilot chat, downloads)
make api                        # http://localhost:8000/docs  (POST /copilot/ask, /summary, /layers/*, /downloads/*)
make pipeline DATE=2026-08-26   # everything from raw data, any AOI/date, one work folder per run
make demo                       # SYNTHETIC town with known answers (labelled) - try the UI before real data exists
```

Optional language model for nicer wording: `COPILOT_LLM=ollama` (Llama 3.1 8B at localhost:11434) or `COPILOT_LLM=openai`
(`OPENAI_API_KEY`). Default is no model: deterministic templates in English and Nepali.

**Why the copilot cannot invent a number.** Numbers come only from `src/facts.py` (reads `data/outputs/*`). A model, if
enabled, only words the JSON it is given; its answer is scanned and rejected if any number is not in that JSON (or a
faithful rounding of one), and the template answer is used instead. Tested in `tests/test_day5.py`.
The Nepali templates were written by hand, not machine-translated, so numbers cannot be altered; **have a native speaker
review them** before submission. The PDF needs a Devanagari font (Noto Sans Devanagari) and pango for WeasyPrint.

### Training on Kaggle / Colab
1. New notebook, GPU on, Internet on. `!git clone <your repo> && cd trishuli-flood-response && pip install -e .[ml]`.
2. `!python -m src.train_data inspect --split train` first: it prints array shapes/units of a few samples. The code
   auto-detects linear vs dB and the event id, but the dataset card does not document either, so confirm they look sane.
3. `!make kuro-data tune train evaluate` (cache is ~5 GB for 8000 patches; shards are streamed, not downloaded whole).
4. Download `models/flood_unet.pt`, `models/baseline_thresholds.json`, `reports/` back to your repo (weights go through Git LFS).

## Layout

| Path | Purpose |
|---|---|
| `config/aoi_*.yaml` | AOI, event date, windows, thresholds (the only thing that changes between AOIs) |
| `src/cdse.py` | CDSE OData search + resumable authenticated download |
| `src/pairing.py` | Same-relative-orbit + same-direction S1 pair selection; S2 pre/post per tile |
| `src/dem.py` | Copernicus DEM GLO-30 from AWS Open Data (anonymous) |
| `src/osm.py` | Pre-event OSM via ohsome (tiled, de-duplicated) → GeoPackage |
| `src/ingest.py` | CLI tying the above together (`--dry-run`, `--skip-s1/s2/dem/osm`) |
| `src/grid.py`, `src/terrain.py` | AOI-derived UTM grid, DEM slope, flood-corridor (height above valley floor) |
| `src/s1_preprocess.py` | SNAP `gpt` graph (orbit, noise, sigma0, Lee Sigma, terrain correction) -> dB, co-registration check |
| `src/s2_preprocess.py` | L2A -> 10 m reflectance stack + SCL cloud/shadow mask |
| `src/flood_map.py` | Classical flood + debris baseline (S1 change, S2 refinement, corridor filter) |
| `src/train_data.py`, `dataset.py`, `features.py` | Kuro Siwo streaming cache, dataset/augmentation, shared input features |
| `src/model.py`, `train.py`, `evaluate.py`, `infer_flood.py` | U-Net/SegFormer, event-held-out training, honest evaluation, tiled inference |
| `src/damage.py`, `src/cutoff.py`, `src/flowpath.py` | Hazard x pre-event OSM; cut-off settlements (graph from the pre-event snapshot); flow path |
| `src/facts.py`, `src/copilot.py`, `src/api.py`, `src/report.py`, `src/pipeline.py`, `src/demo_data.py` | Deterministic tools, verified-number copilot, FastAPI, PDF, orchestration, synthetic demo |
| `dashboard/app.py` | Streamlit dashboard (interactive map + offline static map, EN/NE report, copilot) |
| `tests/` | Offline tests (synthetic valley scene, synthetic WebDataset tar, CPU end-to-end training) |

## Changes from the original 7-day plan

- **`sentinelsat` replaced** by a small CDSE OData client: sentinelsat targets the retired SciHub, not CDSE.
- **`--dry-run` / manifest**: scene selection is inspectable and reproducible before any download.
- **AOI tightened** from ~140×100 km to the flood corridor (fewer tiles, far smaller downloads). Refine when judges give the AOI.
- **Event-time safety**: a same-day S1 pass is not treated as "post-event" unless `event_time_utc` is set.
- **Pair ranking**: post as early as possible after the event, then pre as close as possible; alternatives are logged for the Q&A on "what if no same-orbit pair?".
- **Cloud-limited S2 is allowed with a flag**, not a hard failure (monsoon).
- **OSM tiling + dedup**, and a hard check that `osm_snapshot < event_date`.

## Changes in Days 2-3 (and why)

- **No "Himalaya hold-out".** Neither the Kuro Siwo paper nor its dataset card names a Nepal/Himalaya event, so that split
  cannot be built. Validation is held out *by event*, testing uses the official 10-event test split, and Trishuli vs
  EMSR927 is the only out-of-distribution check. The weights file is `models/flood_unet.pt`, not `segformer_himalaya.pt`.
- **SNAP for terrain correction** instead of pure-Python geocoding; the graph mirrors Kuro Siwo's (sigma0, Lee Sigma 3x3,
  10 m) so training and inference data look alike.
- **Change-aware model**: 5 input channels (post VV/VH, pre VV/VH, VV change), not post-only. Kuro Siwo ships two
  pre-event images; training picks one at random, inference has one.
- **Flood-corridor filter** (height above local valley floor + slope) against wet-snow/hillside false positives.
- **Separate debris mask** (S1 backscatter increase, S2 vegetation loss) because this was an ice-rock debris flow.
- **S2 cloud mask from SCL** (we download L2A; s2cloudless needs L1C).
- **Threshold tuning on train patches only**, never on test or Trishuli.
- DEM slope input dropped: the Kuro Siwo paper's best configuration did not use a DEM.

## Changes in Days 4-5 (and why)

- **No `osmnx.graph_from_bbox`**: it queries *current* OSM, which may contain post-event edits. The road graph is built from
  the ohsome pre-event snapshot.
- **"Cut off" means newly cut off.** A settlement counts only if it could reach a hospital or town *before* the event and
  cannot after. Settlements with no mapped route even before are listed separately (OSM gaps, not flood damage).
- **Two bridge scenarios** (bridges hold / exposed bridges fail), because a 10 m mask cannot say whether a bridge survived.
- **"Exposed", not "destroyed"**: radar/optical masks at 10 m cannot grade structural damage; severity is share of footprint
  inside the hazard, a ranking aid.
- **Road access only** (walking paths excluded, configurable); settlements also come from building clusters, not only `place=*`.
- **Flow path** uses a pure-Python priority-flood + D8 (no richdem to compile). `flow_source` in the AOI config is the
  Rasuwagadhi border crossing (approximate); the real avalanche source is upstream in Tibet.
- **Dashboard offline fallback**: the interactive map loads Leaflet from a CDN; a static map tab works without internet.
- Dashboard "Run analysis" re-runs the fast tail (damage, cut-off, report); a full run downloads many GB and runs from a terminal.

## Attribution

Contains modified Copernicus Sentinel data 2026. Copernicus DEM © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the EU and ESA.
© OpenStreetMap contributors (ODbL). Training data: Kuro Siwo (Bountos et al., NeurIPS 2024, MIT); Sen1Floods11 (Bonafilia et al., CVPR-W 2020, CC BY 4.0).
Evaluation reference only: Copernicus EMS EMSR927.
