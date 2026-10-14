# Trishuli Flood Response

A satellite-based flood assessment project that combines Sentinel-1, Sentinel-2, Copernicus DEM, and pre-event OpenStreetMap data to map flood exposure and changes in road access.

I built this to connect satellite analysis with information that is easier to use: which buildings and roads overlap the mapped hazard, which settlements may lose access, and how to present those results through a dashboard, API, and situation report.

The project is configured through YAML files, so the same workflow can be adapted to another area and event date.

## What it does

- Maps flood water and possible debris/scour using satellite imagery.
- Includes a classical baseline and a trainable flood segmentation model.
- Estimates building, road, and bridge exposure using pre-event OSM data.
- Identifies settlements that lose mapped road access after the event.
- Compares access under two scenarios: exposed bridges remain usable, or exposed bridges fail.
- Produces reports in English and Nepali.
- Provides a Streamlit dashboard, FastAPI endpoints, and a copilot grounded in generated results.
- Includes a synthetic demo for testing without downloading satellite data.

The results describe exposure, not confirmed structural damage. A building overlapping a flood mask is not automatically a destroyed building.

## Project status

The repository includes the processing pipeline, synthetic demo, dashboard, API, training code, and evaluation tools. However, not every execution path has been verified.

Current limitations:

- The Docker configuration has been checked with `docker compose config` and static tests, but the image has not yet been built and tested end to end.
- PDF generation is currently failing in my Windows environment because of native-library issues involving WeasyPrint and its dependencies.
- Some Makefile targets use Unix-style environment variables and need adjustment on Windows.
- Real case-study results and validation metrics should only be reported after the corresponding pipeline stages have completed.
- The Nepali report and copilot templates still need review by a native speaker.

Missing results in the final report are shown as `PENDING`.

## Case-study configuration

The current configuration targets the Bhote Koshi–Trishuli corridor in Nepal, covering parts of Rasuwa, Nuwakot, and Dhading.

The configured event date is `2026-08-26`, and the configured evaluation reference is Copernicus EMS `EMSR927`.

These identifiers are part of the current project setup. The event details, imagery coverage, and reference activation should be checked against the original sources before presenting this as a validated case study.

EMS reference data is reserved for evaluation. It is not used to generate the flood masks or exposure estimates.

## Quick start with Docker

Replace `<repo-url>` with your repository URL:

```bash
git clone <repo-url>
cd trishuli-flood-response
docker compose up
```

Open:

- Dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs

The Docker setup is intended to generate a synthetic demo when no results or credentials are available. Demo outputs are labelled `SYNTHETIC DEMO DATA`, and the startup logic is designed not to overwrite existing real results.

Requirements:

- Docker with Compose 2.24 or newer.
- Enough disk space for the image and dependencies.
- Internet access for the first build.

The first build is large because it includes geospatial dependencies and PyTorch. The Docker image still needs an end-to-end build test.

### Optional environment configuration

Copy the example configuration:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit the values you need:

| Variable | Purpose |
|---|---|
| `CDSE_USER`, `CDSE_PASS` | Credentials for Sentinel downloads from Copernicus Data Space |
| `COPILOT_LLM` | Optional language-model backend; templates are used by default |
| `OLLAMA_*`, `OPENAI_*` | Settings for the selected language-model backend |
| `HF_TOKEN` | Optional token if the training-data download requires authentication |
| `FLOOD_OUT` | Results directory used by the dashboard and API |
| `AUTO_DEMO` | Set to `0` to disable automatic demo generation |

`.env` is git-ignored. Do not commit credentials.

## Local setup

Create and activate the environment:

```bash
micromamba create -n flood -f environment.yml
micromamba activate flood
```

Run the tests:

```bash
make test
```

Generate synthetic data:

```bash
make demo
```

The local demo writes to:

```text
data/demo/outputs/
```

This is different from `data/outputs/`, which several Makefile targets use by default. Make sure the dashboard, API, and report commands point to the results you want to use.

### Run the demo dashboard on Windows

Some Makefile targets currently use Linux/macOS environment-variable syntax. In PowerShell, run the services directly:

```powershell
$env:FLOOD_OUT="data/demo/outputs"
$env:FLOOD_CONFIG="config/aoi_trishuli.yaml"
python -m streamlit run dashboard/app.py
```

### Run the demo API on Windows

Open another terminal, activate `flood`, and run:

```powershell
$env:FLOOD_OUT="data/demo/outputs"
$env:FLOOD_CONFIG="config/aoi_trishuli.yaml"
python -m uvicorn src.api:app --port 8000
```

The API documentation is available at http://localhost:8000/docs.

## Running the satellite pipeline

### 1. Inspect the planned inputs

```bash
make plan
```

This selects the satellite scenes and writes an input manifest before downloading the imagery. Check the selected scenes, acquisition dates, orbit information, and download size first.

### 2. Download the inputs

Add your Copernicus Data Space credentials to `.env`, then run:

```bash
make ingest
```

The input sources are:

- Sentinel-1 radar imagery.
- Sentinel-2 optical imagery.
- Copernicus DEM GLO-30.
- A pre-event OSM snapshot from ohsome.

### 3. Preprocess the imagery

```bash
make preprocess
```

Sentinel-1 preprocessing requires ESA SNAP's `gpt` executable on `PATH`, or an explicit path:

```bash
make preprocess GPT=/path/to/gpt
```

The SNAP graph handles orbit correction, noise removal, calibration, filtering, and terrain correction.

Sentinel-2 preprocessing creates a reflectance stack and uses the scene classification layer to mask clouds and shadows.

### 4. Generate the hazard masks

```bash
make baseline
```

The baseline combines radar change, optical information where usable, and terrain-based filtering.

If trained model weights are available:

```bash
make infer
```

### 5. Calculate exposure and access changes

```bash
make damage cutoff flowpath
```

These stages intersect the hazard with pre-event infrastructure, evaluate changes in road connectivity, and estimate a downstream flow path.

### 6. Generate the situation report

```bash
make report
```

For local demo outputs, use:

```bash
python -m src.report --config config/aoi_trishuli.yaml --outputs data/demo/outputs
```

PDF generation requires a working WeasyPrint installation and a Devanagari font for Nepali text. See the troubleshooting section for the current Windows issue.

## Running the full pipeline

Locally:

```bash
python -m src.pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26
```

With Docker, once the image and SNAP setup are working:

```bash
docker compose run --rm pipeline --aoi config/aoi_trishuli.yaml --date 2026-08-26
```

Useful options:

| Option | Purpose |
|---|---|
| `--dry-run` | Show the planned commands |
| `--from-stage damage` | Restart from the exposure-analysis stage |
| `--bbox W S E N` | Override the area of interest |

Pipeline runs use separate folders under `data/runs/`.

Point `FLOOD_OUT` to the relevant run's `outputs` directory to view its results.

### SNAP in Docker

The Docker image does not include SNAP by default.

An optional build argument is provided:

```text
--build-arg SNAP_INSTALLER_URL=<SNAP Linux installer URL>
```

This installation path has not been tested. Alternatively, run preprocessing locally with SNAP installed.

The demo, dashboard, API, and downstream analysis do not require SNAP.

## Main outputs

Not every file is produced by every command. Demo generation, pipeline runs, and validation use different output directories.

| Output | Description |
|---|---|
| `raw/manifest.json` | Selected satellite scenes and input metadata |
| `outputs/flood_mask.tif` | Baseline flood mask |
| `outputs/flood_mask_ml.tif` | Model prediction, when trained weights are available |
| `outputs/debris_mask.tif` | Possible debris/scour mask |
| `outputs/flood_report.json` | Hazard areas and observation-quality information |
| `outputs/damage.gpkg` | Exposed infrastructure |
| `outputs/damage_summary.json` | Exposure summary |
| `outputs/cutoff_settlements.gpkg` | Settlement-access results |
| `outputs/cutoff_settlements.csv` | Tabular settlement-access results |
| `outputs/cutoff_summary.json` | Access summary for both bridge scenarios |
| `outputs/flowpath.gpkg` | Estimated downstream flow path |
| `outputs/situation_report.pdf` | English/Nepali situation report |
| `reports/validation.json`, `reports/validation.md` | Evaluation results |
| `reports/fig_val_*.png` | Validation figures |
| `reports/final_report.md`, `reports/final_report.pdf` | Final project report |
| `reports/limitations.md` | Interpretation notes and limitations |

## Flood-model training

Training is intended for a GPU machine, such as Kaggle or Colab. The Docker setup does not configure GPU access.

The model uses five input channels:

1. Post-event VV.
2. Post-event VH.
3. Pre-event VV.
4. Pre-event VH.
5. VV change.

The training workflow uses Kuro Siwo data. Threshold tuning uses training patches, validation is separated by event, and testing uses the dataset's official test split.

There is no claimed Nepal or Himalaya hold-out split.

### Kaggle / Colab workflow

Enable a GPU and internet access, then clone and install the project:

```python
!git clone <repo-url>
```

```python
%cd trishuli-flood-response
!pip install -e ".[ml]"
```

Inspect samples before training:

```python
!python -m src.train_data inspect --split train
```

Check the array shapes, radar units, and event identifiers before proceeding.

Run:

```python
!make kuro-data tune train evaluate
```

The configured cache is approximately 5 GB for 8,000 patches. Shards are streamed rather than downloaded in full.

Copy the following back to your local project:

- `models/flood_unet.pt`
- `models/baseline_thresholds.json`
- Generated reports and evaluation results

Model weights are tracked with Git LFS.

## Validation

Place the evaluation vectors in:

```text
data/emsr927/
```

Inspect their structure first:

```bash
python -m src.validate inspect --emsr-dir data/emsr927
```

Then run validation with the appropriate reference date:

```bash
make validate EMSR_DATE=2026-08-27
```

Generate the final report:

```bash
make final-report REPO_URL=[https://github.com/ShivanyaT/trishuli-flood-response.git](https://github.com/ShivanyaT/trishuli-flood-response.git)
```

The evaluation workflow includes mask-overlap metrics and comparisons for buildings, bridges, and settlement access.

`notebooks/06_case_study.ipynb` walks through the case-study workflow. The validation module is kept separate from the processing pipeline to avoid using reference data as an input.

## Copilot

The default copilot uses deterministic English and Nepali templates. It does not require a language model.

Optional backends:

- Ollama, configured through `COPILOT_LLM=ollama`.
- An OpenAI backend, configured through `COPILOT_LLM=openai` and the relevant API settings.

The copilot reads calculated results through `src/facts.py`. When a language model is enabled, it receives those facts for wording rather than calculating new statistics.

The code checks numbers in the generated answer against the supplied facts and falls back to a template if the check fails. This is a guardrail, not a guarantee that every possible answer is correct.

The Nepali templates are handwritten and still need native-speaker review.

## Analysis choices

### Pre-event OSM

The infrastructure snapshot must predate the event. Using current OSM could introduce roads, buildings, or edits added after the flood.

The road graph is built from that snapshot rather than a live OSM query.

### Newly cut-off settlements

A settlement is counted as newly cut off only if it could reach a mapped hospital or town before the event and cannot afterward.

Settlements with no mapped route before the event are listed separately. They may reflect incomplete OSM coverage rather than flood-related isolation.

### Two bridge scenarios

A 10 m hazard mask cannot establish whether a bridge survived. The project therefore reports access under two assumptions:

- Exposed bridges remain usable.
- Exposed bridges fail.

### Exposure rather than destruction

Infrastructure overlap is used to identify possible exposure. The proportion of a building footprint inside the hazard is a prioritisation measure, not a structural damage grade.

### Terrain and observation limits

Terrain filtering helps reduce hillside and wet-snow false positives, but steep terrain, radar shadow, layover, cloud cover, and acquisition timing still limit what can be observed.

The debris mask is separate from the water mask because a debris-flow event may leave deposits and scour rather than persistent standing water.

### Approximate flow source

The configured flow source is near the Rasuwagadhi border crossing. It is a downstream starting point for routing, not a mapped avalanche-origin location.

## Repository structure

| Path | Purpose |
|---|---|
| `config/` | AOI boundaries, dates, thresholds, and analysis settings |
| `src/cdse.py` | Sentinel search and download |
| `src/pairing.py` | Pre/post scene selection |
| `src/dem.py`, `src/osm.py` | DEM and pre-event infrastructure inputs |
| `src/ingest.py` | Input collection |
| `src/grid.py`, `src/terrain.py` | Analysis grid and terrain features |
| `src/s1_preprocess.py`, `src/s2_preprocess.py` | Satellite preprocessing |
| `src/flood_map.py` | Classical flood and debris mapping |
| `src/train_data.py`, `src/dataset.py`, `src/features.py` | Training data and shared features |
| `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/infer_flood.py` | Model training, evaluation, and inference |
| `src/damage.py`, `src/cutoff.py`, `src/flowpath.py` | Exposure, connectivity, and routing |
| `src/facts.py`, `src/copilot.py` | Result access and copilot responses |
| `src/api.py` | FastAPI application |
| `src/report.py`, `src/final_report.py` | Report generation |
| `src/pipeline.py`, `src/demo_data.py` | Pipeline orchestration and synthetic demo |
| `dashboard/app.py` | Streamlit interface |
| `tests/` | Offline tests and synthetic test cases |
| `notebooks/` | Case-study walkthroughs |

## Troubleshooting

### Windows: `'FLOOD_OUT' is not recognized`

Some Makefile targets use Unix-style inline environment variables.

Use PowerShell syntax instead:

```powershell
$env:FLOOD_OUT="data/demo/outputs"
$env:FLOOD_CONFIG="config/aoi_trishuli.yaml"
python -m streamlit run dashboard/app.py
```

### Report says output files are missing

Check the directory passed to `--outputs`.

The local demo uses `data/demo/outputs`, while several default commands look in `data/outputs`.

### WeasyPrint crashes or cannot load DLLs

PDF rendering is currently unresolved in my Windows setup. Importing WeasyPrint successfully does not confirm that rendering works.

Check the environment with:

```bash
python -m weasyprint --info
```

Keep the Python package and native-library setup consistent. Avoid copying DLLs between installations as a workaround.

### Nepali text appears as boxes

Install a Devanagari font such as Noto Sans Devanagari.

### Interactive map is blank offline

The interactive map loads Leaflet from a CDN. Use the static map tab without internet access.

### Docker reports an `env_file` error

Use Docker Compose 2.24 or newer, or create the `.env` file from `.env.example`.

### Docker cannot write output files

The container uses UID 1000. Check ownership and write permissions for the mounted `data`, `models`, and `reports` directories.

## Data sources and attribution

The project uses or includes workflows for:

- Copernicus Sentinel-1 and Sentinel-2.
- Copernicus DEM GLO-30.
- OpenStreetMap data accessed through ohsome.
- Kuro Siwo training data.
- Copernicus EMS reference data for evaluation only.

Attribution:

- Contains modified Copernicus Sentinel data, where processed.
- Copernicus DEM © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the EU and ESA.
- © OpenStreetMap contributors, ODbL.
- Kuro Siwo: Bountos et al., NeurIPS 2024; consult the dataset's original licence and citation instructions.

Check the original providers' attribution and redistribution requirements before publishing derived datasets.
