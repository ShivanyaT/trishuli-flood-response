CONFIG ?= config/aoi_trishuli.yaml
CACHE  ?= data/kuro_siwo
GPT    ?= gpt

.PHONY: validate final-report docker-build docker-up docker-down plan ingest preprocess baseline kuro-data tune train evaluate infer damage cutoff flowpath report pipeline demo dashboard api test lint
plan:        ## select scenes + write data/raw/manifest.json, download nothing
	python -m src.ingest --config $(CONFIG) --dry-run
ingest:      ## download S1, S2, DEM, OSM into data/raw/
	python -m src.ingest --config $(CONFIG)
preprocess:  ## S1 (needs SNAP gpt) + S2 -> data/processed/ on a common UTM grid
	python -m src.s1_preprocess --config $(CONFIG) --gpt $(GPT)
	python -m src.s2_preprocess --config $(CONFIG)
baseline:    ## classical flood + debris masks -> data/outputs/
	python -m src.flood_map --config $(CONFIG)
kuro-data:   ## stream a Kuro Siwo sample into $(CACHE) (run on the training machine)
	python -m src.train_data build --split train --max-patches 8000 --cache $(CACHE)
	python -m src.train_data build --split test --max-patches 3000 --cache $(CACHE)
tune:        ## tune baseline thresholds on TRAIN patches
	python -m src.evaluate --cache $(CACHE) --tune
train:       ## train the change-aware U-Net on the cached Kuro Siwo sample -> models/flood_unet.pt
	python -m src.train --cache $(CACHE)
evaluate:    ## model vs baseline on held-out test patches -> reports/ai_component.md
	python -m src.evaluate --cache $(CACHE)
infer:       ## trained model over the processed scene -> data/outputs/flood_mask_ml.tif
	python -m src.infer_flood --config $(CONFIG)
OUT ?= data/outputs
damage:      ## buildings / roads / bridges exposed to flood + debris -> $(OUT)/damage.gpkg
	python -m src.damage --config $(CONFIG) --outputs $(OUT)
cutoff:      ## cut-off settlements (optimistic / conservative bridge scenarios) -> $(OUT)/cutoff_settlements.*
	python -m src.cutoff --config $(CONFIG) --outputs $(OUT)
flowpath:    ## bonus: D8 flow path from flow_source and settlements along it
	python -m src.flowpath --config $(CONFIG) --outputs $(OUT)
report:      ## one-page PDF (EN + NE) -> $(OUT)/situation_report.pdf
	python -m src.report --config $(CONFIG) --outputs $(OUT)
pipeline:    ## everything from raw data for any AOI/date, e.g. make pipeline DATE=2026-08-26
	python -m src.pipeline --aoi $(CONFIG) $(if $(DATE),--date $(DATE),)
demo:        ## synthetic town with known answers (clearly labelled) -> data/demo/
	python -m src.demo_data
dashboard:   ## http://localhost:8501
	FLOOD_OUT=$(OUT) FLOOD_CONFIG=$(CONFIG) streamlit run dashboard/app.py
api:         ## http://localhost:8000/docs
	FLOOD_OUT=$(OUT) FLOOD_CONFIG=$(CONFIG) uvicorn src.api:app --port 8000
test:
	pytest -q
lint:
	ruff check src tests
EMSR ?= data/emsr927
validate:    ## EMSR927 comparison (evaluation only) -> reports/validation.{json,md}, fig_val_*.png
	python -m src.validate run --config $(CONFIG) --emsr-dir $(EMSR) --outputs $(OUT) $(if $(EMSR_DATE),--emsr-image-date $(EMSR_DATE),)
final-report: ## reports/final_report.{md,pdf} (max 6 pages; missing numbers print as PENDING)
	python -m src.final_report --outputs $(OUT) $(if $(REPO_URL),--repo-url $(REPO_URL),)
docker-build:
	docker compose build
docker-up:   ## same as `docker compose up`
	docker compose up
docker-down:
	docker compose down
