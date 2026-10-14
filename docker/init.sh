#!/usr/bin/env bash
# Runs once before the API and dashboard start. If $FLOOD_OUT has no results yet, fill it with the SYNTHETIC demo town
# (clearly labelled in every output) so `docker compose up` shows a working system with no data and no credentials.
# Real results are never overwritten.
set -euo pipefail
OUT="${FLOOD_OUT:-data/outputs}"
if [ -f "$OUT/flood_report.json" ]; then
  echo "init: found results in $OUT, leaving them alone"
  exit 0
fi
if [ "${AUTO_DEMO:-1}" != "1" ]; then
  echo "init: no results in $OUT and AUTO_DEMO=0; API will report missing outputs until you run the pipeline"
  exit 0
fi
echo "init: no results in $OUT -> building the synthetic demo (not a real event)"
python -m src.demo_data --out data/demo --config "${FLOOD_CONFIG:-config/aoi_trishuli.yaml}"
mkdir -p "$OUT" "$(dirname "$OUT")/raw/osm"
cp -r data/demo/outputs/. "$OUT"/
cp data/demo/osm/osm_pre_event.gpkg "$(dirname "$OUT")/raw/osm/"
cp data/demo/SYNTHETIC_DEMO.txt "$OUT"/
echo "init: done"
