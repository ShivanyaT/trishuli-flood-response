# Limitations

Read this before quoting any number from this system. It is written by hand; the numbers it refers to are in
`reports/eval_metrics.json`, `reports/validation.json` and `data/outputs/*.json`, and are never copied here.

## What the outputs mean
1. **Exposure is not damage.** "Building in hazard zone" means the footprint overlaps the flood or debris mask. It does
   not mean the building collapsed. Precision against EMSR-graded damage is therefore a lower bound.
2. **Lower-bound estimates.** 10 m radar/optical masks miss narrow channels, small debris fans and anything under
   canopy or in shadow. Reported areas and counts are more likely too low than too high.
3. **Cut-off means "no road route", not "no access".** Footpaths, trails, helicopters and fords are not modelled. A
   settlement flagged as cut off may still be reachable on foot; one that is not flagged may still be in trouble.
4. **Two bridge scenarios, not a prediction.** Whether an exposed bridge actually failed is unknown from satellite data.
   "Cut off" (bridges intact) and "cut off if bridges fail" bound the answer; we do not guess between them.
5. **Settlements with no mapped pre-event route are not counted as cut off** ("unreachable_pre"). They are listed
   separately because OSM is incomplete in rural Nepal, and an unmapped road is not an inaccessible village.
6. **No casualty or population estimates.** The system does not estimate deaths, injuries or people affected, and the
   copilot declines such questions.

## Data limitations
7. **Sentinel-1 geometry.** Layover and shadow hide steep Himalayan slopes. Same-relative-orbit pairs make them cancel
   in the change signal but not disappear; the share radar could not observe is reported with every run.
8. **Wet snow and ice** lower radar backscatter like water. The flood-corridor filter (relative elevation and slope)
   reduces these false positives but cannot remove them. High-altitude pixels are the least trustworthy.
9. **Water versus debris.** A debris-laden flood is not always dark in radar. Debris is mapped separately (backscatter
   increase, vegetation loss) and the two classes can be confused at the boundary.
10. **Clouds.** Sentinel-2 is unusable under monsoon cloud. Where optical data are missing, only radar is used and the
    share of pixels with clear optical data on both dates is reported.
11. **Resolution.** 10 m pixels against building footprints of about 8 to 10 m: one pixel of misregistration decides
    whether a house is "hit". Validation therefore reports strict and ±1 pixel tolerant scores.
12. **DEM.** Copernicus DEM (30 m, resampled) and the priority-flood D8 flow path are approximations. The flow path
    follows the DEM, not the observed flood, and cannot represent debris-dam dynamics.
13. **OSM completeness.** The pre-event snapshot is the only road and building source. Mapped coverage is uneven, and the
    match rate of EMSR buildings to OSM footprints is itself reported as a completeness measure.
14. **Pre-image choice.** The pre-event image is chosen automatically within a window. A pre image that already shows
    high water, snow or crop-cycle differences biases the change signal.
15. **Preprocessing shift.** Training data (Kuro Siwo) are preprocessed by their authors (sigma0, DEM and filter
    choices). Our SNAP graph differs, so inputs at inference time are not identically distributed to training data.

## Model limitations
16. **No Himalayan evidence in the training data.** The Kuro Siwo events used contain no Himalayan flash floods or
    glacial-lake outbursts. The model has never seen this terrain type. The held-out evaluation measures
    generalisation to other events, not to Nepal.
17. **Thresholds are not tuned on Trishuli.** The classical baseline thresholds come from the training events and the
    AOI config; tuning them on EMSR927 would make the validation circular.
18. **ML vs baseline.** The ML mask is used only where evaluation shows it beats the baseline; otherwise the baseline is
    kept. The choice is recorded in `flood_report.json`.

## Validation limitations
19. **EMSR927 is an analyst product, not ground truth.** It is drawn from very-high-resolution imagery by people,
    covers only its areas of interest, and is silent elsewhere.
20. **Timing mismatch.** EMSR and Sentinel-1 were acquired at different times. Floods recede and debris moves between
    them; the day gap is recorded in `validation.json`.
21. **EMSR has no cut-off product.** The cut-off comparison reruns the same analysis with an EMSR-derived hazard. It
    cross-checks consistency; it does not measure accuracy.
22. **Evaluation only.** Nothing in `src/` imports `src/validate.py`; a test enforces it. EMSR data never enter the
    pipeline, and the thresholds were not adjusted after seeing them.

## Copilot, language and review
23. **Numbers come from `facts.py` only.** LLM wording that contains a number not in the JSON is rejected and replaced
    by a template. This guarantees the numbers match the outputs, not that the wording is good.
24. **Nepali text has not been reviewed by a native speaker.** Treat it as a draft.
25. **Automated estimate.** Every product is marked "human review required". It supports, and does not replace,
    field assessment.

## Results that this repository does not contain
No real-data run, trained weights, or EMSR927 comparison were produced in the environment where the code was
written (no network access to Copernicus, no SNAP, no GPU, no EMSR files). All tests use a synthetic town with
known answers. The real-data numbers appear only after the commands in the README are run; the final report marks
them `PENDING` until then.
