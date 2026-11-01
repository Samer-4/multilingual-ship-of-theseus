# QUARANTINED: do not use as a base table or as results

Kaggle kernel `samahmed17/sam-sot-mt-madlad-smoke5` version 2 (run_id `madlad-3b-mt-default-2026-10-31T052125Z-fca149`, 2026-10-31).
transformers 5.16.1 loaded MADLAD with a decoder input embedding separate from `shared`, while round 1 (transformers 4.46.3) used one shared module.
All 5 S2 outputs and all 5 T0 -> S1 probes are a single repeated non-Latin token up to the 512-token cap.

The files here are kept unchanged, with their provenance, as evidence. `configs/experiment.yaml: quarantined_runs` makes the runner refuse these records.
Analysis: `reports/stage3/MADLAD_SMOKE_FAILURE.md`.
