# MADLAD pilot remainder: one session for the last 168 documents (prepared 2026-10-31, not launched)

**Kernel:** `samahmed17/sam-sot-mt-madlad-pilot-remainder`. Deployment in `kaggle/madlad_pilot/remainder/`, item list `data/manifests/madlad_pilot_remainder/remainder_01.csv` (168 items, 672 operations; this equals pilot batches 2–8).

**Base:** the downloaded batch-1 export `data/generated/madlad_pilot/b01/xsum/madlad-3b`, 516 records (Stage 2 400 + smoke 20 + batch 1 96).
- It is pinned by sha256 `b05fe1b7…` and verified by `scripts/verify_madlad_pilot.py` before the deployment is written. The checks cover RESUME hashes, staged hashes, Stage 2 unchanged and chains, and confirm that all 420 smoke records are byte-identical.
- The kernel attaches the batch-1 kernel output, and the notebook refuses any other export.

## Configuration
- **Translation:** unchanged. Config `705a8bd4992e`, runtime patch `madlad-embed-decoder-v1` (decoder wiring).
- **Gates:** tokenizer parity, a reference gate with 7 drift probes (one per source label), the quality gate, and validation including the nonblocking content warnings.
- **Hard session limit:** 4 h (`kernels push -t 14400`).
- **Clean stop:** at `TIME_LIMIT_MINUTES = 215` of translation. Setup takes about 4 min and the export under 1 min, so the run ends about 20 min before the hard kill.

**Expected runtime:**
- From batch 1's measured rate (5.16 ops/min), 672 operations take about 2 h 10 min. The batch-1-calibrated runtime model (7730 s × 1119.9/1220) gives about 1 h 58 min.
- Adding about 3 min of fixed setup, the expected session is **about 2.0–2.25 h**. Kaggle queueing and container start are not included.

## Checkpointing and recovery
| Layer | When | Survives |
|---|---|---|
| work-dir checkpoint (`runs/…/generations_delta.parquet`) | every 5 operations | a hard stop, if Kaggle keeps `/kaggle/working` |
| rolling export (`export/…/generations_merged.parquet` + `RESUME.json`) | every 20 operations and at every stage end | a hard stop, if Kaggle keeps `/kaggle/working`; directly resumable |
| partial export on error or interrupt (notebook `except` branch) | on any Python exception | errors (Kaggle keeps the output of ERROR runs: verified with verify1 v1) |
| final export | `stopped:time_limit`, `stopped:quality_gate` or completed | normal end |

The order is stage-major: all 168 S2 operations run first, then T2, S3 and T3. A run that stops early leaves partial chains, which the next session continues; the runner only schedules operations whose predecessor is valid and whose own record is missing.

**Kaggle limitations (not verified here):**
- Whether `/kaggle/working` is saved when a version is killed at the hard timeout, or lost to an infrastructure failure, is unknown. If it is lost, nothing from that session is recoverable, and a rerun from the batch-1 base redoes it (greedy decoding is deterministic). The soft stop is there so the hard limit is never reached.
- A kill during a rolling export (under 1 s) can leave files that no longer match `RESUME.json`. The downloader and `planned_export_sha` reject such an export rather than use it.
- GPU hours count against the weekly Kaggle quota.

**Resuming after an early stop:**
1. Download with `download_kaggle_outputs.py --deploy-dir Sam/kaggle/madlad_pilot/remainder`. Add `--allow-incomplete` if Kaggle did not report COMPLETE, or `--raw-only` if no export exists.
2. Run `prepare_kaggle_deploy.py --preset remainder --resume-attempt 2`. It attaches only the previous attempt's output (pinned), writes to `…/remainder_r2/` and never touches earlier folders.

## Final integrity (after download)
```
Sam/.venv/bin/python Sam/scripts/verify_madlad_pilot.py data/generated/madlad_pilot/remainder/xsum/madlad-3b \
    --contains data/generated/madlad_pilot/b01/xsum/madlad-3b data/generated/smoke5_v2/xsum/madlad-3b --expect-complete
```
The expected result is 1,188 records: 400 Stage 2 and 788 new (197 × 4), with unique ids, correct lineage, the 516 earlier records unchanged, and only the decoder-wired frozen config. Invalid new outputs would reduce the complete count (attrition); they are reported in `invalid_new` and do not count as an integrity failure.
