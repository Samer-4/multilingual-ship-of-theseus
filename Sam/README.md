# Sam/ workspace: Multilingual Ship of Theseus

This is a personal research workspace. It reads team files and never edits them. Background is in `PROJECT_AUDIT.md`, and stage status is in `IMPLEMENTATION_PLAN.md`.

## Contents

| Path | What |
|---|---|
| `src/sot_mt/` | the package: manifest, schema, checks, MADLAD/Qwen legacy import, segmentation, translator adapters, runner |
| `scripts/build_stage2.py` | Stage 2 build (manifest → MADLAD import → validation → regression) |
| `scripts/run_translations.py` | command-line runner (plan, mock dry run, or a real run) |
| `scripts/package_kaggle_inputs.py` | builds the two folders to upload as private Kaggle datasets (uploads nothing; aborts on anything credential-like) |
| `scripts/build_notebook00.py` | generates `notebooks/00_generate_translations.ipynb` (edit this, not the .ipynb) |
| `scripts/select_smoke_items.py` | writes `data/manifests/stage3b_madlad_smoke5.csv` (the five smoke-test items) |
| `notebooks/00_generate_translations.ipynb` | the Kaggle generation notebook |
| `reports/stage3/KAGGLE_SMOKE_TEST.md` | step-by-step runbook for the first Kaggle GPU run (editor or terminal-only) |
| `scripts/prepare_kaggle_deploy.py`, `launch_kaggle.py`, `monitor_kaggle.py`, `download_kaggle_outputs.py` | terminal-only Kaggle workflow: deployment folder `kaggle/madlad_smoke5/`, push, live progress, download and filing |
| `src/sot_mt/progress.py`, `src/sot_mt/kaggle_cli.py` | `SOTMT_PROGRESS` JSON events (measured ETA); Kaggle CLI status/log parsing |
| `configs/` | `experiment.yaml`, `translators.yaml` (the madlad-3b entry is **frozen**), `metric_registry.yaml` |
| `tests/` | pytest; mock-based, CPU-only, no downloads |

## Research notebooks (RQ1-RQ3)

The executed notebooks are in `notebooks/01_RQ1_Translation_Decay.ipynb`, `02_RQ2_Source_Attribution.ipynb` and
`03_RQ3_Translator_Fingerprint.ipynb`. They are built from a shared research layer:
- `src/sot_research/`: canonical corpus, cached features and pair metrics, grouped classifiers, plots;
- `scripts/research/`: `compute_features.py`, `build_notebooks.py`, `run_notebooks.py`.

Committed, text-free outputs are under `results/research/`. Start with `results/research/SUMMARY.md`; the tables are
described in `results/research/README.md`, which also gives the setup (`requirements-research.txt`, Python 3.12) and
how to add a translator.

## Local setup and tests (CPU, no models)

```bash
cd Sam
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                  # 171 tests
```

Run these from the repo root:

```bash
Sam/.venv/bin/python Sam/scripts/build_stage2.py --corpus-src ../Ship_of_theseus_paraphrased_copus   # once
Sam/.venv/bin/python Sam/scripts/run_translations.py --translator madlad-3b --max-items 5 --plan-only
Sam/.venv/bin/python Sam/scripts/run_translations.py --translator nllb --max-items 5 --mock --export   # fake translator
```

- `--mock` writes to `Sam/runs_mock/` (gitignored), and its run_ids start with `MOCK-`.
- A real run refuses any table that contains `MOCK-` records.

## The generation table

Defined in `src/sot_mt/schema.py`.

- **Rows:** one per translation operation (one direction of one round trip). Stages run `T0 → S1 → T1 → S2 → T2 → S3 → T3`, and every row names its `input_stage` and `output_stage`.
- **Pathway:** `(experiment_id, translator, prompt_variant)`. Each pathway has one model checkpoint and one `generation_config_id`.
- **Composite key:** `(experiment_id, translator, prompt_variant, item_id, output_stage)`. `record_id` is a hash of that key.
- **Writes are idempotent.** The same key with different content raises `ConflictError`.
- **Inputs:** a new record can only be built from a **valid** record of the **same pathway**. Round 1 starts from T0, and later rounds never restart from T0.
- **Invalid records are final:** kept with `validation_reason`, never retried, and they block the rest of that item's chain.
  - Blocking codes are `check:` codes (blank, unchanged step, wrong language, repetition) and `engine:` codes (generation_cap, empty_segment, segmentation_error).
  - Warnings (length ratio, possible truncation, meta text, `roundtrip_unchanged`) do not block. An identical English round trip is valid.
- **`generation_meta`** (JSON) holds the segmentation method and budget, the token counts of each segment, EOS flags, batch sizes, OOM retries, seconds and the attempt number. Stage 2 rows carry `None` here.

## How the runner works (`src/sot_mt/runner.py`)

- **Selection:** dataset, article ids, item ids, the first N items, a round range and `stop_after`.
- **Order:** stage-major. All selected items finish S_r before any starts T_r.
- **Writes:** every file in the work dir is written atomically (temp file, fsync, rename).
  - `generations_delta.parquet` holds **completed** operations, valid and invalid.
  - `failures.parquet` holds attempts that raised (CUDA OOM after batch halving, runtime errors). These are not records; they are retried in later runs, up to `max_attempts` (default 2).
  - `runs.parquet` holds settings, runtime versions, library mismatches against the config, and status (`completed`, `interrupted`, `stopped:*`).
- **Checkpoints** happen every `checkpoint_every` operations, at the end of each stage, and on KeyboardInterrupt.
  - A hard crash loses only the unflushed buffer, and greedy decoding redoes it identically.
  - An operation in progress never becomes a record, and a document is only recorded once **all** its segments are translated.
- **Resuming:** the runner reloads its work dir and continues with whatever `eligible_inputs` reports as ready. Rerunning a finished work dir performs 0 operations.
- **Stage 2 data is read, never written.** The staged files are checked against `STAGED_FILES.json`, and new records go only to the work dir and the exports.

## Pathways

| Translator / prompt | Model | Input handling | Status |
|---|---|---|---|
| `madlad-3b / mt-default` | google/madlad400-3b-mt | pilot worker replica: `<2xx>` prefix, regex sentence split + greedy packing to 300 tokens, 4 segments per batch, greedy, 512 new tokens, space-joined | S1/T1 reused from Stage 2 (197 valid); runner continues S2–T3 |
| `qwen3-4b / free` (primary), `/ literal` | Qwen/Qwen3-4B-Instruct-2507 | whole document, legacy prompts verbatim, greedy, legacy max-token rule, `<think>` stripped | from T0 (legacy texts unavailable) |
| `nllb / mt-default` | facebook/nllb-200-distilled-1.3B (provisional) | `src_lang` + `forced_bos_token_id` (`eng_Latn`/`spa_Latn`), one sentence per segment, 256-token budget, 8 segments per batch | from T0 |
| `llama / free` | meta-llama/Llama-3.2-3B-Instruct (provisional, gated) | same prompts and rules as Qwen | from T0; needs `USE_HF_TOKEN=True` + Kaggle Secret `HF_TOKEN` |
| `qwen3-4b-vllm-legacy` | (import only) | legacy Colab vLLM outputs, if ever recovered | can be imported, never continued |

**Segmentation is a confound.** MADLAD and NLLB see sentence-sized pieces, so they keep sentence boundaries by construction. Qwen and Llama translate whole documents. Sentence-length and punctuation metrics (RQ1) and translator fingerprints (RQ3) partly reflect this design choice, and must be reported as such.

## Kaggle

### Input datasets (private)

```bash
Sam/.venv/bin/python Sam/scripts/package_kaggle_inputs.py --kaggle-user <YOU>
```

This builds:

```
Sam/build/kaggle/sam-sot-mt-code/     Sam/src/sot_mt/..., Sam/configs/, Sam/scripts/, Sam/notebooks/, Sam/data/manifests/, dataset-metadata.json
Sam/build/kaggle/sam-sot-mt-staged/   t0_manifest.parquet, generations.parquet, STAGED_FILES.json, dataset-metadata.json
```

Upload them yourself; new Kaggle datasets are private by default.

```bash
kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-code --dir-mode zip
kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-staged
```

The corpus itself is **not** needed for generation, because the manifest already contains T0.

### Notebook setup

1. Import `Sam/notebooks/00_generate_translations.ipynb` into Kaggle.
2. Set **Accelerator = GPU T4 ×1** (or P100), and turn **Internet ON** for the first download of a model.
3. Add both datasets as inputs. The notebook finds them by their marker files, so the placeholder slugs can stay as they are.
4. Llama only: add a Kaggle Secret named `HF_TOKEN` from an account that has accepted the Llama 3.2 licence.

### One model at a time

The configuration cell picks **one** pathway per session, and the runner unloads the model when it finishes. Only one adapter may hold a model at a time unless `allow_multiple_models=True`.

### Five-document smoke test (first GPU run)

The full runbook is **`reports/stage3/KAGGLE_SMOKE_TEST.md`**.

- The notebook's defaults already select the five items in `data/manifests/stage3b_madlad_smoke5.csv`. Set `ALLOW_MODEL_DOWNLOAD = True`, then **Run All**.
- A MADLAD tokenizer parity check (segment counts against round 1) runs before any translation.
- Then 20 operations (5 × S2, T2, S3, T3).
- The export, plus a zip of it, lands in `/kaggle/working/sam_outputs/export/madlad-3b__mt-default/`.

Then repeat with `TRANSLATOR = "nllb"` and `"qwen3-4b"`, which each give 30 operations from T0, in **separate sessions**, or at least after the previous run has unloaded its model. Do `"llama"` with `USE_HF_TOKEN = True` once access is confirmed.

A command-line alternative inside a Kaggle cell:

```bash
!python /kaggle/input/<code>/Sam/scripts/run_translations.py --translator madlad-3b --max-items 5 \
    --staged-dir /kaggle/input/<staged> --work-dir /kaggle/working/sam_outputs/runs/madlad-3b__mt-default \
    --allow-download --export
```

### Checkpoints and resuming

- **Same session:** re-run the "Translate" cell. The runner reloads the work dir and continues.
- **New session:**
  1. "Save Version" so `/kaggle/working` is kept as notebook output.
  2. Add that output as an input dataset.
  3. Keep `PREVIOUS_EXPORTS = "auto"`, which finds attached exports of the same pathway by their `RESUME.json`, or list the folder.
  4. Run again.

  `/kaggle/working` does not survive a new session on its own. Attach exports from one lineage only, because different text for the same record raises `ConflictError`.

  The notebook loads each export's `generations_merged.parquet` (Stage 2 plus everything so far) as the base table, and its `failures.parquet` for retry counts. Duplicates merge, and conflicting content raises.
- **Back to the local workspace:** download the export folder. It can be passed to `run_translations.py --base .../generations_merged.parquet`. Merging into a new staged version is a deliberate later step, and Stage 2 files are never overwritten.

### Hardware and dependency limits

- **GPU:** T4 has 16 GB and **no bf16**, so every config uses fp16. MADLAD round 1 also ran in fp16 (on a V100), and T5-family fp16 overflow would show up as empty or garbage segments, which are flagged invalid.
- **Memory:** MADLAD-3B (~6 GB fp16), NLLB-1.3B (~3 GB) and Qwen3-4B (~8 GB) each fit one T4. Llama-3.2-3B is ~6.5 GB. These are estimates; check them in the smoke test.
- **vLLM is not used.** The team's `config.yaml` records that installing it broke Kaggle's torch. LLMs run one document per `generate` call (batch 1), which is deterministic but slow.
- **Library drift:** Kaggle's transformers version differs from MADLAD round 1 (4.46.3). The runner records the mismatch in `runs.parquet`. The adapters pass `torch_dtype=` before transformers 4.56 and `dtype=` from 4.56 on. `PIN_MADLAD_ROUND1_LIBS = True` installs the round-1 versions before import (notebook §2).
- **No GPU:** the notebook (§3) and the adapters (`NoGPUError`) stop before any download when CUDA is unavailable and `MOCK = False`.
- **Session limits:** keep `TIME_LIMIT_MINUTES` below the session limit, so the run stops cleanly with a checkpoint.

## How the outputs feed RQ1–RQ3

The merged generation table, with `generations_merged.parquet` from all four pathways, plus the corpus paraphrases for the same `item_id`s, becomes the single analysis table (`texts.parquet`, Stage 5). All three notebooks read only that table:

- **RQ1:** cumulative (T0 vs T_r) and step (T_{r−1} vs T_r) metrics on complete cases (`schema.complete_cases`), with attrition reported per pathway and round.
- **RQ2:** source-label classifiers on T0 and T1–T3, grouped by `group_id` (article). Shared-opening removal is a controlled ablation; texts are never modified.
- **RQ3:** translator classifiers on T_r, grouped by article. Segmentation (above) is a named confound.

## Data layout and git

| Path | Contents | Text? | Committed by default |
|---|---|---|---|
| `data/raw/sot_corpus/` | upstream CSVs at `4423d8c` + `STAGED.json` | yes | **no** |
| `data/staged/` | `t0_manifest.parquet`, `generations.parquet`, `STAGED_FILES.json` | yes | **no** |
| `runs/`, `runs_mock/`, `build/` | runner work dirs, mock dry runs, Kaggle upload folders | yes | **no** |
| `data/manifests/`, `data/tables/`, `reports/stage2/` | IDs, configs, summaries | no | yes |

## Paths

`src/sot_mt/paths.py` resolves each root in this order: environment variable, then Kaggle mount, then local default.

| Root | Env var | Kaggle | Local |
|---|---|---|---|
| corpus | `SOT_CORPUS_DIR` | `/kaggle/input/sot-corpus-4423d8c` | `Sam/data/raw/sot_corpus` |
| staged | `SAM_STAGED_DIR` | `/kaggle/input/sam-sot-mt-staged` | `Sam/data/staged` |
| outputs | `SAM_OUTPUT_DIR` | `/kaggle/working/sam_outputs` | `Sam/` |

The notebook sets `SAM_STAGED_DIR` and `SAM_OUTPUT_DIR` itself, from the datasets it finds.
