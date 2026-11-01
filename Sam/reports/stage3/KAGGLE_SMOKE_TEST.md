# Stage 3B: Kaggle smoke test (five MADLAD documents)

**Goal.** Continue five MADLAD chains that already have valid round-1 outputs: T1 → S2 → T2 → S3 → T3. That is 5 items × 4 stages = **20 translation operations**, using the real `google/madlad400-3b-mt` on a Kaggle GPU.

Nothing in this runbook has been run on Kaggle yet. The local preflight (130 CPU tests, including a mock run of the whole notebook from the packaged folders) shows that the plumbing works. It does **not** show that the real model loads and translates correctly on Kaggle. That is what this test is for.

All commands run from the repository root (`multilingual-ship-of-theseus/`).

## Terminal-only workflow (Kaggle CLI 2.2.5, no notebook editor)

The four steps are all run from the repository root.

```bash
# 1. prepare (local only): build the upload folders, write the deployment, validate it
Sam/.venv/bin/python Sam/scripts/prepare_kaggle_deploy.py
Sam/.venv/bin/python Sam/scripts/launch_kaggle.py            # dry run: prints the exact remote commands

# 2. launch: upload the code as a new dataset version, wait until it is ready, push + run the notebook
Sam/.venv/bin/python Sam/scripts/launch_kaggle.py --yes

# 3. monitor: status + live log + progress bar; exits 0 complete / 1 error / 2 cancelled / 3 no notebook_done
Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py

# 4. download + file: raw copy kept, hashes verified, tables to data/generated, reports to results
Sam/.venv/bin/python Sam/scripts/download_kaggle_outputs.py
```

**The deployment folder** is `Sam/kaggle/madlad_smoke5/`. It contains:
- `00_generate_translations.ipynb`, with the configuration cell overridden: `TRANSLATOR="madlad-3b"`, `MAX_ITEMS=5`, the five-item manifest, `MOCK=False`, `ALLOW_MODEL_DOWNLOAD=True`, `DRIFT_PROBE=True`, `TIME_LIMIT_MINUTES=60`, and `EXPECTED_CODE_SHA` set to the code-build digest;
- `kernel-metadata.json`: `samahmed17/sam-sot-mt-madlad-smoke5`, private, GPU `NvidiaTeslaT4`, Internet on, both datasets attached;
- `deploy.json`;
- `last_push.json`, written after a launch.

**The notebook checks its code first.** It compares the attached code dataset with `CODE_MANIFEST.json` and `EXPECTED_CODE_SHA`, and stops at once if they differ, before any model is loaded. The launcher uploads the matching build first for this reason.

**Progress events.** These are one line each, `SOTMT_PROGRESS {json}`, flushed immediately:
- the setup and progress events: `notebook_start`, `phase`, `parity_done`, `probe_model_loaded`, `probe_done`, `run_start`, `model_loaded`;
- one `op_start` and one `op_done` per operation, with `heartbeat` every 30 s of silence;
- `run_end`, `export_done` and `notebook_done` at the end.

Each event carries the model, dataset, current item and stage, done/total and %, valid, invalid, failed and dropped counts, elapsed time, model-load time, ETA (`"ETA calculating"` until two operations have finished, based on measured seconds per input word), throughput, and the phase (loading_model, translating, validating, exporting).

**Download layout**
- `Sam/data/generated/xsum/madlad-3b/`: `generations_merged.parquet`, `generations_delta.parquet`, `failures.parquet` (if any), `runs.parquet`, `RESUME.json`, `PROVENANCE.json`.
- `Sam/results/madlad_smoke5/`: `validation_report.csv`, `attrition.csv`, `tokenizer_parity.csv`, `drift_probe.csv`, `new_translations.csv`, the export zip, the kernel log, `progress_events.jsonl`, `summary.md`, `PROVENANCE.json`.
- `Sam/results/madlad_smoke5/raw/<slug>_v<N>_<UTC>/`: the untouched download, plus `DOWNLOAD.json`.

Generated text, raw downloads, logs and zips are gitignored. Differing existing files are never overwritten; `--replace` moves them to `superseded/` first.

**Stale-code safeguards (after the first launch failed, kernel version 1).** That run mounted code dataset version 1, built before `progress.py` existed, and failed with `ImportError: cannot import name 'progress'`. The launcher had pushed 2 s after uploading version 2, because `datasets status` already said "ready": that status still described version 1. Now:
- `launch_kaggle.py` always rebuilds from the current `Sam/` sources and rewrites the deployment.
- It refuses `owner/slug/N`-pinned code sources.
- It waits until the status is `ready` **and** `current_version_number` has gone up.
- It checks that the file listing of that version, and of the unpinned latest, equals the local build. Kaggle lists the files without the `Sam/` prefix.
- Only then does it push. `last_push.json` records the code dataset version.
- The notebook checks `CODE_MANIFEST.json` and `EXPECTED_CODE_SHA`, and the presence of every `sot_mt` module it needs, *before* the first `sot_mt` import. If several copies of the code are attached, it uses the one whose digest matches.

**Resuming in a later Kaggle session.** Prepare a new kernel with `--resume-from-kernel samahmed17/sam-sot-mt-madlad-smoke5`. This puts the smoke kernel's output into `kernel_sources`, and `PREVIOUS_EXPORTS="auto"` picks it up. Locally, use `run_translations.py --base Sam/data/generated/xsum/madlad-3b/generations_merged.parquet`.

---

## 0. The five test items

The file is `Sam/data/manifests/stage3b_madlad_smoke5.csv`; the notebook reads it by default. It is regenerated by `Sam/scripts/select_smoke_items.py`, and the rule lives in `Sam/src/sot_mt/smoke.py`.

| # | item_id | article_id | source | T1 words | T0 words | round-1 segments (en→es / es→en) | band |
|---|---|---|---|---|---|---|---|
| 1 | `xsum\|OpenAI\|xsum-4710` | xsum-4710 | OpenAI | 50 | 51 | 1 / 1 | short |
| 2 | `xsum\|Tsinghua\|xsum-4719` | xsum-4719 | Tsinghua | 104 | 110 | 1 / 1 | medium |
| 3 | `xsum\|LLAMA\|xsum-4729` | xsum-4729 | LLAMA | 169 | 204 | 2 / 2 | medium |
| 4 | `xsum\|Human\|xsum-4721` | xsum-4721 | Human | 339 | 343 | 2 / 2 | long |
| 5 | `xsum\|BigScience\|xsum-4720` | xsum-4720 | BigScience | 558 | 576 | 3 / 4 | long |

**How they were chosen.**
- The candidates are the 197 items with valid S1 and T1 and no S2 yet.
- For the T1-word quantiles 0.05, 0.30, 0.50, 0.75 and 0.95, the selector takes the nearest candidate, using each source label and each article at most once.
- The valid T1 lengths run from 27 to 940 words, with a median of 169.
- Item 5 has four segments in the es→en direction, so it also exercises a full batch of 4.

**Excluded:** the three invalid chains, `xsum|BigScience|xsum-4713`, `xsum|LLAMA|xsum-4714` and `xsum|LLAMA|xsum-4733`. A test confirms they cannot be continued: 0 operations, and the plan shows them as "blocked".

---

## 1. Build and create the two private Kaggle datasets

**Prerequisites**
- A Kaggle account that is phone-verified. Kaggle requires this for GPU and Internet.
- The Kaggle CLI (`pip install kaggle`) with your API token in `~/.kaggle/kaggle.json`. Never put that file inside the repository; the packaging script refuses to build if it finds one.

```bash
# 1. local tests (CPU only, ~20 s)
Sam/.venv/bin/python -m pytest -q Sam/tests            # expect: 171 passed

# 2. build the upload folders with YOUR Kaggle username (nothing is uploaded)
Sam/.venv/bin/python Sam/scripts/package_kaggle_inputs.py --kaggle-user <your-kaggle-username>
```

This creates the two folders below.

| Folder | Contents | Size |
|---|---|---|
| `Sam/build/kaggle/sam-sot-mt-code/` | `Sam/src/sot_mt/*.py` (16 files), `Sam/configs/*.yaml`, `Sam/scripts/*.py`, `Sam/notebooks/00_generate_translations.ipynb`, `Sam/data/manifests/{stage3b_madlad_smoke5.csv, xsum_madlad200_ids.csv}`, `Sam/requirements.txt`, `dataset-metadata.json` | 29 files, ≈262 KB |
| `Sam/build/kaggle/sam-sot-mt-staged/` | `t0_manifest.parquet`, `generations.parquet`, `STAGED_FILES.json`, `dataset-metadata.json` | 4 files, ≈1.04 MB |

**What the script checks**
- The staged hashes match `STAGED_FILES.json`, both before and after copying.
- No text-bearing table goes into the code dataset.
- Every text file is scanned for credentials: Hugging Face, Kaggle, AWS, GitHub and API keys, private keys, e-mail addresses, and files named like credentials. Any hit aborts the build.
- The licence is declared `"unknown"`: the upstream corpus repository has no LICENSE file, and none is invented.

```bash
# 3. create the datasets. The Kaggle CLI makes them PRIVATE unless you pass --public; do not pass it.
kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-code --dir-mode zip
kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-staged

# 4. check
kaggle datasets list --mine          # both listed; on the website each dataset page shows "Private"
```

`--dir-mode zip` is needed because the code folder has subdirectories; the CLI's default skips folders. Kaggle normally unpacks the uploaded `Sam.zip`. If it does not, the notebook finds the zip and extracts it into `/kaggle/working/sam_code` itself.

To update the code later, rebuild, then upload a new version: `kaggle datasets version -p Sam/build/kaggle/sam-sot-mt-code --dir-mode zip -m "describe change"`.

*Alternative without the CLI:* on kaggle.com, go to Datasets → New Dataset and upload the folder contents, keeping visibility Private.

---

## 2. Import the notebook

1. On kaggle.com, open Code → New Notebook.
2. Choose **File → Import Notebook**, and upload `Sam/notebooks/00_generate_translations.ipynb`. A copy also sits inside the code dataset.
3. Rename it, for example to `sam-sot-mt-generate`. The name becomes the slug of its saved output (§9).

## 3. Attach both datasets

In the right panel, choose **Add Input → Your Work / Datasets**, then add `sam-sot-mt-code` and `sam-sot-mt-staged`.

You don't need to edit `CODE_DATASET` or `STAGED_DATASET`. The notebook searches all of `/kaggle/input` for the marker files `sot_mt/__init__.py` and `STAGED_FILES.json`, so it works whatever mount path Kaggle uses.

## 4. Select a GPU

Under **Settings → Accelerator**, choose **GPU T4 x2**, or GPU T4 x1 if offered. Only the first GPU is used.

- **Avoid P100.** Recent PyTorch CUDA 12.8 builds dropped Pascal (sm_60) kernels. The team's last Kaggle record shows torch 2.11 with cu128, so a P100 may report CUDA as available and then fail with "no kernel image". This is unverified for the current Kaggle image.
- **Without a GPU,** §3 of the notebook stops with "No CUDA GPU in this session", and the adapters raise `NoGPUError` before downloading anything.

## 5. Enable Internet

Turn on **Settings → Internet**. This is needed to download the model on first use (≈12 GB of fp32 safetensors, not verified) and for any `pip install` in §2 of the notebook. The model goes to the Hugging Face cache in `~/.cache/huggingface`, **not** to `/kaggle/working`, so it never ends up in the saved output.

---

## 6. Run the five-item MADLAD test

**Configuration cell (§1).** Change **one** line from the defaults:

```python
ALLOW_MODEL_DOWNLOAD = True
```

Everything else already matches the smoke test:

```python
TRANSLATOR = "madlad-3b"
ITEM_IDS_FILE = "data/manifests/stage3b_madlad_smoke5.csv"
ROUNDS = (1, 3)
MOCK = False
USE_HF_TOKEN = False
PIN_MADLAD_ROUND1_LIBS = False
CHECK_TOKENIZER_PARITY = True
DRIFT_PROBE = False
PREVIOUS_EXPORTS = "auto"
```

Then choose **Run All**. What each section should show:

| Section | Expected |
|---|---|
| §2 Dependencies | versions of torch, transformers, accelerate, sentencepiece, protobuf, pandas, pyarrow and pyyaml, with nothing MISSING. A missing support package is pip-installed and listed; torch never is. |
| §3 Environment | `Tesla T4`, `CUDA: True`, the note "no bf16 … float16", and the transformers version |
| §4 Inputs | the `code:` and `staged:` paths, `previous exports: none`, `manifest: 200 texts, 31 articles \| base table: 400 records`, `5 selected items` and the table above |
| §6 Plan | config `705a8bd4992e`, the frozen MADLAD settings `{num_beams: 1, do_sample: False, max_new_tokens: 512}`, budget 300, batch 4, float16. Plan table: S1 and T1 valid 5; S2 ready_now 5; T2, S3 and T3 waiting 5; blocked 0. |
| §7 Tokenizer parity | `segment counts equal to round 1: 400/400` (tokenizer only, no GPU memory) |
| §8 Translate | `S2: 5 operations`, `T2: 5 …`, `S3: 5 …`, `T3: 5 …`, then `{'status': 'completed', 'operations': 20, 'valid': …, 'invalid': …, 'failed_attempts': 0}` |
| §9 Validation | attrition, the validation report (20 rows), item flags, failures (expected empty) and the five chains printed T0 → T3 |
| §10 Export | the list of files with sizes and hashes, plus the zip path |

**If something goes wrong**
- **Parity is below 400/400.** The notebook stops before translating, because the tokenizer differs from round 1. Set `PIN_MADLAD_ROUND1_LIBS = True`, choose **Run → Restart & clear cell outputs**, then **Run All**. This installs `transformers==4.46.3` and `sentencepiece==0.2.0` (the round-1 versions) before transformers is imported. pip will also downgrade `tokenizers` (<0.21) and `huggingface-hub` (<1.0). If that pin fails on Kaggle's torch, report back rather than skipping the check.
- **Optional drift probe.** Set `DRIFT_PROBE = True` to re-translate T0 → S1 for the five items and compare with the stored round-1 S1. This is 5 extra GPU translations, saved only to `drift_probe.csv` and never written to the generation table. `identical_to_round1_S1 = True` means the Kaggle environment reproduces round 1 exactly. Recommended, but it goes beyond the 20 operations, so it is off by default.
- **The session dies mid-run** (same session). Re-run from §6, or Run All. The work dir `/kaggle/working/sam_outputs/runs/madlad-3b__mt-default/` keeps every completed operation, so only the remainder runs.

The run time has not been measured; §9 reports `seconds` per record.

---

## 7. Inspect the generated translations

- **§9 of the notebook** prints each of the five chains (first 400 characters per stage; raise `SHOW_CHARS` for more).
- **Full text of one item** (new cell):
  ```python
  item = "xsum|BigScience|xsum-4720"
  for st, g in pw[pw.item_id == item].set_index("output_stage").iterrows():
      print(f"--- {st} [{g.validation_status}] {g.validation_reason or ''}\n{g.output_text}\n")
  ```
- **Files:** `new_translations.csv` (input and output text of all 20 new records) and `validation_report.csv` (no text).

**What to check**

| Check | Where | Healthy |
|---|---|---|
| status | `validation_report.csv` `validation_status` | all `valid`, or each `invalid` has a reason worth reading |
| length cap | `finish`, `segment_hit_eos` | `stop`, all `True` (no `engine:generation_cap`) |
| language | read S2/S3 (Spanish) and T2/T3 (English) | no untranslated or mixed-language output (`check:wrong_language`) |
| loops | `validation_reason` | no `check:repetition` |
| length | `length_ratio` | roughly 0.8–1.3 per step; warnings outside 0.6–1.8 |
| segments | `n_segments` | close to the round-1 counts in §0 |
| fp16 trouble | text | no empty or garbage segments (`engine:empty_segment`) |
| memory | `oom_retries`, `failures.parquet` | 0 and empty |
| environment | `runs.parquet` → `runtime` | GPU name, torch and transformers versions, `library_mismatch` against round 1 (`transformers 4.46.3`) |

---

## 8. Export the results

§10 writes to `/kaggle/working/sam_outputs/export/madlad-3b__mt-default/`:

| File | Contents |
|---|---|
| `generations_merged.parquet` | Stage 2's 400 records plus the 20 new ones (**420**). This is the input to the next session. |
| `generations_delta.parquet` | only the 20 new records (valid and invalid), with `generation_meta` JSON: segments, EOS flags, batch sizes, seconds, attempt |
| `failures.parquet` | failed attempts, including earlier sessions'; only written if there are any |
| `runs.parquet` | one row per run: status, settings, runtime versions, GPU, library mismatches |
| `validation_report.csv` | one text-free row per new record |
| `attrition.csv` | valid / invalid / blocked / pending per stage for the MADLAD pathway |
| `new_translations.csv` | item, stage, status, input and output text of the new records |
| `tokenizer_parity.csv` | 400 rows: round-1 vs session segment counts |
| `drift_probe.csv` | only if `DRIFT_PROBE = True` |
| `RESUME.json` | sha256 of every file above, record counts and attrition |

It also writes `/kaggle/working/sam_outputs/export/madlad-3b__mt-default.zip` (the same folder zipped).

The work dir `/kaggle/working/sam_outputs/runs/madlad-3b__mt-default/` holds the same delta, failures and runs files. The staged input dataset is read-only and is never modified.

**Download.** In the right panel, open **Output** (`/kaggle/working`) → `sam_outputs/export/` → `madlad-3b__mt-default.zip`, and choose Download. Keep it outside git, for example in `Sam/runs/kaggle/stage3b/` (gitignored), because it contains corpus-derived text.

---

## 9. Recover and resume in a later session

**`/kaggle/working` does not survive the end of a session.** When an interactive session stops (closed, idle timeout or quota), its files are gone unless you saved a version or downloaded them. Use one of the two methods below.

### Method A: save a notebook version and attach its output (recommended)

1. Choose **Save Version → "Save & Run All (Commit)"**. Kaggle re-runs the whole notebook top to bottom in a fresh background session, with the same accelerator, Internet and inputs. When it finishes, that version's **Output** tab holds `/kaggle/working`, including `sam_outputs/export/madlad-3b__mt-default/`.
   - A commit is a *separate execution*: it performs the 20 operations itself, under its own run_id. Its output, not the interactive one, is the persistent copy.
   - Do not rely on "Quick Save" for outputs.
2. In the next session, in any notebook, choose **Add Input → Your Work → Notebooks**, then pick `sam-sot-mt-generate` and the version.
3. Keep `PREVIOUS_EXPORTS = "auto"`. §4 searches `/kaggle/input` for `RESUME.json` files of **this pathway** (`["madlad-3b", "mt-default"]`) and prints `previous exports: [...]`. You can also list the folder explicitly, for example `["/kaggle/input/sam-sot-mt-generate/sam_outputs/export/madlad-3b__mt-default"]`; the exact mount path may differ, so copy it from the printed path or the input panel.

### Method B: upload the downloaded export as a private dataset

```bash
mkdir -p Sam/runs/kaggle/stage3b/madlad-export && cd Sam/runs/kaggle/stage3b/madlad-export
unzip ~/Downloads/madlad-3b__mt-default.zip
cat > dataset-metadata.json <<'EOF'
{"title": "sam-sot-mt-madlad-export", "id": "<your-kaggle-username>/sam-sot-mt-madlad-export",
 "licenses": [{"name": "unknown"}], "description": "Private working outputs for a CS6120 course project."}
EOF
kaggle datasets create -p .          # private by default
```

Attach it with Add Input, and `"auto"` finds it.

### How resuming avoids duplicate operations

The notebook loads `generations_merged.parquet` (Stage 2 plus everything done so far) as its base table, and `failures.parquet` so retry counts carry over. The runner then:
- plans only what `eligible_inputs` reports as ready (the next stage whose valid predecessor exists in the same pathway);
- skips every operation that already has a record;
- re-running a completed export does **0 operations**. Both are tested: `test_new_session_resumes_from_attached_notebook_output` and `test_export_and_resume_in_new_session`.

Inserts are idempotent, so identical records merge. **Different content under the same key raises `ConflictError`** instead of overwriting.

Consequence: attach exports from **one lineage only**. Suppose you attach both the interactive-session download (Method B) and the committed version (Method A). If GPU nondeterminism made any S2 differ between the two runs, loading fails with `ConflictError`. Choose one, normally the committed version.

**Same session:** re-run from §6. The work dir is still there, and only the remaining operations run.

**Back to the local workspace:** keep the downloaded folder under `Sam/runs/` (gitignored). `Sam/scripts/run_translations.py --base <export>/generations_merged.parquet --prior-failures <export>/failures.parquet` builds on it. Nothing is merged into `Sam/data/staged/` automatically; Stage 2 files are never overwritten.

---

## 10. What to send back for review

Send `runs.parquet`, `validation_report.csv`, `attrition.csv`, `tokenizer_parity.csv` (and `drift_probe.csv` if run), and the §2/§3 printout (versions, GPU), or just the export zip. Before any full run, I will check memory, seconds per operation, flag rates, segment counts, version drift and the text itself.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `project code not found` | code dataset not attached, or still zipped | attach `sam-sot-mt-code`; the notebook lists `/kaggle/input` contents to help |
| `No CUDA GPU in this session` / `NoGPUError` | accelerator off | Settings → Accelerator → GPU T4, restart |
| `OSError … local_files_only` / cannot find model | `ALLOW_MODEL_DOWNLOAD = False` on a fresh session | set it True, with Internet on |
| `MADLAD tokenizer differs from round 1` | transformers/tokenizer drift | `PIN_MADLAD_ROUND1_LIBS = True`, restart, Run All |
| `transformers is already imported … restart` | pin requested after import | Restart & clear outputs, Run All |
| `ConfigMismatchError` | `Sam/configs/translators.yaml` changed for madlad-3b | do not edit the frozen entry; re-upload the code dataset from the repository |
| `ConflictError` on load | two exports with different text for the same record | attach one lineage only (§9) |
| `no kernel image is available` | P100 with a torch build without Pascal support | use GPU T4 |
| `RuntimeError: …sha256… staged inputs changed` | staged dataset modified | re-create it from `package_kaggle_inputs.py` |
