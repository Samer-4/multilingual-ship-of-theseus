# Implementation Plan: Multilingual Ship of Theseus (Sam/ workspace)

This plan builds on the findings in `Sam/PROJECT_AUDIT.md`. Labels used throughout:

- **EXISTS**: already in the repository.
- **REUSE**: existing work, imported or wrapped.
- **NEW**: to be written.
- **DECISION**: needs your sign-off before the code that depends on it is written.

---

## Status (updated after Stage 2, 2026-10-30)

Stage 2 covers milestones M1 and M2 below. It ran CPU-only, with no models and no generations.

### Approved decisions

These replace the matching recommendations in §2, §5 and §8.

| # | Decision |
|---|---|
| 1 | Qwen **free** prompt is the primary pathway (`qwen3-4b/free`); **literal** is a separate pathway (`qwen3-4b/literal`). |
| 2 | NLLB = `facebook/nllb-200-distilled-1.3B` (provisional, configurable in `configs/translators.yaml`). |
| 3 | Fourth model = `meta-llama/Llama-3.2-3B-Instruct` (provisional; access and quality unverified). |
| 4 | MADLAD round 1 is **reused**, not regenerated. |
| 5 | Invalid outputs are kept, with reasons. **An invalid generation is never fed into another round.** Later rounds run only for valid records. Reports show complete-case comparisons plus attrition by model and round. This supersedes the "continue chains on flagged items" recommendation in §2.3. |
| 6 | Full original text is preserved. Shared-opening removal will be a **controlled classifier ablation** later, not a preprocessing step, and translated outputs are never cut by word count. This supersedes the §5 prefix policy. |
| 7 | The article (`group_id = dataset\|article_id`) is the statistical grouping unit. |

### Implemented (Stage 2)

| Component | Files | Verified by |
|---|---|---|
| Path resolution (env → Kaggle → Sam default) | `src/sot_mt/paths.py` | `tests/test_grouping_and_paths.py` |
| Corpus staging (commit-checked, hashed, read-only copy, gitignored) | `src/sot_mt/corpus.py`, `data/raw/sot_corpus/STAGED.json` | build script |
| Immutable T0 manifest: 200 texts, exact legacy order | `src/sot_mt/manifest.py`, `data/staged/t0_manifest.parquet`, `data/manifests/xsum_madlad200_ids.csv` | `tests/test_manifest.py` |
| Long-form generation schema, composite key, idempotent upsert, chain rules, attrition, complete cases | `src/sot_mt/schema.py` | `tests/test_schema.py`, `test_chain.py`, `test_upsert.py` |
| Output checks (blocking and warnings) | `src/sot_mt/checks.py` | `tests/test_checks.py` |
| MADLAD round-1 import (400 records = 200 × S1/T1) | `src/sot_mt/legacy_madlad.py`, `data/staged/generations.parquet` | `tests/test_madlad_import.py` |
| Metric regression, using the worker's own metric code extracted from the notebook | `src/sot_mt/regression.py`, `reports/stage2/metric_regression_*.csv` | `tests/test_metric_regression.py` |
| Metric registry with namespaced ids, formulas, and raw vs normalized kinds | `configs/metric_registry.yaml` | n/a |
| Translator and generation configs (MADLAD and Qwen verified from code; NLLB and Llama provisional) | `configs/translators.yaml`, `data/tables/generation_configs.csv` | n/a |
| Build script and generated report | `scripts/build_stage2.py`, `reports/stage2/STAGE2_REPORT.md` | 65/65 tests pass |

### Stage 2 results

All numbers come from `reports/stage2/STAGE2_REPORT.md`.

**Manifest:** 200 texts, 31 articles, 7 source labels (29/29/29/29/28/28/28), no duplicate texts. The selection reproduces the legacy MADLAD CSV exactly: same IDs, same order, byte-identical T0.

**MADLAD import:** 400 records. Per stage, S1 and T1 each have 197 valid and 3 invalid.

- The 3 flagged items keep their legacy reasons.
- Both of their rows are marked invalid with `legacy:direction_unattributed`, because the legacy `generation_cap` flag does not say which direction failed.
- The new repetition check independently confirms decoding loops in the T1 of all 3 items and in the S1 of `xsum-4713`.
- No legacy-valid row triggered any blocking check.
- 197 items are eligible for round 2.

**Metric regression:**

- ROUGE-L, chrF, word retention, function-word cosine, punctuation F1 and exact match reproduce **bit-for-bit (197/197)**.
- BLEU matches within 7.8e-16 on 41 items and exactly on 156. The most likely cause is `exp`/`log` rounding differences between Linux (the original run) and macOS.
- All README medians match to 3 decimals.
- E5 cosine and BERTScore F1 are **pending**: there is no code, no per-item values, and no weights were downloaded.

**Name clashes:** on these pairs, `pilot.bleu` and `pilot.rougeL` equal the MADLAD versions on every item. `pilot.word_retention` differs on every item (median 0.793 vs 0.836).

## Status (updated after Stage 3A, 2026-10-30): translation engine, no GPU run yet

### Implemented (Stage 3A)

| Component | Files | Verified by |
|---|---|---|
| Deterministic segmentation: `madlad_pack` (exact replica of the pilot worker's `split_chunks`), `sentence` (NLLB), `none` (LLMs); reconstruction by single-space join; per-segment metadata | `src/sot_mt/segmentation.py` | `tests/test_segmentation.py`: identical to the worker's own function, extracted from the notebook, on 1,200 random documents + 40 real texts |
| Four adapters + mock, one interface `translate(text, src, tgt)`. Lazy torch/transformers imports; `local_files_only` unless downloads are enabled; one loaded model at a time; OOM halves the batch, then raises a transient error; no partial outputs | `src/sot_mt/translators.py` | `tests/test_translators.py` (model-touching methods replaced by fakes) |
| Gated-model guard (Llama needs `use_hf_token=True` + `HF_TOKEN`); import-only legacy Qwen pathway | `translators.build_translator`, `configs/translators.yaml` | same |
| Resumable stage-major runner: selection filters, eligible-input planning, per-operation validation, atomic checkpoints, separate failures log with retry limit, run log, export/resume, MOCK-data quarantine | `src/sot_mt/runner.py` | `tests/test_runner.py` (19 tests) |
| Shared setup for the notebook and CLI (staged-hash verification, base loading, overview) | `src/sot_mt/session.py`, `scripts/run_translations.py` | CLI mock run on real staged data |
| Optional legacy Qwen importer (separate pathway; missing file → run from T0) | `src/sot_mt/legacy_qwen.py` | `tests/test_runner.py` |
| Kaggle notebook | `notebooks/00_generate_translations.ipynb` | all code cells executed locally in `MOCK = True` mode against the real staged data |
| Kaggle upload folders (built locally, nothing uploaded) | `scripts/package_kaggle_inputs.py` | built into scratch, hashes verified |
| Schema extensions | optional `generation_meta` column (older tables read with `None`); one `generation_config_id` per pathway; new checks `roundtrip_unchanged` and `meta_text` (warnings), `engine:*` codes (blocking) | `tests/test_schema.py`, `test_chain.py`, `test_checks.py` |

**Totals:** 110 tests pass. The Stage 2 staged files are byte-identical (sha256 `a636b3ed…` / `27ab66d9…`). The MADLAD `generation_config_id` is still `705a8bd4992e`.

### Design decisions made in Stage 3A (please review)

1. **Failures vs invalid records.**
   - Deterministic problems become **invalid records**: segmentation impossible, an empty segment, a missing EOS (`engine:generation_cap`, as in the pilot), or a blocking check. They are final and block the chain.
   - Transient problems (CUDA OOM, runtime errors) become **failed attempts** in `failures.parquet`. They are retried, at most twice by default.
2. **Engine is part of the pathway.** Legacy Qwen outputs (vLLM) can only be imported as `qwen3-4b-vllm-legacy`. The `qwen3-4b` pathway (transformers) starts from T0, so one chain never mixes engines.
3. **NLLB** uses the same sentence regex as MADLAD, one sentence per segment, with a 256-token budget. pysbd was not adopted: it would add a dependency, and the regex keeps the two MT pathways comparable.
4. **LLMs run batch size 1** (no padding effects on greedy output). This is slower; throughput gets measured in the smoke test.
5. **MADLAD config is frozen.** Runtime differences (T4 vs V100, transformers version) are recorded per run in `runs.parquet`, not folded into the config id.

### Remaining

| Stage | Work | Blockers |
|---|---|---|
| 3B | Kaggle smoke test (MADLAD first, see the runbook): 5 documents per pathway (MADLAD 20 ops, others 30 ops each); confirm memory, throughput, output quality, flags | Your approval; private Kaggle datasets uploaded; Internet on; Llama licence + `HF_TOKEN` |
| 4 | Full generation: MADLAD S2–T3 for 197 eligible items; Qwen free (+ literal ablation); NLLB; Llama | Kaggle GPU quota; outcome of 3B |
| 5 | `texts.parquet` (four pathways + corpus paraphrases for the same IDs), RQ1 metrics and notebook | Stage 4 |
| 6–7 | RQ2 (shared-opening ablation), RQ3 (GroupKFold by `group_id`; segmentation confound) | Stage 4 |

## Status (updated after Stage 3B preflight, 2026-10-30): ready for the Kaggle smoke test, not yet run

**Smoke items.** Five valid MADLAD items (`data/manifests/stage3b_madlad_smoke5.csv`, rule in `src/sot_mt/smoke.py`):

| item | T1 words | band |
|---|---|---|
| OpenAI/xsum-4710 | 50 | short |
| Tsinghua/xsum-4719 | 104 | medium |
| LLAMA/xsum-4729 | 169 | medium |
| Human/xsum-4721 | 339 | long |
| BigScience/xsum-4720 | 558 | long |

They are distinct sources and articles, none of them invalid, and they give 20 operations.

**Preflight fixes**
- **Adapters:** `NoGPUError` before any load; a version-aware dtype keyword (`torch_dtype` below 4.56, `dtype` from 4.56; checked against the 4.46.3, 4.55.4, 4.56.0 and 5.16.1 wheel sources); a tokenizer-only `load_tokenizer()`.
- **Runner:** a load failure is recorded as `failed:load:*`, not left as `running`. The export carries earlier sessions' failures and adds `validation_report.csv`, `attrition.csv` and `new_translations.csv`. The `RESUME.json` hint is corrected.
- **Notebook** (now generated by `scripts/build_notebook00.py`):
  - a controlled dependency cell, with an optional round-1 pin;
  - a hard GPU check;
  - an item-ID file;
  - a MADLAD tokenizer parity gate and an optional drift probe;
  - `PREVIOUS_EXPORTS = "auto"`;
  - a zipped export;
  - extraction of the code if Kaggle keeps it as `Sam.zip`.
- **Packaging:** includes `data/manifests`, declares the licence `unknown`, scans for credentials, and re-verifies the staged hashes after copying.
- **CLI:** `--item-ids` / `--item-ids-file`.

**Tests:** 130 pass. Among them, the notebook runs end to end (MOCK) as a subprocess from the packaged folders under a simulated `/kaggle/input`, both unzipped and zipped. A second simulated session resumes from the first session's attached output with 0 operations. The staged files are unchanged.

**Runbook:** `reports/stage3/KAGGLE_SMOKE_TEST.md`.

**Terminal workflow (added before the first launch).** The pieces:
- `scripts/prepare_kaggle_deploy.py` writes `kaggle/madlad_smoke5/`: the deployed notebook plus `kernel-metadata.json`, validated against the CLI 2.2.5 push rules.
- `scripts/launch_kaggle.py` is a dry run unless `--yes`: it uploads the code version, waits until it is ready, then pushes.
- `scripts/monitor_kaggle.py` follows the log stream (falling back to the stored log) and polls status.
- `scripts/download_kaggle_outputs.py` keeps the raw download, verifies it against `RESUME.json` and Stage 2, and files it into `data/generated/xsum/madlad-3b/` and `results/madlad_smoke5/`.
- The runner emits `SOTMT_PROGRESS` JSON events (`src/sot_mt/progress.py`).
- The code dataset carries `CODE_MANIFEST.json`, and the notebook refuses a code build that doesn't match.

160 tests pass. Not yet launched.

---

## 0. Principles

1. **One generation step, one shared dataset.**
   - Only `00_generate_translations.ipynb` calls translation models.
   - It writes to a single store, `Sam/data/translations/`.
   - A consolidation step builds one analysis table, `Sam/data/processed/texts.parquet`.
   - Notebooks 01, 02 and 03 read only that table, plus the corpus for the baseline and the RQ2 training pool. They never regenerate.
2. **Chains are strictly sequential, and each chain uses one model.**
   - Round *r* takes as input the English output of round *r−1* from the **same pathway**.
   - The validators enforce this. It is not left to convention.
3. **One metric code path for translation and paraphrase.**
   - RQ1 scores the four translator pathways and the four English paraphrasers with the same functions, on the same document IDs.
   - The metric functions are imported from the team's `src/pilot/metrics.py`, never copied or edited.
   - The pipeline records the git commit of `pilot` in every output.
4. **Nothing outside `Sam/` is modified.**
   - Team code is imported read-only.
   - Legacy outputs are read and converted into Sam's schema. The original files are left untouched.
5. **Everything runs from a config file.** It records the checkpoint plus its HF revision hash, the prompts, the decoding settings, the seed, the hardware and the library versions.

---

## 1. Directory structure

**As built in Stage 2:** `src/sot_mt/{paths,files,corpus,manifest,schema,checks,legacy_madlad,regression,summaries}.py`, `scripts/build_stage2.py`, `reports/stage2/`, `data/{raw,staged,manifests,tables}/`, `configs/{experiment,translators,metric_registry}.yaml`, and `tests/`.

- The original proposal is kept below for the stages still to come.
- `store.py` and `config.py` are not needed: their jobs are done by `schema.upsert` + parquet, and by `files.load_yaml`.
- Text-bearing tables live in `data/staged/` (gitignored), not `data/processed/`.

```
Sam/
├── PROJECT_AUDIT.md
├── IMPLEMENTATION_PLAN.md
├── README.md                          (M1) how to run locally and on Kaggle
├── .gitignore                         (M1) ignores data/raw/, data/cache/, *.parts/
├── configs/
│   ├── experiment.yaml                seed, corpus commit, sample definition, rounds, languages,
│   │                                  flag thresholds, prefix/flag policies, analysis-set rule
│   ├── translators/
│   │   ├── qwen3_4b.yaml              checkpoint, revision, prompts, decoding, engine
│   │   ├── madlad_3b.yaml
│   │   ├── nllb.yaml                  (DECISION: checkpoint)
│   │   └── llama.yaml                 (DECISION: checkpoint)
│   └── metrics.yaml                   metric models, floor settings, bootstrap n, MFW sizes
├── src/sot_mt/                        package name chosen so it cannot clash with the team's `pilot`
│   ├── __init__.py
│   ├── paths.py                       local vs Kaggle path resolution (no absolute paths in code)
│   ├── config.py                      load + validate YAML; merge translator configs
│   ├── corpus.py                      load corpus/train CSVs; T0 selection; prefix stripping
│   ├── manifest.py                    reproduce the 200-item sample; generalised N-per-dataset sampler
│   ├── schema.py                      column specs, dtypes, validators (§3)
│   ├── store.py                       append-only JSONL writer, resume index, parquet consolidation
│   ├── legacy.py                      import MADLAD chain CSV (and the Qwen chain CSV, if recovered)
│   ├── chunking.py                    sentence split + token-budget packing (MADLAD-compatible)
│   ├── translators/
│   │   ├── base.py                    Translator protocol: translate(texts, src, tgt) -> results + metadata
│   │   ├── seq2seq.py                 MADLAD (<2es> prefix) and NLLB (src_lang / forced_bos)
│   │   ├── chat_llm.py                Qwen and Llama via chat template, greedy
│   │   └── dummy.py                   deterministic fake translator for CPU tests
│   ├── chain.py                       round-major runner: for r in 1..3: EN→ES, then ES→EN; resumable
│   ├── validate.py                    output flags: empty, truncated, wrong language, length ratio,
│   │                                  meta-text, repetition
│   ├── metrics/
│   │   ├── pairs.py                   cumulative/step pairs over texts.parquet (any method)
│   │   ├── lexical.py                 wraps pilot.metrics.lexical_scores (+ chrF)
│   │   ├── semantic.py                wraps pilot LongBertScore + ChunkedSentenceEmbedder
│   │   ├── style.py                   style blocks + char n-grams (logic lifted from sot_analysis.ipynb)
│   │   ├── floors.py                  unrelated-pair floors; retained-share normalisation
│   │   └── legacy_madlad.py           the MADLAD worker's metric set, used only for regression tests
│   ├── attribution.py                 RQ2 features, classifiers, evaluation
│   ├── fingerprint.py                 RQ3 grouped cross-validation, permutation tests, top features
│   ├── stats.py                       key-cluster bootstrap (wraps pilot.bootstrap_mean_ci), paired tests
│   └── plots.py                       shared figure style; every figure also exports its CSV
├── notebooks/
│   ├── 00_generate_translations.ipynb
│   ├── 01_RQ1_metrics.ipynb
│   ├── 02_RQ2_attribution.ipynb
│   └── 03_RQ3_fingerprints.ipynb
├── data/
│   ├── raw/                           (gitignored) corpus snapshot at commit 4423d8c
│   ├── manifests/
│   │   └── xsum_madlad200.csv         doc_uid, dataset, source, key, sample_rank, t0_sha1, ...
│   ├── translations/
│   │   └── <pathway_id>/generations.jsonl   append-only; one line per generation call
│   ├── processed/
│   │   ├── translations.parquet       consolidated long table (§3.2)
│   │   ├── texts.parquet              analysis table read by 01/02/03 (§3.3)
│   │   └── validation_report.csv
│   └── cache/                         (gitignored) metric caches, embeddings
├── tests/
│   ├── conftest.py                    tiny fixture corpus (a few rows per source), no network
│   └── test_*.py                      (§6)
└── figures/
    ├── rq1/ rq2/ rq3/                 PNG + matching CSV for every figure
```

**Data size.** 200 docs × 4 pathways × 3 rounds × 2 directions comes to **4,800 generations**, a few MB of text. The processed tables are small enough to commit. **[ASSUMED]** The corpus licence is unclear (audit §9). Keep the corpus snapshot in `data/raw/` gitignored. **DECISION:** whether derived texts get committed to this (private?) repository.

---

## 2. Translation architecture

### 2.1 Pathways

| `pathway_id` | Checkpoint | Family | Input unit | Status |
|---|---|---|---|---|
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | General LLM | Whole document, chat prompt | EXISTS: round-1 metrics only; texts on Drive |
| `madlad-3b` | `google/madlad400-3b-mt` | Specialised MT | Sentence-split, ≤ 300-token chunks | EXISTS: round 1 (S1/T1) for 200 items |
| `nllb` | **Recommended:** `facebook/nllb-200-3.3B` (already PROVISIONAL in the team `config.yaml`). Fallback: `facebook/nllb-200-distilled-1.3B` if T4 throughput is a problem. | Specialised MT | Sentence-split (pysbd), chunk-packed like MADLAD | NEW |
| `llama` | **Recommended:** `meta-llama/Llama-3.2-3B-Instruct`. It fits one T4 in fp16, lists Spanish as supported, and sits at the same ~3–4B scale as the other three, so the 2 LLM vs 2 MT comparison is not confounded by model size. Alternative: `meta-llama/Llama-3.1-8B-Instruct`, which is stronger but needs 2×T4 and gives up size parity. | General LLM | Whole document, chat prompt | NEW |

Both recommendations are **DECISIONS**. The licence and model-card facts in this table are **[ASSUMED]** and must be confirmed on Hugging Face before the configs are written.

### 2.2 Prompts and decoding

- **LLM pathways (Qwen, Llama).** Use the Qwen chain notebook's **F (free)** forward prompt and its **back** prompt word for word, so that both LLMs receive identical instructions. Wording is in audit §4.2.
  - **DECISION:** F as the main pathway.
  - **Recommendation:** F, because it is the natural-translation condition and the fairest counterpart to MT systems. L stays an optional round-1 ablation for Qwen only.
  - Decoding: greedy (`temperature=0`), `max_new_tokens = 2·n_in + 64` capped by the context length, `<think>` stripping, and the same system/user role layout.
- **MT pathways (MADLAD, NLLB).**
  - Greedy (`num_beams=1`).
  - `max_new_tokens=512` per chunk.
  - MADLAD's regex sentence split plus 300-token packing, reproduced exactly in `chunking.py`.
  - NLLB codes `eng_Latn` and `spa_Latn`. NLLB was trained on sentence-level data, so it uses the same chunker with a pysbd split.
  - **DECISION:** keep MADLAD's regex splitter for MADLAD, required for round-1 consistency, and use pysbd for NLLB. Or use one splitter for both.
- **Seeds:** `seed: 6120` (the team convention) is set globally. Greedy decoding makes the seed nearly irrelevant, but it is recorded anyway.
- **Precision:** fp16 on T4 (T4 has no bf16). Record the dtype per row.

### 2.3 Chain runner (`chain.py`)

```
for round r in 1..3:
    inputs = T0 (r=1) or the same pathway's T_{r-1} (r>1)   ← validated by doc_uid + input_sha1
    S_r = translate(inputs, en→es)   checkpoint every item
    T_r = translate(S_r,   es→en)    checkpoint every item
```

- **Round-major order.** All 200 items finish round *r* before round *r+1* starts. This allows batching and gives a clean resume point.
- **Resume.** The runner skips any `(pathway_id, doc_uid, round, direction)` already in `generations.jsonl` whose `input_sha1` matches the expected input. If an input hash disagrees, the runner raises an error instead of silently mixing chains. The MADLAD worker followed the same rule.
- **Flagged items.** **DECISION.** The recommended policy:
  - Always continue the chain, so that every pathway has 200 × 3 rounds.
  - Mark each row `valid` and set `chain_valid_through`.
  - The **core analysis set** is the documents valid at every round in all four pathways. Sensitivity tables report per-pathway n.
  - Never change decoding settings for individual items inside a pathway.
- **Legacy reuse (MADLAD).** **DECISION.** The recommendation:
  - Import the existing V100 round 1 as `provenance="legacy:madlad_explorer_v100"`, then generate rounds 2–3 on Kaggle with identical settings.
  - Add a drift check: regenerate round 1 for about 20 items on Kaggle and report the exact-match rate and chrF against the V100 outputs.
  - If drift is material, regenerate round 1 for all 200 so that every round shares one hardware and software stack.
- **Legacy reuse (Qwen).** If `chain_es_Qwen3-4B-Instruct-2507.csv` and `run_config.json` are recovered, import `F`/`B_F` for the 200 IDs as round 1 and check them against the saved `metrics_items.csv` (§6). If not, regenerate round 1.

---

## 3. Shared data schema

The schema is general across datasets, sources, pathways, paraphrasers and rounds. Expanding to all seven datasets or adding a language means adding rows, not columns.

### 3.1 `data/manifests/<sample_id>.csv`: which documents

| column | type | example / note |
|---|---|---|
| `doc_uid` | str, PK | `xsum\|Human\|xsum-2638` (identical to the legacy `item_id`) |
| `sample_id` | str | `xsum_madlad200` |
| `sample_rank` | int | position in the round-robin order (0–199) |
| `dataset` | str | `xsum` (any of the 7) |
| `source` | str | `Human`, `OpenAI`, … (**generator/source label, not author**) |
| `source_is_human` | bool | |
| `key` | str | `xsum-2638` (the article; used for group splits) |
| `t0_text` | str | original corpus text |
| `t0_sha1` | str | |
| `t0_words` | int | |
| `prefix_words` | int | shared prefix length with the key's Human text (0 for Human) |
| `corpus_repo`, `corpus_commit` | str | `tripto03/Ship_of_theseus_paraphrased_copus`, `4423d8c` |

### 3.2 `data/processed/translations.parquet`: one row per generation call

| column | type | note |
|---|---|---|
| `pathway_id` | str | `qwen3-4b` \| `madlad-3b` \| `nllb` \| `llama` |
| `doc_uid` | str | FK → manifest |
| `round` | int | 1, 2, 3 |
| `direction` | str | `en-es` \| `es-en` |
| `input_stage` / `output_stage` | str | `T0→S1`, `S1→T1`, `T1→S2`, … |
| `input_text`, `input_sha1` | str | |
| `output_text`, `output_sha1` | str | |
| `model_checkpoint`, `model_revision` | str | HF repo + commit hash |
| `prompt_id`, `system_prompt`, `user_template` | str | null for MT models |
| `decoding` | JSON str | num_beams, do_sample, temperature, max_new_tokens rule, dtype |
| `seed` | int | |
| `chunking` | str | `none` \| `regex_sent_300tok` \| `pysbd_sent_300tok` |
| `n_chunks`, `n_in_tokens`, `n_out_tokens` | int | |
| `finish_reason` | str | `stop` \| `length` (no EOS) |
| `flags` | str | `;`-joined: `empty`, `truncated`, `wrong_language`, `compression`, `expansion`, `meta_text`, `repetition` |
| `valid` | bool | no blocking flags |
| `seconds` | float | |
| `hardware`, `lib_versions` | str / JSON | GPU name; torch / transformers / vllm versions |
| `provenance` | str | `generated:kaggle:<kernel-version>` \| `legacy:madlad_explorer_v100` \| `legacy:qwen_colab` |
| `created_at` | str | ISO timestamp |

Unique key: `(pathway_id, doc_uid, round, direction)`.

**Chain invariant, enforced in tests:**

- For `en-es` at round *r* > 1, `input_sha1` equals `output_sha1` of `es-en` at round *r−1* for the same `(pathway_id, doc_uid)`.
- For `es-en` at round *r*, `input_sha1` equals `output_sha1` of `en-es` at round *r*.

### 3.3 `data/processed/texts.parquet`: the analysis table used by 01, 02 and 03

Every text that any RQ notebook analyses, in one long table:

| column | type | note |
|---|---|---|
| `doc_uid`, `dataset`, `source`, `key` | | from manifest |
| `method_family` | str | `original` \| `translation` \| `paraphrase` |
| `method` | str | `original`, the 4 `pathway_id`s, and `chatgpt`, `palm`, `dipper`, `pegasus(full)` (optionally `dipper(low/high)`, `pegasus(slight)`) |
| `round` | int | 0 for T0; 1–3 |
| `language` | str | `en` for T_r; `es` for S_r rows (kept for optional Spanish-side analyses) |
| `stage` | str | `T0`, `S1`, `T1`, … |
| `text`, `text_sha1` | str | |
| `text_noprefix` | str | the same text with the shared 30-word prefix removed (§5) |
| `valid` | bool | translation flags; for paraphrases, the text exists and is non-empty |
| `chain_valid_through` | int | last round at which the chain is still valid |
| `in_core_set` | bool | the document is valid at T1–T3 in all 4 pathways **and** has T1–T3 for all 4 paraphrasers |

The paraphrase rows are read directly from the corpus for the same `doc_uid`s. They are never regenerated.

### 3.4 Metric outputs (`data/cache/metrics/` plus exported CSVs)

`pair_scores.parquet` columns:

- `doc_uid`, `method`, `round`, `kind` (`cumulative` = T0 vs T_r; `step` = T_{r−1} vs T_r)
- one column per metric (raw)
- `<metric>_ret`: the retained share
- `pilot_commit`, `metric_config_hash`

`floors.parquet` holds per-dataset unrelated-pair floors.

---

## 4. Kaggle execution strategy

### 4.1 What runs where

| Step | Where | Why |
|---|---|---|
| Manifest, legacy import, consolidation, tests | Local CPU | No models needed |
| 00 generation (all 4 pathways) | Kaggle GPU (T4 ×1 or ×2) | Model inference |
| 01 RQ1: BERTScore and sentence embeddings | Kaggle GPU or local | About 200 docs × 8 methods × 3 rounds × 2 kinds, roughly 10k pairs. Fast on GPU; feasible on CPU or MPS. **[ASSUMED]** |
| 01 RQ1: lexical and style metrics | Local CPU or Kaggle CPU | spaCy plus sacrebleu |
| 02 and 03 classifiers | CPU (local or Kaggle) | Linear models on n-gram features |

### 4.2 Path handling (`paths.py`)

- Detect Kaggle by checking for `KAGGLE_KERNEL_RUN_TYPE` in the environment or for the existence of `/kaggle/input`.
  - **Local:** paths are relative to `Sam/`, found by walking up from `__file__`. The corpus comes from `Sam/data/raw/` or an env var `SOT_CORPUS_DIR`. It never comes from a hard-coded sibling path.
  - **Kaggle:** inputs are read-only from `/kaggle/input/<dataset-slug>/…`; outputs go to `/kaggle/working/sam_outputs/…`.
- All paths in the configs are relative. The notebooks call `paths.resolve(...)` and nothing else.

### 4.3 Kaggle inputs

All three are private Kaggle Datasets. **DECISION:** the names.

1. **`sot-corpus-4423d8c`.** Snapshot of the upstream CSVs at commit `4423d8c`, kept private because of the licence question. It removes the dependency on `raw.githubusercontent.com/.../main`.
2. **`sam-sot-mt-code`.** Contains `Sam/src`, `Sam/configs`, `Sam/data/manifests`, the legacy round-1 imports, and a **read-only copy of the team's `src/pilot/`** with its git commit recorded, so imports work identically. The dataset is re-versioned whenever code changes, and the notebook prints the version it used.
3. **`sam-sot-mt-generations`.** Holds the previous run's `generations.jsonl` files. Each `00` run attaches the latest version, resumes, and writes updated files to `/kaggle/working`. Its output is then published as the next version. Kaggle does not persist `/kaggle/working` across sessions except as a saved notebook version.

### 4.4 Running `00_generate_translations.ipynb`

- A parameter cell sets `PATHWAY = "madlad-3b" | "nllb" | "qwen3-4b" | "llama"`, `ROUNDS = [1,2,3]`, `MAX_ITEMS` and `DRY_RUN`. **There is one pathway per session**, which isolates GPU memory, keeps logs clean and makes model mixing impossible.
- Use "Save & Run All" (background commit) for full runs, so a dropped browser does not kill the job. Interactive sessions are only for 5-item smoke tests.
- `HF_TOKEN` comes from Kaggle Secrets (`kaggle_secrets.UserSecretsClient`). Llama needs it; the other three are expected to be ungated. **[ASSUMED]**
- Enable internet for Hugging Face downloads. The cache stays in `/kaggle/working/hf_cache` or the default HF cache, and is never committed.
- **Engine:** use plain `transformers` for all four models. The team's `config.yaml` records that `pip install vllm` broke Kaggle's torch. vLLM is optional for Qwen and Llama only if a pinned, tested install is confirmed in a throwaway session first. The engine is recorded per row.
- **Environment:** no `pip install -U` of torch or transformers. Print and record the preinstalled versions. Only small packages are installed (`sacrebleu`, `pysbd`, `lingua-language-detector`).
- **Checkpointing:** append and fsync after every item, as the MADLAD worker did. A session that times out loses at most one item.
- The notebook ends with a validation summary: counts per round and direction, the flag table, and chain-invariant checks.

### 4.5 Compute estimate

All figures are **[ASSUMED]** and will be re-measured during the smoke tests.

- **MADLAD.** About 16 s per item per round trip on a V100 (measured median 17.85 s). Rounds 2–3 for 200 items come to about 1.8 h on a V100, or roughly 3–5 h on a T4. That may need 1–2 sessions.
- **NLLB-3.3B.** Probably similar to MADLAD.
- **Qwen3-4B and Llama-3.2-3B** with `transformers` batched greedy decoding, 3 rounds × 2 directions × 200 docs. Expect a few hours each on a T4. vLLM would be much faster if it can be used safely.
- **Total:** about 10–20 T4-hours, which should fit the weekly Kaggle quota, spread over several sessions.

---

## 5. Research-question designs

Every metric section in notebooks 01–03 follows the same five-part template:

1. **What it measures:** plain English, 2–4 sentences, with a toy example.
2. **How it is computed:** a short code cell calling `sot_mt.metrics.*`, with no inline math beyond a formula.
3. **Result:** a table and a figure, the latter exported to `Sam/figures/rqN/` as PNG plus CSV.
4. **What this means:** 2–4 beginner-readable sentences.
5. **Limitations and warnings:** what the metric cannot tell you.

### Preprocessing policy for all RQs

**DECISION.** The recommended policy:

- Compute everything on `text_noprefix`, removing the shared 30-word Human prefix (plus the sentence fragment it cuts into) from every source's text, Human included. `config.yaml` proposed the same symmetric treatment.
- Report the with-prefix variant as a sensitivity check.

### RQ1: decay across rounds (`01_RQ1_metrics.ipynb`)

**Methods compared**

- The 4 translator pathways.
- The 4 primary paraphrasers (ChatGPT, PaLM, Dipper, Pegasus-full) on the **same `doc_uid`s**.
- The paraphraser texts come from the corpus.

**Pairs**

- *Cumulative:* T0 vs T_r.
- *Step:* T_{r−1} vs T_r.

**Metrics, all from the same code path**

| Family | Metrics | Source |
|---|---|---|
| Lexical | BLEU, ROUGE-L, chrF, word retention (pilot definition) | REUSE `pilot.metrics.lexical_scores`; add chrF |
| Semantic | BERTScore F1 (roberta-large L17, rescaled, windowed), mpnet cosine | REUSE `LongBertScore`, `ChunkedSentenceEmbedder` |
| Style | Stylometric block cosines (surface, punctuation, function words, POS), averaged into `style`; char n-gram TF-IDF cosine | REUSE logic from `sot_analysis.ipynb` cell 9, moved into `metrics/style.py` |
| Extra | Length ratio, entity/number recall | REUSE definitions from the Qwen metrics notebook |
| Optional | COMETKiwi QE of S_r, multilingual-e5 cross-lingual T0↔S_r | Requires a gated licence; stretch goal |

**Normalisation:** retained = (s − floor) / (1 − floor), using an XSum floor of unrelated reference pairs. This reproduces the baseline notebook's seeded procedure, so the two scales match.

**Statistics:**

- Means with key-cluster bootstrap 95% CIs (`pilot.bootstrap_mean_ci`).
- Paired per-document differences: translation vs paraphrase, and pathway vs pathway, at each round.
- Decay curves in the style of `decay_curves.png`.
- Context only: show the full-corpus baseline curves (`results/sot_analysis/decay_curves.csv`) next to the matched 200-item curves, labelled as different populations.

**Warning to state explicitly:** chunked MT pathways keep sentence boundaries by construction. Style metrics that depend on sentence length partly reflect that design choice.

### RQ2: source attribution after translation (`02_RQ2_attribution.ipynb`)

**Framing.** RQ2 asks whether a classifier can still tell which *source* produced a text: Human, or one of six LLM families. It does not identify individual authors, and every caption must say so.

**Protocol A (primary):**

- Train on `xsum_train.csv`: 3,243 texts, keys disjoint from the test set (verified).
- Test on the 200 sample documents at T0 (the reference accuracy), then at T1, T2 and T3 for every translator pathway and every paraphraser.
- The question it answers: does attribution trained on clean text survive repeated rewriting?

**Protocol B (secondary, cheap):** grouped cross-validation by `key` within the sample, training and testing at the same round. This shows whether source signal still *exists* after rewriting, even if it shifted.

**Labels:** 7-class, plus binary Human vs machine.

**Models:**

- Logistic regression on char n-gram TF-IDF.
- Logistic regression on stylometric features (function words, punctuation, POS).
- Combined.
- Stretch: a fine-tuned RoBERTa on Kaggle.

**Required baselines:**

- Majority class and chance (1/7).
- **A length-only classifier**, because source is strongly confounded with length (audit §2).

**Reporting:**

- Accuracy and macro-F1 with key-bootstrap CIs.
- Confusion matrices at T0 and T3.
- Per-source recall.
- A warning about n ≈ 28 per class.

### RQ3: translator fingerprints (`03_RQ3_fingerprints.ipynb`)

**Task:** 4-class prediction of `pathway_id` from the reconstructed English T_r, run separately for r = 1, 2, 3 and also pooled.

**Content control:** every document appears once per class, so topic is balanced by design.

**Splits:** `GroupKFold` by `key`. All four pathway outputs of a document, and all sources of the same article, always land in the same fold. This is a hard requirement, tested in §6.

**Features:**

- char n-grams
- word n-grams
- function words
- punctuation
- stylometric features
- (variant) features of the change relative to T0, which is legitimate because T0 is identical across classes

**Baselines and controls:**

- Chance (25%).
- Length-only and length-ratio-only classifiers.
- A label-permutation test.
- A "MT vs LLM" binary task, to separate family-level from model-level fingerprints.

**Interpretability:** top-weighted n-grams per pathway (for example spelling variants, quote styles, number formats), with examples from the actual texts.

**Optional:** the same analysis on the Spanish S_r, and on 8 classes with the paraphrasers added.

---

## 6. Testing approach (`Sam/tests/`, pytest, CPU-only, no network)

| Test | Checks |
|---|---|
| `test_manifest.py` | The procedure reproduces the 200 `doc_uid`s **in order** from the corpus snapshot, and they equal `results/madlad_translation/chain_es_madlad4003b.csv`; the T0 text hashes match |
| `test_schema.py` | Required columns and dtypes; the unique key `(pathway_id, doc_uid, round, direction)`; no nulls in required fields |
| `test_chain_invariants.py` | The round *r* input equals the same pathway's round *r−1* output (by hash); round > 1 inputs never equal T0 (unless the output was identical); one `model_checkpoint` and `model_revision` per pathway; the same `doc_uid` set across all pathways |
| `test_legacy_import.py` | The MADLAD import preserves S/F/B_F byte-for-byte; re-running the **MADLAD worker's metric functions** on the imported texts reproduces `matched_item_metrics.csv` (197 rows, tolerance 1e-9); if the Qwen texts are recovered, the e5 `content` distance on imports reproduces `metrics_items.csv` within a tolerance |
| `test_chunking.py` | No chunk exceeds the token budget (with a mock tokenizer); split and join loses no characters other than whitespace |
| `test_validate.py` | Synthetic cases trigger each flag: empty, truncated, wrong language, compression, expansion, meta text, repetition |
| `test_prefix.py` | Prefix stripping removes exactly the shared prefix; Human texts are treated symmetrically |
| `test_metrics.py` | Identical texts give retained = 1; floor normalisation is correct on toy numbers; the wrappers return the same values as direct `pilot.metrics` calls |
| `test_splits.py` | GroupKFold folds share no `key`; every fold contains all 4 classes (RQ3) |
| `test_paths.py` | Local vs Kaggle resolution with a monkeypatched environment; no absolute paths in configs |
| `test_end_to_end_dummy.py` | 5 documents × 3 rounds through `chain.py` with the `dummy` translator, then consolidation, `texts.parquet` and one metric. This covers the whole pipeline without a GPU |

A one-cell **GPU smoke test** (5 items × 1 round per pathway) runs on Kaggle before any full run. It goes ahead only after your approval, because it downloads models.

---

## 7. Milestones

Each milestone ends with a stop for your review.

| # | Milestone | Deliverables | Needs GPU? |
|---|---|---|---|
| **M0** | Decisions | Your answers to §8 | No |
| **M1** | Foundations | `configs/experiment.yaml`, `src/sot_mt/{paths,config,corpus,manifest,schema}.py`, `data/manifests/xsum_madlad200.csv`, `Sam/README.md`, `Sam/.gitignore`, tests: manifest, schema, paths, prefix | No |
| **M2** | Legacy import | `legacy.py`; MADLAD round 1 in `translations.parquet`; regression test against `matched_item_metrics.csv`; Qwen import if the Drive files arrive | No |
| **M3** | Translator backends and runner | `translators/*`, `chunking.py`, `chain.py`, `validate.py`, `store.py`; dummy end-to-end test; translator configs with checkpoints and revisions | No (local); Kaggle smoke test after approval |
| **M4** | Generation | `00_generate_translations.ipynb`; full runs per pathway (MADLAD rounds 2–3 plus the drift check; Qwen; NLLB; Llama); `translations.parquet`, `texts.parquet`, `validation_report.csv` | **Yes** |
| **M5** | RQ1 | `metrics/*`, `01_RQ1_metrics.ipynb`, `figures/rq1/` | Optional (BERTScore) |
| **M6** | RQ2 | `attribution.py`, `02_RQ2_attribution.ipynb`, `figures/rq2/` | No |
| **M7** | RQ3 | `fingerprint.py`, `03_RQ3_fingerprints.ipynb`, `figures/rq3/` | No |
| **M8** | Merge review | Summary of results, limitations, and which pieces are worth merging into the team submission | No |

**Future expansion to all 7 datasets.** `manifest.py` generalises the round-robin sampler to N items per dataset. The schema already carries `dataset`, the floors are per dataset, and the runner is dataset-agnostic. Compute scales linearly: about 7× for 200 items per dataset.

---

## 8. Decisions needed

Decisions 1–7 were resolved at Stage 2 approval (see "Status" at the top). Decisions 8–10 remain open.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Qwen main prompt: free (F) or literal (L) | **F**; keep L as an optional round-1 ablation |
| 2 | NLLB checkpoint | `facebook/nllb-200-3.3B` (fallback: distilled-1.3B) |
| 3 | Llama checkpoint | `meta-llama/Llama-3.2-3B-Instruct` (size parity); alternative 3.1-8B |
| 4 | MADLAD round 1: reuse V100 outputs or regenerate on Kaggle | Reuse, with a 20-item drift check; regenerate if drift is material |
| 5 | Qwen round 1 | Recover the Drive files from the teammate; otherwise regenerate for the 200 IDs |
| 6 | Flagged items | Continue chains; flag; analyse the core set valid across all pathways and rounds; report per-pathway n |
| 7 | Prefix stripping | Strip the shared 30-word prefix for all metrics and classifiers; report with-prefix sensitivity |
| 8 | Sentence splitter for NLLB | pysbd for NLLB; MADLAD keeps its original regex splitter (consistency with round 1) |
| 9 | Commit derived texts to git? | Commit `data/manifests` and `data/processed` (small); never commit the corpus snapshot or caches |
| 10 | Kaggle account and dataset names | Your account, private datasets as in §4.3 |

## 9. Recommended next step

**M1 plus M2, done together. Both run locally on CPU, with no models.**

1. Build the manifest generator and lock `xsum_madlad200.csv`.
2. Write the schema validators.
3. Import MADLAD round 1 into the shared schema.
4. Pass the regression test that reproduces `matched_item_metrics.csv`.

This turns the only existing translation texts into the shared format and proves that the IDs, texts and metric code line up, before any GPU time is spent.
