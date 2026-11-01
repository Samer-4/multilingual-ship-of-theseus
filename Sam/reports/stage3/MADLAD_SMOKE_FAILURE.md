# MADLAD smoke test failure (kernel `sam-sot-mt-madlad-smoke5` v2, 2026-10-31)

**Status:** diagnosed from the run's own outputs and from the transformers source code. A fix has been implemented and is waiting for one-document GPU verification. Nothing has been re-run.

## 1. What happened

| | Observed |
|---|---|
| Environment | Kaggle, 2 × Tesla T4, Python 3.13.15, torch 2.11.0+cu128, **transformers 5.16.1**, accelerate 1.14.0, sentencepiece 0.2.2. Code dataset v3, digest `b80f8223cce2404d`. |
| Tokenizer | `T5Tokenizer` (tokenizers backend). Segment counts 400/400 equal to round 1. |
| Model load | 87.7 s, fp16, `cuda:0`. Warning from transformers: *"The tied weights mapping and config for this model specifies to tie shared.weight to decoder.embed_tokens.weight, but both are present in the checkpoints with different values, so we will NOT tie them."* |
| Drift probes (T0 → S1, five texts that translated fine in round 1) | All five degenerate: a repeated Thai `ง`, Hebrew `ך`/`ה`, Georgian `ნ` or `ן` from the first token. finish = length, no EOS. |
| S2 (input = round-1 T1) | All five degenerate, the same way: Georgian/Cyrillic `ოე ое е…`, Korean `은 은 …`, Thai `ง…`, Arabic-script `ە…`. Every segment produced 513 tokens (512 new + start) with `segment_hit_eos = False`. 0 Latin characters in any output. |
| Validation | `engine:generation_cap;check:repetition` on all 5, so T2/S3/T3 were dropped (15 operations). The pipeline did the right thing. |

**These are not small wording changes.** In every probe and every S2 output the model emits a single non-Latin glyph, or two alternating, until the cap. It does so for every input, including inputs round 1 translated correctly. A model that sees its input correctly but decodes through broken input embeddings produces exactly this.

## 2. Hypotheses checked

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Repeats tokens until `max_new_tokens` | **Yes (symptom)** | 513 tokens in every segment; one glyph repeated |
| Fails to generate EOS | **Yes (symptom)** | `segment_hit_eos` all False |
| Wrong target-language prefix | Unlikely | `<2es>`/`<2en>` built exactly as in the worker; token counts (prefix included) equal round 1 on 400 texts. A wrong but valid prefix gives fluent text in another language, not one repeated glyph across random scripts. The diagnostic run prints the prefix tokens. |
| Tokenizer version difference | Unlikely to be the cause | 400/400 segment-count parity; output glyphs decode as real vocabulary pieces. Not fully excluded: parity compares counts, not ids. The diagnostic prints the tokens. |
| **Model loading under transformers 5.x** | **Most likely root cause** | See §3 |
| dtype (fp16 overflow) | Possible contributor, not the primary one | Round 1 also used fp16 (V100) and worked. T5's `_keep_in_fp32_modules = ["wo"]` exists in both 4.46.3 and 5.16.1. The diagnostic records the `wo` dtype. |
| Incompatible generation parameters | No | Identical arguments, `num_beams=1, do_sample=False, max_new_tokens=512` (§4) |
| Different checkpoint revision | Not excluded | Neither round pinned a revision. The diagnostic records `config._commit_hash`. |

## 3. Root cause

I compared the T5 source code of the two versions, from the 4.46.3 and 5.16.1 wheels (downloaded for reading only, never installed):

- **transformers 4.46.3** (round 1), `T5ForConditionalGeneration.__init__`:
  ```python
  self.shared = nn.Embedding(config.vocab_size, config.d_model)
  self.encoder = T5Stack(encoder_config, self.shared)
  self.decoder = T5Stack(decoder_config, self.shared)
  ```
  The encoder, the decoder and `shared` hold **one and the same module**, so the decoder cannot use a different matrix.
- **transformers 5.16.1** (smoke run):
  ```python
  self.shared = nn.Embedding(...)
  self.encoder = T5Stack(encoder_config)   # T5Stack: self.embed_tokens = nn.Embedding(...)
  self.decoder = T5Stack(decoder_config)
  _tied_weights_keys = {"lm_head.weight": "shared.weight", "encoder.embed_tokens.weight": "shared.weight",
                        "decoder.embed_tokens.weight": "shared.weight"}
  ```
  The modules are separate and are tied only after loading, via `_tied_weights_keys`. `modeling_utils` (around line 2644) skips a tie when the checkpoint contains both tensors and `not torch.equal(source, target)`, which is the warning printed in the log.

The MADLAD checkpoint stores a `decoder.embed_tokens.weight` that differs from `shared.weight`. Under 5.16.1 the decoder therefore embeds its own previous tokens with a matrix that round 1 never used, and the decoder degenerates at once whatever the input. That also explains why the encoder-side checks (tokenizer, segmentation, prefix) all looked right.

**Not verifiable offline:** which of the two stored tensors 4.46.3 ended up with. In 4.46.3 every checkpoint key writes into the same tensor, so the last one loaded wins. The likely answer is `shared`, since the encoder tie succeeded under 5.16.1 and the mapping's source is `shared`. Telling it apart from `decoder` requires the checkpoint, which I did not download. The one-document diagnostic settles it by translating with both wirings and comparing each with the stored round-1 S1.

## 4. Round-1 worker vs current adapter, line by line

| Step | Round 1 (`experiments/madlad_translation/madlad_explorer.ipynb`, worker) | Current (`Sam/src/sot_mt/translators.py`) | Same? |
|---|---|---|---|
| Checkpoint | `MODEL_ID = 'google/madlad400-3b-mt'`, no revision | `google/madlad400-3b-mt`, `model_revision: null` | yes (revision unpinned in both) |
| Libraries | transformers 4.46.3, accelerate 1.2.1, sentencepiece 0.2.0, V100 | transformers 5.16.1, accelerate 1.14.0, sentencepiece 0.2.2, T4 | **no** |
| Tokenizer | `AutoTokenizer.from_pretrained(model)` | same, plus `local_files_only`/`revision`/`token` kwargs | yes |
| Model | `AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=float16, low_cpu_mem_usage=True).to('cuda:0')`, `model.eval()` | same; `dtype=` on ≥4.56 (`torch_dtype` deprecated); `low_cpu_mem_usage` ignored by 5.x | yes in code; **the resulting module structure differs (§3)** |
| Embedding wiring | one module (by construction) | 5.x: three modules, decoder untied (fix: re-tie) | **no → fixed** |
| Prefix | `f'<2{lang}> ' + x`, lang = target | `f"<2{tgt}> " + x` | yes |
| Token count | `len(tokenizer('<2xx> '+s, add_special_tokens=True).input_ids)` | identical | yes |
| Segmentation | `split_chunks` (regex sentence split, word fallback, greedy pack, budget 300) | `madlad_pack`, tested equal to the extracted `split_chunks` | yes |
| Batching | 4 segments per `generate` | 4 (halved only on OOM) | yes |
| Encoding | `tokenizer([...], padding=True, return_tensors='pt', truncation=False).to('cuda:0')` | identical | yes |
| `generate` | `num_beams=1, do_sample=False, max_new_tokens=512` under `torch.inference_mode()` | identical | yes |
| EOS / cap | `generation_config.eos_token_id` (or the tokenizer's); no EOS in ids → truncated | identical (no EOS → `finish="length"`) | yes |
| Decode | `tokenizer.decode(ids, skip_special_tokens=True).strip()`; empty → error | identical (empty → `engine:empty_segment`) | yes |
| Reconstruction | `' '.join(outputs)` | `" ".join` | yes |

The only differences are the library versions and, as a result, the embedding wiring.

## 5. Changes

1. **Fix (`translators.py`):** `MadladTranslator._load` re-ties the encoder, decoder and `shared` to one embedding module (`embedding_tie = "shared"`), restoring the 4.46.3 structure. It records a `load_report` before and after: which modules are shared or equal, `tie_word_embeddings`, the `wo` dtype and the commit hash.
   - The report goes into `runs.parquet` (`load_report`), and `generation_meta.embedding_tie` goes into every record.
   - It is a no-op when the model is already one shared module, as under 4.46.3.
   - `configs/translators.yaml` is unchanged, and the config id is still `705a8bd4992e`.
   - `max_new_tokens` is unchanged and validation is unchanged.
2. **Fail-fast quality gate (`runner.py`):** the run stops with status `stopped:quality_gate` once at least 3 of the last 10 completed operations carry a degenerate code (`generation_cap`, `repetition`, `empty_segment`, `blank`) and they make up ≥50% of that window. Those outputs are still stored as invalid records.
3. **Reference gate (notebook):** with `DRIFT_PROBE`, a probe that is degenerate, or has <0.5 word similarity to round-1 S1, stops the notebook **before any S2 work**, after the first bad probe. Exact identity is reported, not required (`smoke.probe_verdict`).
4. **Diagnostic cell (notebook, `MADLAD_DIAGNOSTIC`):** loads the model as transformers builds it, then translates T0 → S1 under three wirings: `as_loaded`, `shared` and `decoder`. It compares each with round 1 and writes `madlad_diagnostic.json`, which also holds versions, prefix tokens, the generation config, the safetensors weight map and the embedding report. It stops the notebook unless `shared` passes.
5. **Quarantine:** `experiment.yaml: quarantined_runs` lists the failed run, and the runner refuses any base containing its records.
   - The failed outputs are untouched: 13 raw files and the filed tables are hash-verified.
   - `QUARANTINED.md` notes were added beside them, and the push record was copied to `results/madlad_smoke5/failed_kernel_v2_last_push.json`.
6. **One-document deployment:** `prepare_kaggle_deploy.py --preset verify1` writes `Sam/kaggle/madlad_verify1/`.
   - Kernel `samahmed17/sam-sot-mt-madlad-verify1`; item `xsum|LLAMA|xsum-4729`, which has 2 segments per direction, so batching and padding in fp16 are exercised.
   - Diagnostic, drift-probe gate and quality gate are all on.
   - Results go to `results/madlad_verify1` and `data/generated/verify1`.

**Tests:** 187 pass (16 new in `tests/test_madlad_engine_failure.py`, which use the real failed outputs as fixtures).

## 6. One-document verification plan

```bash
cd /Users/samerahmed/multilingual-ship-of-theseus
Sam/.venv/bin/python -m pytest -q Sam/tests                                          # 187 passed
Sam/.venv/bin/python Sam/scripts/launch_kaggle.py --deploy-dir Sam/kaggle/madlad_verify1          # dry run
Sam/.venv/bin/python Sam/scripts/launch_kaggle.py --deploy-dir Sam/kaggle/madlad_verify1 --yes    # GPU run
Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py --deploy-dir Sam/kaggle/madlad_verify1
Sam/.venv/bin/python Sam/scripts/download_kaggle_outputs.py --deploy-dir Sam/kaggle/madlad_verify1
```

**The session runs, in order:**
1. Tokenizer parity.
2. Diagnostic: 3 T0 → S1 translations.
3. Drift probe with the `shared` wiring: 1 translation.
4. Only if both pass, the item's 4 operations S2 → T3.

The cost is at most 8 translations of one 204-word text plus about 90 s of model load.

**Pass criteria** (from `results/madlad_verify1/madlad_diagnostic.json`, `drift_probe.csv`, `validation_report.csv`):

| Check | Pass |
|---|---|
| `as_loaded` variant | degenerate (reproduces the failure) |
| `shared` variant | `ok`: EOS reached, no blocking check, similarity ≥ 0.5 to round-1 S1 (ideally identical or ≥ 0.9) |
| `load_report_as_loaded.decoder_equals_shared` | `false` (confirms the mechanism) |
| drift probe | `ok`; similarity reported |
| S2/T2/S3/T3 | 4 valid records, `finish = stop`, every `segment_hit_eos` true, length ratios ~0.8–1.3, readable Spanish/English |

**Decision rules**
- **`shared` passes:** run the five-item smoke test with `--deploy-dir Sam/kaggle/madlad_smoke5`. It has its own reference gate and quality gate, and its download needs `--replace`, which moves the quarantined files into `superseded/` instead of overwriting them.
- **Only `decoder` passes:** switch `MadladTranslator.embedding_tie` to `"decoder"` (one line) after review, and verify again.
- **Neither passes:** pin `PIN_MADLAD_ROUND1_LIBS = True` (transformers 4.46.3, the round-1 build) in a verification run. That pin is unverified on Python 3.13 with torch 2.11.

---

## 7. One-document diagnostic result (kernel `samahmed17/sam-sot-mt-madlad-verify1` v1, status ERROR by design)

**Retrieved with** `kaggle kernels output samahmed17/sam-sot-mt-madlad-verify1 -p …`. The run failed before exporting, so the regular downloader could not file it.
- The raw files and `DOWNLOAD.json` (sha256 of each file) are in `results/madlad_verify1/raw/sam-sot-mt-madlad-verify1_failed_20261031T061019Z/`.
- Copies of the diagnostic, tokenizer parity and log, with provenance, are in `results/madlad_verify1/failed_v1/`.
- `download_kaggle_outputs.py --raw-only` now does this for any failed run.

**Setup:** transformers 5.16.1, torch 2.11.0+cu128, T4, fp16; checkpoint commit `fa184c675da0b5c9e1c8694fccd4e12e2d422094`; item `xsum|LLAMA|xsum-4729` (2 segments). Prefix tokens are `▁ <2es> ▁Hello ▁world . </s>`, so the prefix is a single token as intended. The weight-map index file does not exist in the repo (404), so the shard layout is unknown.

**Embedding state as loaded:** `tie_word_embeddings = true`; encoder equals `shared` (by value, not the same module); **decoder differs from `shared` and from the encoder**; `lm_head` equals `shared`; parameters fp16, `wo` fp32.

| Variant (run order) | Input embeddings | lm_head | finish / tokens | Output | vs round-1 S1 |
|---|---|---|---|---|---|
| `as_loaded` (29.0 s) | encoder = shared values, decoder = decoder values | shared values | length, 513 + 513, no EOS | `ง ง ง … 们 们 …` | 0.0, generation_cap + repetition |
| `shared` (12.5 s) | all = shared values | shared values | **stop**, 228 + 40 | `…o…)aaa  εαααεόσwwnw semneknie …` | **0.0, yet no blocking check fired** |
| `decoder` (14.4 s) | all = decoder values | shared values | stop, 263 + 53, EOS in both | "La energía viajará a través del interconector de electricidad submarina más largo del mundo. …" | **identical (1.0)** |

**Reading:** the checkpoint's `decoder.embed_tokens.weight` is MADLAD's input-embedding table, for the encoder **and** the decoder. `shared.weight`, which `lm_head` equals, works as the output projection but not as an input embedding.
- The `shared` variant fed both stacks the output matrix: fluent-shaped garbage that still emits EOS.
- `as_loaded` fed the encoder the wrong matrix and the decoder the right one: an immediate loop.
- Only `decoder` gives both stacks the right table and leaves the output side alone, and it reproduces round 1 **exactly**.

So the round-1 build (transformers 4.46.3, one module for encoder, decoder and `shared`) must have ended up holding the decoder tensor. My first guess, `shared`, was wrong, and the diagnostic gate correctly refused it.

**State between variants:** the cell restored the as-loaded module references before each variant and never touched `lm_head`. Evidence that nothing leaked:
- `as_loaded`, run first, reproduces the smoke-run degeneration;
- `decoder`, run *after* `shared`, is bit-identical to round 1.

The notebook now also records the wiring report per variant and after the restore.

## 8. Fix (verified configuration)

- **`configs/translators.yaml`:** a new top-level `runtime_patches.madlad-3b` entry, `patch_id madlad-embed-decoder-v1`, with `embedding_tie: decoder` and `lm_head: unchanged`.
  - It sits outside the frozen translator entry, so `generation_config_id` is still `705a8bd4992e` and the Stage 2 records and config are untouched.
- **`MadladTranslator` reads the patch.** On load it saves the as-loaded modules and applies the wiring through `rewire()`, which:
  - restores the loaded modules first;
  - makes encoder, decoder and `shared` one module from the named source;
  - verifies that, and that `lm_head` is the same tensor as loaded; otherwise `EngineWiringError`.

  It is a no-op if the model already uses one module (4.46.3). Translation is refused while the wiring is unverified.
- **Tracking:**
  - `runs.parquet`: `load_report` (patch id, applied wiring, before/after embedding reports);
  - every new record's `generation_meta`: `embedding_wiring`, `runtime_patch`.
- **New blocking check `check:not_target_language`** (thresholds in `experiment.yaml validation.target_language`): fails if more than 5% of the letters are non-Latin, or if an output of 20+ words has a target function-word rate below 0.01.
  - Measured margins: every real Stage 2 output and T0 text has ≤ 0.001 non-Latin letters and a rate ≥ 0.041 (MT outputs) or ≥ 0.015 (T0); every garbage output has ≥ 0.222 non-Latin letters or a rate of 0.000.
  - It also counts towards the quality gate.
- **Diagnostic gate:** requires the **configured** wiring to pass and lists every passing wiring. The reference similarity threshold was **raised** from 0.5 to 0.8 (verified wiring 1.0, broken ones 0.0).
- **`verify1` now runs one gated reference check** (diagnostic off, drift probe on), then S2 → T3 for the item.
