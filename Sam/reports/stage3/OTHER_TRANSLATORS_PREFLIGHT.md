# NLLB, Qwen and Llama: audit, preflights and full-run plan (2026-10-31, nothing launched)

## Rule
Every model translates the same 200 original T0 documents through T0 → S1 → T1 → S2 → T2 → S3 → T3, on its own pathway.
- **Lineage:** S1 reads the manifest T0. Every later stage reads the same model's valid predecessor; the runner never chains across pathways. `verify_madlad_pilot.py --translator X` checks this.
- **No reuse of MADLAD output:** MADLAD records serve only as the shared, unchanged Stage 2 base table. They are never input to another model.
- **No exclusions:** none of the 200 documents is excluded, including those where MADLAD failed.

## Per-model configuration (`configs/translators.yaml`)
| | NLLB | Qwen | Llama |
|---|---|---|---|
| Checkpoint | `facebook/nllb-200-distilled-1.3B` | `Qwen/Qwen3-4B-Instruct-2507` | `meta-llama/Llama-3.2-3B-Instruct`, loaded from Kaggle Models `metaresearch/llama-3.2/transformers/3b-instruct/1` |
| Revision | not pinned; the hub commit is recorded at load | same | Kaggle version 1; a folder fingerprint is recorded |
| Access | public | public | gated: Llama 3.2 licence on Kaggle (**currently 403**) |
| Config id | `c63f99caa212` | `747d97077f38` | `8348c2bc3f50` |
| Languages | `eng_Latn` / `spa_Latn` (`src_lang` + forced BOS) | system prompt per direction | same prompts as Qwen |
| Prompt variant | `mt-default` | `free` (primary) | `free` |
| Segmentation | one sentence per segment (split at 256 tokens) | whole document | whole document |
| Batching | 8 segments per call (padded) | 1 document | 1 document |
| Decoding | greedy, 512 new tokens per segment | greedy, max(16, min(2·n_in + 64, 8192 − n_prompt − 8)) | same as Qwen |
| dtype / device | fp16, GPU 0 | fp16, GPU 0 | fp16, GPU 0 |
| Weights in GPU memory (estimated) | ≈ 2.6 GB | ≈ 8 GB | ≈ 6.4 GB |

MADLAD for reference: packed sentences up to 300 tokens, batch 4, greedy, 512 new tokens; config `705a8bd4992e` is unchanged.

## Changes in this stage
1. **Llama chat template:** Llama 3.x writes "Today Date: <today>" into the system prompt. It is now fixed to the template's own fallback (`date_string: "26 Jul 2024"`). Otherwise every session, and every resume, would get a different prompt.
2. **Plain greedy decoding:** generation uses `repetition_penalty=1.0` and `no_repeat_ngram_size=0` for NLLB, Qwen and Llama, so a checkpoint's `generation_config.json` cannot add a penalty implicitly. Its defaults are recorded in `load_report.default_generation_config`. MADLAD is untouched.
3. **NLLB embeddings:** after loading, the encoder/decoder input embeddings and the LM head must be the tied shared matrix, otherwise the run stops. This guards against the transformers 5 failure that broke MADLAD.
4. **Load report** (`runs.parquet`): weights source, resolved hub commit or folder fingerprint (`weights_id`), default generation config, device, dtype and GPU memory.
5. **Single GPU for the LLMs:** they are placed on one GPU (`device_map={"": 0}`) instead of being split over both T4s. This changes speed only, not the arithmetic of the outputs.
6. **Notebook:**
   - a sanity gate before any record: model load (timed) plus one fixed sentence per direction, which must be clean, in the target language and contain the expected words (`sanity_probe.csv`);
   - `EXPECTED_WEIGHTS_ID`, so a full run refuses weights other than the preflight's;
   - peak GPU memory per device (`gpu_memory.json`);
   - `checks` added to the import guard.
7. **Deployments:** `kernel-metadata.json` gains `model_sources` (Llama).

## Comparison confounds to report in RQ1–RQ3
- **Unit of translation:** NLLB translates single sentences, MADLAD packed sentence groups, and Qwen and Llama whole documents. Cross-sentence context is available to the LLMs only.
- **Padding:** NLLB and MADLAD batch padded segments in fp16. Their greedy outputs could differ slightly from unbatched decoding; this is fixed per model.
- **Length caps:** the MT models have 512 new tokens per segment; the LLMs scale with the input length.
- **Prompting:** the LLMs are instructed ("naturally and idiomatically" for en→es); the MT models are not.
- **Hardware drift:** all models run on T4 fp16 with transformers 5.x on Kaggle, except MADLAD round 1 (V100, 4.46.3).

## Preflight
- **Documents:** the same 3 T0 documents for each model (`data/manifests/preflight3.csv`):
  - `xsum|Eleuther-AI|xsum-4714`, 76 words, short;
  - `xsum|OpenAI|xsum-4715`, 176 words, median;
  - `xsum|Human|xsum-4709`, 916 words, the longest, as the worst case for memory and token caps.
- **Gates and outputs:** each preflight does 18 operations, runs every gate, writes a rolling export every 6 operations, records GPU memory and has a 90 min timeout.
- **Kernels:** `samahmed17/sam-sot-mt-{nllb,qwen3-4b,llama}-preflight`.

## Full runs (prepared only after each preflight is downloaded)
`prepare_kaggle_deploy.py --preset full --model X` builds on the model's own preflight export, pinned by sha256; its 18 records are reused, never redone. It then:
- pins the weights to the preflight's `weights_id`;
- fits seconds per operation as a + b·words on the 18 measured operations, and uses the measured setup time and length ratios;
- sets the session timeout from that estimate, capped at 12 h with a 20 min margin;
- if the estimate needs more than one session, stops cleanly and continues as `--resume-attempt 2`, `3`, …;
- writes a rolling export every 20 operations.
