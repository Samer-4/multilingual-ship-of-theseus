# Project Audit: Multilingual Ship of Theseus

**Audit date:** 2026-10-30
**Repository:** `multilingual-ship-of-theseus` (local checkout, branch `feature/sam-experiments`, working tree clean at audit time; latest commit `5fed882`)
**Scope:** read-only inspection of every tracked file, plus the sibling upstream corpus clone at `../Ship_of_theseus_paraphrased_copus`. No file outside `Sam/` was modified. No model was downloaded and no inference was run.

Each finding carries one of these labels:

- **[VERIFIED]**: checked directly in a file or by running a read-only pandas check during this audit.
- **[REPORTED]**: stated in a README or notebook, but the evidence (code or per-item output) is not in the repository.
- **[ASSUMED]**: my inference or general knowledge. It needs confirmation.

---

## 1. Repository inventory

The repository holds 31 tracked files, about 26 MB. **[VERIFIED]**

```
.
├── README.md                 project overview + MADLAD summary table
├── config.yaml               "pilot v2" config (translate-then-paraphrase; NOT the current design, see §7)
├── pyproject.toml            package "pilot", src layout, requires-python >=3.11
├── requirements.txt          unpinned (>=) deps, incl. zh tooling, anthropic, mlx-lm, kaggle
├── .gitignore                no data/ rule; caches/venv/.env ignored
├── src/pilot/
│   ├── __init__.py
│   ├── config.py             load_config() (paths relative to repo root), set_seeds()
│   └── metrics.py            corpus loaders + lexical/semantic/style metrics + bootstrap (reusable)
├── scripts/
│   ├── score_lexical.py      background runner → data/derived/sot_pairs_lexical_*.parquet
│   ├── score_semantic.py     background runner → data/derived/sot_pairs_semantic.parquet
│   └── style_features.py     background runner → data/derived/sot_style_features.parquet
├── notebooks/
│   └── sot_analysis.ipynb    English paraphrase baseline analysis (outputs saved in notebook)
├── experiments/
│   ├── madlad_translation/
│   │   ├── README.md
│   │   └── madlad_explorer.ipynb        Explorer HPC (V100) notebook; embeds the worker script
│   └── qwen_translation/
│       ├── README.md
│       ├── degree_zero_spanish_pilot.ipynb   Colab, 40-item prompt pilot
│       ├── degree_zero_chain.ipynb          Colab + vLLM, T0→S1→T1 generation
│       ├── degree_zero_metrics.ipynb        Colab, metrics for chain_*.csv
│       └── degree_zero_qe.ipynb             Colab, COMETKiwi QE
└── results/
    ├── sot_analysis/          decay_curves.{csv,png}, floors_by_dataset.csv, style_blocks_T3.csv,
    │                          style_blocks_and_sources.{csv,png}, style_vs_content.{csv,png}
    ├── madlad_translation/    chain_es_madlad4003b.csv, matched_item_metrics.csv
    └── qwen_translation/      metrics_items.csv, metrics_summary.json
```

These paths are referenced in the code but **do not exist** in the repository, and the git history shows they were never committed **[VERIFIED]**:

- `docs/sot_analysis_plan.md` and `docs/pilot_execution_plan.md`, cited by `metrics.py`, `config.yaml`, and `sot_analysis.ipynb`.
- `data/sot/`, `data/derived/`, `data/calls/`, `data/kaggle/`, all named in `config.yaml` → `paths`.
- `notebooks/archive/multitude/config_v1.yaml`, cited in `config.yaml`.
- `round_trip_stats.py`, cited by the Qwen chain and metrics notebooks.
- The `Sam/` folder existed but was empty before this audit.

**Upstream corpus (outside the repository).** **[VERIFIED]**

`../Ship_of_theseus_paraphrased_copus` is a git clone of `https://github.com/tripto03/Ship_of_theseus_paraphrased_copus` at commit `4423d8c`, about 693 MB.

- It contains `paraphrased_datasets/{cmv,eli5,sci_gen,tldr,wp,xsum,yelp}_paraphrased.csv` and `train_datasets/{...}_train.csv`.
- **It has no LICENSE file.**
- `config.yaml` points at it as `paths.corpus_src: ../Ship_of_theseus_paraphrased_copus`. That path only resolves on a machine with the same layout.

---

## 2. Corpus structure

All of the following is **[VERIFIED]** from the local clone.

**Paraphrased files** have the columns `source, key, text, version_name`.

- `version_name` takes 22 values: `original`, plus 7 paraphrasers × 3 chained rounds.
- The paraphrasers are `chatgpt`, `palm`, `dipper`, `dipper(low)`, `dipper(high)`, `pegasus(full)`, `pegasus(slight)`.
- Rounds are encoded by repetition: `chatgpt`, `chatgpt_chatgpt`, `chatgpt_chatgpt_chatgpt`.
- There are 7 sources: `Human`, `OpenAI`, `LLAMA`, `PaLM`, `Tsinghua`, `BigScience`, `Eleuther-AI`. These are generator families plus one human class. **They are not individual authors.**

| dataset | rows | originals (T0) | keys |
|---|---:|---:|---:|
| cmv | 35,405 | 1,626 | 256 |
| eli5 | 69,270 | 3,160 | 475 |
| sci_gen | 60,166 | 2,797 | 471 |
| tldr | 53,519 | 2,444 | 382 |
| wp | 58,417 | 2,768 | 469 |
| xsum | 70,388 | 3,227 | 476 |
| yelp | 72,847 | 3,320 | 491 |

- **XSum quirk:** `xsum|OpenAI|xsum-492` has paraphrases but no `original` row. That is why the non-original versions have 3,228 rows against 3,227 originals.
- **Train files** have the columns `source, key, text`. `xsum_train.csv` holds 3,243 texts across all 7 sources. **Its keys and its texts are fully disjoint from `xsum_paraphrased.csv`** (0 shared keys, 0 shared texts). It is therefore a clean held-out training pool for RQ2.
- **Shared prompt prefix:** LLM-source texts in the MADLAD sample begin with the opening words of the Human text of the same key. The median shared prefix is 30 words for all six LLM sources. Stage 2 measured this precisely: of the 171 LLM texts, 95.9% share at least 10 leading words and 69% share exactly 30. Stage 1 overstated this as "every … 30 words". This matches the upstream README ("prompting the first 30 tokens") and `config.yaml` → `strip_prompt_prefix`. In the sample, 18 of 171 LLM texts have fewer than 40 words of their own once that prefix is removed.
- **Length differs strongly by source.** In the 200-item sample, median T0 words are: Human 355, PaLM 318, LLAMA 299, BigScience 221, Tsinghua 124, OpenAI 106, Eleuther-AI 99.

---

## 3. MADLAD XSum experiment (`experiments/madlad_translation/`)

### 3.1 Which 200 documents were sampled? Reproducible?

**Yes, exactly.** **[VERIFIED]**

The worker script embedded in `madlad_explorer.ipynb` (cell 5) selects the sample this way:

1. Load `xsum_paraphrased.csv` from the raw GitHub URL on branch `main`.
2. Keep the `original` rows, drop rows with no text, and de-duplicate on `(key, source)`.
3. Build `item_id = "xsum|<source>|<key>"`.
4. Sort by `(source, item_id)`, with `item_id` compared as a string.
5. Number each source's items in that order with `cumcount`.
6. Re-sort by `(position, source, item_id)`. This is a source-balanced round-robin.
7. Take the first 200.

I re-ran this procedure on the local clone at commit `4423d8c`:

- It returns **the same 200 `item_id`s in the same order** as `results/madlad_translation/chain_es_madlad4003b.csv`.
- The source text `S` is **byte-identical for 200/200**.

Properties of the sample:

- 200 items across **31 keys**. 23 keys have all 7 sources, 6 keys have 6, one has 2 and one has 1.
- Per source: 29 each for BigScience, Eleuther-AI, Human and LLAMA; 28 each for OpenAI, PaLM and Tsinghua.
- It is **not a random sample**. Because `item_id` is sorted as a string, the keys cluster around `xsum-2638`, `xsum-3981` and `xsum-4708`…`xsum-4738`. The notebook itself says so: "not a random representative sample".
- The notebook pulls the corpus from branch `main`, not from a pinned commit. If upstream ever changes, the procedure would silently select different items. The worker has one guard: it raises an error if the count is not 3,227.

The term "document" is ambiguous here. The unit of the sample is a **`(source, key)` text**, not a news article. There are 200 texts about 31 underlying articles.

### 3.2 Are IDs and source labels available?

**Yes.** **[VERIFIED]**

- `chain_es_madlad4003b.csv` (200 rows) has the columns `key, source, S, domain, item_id, words, F, B_F, bad, bad_reason, chunks_en, chunks_es, seconds`.
- `item_id` is unique and always equals `domain|source|key`.

### 3.3 Are the Spanish and reconstructed English texts saved?

**Yes, for round 1 only.** **[VERIFIED]**

- `S` holds T0. `F` holds the Spanish S1. `B_F` holds the back-translated English T1. None of these columns is null.
- There is no T2, T3, S2 or S3. The README confirms that only one round trip was run.
- **Three rows are flagged** `bad=True`, and their texts are still stored:

| item_id | bad_reason | T0 words | T1 words |
|---|---|---:|---:|
| `xsum\|BigScience\|xsum-4713` | generation_cap | 411 | 1065 |
| `xsum\|LLAMA\|xsum-4714` | generation_cap;severe_expansion | 95 | 314 |
| `xsum\|LLAMA\|xsum-4733` | generation_cap | 460 | 475 |

- That leaves 197 valid rows, which matches the report.
- The worker also wrote `input_manifest.csv` and `slide_metrics_summary.csv` on Explorer (`~/degree_zero_madlad_v100/output/`). **Neither is in the repository.** The manifest can be rebuilt exactly (§3.1).

### 3.4 Decoding settings

All of these are **[VERIFIED]** from the worker code.

| Setting | Value |
|---|---|
| Model | `google/madlad400-3b-mt` (no HF revision hash recorded) |
| Precision | fp16 |
| Chunking | Sentence-split with regex `(?<=[.!?])\s+`, packed into chunks of ≤ 300 tokens; the `<2xx>` prefix counts toward the budget |
| Batch size | 4 |
| Decoding | Greedy (`num_beams=1`, `do_sample=False`) |
| Length cap | `max_new_tokens=512` per chunk |
| `generation_cap` flag | Set when a chunk ends without EOS |
| Length-ratio flags | Ratio < 0.35 flags compression; ratio > 3 flags expansion |
| Pinned stack | transformers 4.46.3, accelerate 1.2.1, sentencepiece 0.2.0, sacrebleu 2.5.1, rouge-score 0.1.2 |
| Hardware | V100 32 GB (Explorer HPC) |

No seed is set. Greedy decoding is deterministic up to GPU numerics.

### 3.5 Metrics already implemented

`results/madlad_translation/matched_item_metrics.csv` has 197 rows, one per valid item. **[VERIFIED]** It contains:

- `bleu`: sacrebleu sentence BLEU with `effective_order=True`, divided by 100.
- `rougeL`: F1 with stemming.
- `chrf`: divided by 100.
- `word_retention`: clipped multiset overlap over **all** `\w+` tokens, function words included.
- `function_word_cosine`: computed over a fixed list of 52 words (corrected in Stage 2; Stage 1 said 48).
- `punctuation_f1`
- `exact_match`

I recomputed the medians from the CSV. They match the README exactly: BLEU 0.604, ROUGE-L 0.831, chrF 0.793, word retention 0.836, function-word cosine 0.972, punctuation F1 0.853. **[VERIFIED]**

The README also reports **multilingual-E5 cosine 0.9818 and BERTScore F1 0.9217**. **[REPORTED]** Neither the code nor the per-item values are in the repository, and the README says so. These numbers cannot be checked or reused.

---

## 4. Qwen experiment (`experiments/qwen_translation/`)

### 4.1 Which document IDs were used?

**[VERIFIED]** from `degree_zero_chain.ipynb` and `metrics_items.csv`.

- The run used `DOMAINS=["xsum"]`, `SOURCES=None` and `MAX_W=None`. So it covered **every XSum original: 3,227 items**. It used the same `item_id` format, `xsum|<source>|<key>`.
- `metrics_items.csv` has 22,589 rows = 3,227 items × 7 versions. The versions are `T1 free`, `T1 literal`, `chatgpt×1/×2/×3`, `pegasus(full)` and `dipper`.
- **All 200 MADLAD item IDs are present** in the Qwen `T1 free` and `T1 literal` rows, each with a non-null `content` value.
- The "3,208 valid documents" in the README matches `metrics_summary.json` → `F_vs_L.n = 3208`. Which 19 items were excluded is **not saved**. The `bad` flags lived only in the chain CSV.

### 4.2 Are literal and free prompt outputs available?

**The texts are not available. Only metrics are.** **[VERIFIED]**

`degree_zero_chain.ipynb` wrote everything to Google Drive (`/content/drive/MyDrive/degree_zero/`):

- `generations.jsonl`
- `run_config.json`
- `chain_es_Qwen3-4B-Instruct-2507.csv`, with the columns `S, G, L, F, F2, B_L, B_F`, the `finish_*` flags and `bad`/`bad_reason`.

**None of these files is in the repository.** The Qwen README confirms this: "intermediate translation-chain CSV files are not yet available". The metrics notebook also wrote `metrics_intervention.csv` and `qe_scores.csv`. Neither is in the repository.

What *is* recorded in the notebook code **[VERIFIED]**:

**Model and decoding**

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507`, run with vLLM |
| Precision | `dtype="half"` |
| Context | `max_model_len=8192` |
| Seed | `seed=0` |
| Temperature | `temperature=0` |
| Length cap | `max_tokens = max(16, min(2·n_in + 64, 8192 − n_prompt − 8))` |
| Input | The **whole document**; no chunking |
| Output cleanup | `<think>` blocks stripped |
| Chat format | System prompt = instruction; user turn = text |

**Prompts** (`prompts_for("es")`)

- **L**: "Translate the following English text into Spanish word-for-word. Preserve the source word order, sentence boundaries and structure as closely as Spanish grammar allows. Do not rephrase, merge, split, or reorder anything. Output only the Spanish text, with no notes, quotes, or explanations."
- **F**: "Translate the following English text into Spanish naturally and idiomatically, as a skilled professional translator would for a Spanish-speaking readership. Output only the Spanish text, with no notes, quotes, or explanations."
- **back**: "Translate the following Spanish text into English. Output only the translation, with no notes."
- The vLLM version was written to `run_config.json` on Drive, so it is not known here.

### 4.3 Can its outputs be aligned with the MADLAD sample?

- **At the ID level: yes.** All 200 IDs match. **[VERIFIED]**
- **At the text level: no**, until the Drive files are recovered.
- **At the metric level: not directly comparable.** **[VERIFIED]** The Qwen metrics measure *change*, where higher means more changed:
  - `content` = ‖e5(T1) − e5(T0)‖
  - `surface_chrF` = 1 − chrF
  - `style_funcwords` = cosine *distance*
  - Burrows' Delta
  - and so on

  Each also has a `_norm` version, divided by a random-pair baseline built with shift 7 inside each domain × source. The MADLAD metrics measure *preservation* and use different function-word lists and different word-retention definitions. The Qwen README warns about this too.

### 4.4 Also present

- **Gloss baseline G:** a MUSE en–es word-for-word gloss, built from an external dictionary URL.
- **F2 noise floor:** 20 items per source.
- **COMETKiwi QE:** `Unbabel/wmt22-cometkiwi-da`. It needs a licence click-through and an `HF_TOKEN`.
- **The 40-item Spanish pilot:** Human and OpenAI sources, 80–250 words. Its outputs went to `/content/results.csv`, which is not in the repository.

### 4.5 What to reuse versus regenerate

- **Reuse** the prompt wording, decoding settings, output-validation heuristics (`problems()`), and model choice. Reproduce them exactly.
- **Reuse if recovered:** the T1 texts (`F`, `B_F`, plus `L`, `B_L`) for the 200 MADLAD IDs, taken from `chain_es_Qwen3-4B-Instruct-2507.csv` together with `run_config.json`. These would become round 1 of the Qwen pathway.
- **Regenerate if not recovered:** round 1 for the 200 IDs with the identical prompts and settings, plus rounds 2–3 in either case.
- **Do not reuse the Qwen `metrics_items.csv` numbers in the final RQ1 tables.** They come from a different metric code path. They are fine as a sanity cross-check, for example that T1 free changes less than chatgpt×1. The 200-item medians support that check: `content` is 0.200 for T1 free against 0.261 for chatgpt×1.

---

## 5. English paraphrasing baseline

### 5.1 Are the outputs stored and usable?

**[VERIFIED]**

- **The texts:** **yes**, in the upstream corpus clone. They are not in the repository. For the 200 sampled IDs, coverage by variant and round is:

  | Variant | T1 | T2 | T3 |
  |---|---:|---:|---:|
  | `dipper`, `dipper(low)`, `dipper(high)` | 200 | 200 | 200 |
  | `pegasus(full)`, `pegasus(slight)` | 200 | 200 | 200 |
  | `chatgpt` | 199 | 197 | 196 |
  | `palm` | 192 | 191 | 188 |

- **Per-pair metric caches:** **no.** The scripts and the notebook write the per-pair scores (`data/derived/sot_pairs_{lexical,semantic,style}*.parquet`, `sot_sample.parquet`, `sot_floors*.parquet`, `sot_style_features.parquet`, `sot_function_words.json`). None of those files is in the repository.
- **What is saved:** only aggregate tables under `results/sot_analysis/`.
  - `decay_curves.csv`: variant × round × metric mean with a bootstrap CI, over **16,682 "completer" originals from all 7 datasets**.
  - `floors_by_dataset.csv`
  - `style_*.csv`
- These aggregates cover a different population from the 200-item XSum sample. They can be shown as context, but a matched RQ1 comparison needs the baseline recomputed on the same 200 IDs. That is cheap, because the texts exist.

### 5.2 Reusable metric code (`src/pilot/metrics.py`)

**[VERIFIED]**

| Function or class | What it does | Reusable as-is? |
|---|---|---|
| `load_corpus`, `load_reference` | Long table `dataset, source, key, variant, round, version_name, text` | Yes. Needs a configurable `sot_dir`. |
| `lexical_scores` | BLEU (sentence, exp smoothing), ROUGE-1/2/L (stemmed), **content-word type retention** (sklearn stop words removed) | Yes |
| `LongBertScore` | Matches bert_score 0.3.13: roberta-large, layer 17, baseline-rescaled; sliding windows instead of truncation | Yes. Downloads roberta-large (~1.4 GB). |
| `ChunkedSentenceEmbedder` | all-mpnet-base-v2 cosine, mean over ~200-word chunks | Yes |
| `style_features`, `extract_style`, `top_function_words`, `reference_subset` | spaCy `en_core_web_sm`: surface, punctuation, top-100 function words (from reference texts), UPOS, dependency depth | Yes |
| `build_pairs`, `chain_jobs`, `score_semantic`, `score_lexical_chunked` | Cumulative (T0 vs Tn) and step (Tn−1 vs Tn) pairs; resumable caches | The logic is reusable; it is hard-wired to corpus `variant` names. |
| `bootstrap_mean_ci` | Cluster (key) bootstrap of group means | Yes |

The notebook defines more logic inline (`sot_analysis.ipynb` cells 6, 9 and 11) that is not in the module:

- floor pairs: 200 random reference pairs per dataset, seeded `[6120, 32]`
- style z-scoring on a 300-per-dataset reference subset
- block cosines (surface, punctuation, function words, syntax/POS), averaged into `style`
- char n-gram TF-IDF (`char_wb` 2–4, `min_df=3`, sublinear)
- the "retained share" normalisation `(s − floor) / (1 − floor)`

All of this needs extracting into reusable code.

### 5.3 Three incompatible metric implementations

The repository holds three different implementations that use **the same metric names**. **[VERIFIED]**

| Metric name | `pilot.metrics` (baseline) | MADLAD worker | Qwen metrics notebook |
|---|---|---|---|
| word retention | content-word **types**, stop words removed | all-token **multiset** recall | (not used) |
| function words | top-100 from reference texts, z-scored, cosine | fixed 52-word list, raw-count cosine | fixed 75-word list, rate cosine **distance** |
| semantic | BERTScore (rescaled) + mpnet cosine | (E5/BERTScore reported, code absent) | e5-base Euclidean distance |
| normalisation | floor of unrelated reference pairs | none | random-pair shift-7 baseline |

---

## 6. Dependencies, checkpoints, preprocessing

**Dependencies.** **[VERIFIED]**

- `requirements.txt` is unpinned (`>=`).
- It includes packages left over from a previous English–Chinese pilot: `opencc`, `jieba`, `spacy-pkuseg`, `mlx-lm` and `anthropic`.
- `pyproject.toml` describes the package as "Helpers for the English-Chinese paraphrase-decay pilot".
- The local Python is **3.9.6**, but `pyproject` requires ≥ 3.11. Locally, torch 2.6.0, pandas 2.2.3 and pytest 8.2.0 are installed.

**Model checkpoints referenced anywhere in the repository.** **[VERIFIED]**

| Checkpoint | Where | Status |
|---|---|---|
| `google/madlad400-3b-mt` | MADLAD worker | Used (round 1) |
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen notebooks | Used (round 1) |
| `facebook/nllb-200-3.3B` | `config.yaml` (PROVISIONAL, "CC-BY-NC-4.0; sentence-split input") | Never run in the repository |
| `Qwen/Qwen3.5-9B`, `claude-haiku-4-5` | `config.yaml` paraphrasers/translators (old v2 design) | Not part of the current plan |
| `roberta-large`, `sentence-transformers/all-mpnet-base-v2`, `en_core_web_sm` | `pilot.metrics` | Baseline metric models |
| `intfloat/multilingual-e5-base`, `sentence-transformers/LaBSE`, `Unbabel/wmt22-cometkiwi-da`, `BAAI/bge-m3` | Qwen notebooks / config | Optional metric models |
| A Llama checkpoint | Nowhere | Must be chosen |

**Preprocessing.** **[VERIFIED]**

- MADLAD: regex sentence split, plus token-budget packing.
- Qwen: none; the whole document goes in as one input.
- `requirements.txt` lists `pysbd` "for NLLB input", but no NLLB code exists.
- `config.yaml` describes `strip_prompt_prefix` (removing the shared 30-word prefix), but no code implements it.

**Kaggle.** **[VERIFIED]** from `config.yaml`:

- Kaggle user `chwong0627`, dataset `cs6120-pilot-jobs`, kernel `cs6120-pilot-runner`.
- A note dated Oct 11 2026 says a 2×T4 smoke test passed with transformers 5.16.1 and torch 2.11 cu128.
- It also warns that "`pip install vllm` replaced Kaggle's CUDA 12.8 torch and broke it".
- None of that Kaggle job code is in the repository.

---

## 7. Configuration drift

`config.yaml` describes a **different experiment** ("pilot v2: translate-then-paraphrase" with zh/es, Claude and Qwen3.5-9B paraphrasers, xsum+yelp, 50 keys per dataset). It is not the current four-translator round-trip design. Every value is marked PROVISIONAL, and `go_no_go.locked: false`. **[VERIFIED]**

For `Sam/` the takeaway is:

- **Keep** `seed: 6120`, the corpus path convention, the NLLB language codes (`eng_Latn`, `spa_Latn`) and the T4 precision notes.
- **Do not inherit** the rest of the file.

---

## 8. Kaggle portability blockers

Everything that ties the existing code to one machine **[VERIFIED]**:

- **MADLAD:** `Path.home()/'degree_zero_madlad_v100'`, `/scratch/$USER`, a private venv, and an `nvidia-smi` check.
- **Qwen:** `google.colab.drive.mount`, `/content/drive/MyDrive/degree_zero`, `google.colab.userdata`, `files.download`/`files.upload`.
- **Corpus:** read from `raw.githubusercontent.com/.../main/...`, which is not pinned to a commit and needs internet; or from the sibling folder `../Ship_of_theseus_paraphrased_copus`.
- **`pilot.config.ROOT`:** `Path(__file__).parents[2]`. This works if the repository is cloned or attached on Kaggle, but `config.yaml` → `corpus_src` is a relative sibling path.
- **Downloads during runs:** the MUSE dictionary via `wget`; COMETKiwi needs a gated licence plus `HF_TOKEN`.

---

## 9. Risks

### Data leakage and validity

1. **Shared 30-word prompt prefix.** Every LLM-source text begins with the Human text's first 30 words. **[VERIFIED]**
   - For RQ2 this text is human-written inside "LLM" documents. It dilutes source signal, and it is identical across the 7 classes of a key.
   - For RQ1 it is easy text to preserve, so it inflates retention for LLM sources.
   - Decide whether to strip it, consistently, before any metric or classifier.
2. **Key-level grouping.** **[VERIFIED]**
   - The 200 texts share 31 articles. RQ2 and RQ3 classifiers must split by `key`, not by row.
   - In RQ3 the same document appears under all four translators. A row-level split would put a document's T1 from model A in training and its T1 from model B in test, so the classifier would learn content.
3. **Source is confounded with length** (median 99 to 355 words by source). **[VERIFIED]** A length-only baseline classifier is required before any stylometric accuracy can be read.
4. **"Source" ≠ author.** Six of the seven classes are LLM families (OpenAI, LLAMA, PaLM, Tsinghua, BigScience, Eleuther-AI) and one is "Human" (many BBC journalists). RQ2 must be framed as *source/generator attribution*.
5. **Non-random, clustered sample.** 31 keys, mostly consecutive IDs from `xsum-4708` to `xsum-4738`. Generalisation claims must be limited to "this XSum subset".
6. **Small RQ2 test set.** About 28 items per class, so confidence intervals will be wide. RQ3 is better powered (about 197 × 4 per round).

### Comparison validity

7. **Chunking differs between pathways.** MADLAD splits by sentence and packs to 300 tokens; NLLB will need sentence-level input; the LLMs translate whole documents. This changes sentence-boundary behaviour, so it is part of the "fingerprint" (RQ3) and of style decay (RQ1). Record it per pathway and discuss it. Do not hide it.
8. **Flagged items propagate.** A truncated or expanded T1 becomes the input for round 2. Define now whether chains continue on flagged items, and define the analysis set as items valid in *all* pathways and *all* rounds.
9. **Baseline coverage gaps.** PaLM reaches T3 for only 188 of the 200 items, ChatGPT for 196. Matched comparisons must use the intersection, or report the n for each cell.
10. **Hardware or library drift in round 1 reuse.** MADLAD round 1 ran on a V100 with transformers 4.46.3. Kaggle T4/P100 with newer libraries may produce slightly different greedy outputs. The chain stays internally consistent (one model, one set of settings), but record hardware and versions for each round.
11. **The MADLAD E5 and BERTScore numbers** in the README are unverifiable. Do not cite them as results of the new pipeline.

### Reproducibility

12. The corpus is pulled from upstream `main`, not a pinned commit. Pin commit `4423d8c` or snapshot the files into a Kaggle Dataset.
13. No HF model revision hashes are recorded for MADLAD or Qwen. Record `revision=` from now on.
14. The Qwen texts live only on a teammate's Google Drive. If they cannot be recovered, round 1 must be regenerated.
15. `requirements.txt` is unpinned and includes unrelated packages.

### Licensing

All of these are **[ASSUMED]** unless marked; confirm on each model card.

16. **Upstream corpus:** no licence file **[VERIFIED]**. The XSum texts derive from BBC articles. Treat them as research-use only. Do not publish a public Kaggle Dataset containing the corpus; keep it private.
17. **NLLB-200:** CC-BY-NC-4.0, which is non-commercial. Academic coursework is fine; record it. `config.yaml` states the same licence.
18. **Llama:** gated on Hugging Face and covered by the Llama Community License. Each runner needs an approved account plus an `HF_TOKEN` stored as a Kaggle Secret.
19. **MADLAD-400 and Qwen3:** Apache-2.0 (assumed).
20. **COMETKiwi:** a gated licence that is non-commercial.

### Compute

21. **MADLAD round-trip time:** about 16 s per item on a V100 (median 17.85 s, from the `seconds` column) **[VERIFIED]**. So rounds 2–3 for 200 items take about 1.8 GPU-hours on a V100. **[ASSUMED]** A T4 could be 2–3× slower.
22. **vLLM on Kaggle:** reported to break Kaggle's torch **[VERIFIED, `config.yaml` note]**. Plan the LLM pathways with plain `transformers` batched generation, or with a vLLM version pinned and tested in an isolated step.
23. **Kaggle session limits:** about 12 hours per session and a weekly GPU quota **[ASSUMED]**. Every generator must checkpoint after each item and resume.
24. **fp16 on T4 (no bf16):** T5-family models such as MADLAD can overflow in fp16 **[ASSUMED]**. MADLAD round 1 used fp16 successfully on a V100. Keep fp16 for consistency and watch for empty or garbage outputs.

---

## 10. Status

| Item | Exists | Missing |
|---|---|---|
| 200-item sample definition | Exact, reproducible procedure; IDs, sources and T0 texts in `chain_es_madlad4003b.csv` | A standalone manifest file; a pinned corpus snapshot for Kaggle |
| MADLAD pathway | Round 1 (S1, T1) for 200 items, 3 flagged; lexical and stylistic metrics for 197 | Rounds 2–3; HF revision; E5 and BERTScore code and per-item values |
| Qwen pathway | Prompts, decoding settings, round-1 *metrics* for all 3,227 items incl. the 200 | **Round-1 texts** (Drive only); `bad` flags; `run_config.json`; rounds 2–3 |
| NLLB pathway | Checkpoint candidate in `config.yaml`; `pysbd` in requirements | Everything else |
| Llama pathway | Nothing | Everything, including the checkpoint choice |
| English baseline texts | Full corpus in the sibling clone | A Kaggle-accessible copy |
| English baseline metrics | Reusable code (`pilot.metrics`); aggregate CSVs over the full corpus | Per-item scores for the 200 IDs; the inline notebook logic as a module |
| RQ2 source attribution | Labels; a disjoint `xsum_train.csv` training pool | All classifier code |
| RQ3 translator fingerprinting | Nothing | All classifier code |
| Tests | Nothing | Everything |

### Blockers that need your input

1. **Qwen round-1 texts:** ask the teammate who ran `degree_zero_chain.ipynb` for `chain_es_Qwen3-4B-Instruct-2507.csv`, `generations.jsonl` and `run_config.json` from `MyDrive/degree_zero/`. Otherwise round 1 has to be regenerated.
2. **Qwen prompt for the main pathway:** free (F) or literal (L)?
3. **NLLB and Llama checkpoints** (recommendations in the plan).
4. **Flagged-item policy** (continue chains, or stop them) and the **prefix-stripping policy**.
5. **Whether MADLAD round 1 is reused** (V100 outputs) or regenerated on Kaggle.
