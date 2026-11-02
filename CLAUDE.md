# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 6120 (Northeastern) NLP research: does rewriting text through another language (English → Spanish → English round-trip translation) preserve meaning and style differently from English-only paraphrasing? The baseline is the Ship of Theseus (SoT) paraphrased corpus (Tripto et al.): 7 datasets (`cmv, eli5, sci_gen, tldr, wp, xsum, yelp`), rounds T0 (original) through T3, with paraphrasers ChatGPT, PaLM, Pegasus and Dipper. This is research code. There are no tests, linter or CI.

## Where to run compute

**Anything expected to take more than 10 minutes must run on Google Colab (the user has Colab Pro), not locally.** Estimate the runtime first, for example by timing a small `--limit` or sample run. If it's over 10 minutes, write or extend a Colab runner instead of running it on the Mac. Runners keep their caches on Drive so a disconnect only means re-running cells (see `Sam/notebooks/colab_semantic_scoring.ipynb` for the pattern). Short checks, smoke tests and assembling results from cached scores stay local.

## Setup

The repo has one folder per contributor: **`Sam/`** (Sam Wong: the `pilot` library, scripts, the SoT analysis and the metric standard) and **`Samer/`** (Samer Ahmed: round-trip translation RQ1-RQ3). Run Sam's code from inside `Sam/`; every path below without a prefix is relative to `Sam/`.

```bash
cd Sam
pip install -r requirements.txt
pip install -e .                      # installs src/pilot as the `pilot` package (Python >= 3.11)
python -m spacy download en_core_web_sm
```

The SoT corpus is not in the repo. `config.yaml` expects an upstream clone next to the repo (`../../Ship_of_theseus_paraphrased_copus` relative to `Sam/`), copied to `Sam/data/sot/`. Derived data goes in `Sam/data/derived/`, which is not committed. Only final result tables and figures are committed (`Sam/results/`, `results/`, `Samer/results/`, `Samer/figures/`).

## Layout and how the pieces connect

- **`src/pilot/`**: the shared library, used only by the SoT baseline (`notebooks/sot_analysis.ipynb` and `scripts/`).
  - `config.py`: `load_config()` reads `config.yaml` and resolves every `paths.*` entry against `ROOT`, the folder that holds `src/` (i.e. `Sam/`). `set_seeds()` seeds the RNGs.
  - `metrics.py`: corpus loading (`load_corpus` returns a long table with `dataset, source, key, variant, round, text`; `round` and `variant` are parsed from `version_name`), pair building (`cumulative` = T0 vs Tn, `step` = Tn-1 vs Tn), lexical scores, `LongBertScore`, `SentenceChunkEmbedder` (MPNet), spaCy style features plus `StyleScorer`, the retained scale (`retained`, `add_style_composite`), the unrelated-T0 floors (`t0_floor_pairs`), and an article-clustered bootstrap CI.
- **`docs/metric_standard.md`**: the agreed team metric definitions (content-word retention, rescaled BERTScore, length-weighted MPNet, combined style score from the union of both pipelines, floors from unrelated T0 pairs, everything reported on the retained scale). Any analysis, including Samer's `sot_research` package, must follow it before results can be pooled.
- **`scripts/`**: background runners for the slow stages of `sot_analysis.ipynb`. They call the same `pilot.metrics` functions and write the same caches, so the notebook just loads the results. All of them read `data/derived/sot_sample.parquet` (and `sot_floors.parquet`), which the notebook creates first.
  ```bash
  python scripts/score_lexical.py [scope]   # scope defaults to "full" -> sot_pairs_lexical_<scope>.parquet
  python scripts/score_semantic.py          # rescaled BERTScore + MPNet -> sot_pairs_semantic_v2.parquet, sot_floors_semantic_v2.parquet
  python scripts/style_features.py          # spaCy features keyed by text_hash -> sot_style_features_v2.parquet
  python scripts/score_extra.py lexical     # chrF, word edit rate, exact equality (CPU)
  # ~1 h+ together, so these run on Colab via notebooks/colab_extra_scoring.ipynb (code bundle: zip src scripts):
  python scripts/info_features.py           # spaCy entities/numbers/sentences -> sot_info_features.parquet, sot_sentences.parquet
  python scripts/score_extra.py information # entity/number/quantity recall (needs info_features)
  python scripts/score_neural_extra.py      # E5 cosine + sentence coverage (GPU)
  python scripts/import_samer_rq1.py        # Samer's text-free translation pair scores from his branch + agreement check (~3 min) -> notebook §5
  ```
  Semantic scoring takes ~8 h on a laptop; `notebooks/colab_semantic_scoring.ipynb` runs the same script on a Colab GPU (batched across originals, fp16 only if it stays within 0.002 of fp32) with caches on Drive, while style parsing runs locally. Run the scripts with `PYTHONPATH=src` if the active environment has another `pilot` installed (for example the `../../cs6120-pilot` venv, relative to `Sam/`). Every stage can be resumed: lexical saves chunk files to a `.parts/` folder, semantic skips ids already in its parquet cache, and style skips text hashes it has already processed. To rerun from scratch, delete the cache.
- **`experiments/{qwen,madlad}_translation/`** and **`Samer/notebooks/`** (repo root, not under `Sam/`): self-contained round-trip translation notebooks that do **not** import `pilot`. They ran on remote GPUs (Northeastern Explorer HPC, V100), and the MADLAD notebook builds its own isolated environment. Results go to `results/{qwen,madlad}_translation/` and `Samer/results/`. `Samer/` covers three translators (MADLAD-400-3B, NLLB-200-1.3B, Qwen3-4B) over three round trips on 200 XSum T0 documents, with a matched cohort of 193; see `Samer/SUMMARY.md` for the findings.

## Gotchas

- spaCy `n_process > 1` deadlocks inside Jupyter on macOS. Run style extraction through `scripts/style_features.py`, not from inside the notebook.
- `score_lexical_chunked` processes data in chunks because passing 500k+ pairs to `pool.map` all at once runs out of memory. Keep it chunked.
- The statistical unit is the original article/key. Bootstrap CIs resample keys, and classifier splits must be grouped by article so that no article appears in both train and test.
- Invalid generations (repetition loops, token-cap failures) are kept as attrition. They are never repaired or fed into a later round.
- The `*_v2` caches follow the metric standard. The older unversioned caches (in `cs6120-pilot/data/derived/`, a sibling of this repo) use the previous floors (reference-text pairs), the previous MPNet chunking and the previous style score, so don't mix the two.
- Qwen results use normalized *change* scores (lower = more preserved). MADLAD results use *preservation* scores (higher = better). Don't compare the two directly.
- Parts of `config.yaml` and the `pyproject.toml` description are left over from an earlier English-Chinese/Kaggle pilot. They mention `docs/`, `notebooks/archive/`, Kaggle jobs and `zh` targets, none of which exist in this repo. The `paths` and `seed` entries are still in use.
- Notebooks are live documents in Jupyter. Edit and run them through the Jupyter MCP tools, not by writing the `.ipynb` JSON directly.
