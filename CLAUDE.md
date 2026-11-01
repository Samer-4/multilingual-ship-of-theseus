# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS 6120 (Northeastern) NLP research: does rewriting text through another language (English → Spanish → English round-trip translation) preserve meaning and style differently from English-only paraphrasing? The baseline is the Ship of Theseus (SoT) paraphrased corpus (Tripto et al.): 7 datasets (`cmv, eli5, sci_gen, tldr, wp, xsum, yelp`), rounds T0 (original) through T3, with paraphrasers ChatGPT, PaLM, Pegasus and Dipper. This is research code. There are no tests, linter or CI.

## Setup

```bash
pip install -r requirements.txt
pip install -e .                      # installs src/pilot as the `pilot` package (Python >= 3.11)
python -m spacy download en_core_web_sm
```

The SoT corpus is not in the repo. `config.yaml` expects an upstream clone at `../Ship_of_theseus_paraphrased_copus`, copied to `data/sot/`. Derived data goes in `data/derived/`, which is not committed. Only final result tables and figures are committed (`results/`, `Samer/results/`, `Samer/figures/`).

## Layout and how the pieces connect

- **`src/pilot/`**: the shared library, used only by the SoT baseline (`notebooks/sot_analysis.ipynb` and `scripts/`).
  - `config.py`: `load_config()` reads `config.yaml` and resolves every `paths.*` entry against the repo root (`ROOT`). `set_seeds()` seeds the RNGs.
  - `metrics.py`: corpus loading (`load_corpus` returns a long table with `dataset, source, key, variant, round, text`; `round` and `variant` are parsed from `version_name`), pair building (`cumulative` = T0 vs Tn, `step` = Tn-1 vs Tn), lexical scores, `LongBertScore` (roberta-large layer 17, baseline-rescaled; long texts use sliding windows instead of truncation), the chunked SBERT cosine, spaCy style features, and an article-clustered bootstrap CI.
- **`scripts/`**: background runners for the slow stages of `sot_analysis.ipynb`. They call the same `pilot.metrics` functions and write the same caches, so the notebook just loads the results. All of them read `data/derived/sot_sample.parquet` (and `sot_floors.parquet`), which the notebook creates first.
  ```bash
  python scripts/score_lexical.py [scope]   # scope defaults to "full" -> sot_pairs_lexical_<scope>.parquet
  python scripts/score_semantic.py          # BERTScore + SBERT -> sot_pairs_semantic.parquet, sot_floors_semantic.parquet
  python scripts/style_features.py          # spaCy features keyed by text_hash -> sot_style_features.parquet
  ```
  Every stage can be resumed: lexical saves chunk files to a `.parts/` folder, semantic skips ids already in its parquet cache, and style skips text hashes it has already processed. To rerun from scratch, delete the cache.
- **`experiments/{qwen,madlad}_translation/`** and **`Samer/notebooks/`**: self-contained round-trip translation notebooks that do **not** import `pilot`. They ran on remote GPUs (Northeastern Explorer HPC, V100), and the MADLAD notebook builds its own isolated environment. Results go to `results/{qwen,madlad}_translation/` and `Samer/results/`. `Samer/` covers three translators (MADLAD-400-3B, NLLB-200-1.3B, Qwen3-4B) over three round trips on 200 XSum T0 documents, with a matched cohort of 193; see `Samer/SUMMARY.md` for the findings.

## Gotchas

- spaCy `n_process > 1` deadlocks inside Jupyter on macOS. Run style extraction through `scripts/style_features.py`, not from inside the notebook.
- `score_lexical_chunked` processes data in chunks because passing 500k+ pairs to `pool.map` all at once runs out of memory. Keep it chunked.
- The statistical unit is the original article/key. Bootstrap CIs resample keys, and classifier splits must be grouped by article so that no article appears in both train and test.
- Invalid generations (repetition loops, token-cap failures) are kept as attrition. They are never repaired or fed into a later round.
- Qwen results use normalized *change* scores (lower = more preserved). MADLAD results use *preservation* scores (higher = better). Don't compare the two directly.
- Parts of `config.yaml` and the `pyproject.toml` description are left over from an earlier English-Chinese/Kaggle pilot. They mention `docs/`, `notebooks/archive/`, Kaggle jobs and `zh` targets, none of which exist in this repo. The `paths` and `seed` entries are still in use.
- Notebooks are live documents in Jupyter. Edit and run them through the Jupyter MCP tools, not by writing the `.ipynb` JSON directly.
