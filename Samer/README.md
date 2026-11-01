# Multilingual Ship of Theseus: round-trip translation research (Samer Ahmed)

Three research notebooks on what happens to English texts that are repeatedly translated English → Spanish → English.
The main findings are in **[SUMMARY.md](SUMMARY.md)**.

## Notebooks
The notebooks are saved with their outputs and are readable directly on GitHub.

| Notebook | Question |
|---|---|
| [01_RQ1_Translation_Decay](notebooks/01_RQ1_Translation_Decay.ipynb) | How do lexical, semantic, stylistic and information preservation decay over three round trips, and do translators differ? |
| [02_RQ2_Source_Attribution](notebooks/02_RQ2_Source_Attribution.ipynb) | Can the original source of a text (Human or one of six LLMs) still be identified after translation? |
| [03_RQ3_Translator_Fingerprint](notebooks/03_RQ3_Translator_Fingerprint.ipynb) | Can the translation model be identified from the translated English text? |

## Data and sample sizes
- **Corpus:** XSum only. There are 200 original English documents (T0): 31 original news articles, each written by
  up to 7 sources. The **seven source labels** are Human (the XSum article) and six LLM generators: BigScience,
  Eleuther-AI, LLAMA, OpenAI, PaLM and Tsinghua.
- **Translators:** three translators, each run independently from T0 through three round trips
  (T0 → S1 → T1 → S2 → T2 → S3 → T3):
  - MADLAD-400-3B: 196/200 complete chains;
  - NLLB-200-1.3B: 197/200;
  - Qwen3-4B-Instruct: 200/200.
  Phi-4-mini is **not yet included**.
- **Matched cohort:** 193 documents from all 31 articles, complete for every translator. It is used for every
  cross-translator comparison.
- **Attrition:** invalid generations (repetition loops, token-cap failures) are kept as attrition. They are never
  repaired and never fed into a later round.
- **Statistical unit:** the original article. Confidence intervals resample articles. Classifier splits are grouped by
  article, so no article appears in both training and test data, regardless of source, translator or round.

## Contents
| Folder | What |
|---|---|
| `notebooks/` | the three executed notebooks |
| `figures/` | the 13 figures shown in the notebooks (`rq1_*`, `rq2_*`, `rq3_*`) |
| `results/coverage/` | translators, complete chains, counts by stage and by source label |
| `results/rq1/` | metric means and medians with 95% article-cluster CIs, per translator × stage. The tables are: `summary_cumulative` (T0→Tk, matched and all-available cohorts), `summary_step` (Tk-1→Tk), `paired_differences`, `convergence`, `information_flags`, `baselines` and `bertscore_T3_by_source` |
| `results/rq2/` | source-attribution accuracy and macro-F1 (mean ± SD over 5 repeats, 95% CI, n) per model × translator × round, the opening-removed ablation, per-class P/R/F1 and confusion counts |
| `results/rq3/` | translator-identification scores: main, cross-round, leave-one-source-out, by source, typography controls, per-class, confusion counts, top features, and length/segmentation descriptives |

All tables are aggregates. No document text or per-document predictions are published here. RQ1's examples section
shows a few 25-word excerpts only.

## Metrics
- **Lexical:**
  - BLEU (sacrebleu);
  - chrF;
  - ROUGE-L;
  - word retention;
  - word edit rate;
  - length ratio;
  - exact equality.
- **Semantic:**
  - E5 cosine (`intfloat/e5-base-v2`);
  - MPNet cosine (`all-mpnet-base-v2`);
  - BERTScore F1 (`roberta-large`, layer 17), raw and baseline-rescaled.
  Long texts are embedded in chunks and windows, never truncated.
- **Style:**
  - function-word cosine;
  - punctuation distance;
  - POS and dependency-label Jensen-Shannon divergence;
  - character 3-gram cosine;
  - Burrows' Delta;
  - sentence-length and sentence-count ratios.
- **Information preservation:** entity recall, number recall and sentence coverage. These are **heuristic warnings,
  not validated factual errors**.
- **Comparisons:** every metric is computed against the original (T0 → Tk) and between consecutive rounds
  (Tk-1 → Tk). Two reference pairs give scale:
  - unrelated texts;
  - the same article written by a different author.

## Reproducibility and limitations
- **Executed outputs:** the notebooks were executed top to bottom from a clean kernel (Python 3.12, CPU). Seed 6120,
  5 article-grouped folds × 5 repeats, 2,000 bootstrap samples.
- **What re-running needs:**
  - the shared feature pipeline;
  - the local text tables (XSum texts and translations).
  These are kept out of this public folder: the upstream corpus has no licence file, and the translations derive
  from it. They live in the development workspace (branch `feature/sam-experiments`, folder `Sam/`), which
  documents the full pipeline.
- **Limitations:**
  - only 31 articles, so intervals are wide;
  - one language pair, one dataset and greedy decoding;
  - the MT systems translate sentence by sentence while Qwen translates whole documents (a segmentation
    confound);
  - the information-loss flags are unvalidated heuristics.

Contributor: Samer Ahmed.
