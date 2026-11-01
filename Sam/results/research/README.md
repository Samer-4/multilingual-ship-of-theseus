# Research results (RQ1-RQ3): MADLAD, NLLB, Qwen round trips on 200 XSum texts

Text-free, committed outputs of the shared research layer and the three notebooks. For a quick overview without
running anything, start with [`SUMMARY.md`](SUMMARY.md). Then open the executed notebooks on GitHub:
- [`Sam/notebooks/01_RQ1_Translation_Decay.ipynb`](../../notebooks/01_RQ1_Translation_Decay.ipynb)
- [`Sam/notebooks/02_RQ2_Source_Attribution.ipynb`](../../notebooks/02_RQ2_Source_Attribution.ipynb)
- [`Sam/notebooks/03_RQ3_Translator_Fingerprint.ipynb`](../../notebooks/03_RQ3_Translator_Fingerprint.ipynb)

No file here contains corpus or generated text. The notebooks show short excerpts (at most 70 words) in their
example cells only.

## Join keys
| Key | Meaning |
|---|---|
| `item_id` | one original T0 document = `dataset\|source_label\|article_id` (200 documents) |
| `article_id` | the ORIGINAL XSum article (31). **The statistical grouping unit**: all CIs resample articles, and all classifier folds split by article |
| `source_label` | who wrote T0: Human, BigScience, Eleuther-AI, LLAMA, OpenAI, PaLM, Tsinghua |
| `translator` | `madlad-3b`, `nllb`, `qwen3-4b` (discovered from `data/generated/*_final/`); `original` for T0; `baseline` for reference pairs |
| `stage` | T0 (original), S1-S3 (Spanish), T1-T3 (English after 1-3 round trips) |
| `text_id` | `item_id\|translator\|stage`, unique per text |
| `ref_sha256`, `hyp_sha256` | sha256 of the two texts of a pair; the metric caches are keyed by these |
| `matched` | the document's chain is complete (all six stages valid) for **every** translator: 193 documents |

## Contents
| Path | What one row is |
|---|---|
| `data_coverage/translators.csv` | one translator: checkpoint, generation config id, weights id, final dataset sha256, complete chains |
| `data_coverage/chains.csv` | (document, translator): status of S1..T3 (valid / invalid / blocked), complete, first failure, matched |
| `data_coverage/cohort.csv` | document: complete flag per translator, matched flag |
| `data_coverage/coverage_by_stage.csv`, `coverage_by_source.csv` | counts by translator × stage (× source) |
| `data_coverage/lineage_report.json` | lineage check (every S1 reads its T0; every later stage reads the same chain's previous valid stage), config ids, texts sha256 |
| `shared_features/text_features.parquet` | English text: lengths, sentence statistics, MATTR, punctuation / function-word / UPOS / dependency rates, rates of the 150 most frequent T0 words, entity / number counts |
| `shared_features/reference_lists.json` | function words and Burrows MFW list + T0 means/SDs (fixed from the 200 originals only) |
| `metrics/pair_metrics.parquet` (+ `.csv.gz`) | one comparison: `kind` = cumulative (T0→Tk), step (Tk-1→Tk), unrelated / cross_source (baselines), with every lexical, semantic, style and information metric and the heuristic `flag_*` columns |
| `rq1/summary_cumulative.csv`, `summary_step.csv` | mean, median, 95% article-cluster CI, n pairs, n articles per (translator, stage, metric); `cohort` = matched / all_available |
| `rq1/paired_differences.csv` | translator a − b on the same documents at T1 and T3, with CI |
| `rq1/convergence.csv`, `information_flags.csv`, `baselines.csv`, `bertscore_T3_by_source.csv`, `examples.csv` | convergence, flag rates, reference pairs, by-source semantic retention, example ids |
| `rq2/predictions.parquet` | one out-of-fold prediction: text, model, repeat, fold, eval set, y_true, y_pred, `ablation` (full_text / opening_removed) |
| `rq2/scores.csv`, `scores_opening_removed.csv`, `per_class.csv`, `confusion_counts.csv` | accuracy / macro-F1 (mean ± SD over repeats, 95% article bootstrap CI, n), per-class P/R/F1, confusion counts |
| `rq2/splits.csv`, `config.json`, `shared_opening_audit.csv` | article → fold per repeat and seed; labels and models; measured shared-opening lengths |
| `rq3/predictions.parquet`, `scores_main.csv`, `scores_cross_round.csv`, `scores_leave_one_source_out.csv`, `scores_by_source.csv`, `scores_typography_controls.csv`, `per_class.csv`, `confusion_counts.csv`, `top_features.csv`, `length_segmentation_descriptives.csv`, `splits.csv`, `config.json` | the same, for translator identification |
| `figures/*.png` | every figure shown in the notebooks |
| `run_manifests/feature_run.json` | fingerprints, model revisions, devices, library versions, counts and timings of the feature run |
| `run_manifests/notebook_runs.json` | last clean-kernel execution of each notebook: return code, errors, figures, seconds |

Example join (pandas):
```python
p = pd.read_parquet("Sam/results/research/metrics/pair_metrics.parquet")
f = pd.read_parquet("Sam/results/research/shared_features/text_features.parquet")
t3 = p[(p.kind == "cumulative") & (p.hyp_stage == "T3") & p.matched]
t3 = t3.merge(f[["text_id", "n_sentences"]], left_on="hyp_text_id", right_on="text_id")
```

## Metrics (definitions in `src/sot_research/metrics.py`, `neural.py`, `features.py`)
- **Lexical:**
  - BLEU (sacrebleu sentence BLEU, exp smoothing);
  - ROUGE-L F (stemmed);
  - chrF;
  - word retention (clipped multiset recall);
  - word edit rate (word Levenshtein / reference words);
  - length ratio;
  - exact equality.
- **Semantic:**
  - E5 cosine (`intfloat/e5-base-v2`, "query: " prefix);
  - MPNet cosine (`sentence-transformers/all-mpnet-base-v2`);
  - BERTScore P/R/F1 (`roberta-large`, layer 17, no idf; F1 also baseline-rescaled).
- **Long texts:** never truncated. E5 and MPNet use sentence-packed chunks averaged by token count. BERTScore uses
  overlapping 510-token windows (281 texts needed them).
- **Style:**
  - function-word cosine;
  - punctuation L1;
  - UPOS and dependency-label Jensen-Shannon divergence;
  - character 3-gram cosine;
  - Burrows' Delta (z-scores from T0);
  - sentence-length and sentence-count ratios.
- **Information (HEURISTIC warnings, never validated errors):**
  - entity recall (spaCy entities of the reference found verbatim in the hypothesis);
  - number recall and added numbers;
  - quantity recall;
  - sentence coverage (best MPNet match ≥ 0.6).
  The `flag_*` columns are true when anything was lost.

Model revisions, devices and library versions are recorded in `run_manifests/feature_run.json`. The encoder snapshots
used were:
- `intfloat/e5-base-v2` at `f52bf8ec8c`;
- `all-mpnet-base-v2` at `e8c3b32edf`;
- `FacebookAI/roberta-large` at `722cf37b1a`.

All were run on CPU.

## Reproduce
The text tables are not committed:
- the T0 corpus copy has no licence file upstream;
- the generated translations derive from it.

They live locally under `Sam/data/` (gitignored):
- `data/staged/t0_manifest.parquet`;
- `data/generated/<translator>_final/`;
- `data/research/texts.parquet` and caches.

Teammates can obtain them in either of two ways:
- rebuild Stage 2 from the Ship of Theseus corpus (`Sam/scripts/build_stage2.py`) and download the translation runs
  from Kaggle (`Sam/scripts/download_kaggle_outputs.py`, which needs access to the `samahmed17/sam-sot-mt-*`
  kernels);
- ask Sam for the private Kaggle dataset.

Then, from the repo root:
```bash
/opt/homebrew/bin/python3.12 -m venv Sam/.venv-research
Sam/.venv-research/bin/pip install -r Sam/requirements-research.txt
Sam/.venv-research/bin/python -m spacy download en_core_web_sm
Sam/.venv-research/bin/python -m ipykernel install --prefix Sam/.venv-research --name sot-research
# encoders (about 2.3 GB, Hugging Face cache): intfloat/e5-base-v2, sentence-transformers/all-mpnet-base-v2, FacebookAI/roberta-large
Sam/.venv-research/bin/python Sam/scripts/research/compute_features.py      # corpus + features + metrics (resumes; about 30 min on CPU)
Sam/.venv-research/bin/python Sam/scripts/research/build_notebooks.py       # regenerate the .ipynb from the builder
Sam/.venv-research/bin/python Sam/scripts/research/run_notebooks.py         # clean-kernel execution, all three in parallel
Sam/.venv-research/bin/python -m pytest Sam/tests/research                  # research-layer tests
```
- **Determinism:** seeds are in `Sam/configs/research.yaml` (`seed: 6120`, 5 folds × 5 repeats, 2000 bootstrap
  samples).
- **Long runtime:** the feature run took about 28 minutes on an 8-core M1 CPU, mostly BERTScore. Apple MPS is
  disabled by default because torch 2.15 aborts on macOS 14.3; set `SOT_RESEARCH_DEVICE=mps` to try it.

## Adding a translator (e.g. Phi-4-mini)
1. Finalize it: `Sam/.venv/bin/python Sam/scripts/finalize_madlad.py --translator phi4-mini`. This writes
   `data/generated/phi4-mini_final/`, verified for lineage, hashes and attrition.
2. Run `compute_features.py`. It discovers the new translator, re-checks lineage and rebuilds the cohorts. Only the
   new texts and the pairs that involve them are computed; everything else comes from the caches.
3. Run `run_notebooks.py`.
   - RQ1 and RQ3 include the new translator automatically: colours, legends, paired comparisons and the
     1/K chance level.
   - RQ2 keeps its seven source labels and gains three evaluation sets.
4. The matched cohort may shrink if the new translator has invalid stages. The written interpretation cells
   (`Sam/scripts/research/interpretations.md`) quote the current numbers and should be re-read and updated.
