# Protocol: RQ3, rewriter fingerprint

**Question.** Given a rewritten text, can a classifier tell **which model rewrote it**? Do English paraphrasers and
round-trip translators leave different kinds of fingerprints?

This follows Samer Ahmed's RQ3 protocol (`sot_research`, branch `feature/sam-experiments`), ported verbatim into
`src/pilot/attribution.py`. The same code reproduced his RQ2 predictions 100% (see `docs/protocol_rq2.md` §6). The
RQ2 protocol is in `docs/protocol_rq2.md`, and RQ1 metrics are in `docs/metric_standard.md`.

| | |
|---|---|
| Label | `rewriter`. Paraphrasers: ChatGPT, PaLM, Dipper, Pegasus (K = 4, chance 0.25). Translators: MADLAD-400-3B, NLLB-200-1.3B, Qwen3-4B (K = 3, chance 0.333) |
| Inputs | The rewritten English text only. Metadata and source labels are never features |
| Trained on | Rewritten texts: T1-T3 pooled, or a single round |
| Evaluated on | Held-out articles' T1, T2, T3 (per round and pooled) |
| Notebook | `notebooks/rq3_rewriter_fingerprint.ipynb` |
| Results | `results/rq3_fingerprint/` |

---

## 1. Data and scopes

| Scope | Documents | Used for | Status |
|---|---|---|---|
| **Matched XSum** | Samer's XSum documents that **all 4 paraphrasers and all 3 translators** completed to T3: **178 documents, 31 articles**. That gives 2,136 paraphrase texts and 1,602 translator texts | Paraphrasers vs translators | Done |
| **Full corpus** | Every completer original of the 7 SoT datasets: 16,682 originals, 2,996 articles, about 200k paraphrase texts | Paraphraser identification with statistical power, per dataset | Runner ready (§8), not yet run |

- **One family at a time:** paraphrasers are classified on our texts. Translators come from Samer's published predictions; the translations themselves are not in the repo (corpus licence).
- **Balance:** every class contributes the same documents at every round, checked by assertion. Original sources are therefore balanced across classes by construction.
- **Identical originals:** Samer's T0 texts are byte-identical to ours (SHA-256 checked).

## 2. Splitting (leakage rule)

- **Folds:** formed over **original article IDs**. An article's texts (every source, rewriter and round) are always on one side of a split, so no test text can be matched to a training text with the same content.
- **Assignment** (`article_folds`): the unique article IDs are sorted, shuffled with `numpy.random.default_rng(seed + repeat)`, then dealt round-robin into 5 folds.
- **Repeats:** **5 folds × 5 shufflings**, seed **6120**, except where noted in §4.
- **Full corpus:** folds and classifiers per dataset.

## 3. Models (fixed hyperparameters, never tuned)

| Name | Features | Classifier |
|---|---|---|
| `tfidf_word` | TF-IDF word 1-2-grams, `min_df=2`, sublinear tf | Logistic regression, C = 10, balanced class weights |
| `tfidf_char` | TF-IDF `char_wb` 2-5-grams, `min_df=2`, sublinear tf, ≤ 200,000 features | Logistic regression, C = 10, balanced |
| `stylometric` | Samer's 187 per-text stylometric features, median-imputed, standardised | Logistic regression, C = 1, balanced |
| `stylometric_no_sentence` | `stylometric` without sentence-boundary features: sentence-length mean/SD, full-stop rate, ROOT-dependency rate, capitalisation rate | as `stylometric` |
| `length_only` | `log(1+words)`, `log(1+sentences)`, mean sentence length, mean word length | as `stylometric` |

**Stylometric features** (spaCy `en_core_web_sm`). All rates are per token:
- surface scalars;
- rates of 17 punctuation marks;
- rates of 100 function words;
- rates of 17 UPOS tags;
- rates of every dependency label.

**Function-word list:**
- **matched XSum:** Samer's XSum-fitted list;
- **full corpus:** the team standard's held-out list (`data/derived/sot_style_vocab.json`).

## 4. Experiments

| Experiment | Train | Evaluate | Models | Repeats |
|---|---|---|---|---|
| **Main** | pooled T1-T3 of training articles | each of T1, T2, T3, and pooled | all 5 | 5 × 5 |
| **Cross-round** | one round (T1, T2 or T3) | every round | char n-grams, stylometric | 5 × 5 |
| **Leave one source out** | all original sources except one | texts of the held-out source | char n-grams, stylometric | 5 folds × 2 |
| **Typography** | pooled T1-T3 | each round | char n-grams on texts with ‘ ’ “ ” – — mapped to ASCII; stylometric without those 6 punctuation rates | 5 × 5 |

**What each check tests:**
- **Cross-round:** is the fingerprint the same at every round, or does it change as rewriting accumulates?
- **Leave one source out:** does it transfer to authors the classifier never saw?
- **Typography:** does it survive without orthographic cues such as curly quotes and long dashes?
- **No sentence features:** does it depend on how the rewriter segments sentences?
- **Length only:** how much of it is just length?

## 5. Scoring and uncertainty

- **Primary metric:** macro-F1 over the K classes. Accuracy is also reported.
- **Point estimate:** mean over the repeats. The SD over repeats is reported.
- **95% CI:** article-cluster bootstrap over predictions pooled across repeats, 1,000 resamples (`scores`). Leave-one-source-out uses 300 resamples and cross-round uses 500, as in Samer's RQ3.
- **Per class:** precision, recall and F1 per rewriter, plus row-normalised confusion matrices (pooled T1-T3).

### Comparing paraphrasers with translators
- **Chance-corrected macro-F1:** the families have different K, so raw macro-F1 is not comparable. Both are reported on `(F1 − 1/K) / (1 − 1/K)`, where 0 = chance and 1 = perfect.
- **Main experiment:**
  - Samer's translator predictions (trained on his 193 matched documents) are re-scored on the shared 178. The slightly larger training set, if anything, favours the translators.
  - The paraphrase − translation difference, pooled T1-T3, gets a **paired article bootstrap** CI: the same articles are resampled for both families, 1,000 resamples.
- **Secondary experiments:** his translator predictions for cross-round, leave-one-source-out and typography were not published. The comparison uses his summary tables on 193 documents and is **approximate**.

### Descriptive confound checks
Per rewriter and round:
- word and sentence count relative to the same document's T0;
- mean sentence length, comma rate and MATTR;
- share of texts containing curly quotes or long dashes (originals: 0.5%).

## 6. Running it

From `Sam/`, with `PYTHONPATH=src`:
```bash
python scripts/import_samer_rq1.py --files-only   # Samer's RQ3 predictions and folds (seconds)
python scripts/rq2_attribution.py                 # computes the stylometric features RQ3 reuses (~3 min)
python scripts/rq3_fingerprint.py                 # all four experiments on matched XSum (~10 min, local)
# full corpus, on Colab (> 10 min): notebooks/colab_rq3_fingerprint.ipynb runs scripts/rq3_full.py (~1 h)
```
Then run `notebooks/rq3_rewriter_fingerprint.ipynb` top to bottom. It only reads cached predictions.

| Output (in `data/derived/`, not committed) | Content |
|---|---|
| `rq3_paraphrase_predictions.parquet` | all four experiments, one row per repeat × text × model |
| `rq3_pool_features.parquet` | text-free descriptives per text (lengths, stylometric features, typography flag) |
| `rq3_full_predictions.parquet`, `rq3_full_pool_features.parquet` | full corpus: all four experiments per dataset, and text-free descriptives (T0 and pool) |

Committed tables and figures are in `results/rq3_fingerprint/`.

## 7. Deviations from Samer's protocol, and why

| Ours | Samer's | Reason |
|---|---|---|
| Chance-corrected macro-F1 alongside raw | Raw macro-F1 | The families have different K |
| Common cohort of 178 documents | 193 (his matched set) | Both families must be scored on the same documents |
| Paired article bootstrap for the family difference | — | Same documents and articles in both families |
| Full corpus: per-dataset classifiers, team-standard function words, possibly 1 repeat | — (XSum only) | Genre differences, held-out vocabulary per the metric standard, compute (§8) |

## 8. Full corpus

- **Runner:** `scripts/rq3_full.py`, run on Colab by `notebooks/colab_rq3_fingerprint.ipynb`.
  - 189 jobs: 7 datasets × (5 main models + 6 cross-round + 14 leave-one-source-out + 2 typography).
  - Each job saves its own part file, so a disconnect only means re-running the cells.
  - `--repeats` defaults to 1. With `--repeats 5`, leave-one-source-out uses 2 shufflings, as in Samer's protocol.
  - `--limit N` runs on the RQ2 smoke-test files (`rq2_full_*_limitN`). With 15 articles per dataset this takes 2.3 min locally.
- **Cohort:** per dataset, originals that all 4 paraphrasers completed at T1-T3. Balance is asserted per dataset and round. Every original in `sot_sample.parquet` qualifies (the sample already holds completers only).

- **Scope:** paraphraser identification per dataset on all 7 datasets, about 200k rewritten texts, with all four experiments.
- **Inputs:** reuses the texts and the `full_text` stylometric features saved by the RQ2 full-corpus run, so no new spaCy pass.
- **Where:** on Colab, in the same session after RQ2 finishes.
- **Measured cost:** one char n-gram fit on the largest dataset (XSum, 29k training texts) takes 74 s; one word n-gram fit takes 36 s.

| Option | CPU hours | Wall time (12 vCPUs) |
|---|---|---|
| Samer's full protocol (5 × 5; leave-one-source-out 5 × 2) | ~27 h | ~2.5-3 h |
| **Recommended: 5 folds × 1 repeat, all four experiments** | ~9 h | ~1 h |
| Main experiment only (5 × 1) | ~2 h | ~15 min |

**Why one repeat:** with about 2,400 articles per dataset, one fold split is already stable, and the CIs come from the article bootstrap, not from the repeats. The open decision is whether to use 1 or 5 repeats.

## 9. Limitations
- **Matched XSum is small:** 31 articles, so CIs are about ±0.04-0.05.
- **Approximate secondary comparisons:** the translator-side cross-round, leave-one-source-out and typography numbers come from Samer's summary tables on 193 documents.
- **Unequal class counts:** 4 paraphrasers vs 3 translators. The chance correction aligns the scales, but a 4-way task is not identical to a 3-way task.
- **Mixed families:** Dipper and Pegasus are dedicated paraphrase models, while ChatGPT and PaLM are general LLMs. Likewise, two of the translators are MT systems and one is an LLM.
- **Feature importances** describe the classifier, not causes.
