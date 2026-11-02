# Protocol: RQ2, source attribution

**Question.** After a text has been rewritten, can a classifier trained on **original** texts still name its original
**source**? The sources are Human or one of six LLMs. And does round-trip translation preserve that signal
differently from English paraphrasing?

This follows Samer Ahmed's RQ2 protocol (`sot_research`, branch `feature/sam-experiments`), ported verbatim into
`src/pilot/attribution.py`. Paraphrase results (ours) and round-trip translation results (Samer's) therefore come from
identically defined experiments. The RQ3 protocol is in `docs/protocol_rq3.md`, and RQ1 metrics are in
`docs/metric_standard.md`.

| | |
|---|---|
| Label | `source_label`: Human, BigScience, Eleuther-AI, LLAMA, OpenAI, PaLM, Tsinghua (K = 7, chance 0.143) |
| Trained on | T0 originals only. The model never sees a rewrite |
| Evaluated on | Held-out articles' T0, and each rewriter's T1, T2, T3 |
| Rewriters | Paraphrasers ChatGPT, PaLM, Dipper, Pegasus (ours). Translators MADLAD-400-3B, NLLB-200-1.3B, Qwen3-4B (Samer's predictions) |
| Notebook | `notebooks/rq2_source_attribution.ipynb` |
| Results | `results/rq2_attribution/` |

---

## 1. Data and scopes

| Scope | Documents | Used for | Status |
|---|---|---|---|
| **Matched XSum** | Samer's 200 XSum T0 documents (31 articles × up to 7 sources), restricted to those **all 3 translators and all 4 paraphrasers** completed to T3: **178 documents, 31 articles** | Paraphrase vs translation | Done |
| **Full corpus** | Every completer original of the 7 SoT datasets (`data/derived/sot_sample.parquet`): **16,682 originals, 2,996 articles**, with the ChatGPT / PaLM / Dipper / Pegasus rewrites T1-T3 | Paraphrase only, with statistical power and per-dataset results | Done (13.0M predictions) |

- **Identical originals:** Samer's T0 texts are **byte-identical** to ours (SHA-256 checked for all 200).
- **Translations:** they exist only for the 200 XSum documents and are **not in the repo** (corpus licence). Only Samer's text-free per-text predictions are used (`scripts/import_samer_rq1.py`).
- **Paraphrasers:** the four primary variants. The intensity variants (Dipper low/high, Pegasus slight) are not used.
- **Attrition:** invalid generations count as attrition. They are never repaired, and the cohort rules above exclude documents a rewriter did not complete.

## 2. Splitting (leakage rule)

- **Folds:** formed over **original article IDs**. Every text derived from an article (all sources, every rewriter, every round) is on the same side of a split.
- **Assignment** (`article_folds`): the unique article IDs are sorted, shuffled with `numpy.random.default_rng(seed + repeat)`, then dealt round-robin into 5 folds.
- **Repeats:** **5 folds × 5 shufflings**, seed **6120**. Out-of-fold predictions are pooled over the repeats.
- **Full corpus:** each dataset has its own folds and its own classifiers, since genres and Human texts differ between datasets.

## 3. Models (fixed hyperparameters, never tuned)

| Name | Features | Classifier |
|---|---|---|
| `tfidf_word` | TF-IDF word 1-2-grams, `min_df=2`, sublinear tf | Logistic regression, C = 10, balanced class weights |
| `tfidf_char` | TF-IDF `char_wb` 2-5-grams, `min_df=2`, sublinear tf, ≤ 200,000 features | Logistic regression, C = 10, balanced |
| `stylometric` | 187 per-text features (below), median-imputed, standardised | Logistic regression, C = 1, balanced |
| `length_only` | `log(1+words)`, `log(1+sentences)`, mean sentence length, mean word length | as `stylometric` |

**Stylometric features** (spaCy `en_core_web_sm`, Samer's `doc_features`). All rates are per token:
- surface scalars: sentence length mean/SD, word length, MATTR, TTR, capitalisation rate, dependency distance and depth;
- rates of 17 punctuation marks;
- rates of 100 function words;
- rates of 17 UPOS tags;
- rates of every dependency label.

**Function-word list:**
- **matched XSum:** Samer's list, fitted on the 200 XSum T0 texts, so his classifiers are reproduced exactly;
- **full corpus:** the team standard's list, fitted on held-out reference texts (`data/derived/sot_style_vocab.json`), because his XSum list does not suit the other six datasets.

## 4. Experiment

For each scope, text variant and model:
1. Train on **T0 texts of the training articles** (all 7 sources).
2. Predict the held-out articles' T0 texts and each rewriter's T1, T2 and T3 texts.

Evaluation sets are `original|T0` and `<rewriter>|Tk`.

### Text variants

| Variant | Definition | Role |
|---|---|---|
| `full_text` | texts as they are | supplement |
| `opening_removed` | the **measured shared opening** removed from every text, training and test | **primary** |
| `truncated_50` | `opening_removed`, then every text cut to its first 50 whitespace words | length control (paraphrase only; needs the texts) |

**Shared opening** (`shared_opening_lengths`): the six LLM sources were generated from the first ~30 words of the human article. For each text, it is the longest whitespace-word prefix shared verbatim with another source's text of the same article, rewriter and stage.
- It is measured per text, never assumed.
- At T0 it recovers the ~30-word human opening.
- In rewrites it is shorter, because paraphrasers and LLM translators render the opening differently in each text.
- It carries article content, not source style. On the matched set, removing it raises T0 macro-F1 (char n-grams 0.42 → 0.47).

## 5. Scoring and uncertainty

- **Primary metric:** macro-F1 over the 7 sources. Accuracy is also reported.
- **Point estimate:** mean over the 5 repeats. The SD over repeats is reported.
- **95% CI:** article-cluster bootstrap over the predictions pooled across repeats; articles are resampled with replacement, 1,000 resamples.
  - Matched XSum: `scores` (Samer's implementation).
  - Full corpus: `scores_fast`, vectorized. Its point estimates and SDs are identical to `scores`; CI bounds differ only by bootstrap noise (≤ 0.009 on the XSum set).
- **Per class:** precision, recall and F1 per source, plus row-normalised confusion matrices, pooled over repeats.

### Derived measures
- **Retained attribution:** `(F1_Tk − chance) / (F1_T0 − chance)`, with T0 from the same model and variant. 1 = as identifiable as the original, 0 = chance. This is RQ1's retained scale.
- **Gain over length-only:** `F1(model) − F1(length_only)` at the same rewriter, stage and variant.
- **Family gap (matched XSum):** macro-F1 at T3 of all translator predictions pooled, minus all paraphraser predictions pooled. CI from a **paired** article bootstrap: the same articles are resampled for both families.
- **Rewriter takeover:** share of texts attributed to OpenAI, counting only texts whose true source is *not* OpenAI. ChatGPT rewrites are compared with the T0 rate and the other rewriters; PaLM → PaLM likewise. Uses char n-grams, opening removed.
- **Full corpus:** every measure is computed per dataset and for "all" (the out-of-fold predictions of the 7 datasets pooled; the CI resamples articles).

## 6. Validation before pooling with Samer

On the matched XSum set, the port must reproduce Samer's outputs on identical inputs (the T0 texts):

| Check | Result |
|---|---|
| Stylometric and length features, 200 T0 texts × 187 columns | identical (max abs diff 0) |
| Article-to-fold assignment, 31 articles × 5 repeats | identical to his `splits.csv` |
| Out-of-fold T0 predictions, 4 models × 2 text variants (8,000 predictions) | **100% identical** |

The checks are in notebook §1 and `results/rq2_attribution/port_agreement_T0.csv`. Because the classifiers are identical, his translator predictions and our paraphrase predictions are pooled directly on the 178-document cohort.

## 7. Running it

From `Sam/`, with `PYTHONPATH=src`:
```bash
python scripts/import_samer_rq1.py --files-only   # Samer's predictions, folds, reference lists, features (seconds)
python scripts/rq2_attribution.py                 # matched XSum (~3 min, local)
# full corpus, on Colab (> 10 min): notebooks/colab_rq2_attribution.ipynb runs scripts/rq2_full.py (~2-2.5 h)
```
Then run `notebooks/rq2_source_attribution.ipynb` top to bottom: §1-§5 cover matched XSum and §6 the full corpus. The notebook only reads cached predictions.

| Output (in `data/derived/`, not committed) | Content |
|---|---|
| `rq2_paraphrase_predictions.parquet` | matched XSum: one row per repeat × text × model × variant |
| `rq2_features.parquet` | Samer's stylometric features per text and variant |
| `rq2_full_predictions.parquet`, `rq2_full_texts_meta.parquet` | full corpus (dictionary-encoded, ~35 MB) |

Committed tables and figures are in `results/rq2_attribution/`.

## 8. Deviations from Samer's protocol, and why

| Ours | Samer's | Reason |
|---|---|---|
| Opening-removed texts are the **primary** analysis | Full text primary, opening removal as an ablation | Removing the opening raises T0 scores, and paraphrasers rewrite the opening differently in each text (as Qwen does), so full text treats the families inconsistently |
| Retained attribution, gain over length-only, 50-word truncation, rewriter takeover | — | Paraphrasers change length, and ChatGPT and PaLM have matching source classes |
| Common cohort of 178 documents | 193 (his matched set) | Both families must be scored on the same documents |
| Full corpus: per-dataset classifiers, team-standard function words, `scores_fast` | — (XSum only) | Genre differences, held-out vocabulary per the metric standard, compute |
| Bootstrap: 1,000 resamples | 1,000 in his RQ2 (2,000 in his RQ1) | Team decision still open (`docs/metric_standard.md`) |

## 9. Limitations
- **Matched XSum is small:** 31 articles (about 6 per test fold), so per-rewriter CIs are about ±0.05. Only the family gap and large differences are reliable.
- **Classical models only:** fixed, untuned classical models measure the *relative* survival of the source signal, not the maximum achievable attribution.
- **Length control is incomplete:** the 50-word truncation leaves short texts and sentence-level statistics. Length-only still reaches 0.32 at T0.
- **Takeover can only be measured for ChatGPT and PaLM:** six of the seven sources are LLM outputs, and only these two rewriters have a matching source class.
- **Uneven translator evidence:** the translation side relies on Samer's predictions. Text-dependent controls (truncation) exist for paraphrases only.
