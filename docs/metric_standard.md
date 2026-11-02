# Team metric standard

Every rewriting analysis (English paraphrase chains, round-trip translation chains, all 7 Ship of Theseus
datasets) reports its metrics this way, so that results can be pooled and compared directly. The reference
implementation is `src/pilot/metrics.py`. `ref` = the earlier text (T0 for cumulative pairs), `hyp` = the
rewritten text.

## Decisions

| Topic | Standard |
|---|---|
| Word retention | Content words only: type (set) recall of `[A-Za-z]+(?:'[a-z]+)?` tokens, lowercased, with sklearn `ENGLISH_STOP_WORDS` removed from the reference. `lexical_scores` |
| Reported scale | Every metric is reported **normalized** on the retained scale (below). Raw scores are kept in the caches. |
| Floor | Per dataset: the mean score of **200 unrelated T0 pairs**, two originals from different keys (different articles), seed 6120. `t0_floor_pairs` |
| BERTScore | **Baseline-rescaled**: roberta-large, layer 17, no idf, bert_score 0.3.13 baseline. Texts over 510 tokens are embedded in overlapping windows (core 384 tokens) rather than truncated. `LongBertScore` → `bertscore_f1_rescaled` |
| MPNet cosine | `all-mpnet-base-v2`. Texts are split into sentences and packed into chunks of at most 384 tokens; each chunk is mean-pooled and L2-normalized; the document vector is the **token-count-weighted mean** of its chunks, re-normalized. Nothing is truncated. `SentenceChunkEmbedder` → `mpnet_cosine` |
| Style | One **combined score**, built from the union of both team pipelines (below). `StyleScorer`, `add_style_composite` |

## Retained scale

- Similarities (1 = identical): `retained = (s - floor) / (1 - floor)`.
- Distances (0 = identical; `punct_l1`, `pos_jsd`, `dep_jsd`, `burrows_delta`, `word_edit_rate`): `retained = (floor - d) / floor`.

The floor is the per-dataset **mean**, except for `word_edit_rate`, which uses the **median**: it is unbounded
(edits / reference words), and unrelated pairs with a very short reference reach 30-85, which pulls the mean to
~2.6 while the typical unrelated pair sits at ~1.0. With the mean, a near-total rewrite (Dipper high) would
score 0.62 retained; with the median it scores 0.04, in line with the other lexical metrics.

`retained = 1` means identical to T0, and `0` means no closer than an unrelated original from the same dataset.
Values can fall below 0. `retained(score, floor, metric)` applies the correct formula.

## Combined style score

There are 11 components, grouped into 5 families. A family score is the mean of its components' retained scores,
and `style` is the mean of the 5 family scores, so overlapping measures don't get extra weight.

| Family | Components |
|---|---|
| surface | `style_surface`: z-scored cosine of sentence-length mean/SD, word length, MATTR, capitalization rate, dependency depth |
| punctuation | `style_punctuation` (z-scored cosine of punctuation rates), `punct_l1` (L1 distance of punctuation rates) |
| function_words | `style_function_words` (z-scored cosine), `fw_cosine` (raw-rate cosine) over the 100 most frequent stop words; `burrows_delta` (mean absolute z-difference over the 150 most frequent words) |
| syntax | `style_syntax_pos` (z-scored UPOS cosine), `pos_jsd` (UPOS Jensen-Shannon divergence), `dep_jsd` (dependency-label JSD) |
| character | `style_charngram` (char_wb 2-4-gram TF-IDF cosine), `char3_cosine` (raw char 3-gram cosine) |

Everything that needs fitting is fitted on **held-out reference texts** (`train_datasets/`, 300 per dataset,
never paraphrased): the z-score means and SDs, the function-word and most-frequent-word lists, and the char
n-gram TF-IDF vocabulary. Fitting on the analysed T0 texts would let the evaluated texts shape the measure.

## Samer's extra metrics (adopted)

Ported from `sot_research` with identical definitions (checked: word edit rate, chrF, number/quantity
extraction and entity normalization give zero difference on test pairs).

| Metric | Definition | Kind | Computed by |
|---|---|---|---|
| `chrf` | sacrebleu CHRF defaults (char order 6, beta 2) / 100 | similarity | `scripts/score_extra.py` (CPU) |
| `word_edit_rate` | word-level Levenshtein distance over whitespace tokens / ref words (can exceed 1; rapidfuzz) | distance | `scripts/score_extra.py` |
| `exact_equal` | stripped texts identical; a convergence indicator for step pairs, **not normalized** | share | `scripts/score_extra.py` |
| `e5_cosine` | `intfloat/e5-base-v2`, "query: " prefix, 512-token sentence-packed chunks, token-weighted mean | similarity | `scripts/score_neural_extra.py` (GPU) |
| `entity_recall` | share of normalized spaCy entities (PERSON, ORG, GPE, LOC, NORP, FAC, EVENT, WORK_OF_ART, LAW, PRODUCT) found verbatim in the rewrite | similarity | `scripts/info_features.py` + `score_extra.py` |
| `number_recall`, `quantity_recall` | clipped multiset recall of normalized numbers / number+unit quantities | similarity | same |
| `sent_coverage` | share of original spaCy sentences whose best MPNet match in the rewrite reaches cosine 0.6 | similarity | `scripts/score_neural_extra.py` |

All except `exact_equal` are reported on the retained scale against the same unrelated-T0 floors. Recalls are
undefined (NaN) when the original has no entities, numbers or quantities. The content-warning flags
(`flag_entity_loss`, `flag_number_change`, `flag_sentence_omission`) are raw shares of pairs, as in Samer's RQ1.
They are **heuristic warnings, not validated factual errors**.

## Unchanged

- Statistical unit: the original article (`key`). Bootstrap CIs resample articles.
- Pairs: cumulative (T0 → Tk) and step (Tk-1 → Tk).
- Comparisons use only documents that every compared rewriter completed to T3.

## Still open

- Number of bootstrap resamples: 1,000 in `sot_analysis.ipynb`, 2,000 in Samer's RQ1.
- Samer's `sot_research` package (branch `feature/sam-experiments`) still uses multiset word retention over all
  tokens, raw BERTScore as the headline, and T0-fitted style vocabularies. It needs the same changes before RQ1
  numbers can be pooled with the paraphrase baseline.
- Until then, `sot_analysis.ipynb` §5 pools only the metrics his published per-pair scores already share with
  this standard. On his 368 baseline pairs (identical T0 texts) our code reproduces BLEU, ROUGE-L, chrF, word edit
  rate, MPNet, E5, entity/number/quantity recall and sentence coverage exactly, and rescaled BERTScore within 0.0005
  (`results/sot_analysis/samer_metric_agreement.csv`). Word retention and style are excluded.
