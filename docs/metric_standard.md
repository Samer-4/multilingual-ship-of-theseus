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
- Distances (0 = identical; `punct_l1`, `pos_jsd`, `dep_jsd`, `burrows_delta`): `retained = (floor - d) / floor`.

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

## Unchanged

- Statistical unit: the original article (`key`). Bootstrap CIs resample articles.
- Pairs: cumulative (T0 → Tk) and step (Tk-1 → Tk).
- Comparisons use only documents that every compared rewriter completed to T3.

## Still open

- Number of bootstrap resamples: 1,000 in `sot_analysis.ipynb`, 2,000 in Samer's RQ1.
- Whether to add Samer's extra metrics (chrF, E5, word edit rate, exact equality, entity/number/sentence
  coverage) to the 7-dataset analysis.
- Samer's `sot_research` package (branch `feature/sam-experiments`) still uses multiset word retention over all
  tokens, raw BERTScore as the headline, and T0-fitted style vocabularies. It needs the same changes before RQ1
  numbers can be pooled with the paraphrase baseline.
