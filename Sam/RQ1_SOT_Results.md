# RQ1: How words, meaning, style and information decay under repeated rewriting

**Question.** When a text is rewritten three times in a row (T0 → T1 → T2 → T3), how much of its wording, meaning, style and factual content survives? Does English-only paraphrasing behave differently from English → Spanish → English round-trip translation?

**Short answer.**
- **Meaning outlasts wording and style.** Paraphrasers keep most of the meaning but replace much of the wording, and style decays faster than meaning.
- **Most of the loss happens in round 1.**
- **Round-trip translation preserves more than paraphrasing on every comparable metric.** On the same 178 XSum originals, translation also settles into a near-fixed point after one round, while paraphrasers keep drifting.

Source: `notebooks/sot_analysis.ipynb`. Figures and tables: `results/sot_analysis/`. Metric definitions: `docs/metric_standard.md`.

---

## Data and method

| | |
|---|---|
| Corpus | Ship of Theseus paraphrased corpus (Tripto et al.): 7 datasets (cmv, eli5, sci_gen, tldr, wp, xsum, yelp), 7 sources (Human plus 6 LLMs) |
| Paraphrasers | ChatGPT, PaLM, Dipper, Pegasus (main analysis), plus Dipper low/high and Pegasus slight intensity variants |
| Sample | **16,682 originals** that every paraphraser completed to T3, on 2,996 articles, giving **583,870 pairs** |
| Pairs | cumulative (T0 → Tk) and step (Tk-1 → Tk) |
| Statistical unit | the original article (`key`). 95% CIs come from a bootstrap that resamples articles (1,000 resamples) |

**Retained scale.** Every metric is rescaled so that **1 = identical to T0** and **0 = no closer than an unrelated T0 text from the same dataset**. The floor is the per-dataset mean over 200 unrelated T0 pairs; word edit rate uses the median, because it has no upper limit. Values below 0 are possible. Formulas:
- similarities: `(s − floor) / (1 − floor)`
- distances: `(floor − d) / floor`

**Metrics.**
- **Lexical:** BLEU, ROUGE-L, content-word retention (stop words removed), chrF, word edit rate.
- **Semantic:** baseline-rescaled BERTScore F1 (roberta-large L17, long texts windowed), length-weighted MPNet cosine, E5 cosine.
- **Style:** one combined score from 11 components in 5 families (surface, punctuation, function words, syntax, character n-grams), fitted on held-out reference texts.
- **Information:** entity, number and quantity recall, and sentence coverage. These are heuristic warnings, not validated factual errors.

---

## 1. Decay at T3 (all 7 datasets)

Retained at T3, mean [95% CI]. Higher means more preserved.

| Paraphraser | BLEU | Word retention | chrF | BERTScore F1 | MPNet | E5 | **Style** |
|---|---|---|---|---|---|---|---|
| ChatGPT | 0.17 | 0.45 [0.44, 0.45] | 0.35 | 0.55 [0.55, 0.56] | 0.86 | 0.81 | 0.45 [0.45, 0.46] |
| PaLM | 0.31 | 0.52 [0.52, 0.53] | 0.43 | 0.62 [0.61, 0.62] | 0.85 | 0.81 | 0.57 [0.56, 0.57] |
| Dipper | 0.07 | 0.24 [0.24, 0.24] | 0.18 | 0.37 [0.37, 0.37] | 0.68 | 0.63 | 0.36 [0.35, 0.36] |
| Pegasus | 0.26 | 0.51 [0.50, 0.51] | 0.36 | 0.53 [0.53, 0.54] | 0.80 | 0.78 | 0.47 [0.46, 0.48] |
| *Pegasus (slight)* | 0.64 | 0.79 | 0.71 | 0.78 | 0.92 | 0.92 | 0.80 |
| *Dipper (low)* | 0.32 | 0.59 | 0.48 | 0.65 | 0.88 | 0.85 | 0.64 |
| *Dipper (high)* | −0.00 | −0.04 | −0.11 | −0.01 | −0.11 | −0.20 | −0.23 |

- **Wording goes first, meaning stays.** Every paraphraser keeps far more meaning (MPNet 0.68–0.86) than wording (BLEU 0.07–0.31).
- **PaLM preserves the most** on the headline content and style measures. **Dipper preserves the least.** ChatGPT changes the wording heavily (BLEU 0.17) but keeps meaning about as well as PaLM (MPNet 0.86).
- **Intensity matters.** Pegasus-slight keeps 80% of style. Dipper-high produces text no closer to the original than an unrelated document, and below the floor on style.
- **The ranking depends on the metric.** Embedding cosines (MPNet, E5) compress the differences; BERTScore and the lexical metrics separate the paraphrasers far more. Figure: `decay_curves.png`.

## 2. Style decays faster than meaning

The decay gap is retained content (BERTScore) minus retained style, at T3, mean [95% CI]:

| ChatGPT | PaLM | Pegasus | Dipper |
|---|---|---|---|
| **+0.099** [0.095, 0.102] | +0.053 [0.050, 0.057] | +0.063 [0.059, 0.068] | +0.014 [0.010, 0.018] |

- For every paraphraser, style is lost faster than content by T3, and all four gaps exclude zero.
- **The gap grows over rounds.** Pegasus's gap rises from 0.005 at T1 to 0.063 at T3. Dipper's starts slightly negative at T1 (−0.008).
- **ChatGPT has the widest gap:** it keeps meaning but rewrites in its own voice.

**Which style features go first** (retained at T3, by family):

| | Surface | Punctuation | Function words | Syntax | Character |
|---|---|---|---|---|---|
| ChatGPT | 0.40 | **0.33** | 0.45 | 0.56 | 0.54 |
| PaLM | 0.64 | **0.39** | 0.56 | 0.62 | 0.62 |
| Dipper | 0.45 | **0.18** | 0.34 | 0.43 | 0.37 |
| Pegasus | 0.52 | **0.24** | 0.54 | 0.43 | 0.62 |

- **Punctuation is the most fragile style family for every paraphraser.** Syntax and character n-grams hold best (Pegasus's syntax is the exception).
- **By source,** texts originally written by Eleuther-AI lose the most style (Dipper keeps only 0.08). OpenAI- and LLAMA-authored texts keep the most. Figure: `style_blocks_and_sources.png`.

## 3. Information preservation

Retained at T3:

| | Entity recall | Number recall | Quantity recall | Sentence coverage | Any content warning (raw share) |
|---|---|---|---|---|---|
| ChatGPT | 0.81 | 0.65 | 0.53 | 0.79 | 77% |
| PaLM | 0.73 | 0.58 | 0.41 | 0.76 | 75% |
| Pegasus | 0.51 | 0.37 | 0.49 | 0.83 | 87% |
| Dipper | 0.38 | 0.14 | 0.07 | 0.55 | 97% |

- **ChatGPT keeps the most facts:** 81% of named entities and 65% of numbers. **Dipper loses nearly all** quantities.
- **Pegasus covers most original sentences** (0.83) but drops entities and numbers inside them.
- **Warnings are common.** 75–97% of T3 rewrites trigger at least one content warning (a lost entity, a changed number or an uncovered sentence). These are heuristic flags, not confirmed errors. Figure: `information_curves.png`.

## 4. Most of the change happens in round 1

| | Round 1's share of the total T3 loss (chrF) | Round 1's share (BERTScore) | Word edit rate T0→T1 | T1→T2 | T2→T3 |
|---|---|---|---|---|---|
| ChatGPT | 90% | 85% | 0.71 | 0.41 | 0.36 |
| PaLM | 80% | 79% | 0.53 | 0.26 | 0.19 |
| Pegasus | 71% | 74% | 0.53 | 0.39 | 0.33 |
| Dipper | 79% | 78% | 0.88 | 0.86 | 0.83 |

- **71–90% of the T3 loss is already present at T1.** Later rounds change less, except Dipper, which keeps rewriting about 85% of words every round.
- **PaLM and Pegasus start to converge:** about 2.5% of their T2 → T3 outputs are identical to their input.

## 5. Round-trip translation vs paraphrase (same 178 XSum originals)

Samer's RQ1 translated 200 XSum T0 documents from this corpus through three English → Spanish → English round trips, using MADLAD-400-3B, NLLB-200-1.3B and Qwen3-4B.
- **Pooling with our results is valid:** his originals are byte-identical to ours, and our code reproduces 11 of his per-pair metrics on identical texts. BERTScore is within 0.0005; the other 10 match exactly.
- **Comparison set:** 178 documents (31 articles) that every translator and paraphraser completed, all on the same retained scale.
- **Not compared:** word retention and style, because they need his translated texts.

Translation minus paraphrase at T3 (per-document family means, CI resamples articles):

| Metric | Translation (3 models) | Paraphrase (4 models) | Gap [95% CI] |
|---|---|---|---|
| BLEU | 0.53 | 0.24 | +0.29 [0.27, 0.30] |
| chrF | 0.67 | 0.38 | +0.29 [0.27, 0.30] |
| BERTScore F1 | 0.81 | 0.58 | +0.23 [0.22, 0.24] |
| MPNet | 0.95 | 0.85 | +0.10 [0.09, 0.11] |
| E5 | 0.92 | 0.80 | +0.12 [0.11, 0.13] |
| Entity recall | 0.88 | 0.65 | +0.23 [0.19, 0.26] |
| Number recall | 0.88 | 0.52 | +0.36 [0.32, 0.39] |
| Sentence coverage | 0.94 | 0.79 | +0.16 [0.13, 0.18] |

- **Translation preserves more on every metric.** The gap is largest for exact wording and numbers, and smallest for overall meaning.
- **NLLB preserves the most of any rewriter on every metric** (BERTScore 0.84, entity recall 0.94).
- **Not a clean split.** The best paraphraser overlaps the weakest translator: PaLM's BERTScore of 0.76 sits close to Qwen3-4B's 0.78, and PaLM's BLEU (0.45) exceeds Qwen's (0.42).
- **Translation converges, paraphrase drifts.** After round 1, translators change only 2–10% of words per round (2–4% by T2 → T3), while paraphrasers keep changing 15–42% (Dipper 86–100%). Figure: `translation_vs_paraphrase.png`.

---

## Caveats

- **Some originals are themselves LLM-written.** Six of the seven sources are LLM-generated, so a paraphraser may push a text toward a style it already had.
- **Cohort restriction.** Comparisons use only documents that every compared rewriter completed. Invalid generations are counted as attrition and excluded, never repaired.
- **The translation comparison is small and narrow:** XSum only, 31 articles, one language pair and greedy decoding. The machine-translation models translate sentence by sentence, while Qwen translates whole documents.
- **Information metrics are heuristics.** A paraphrased entity name or a reformatted number counts as a loss.
- **Still open:** pooling word retention and style for translation needs Samer's texts, or his package following the standard. The team has not yet agreed on 1,000 vs 2,000 bootstrap resamples.

## Files

| File | Content |
|---|---|
| `results/sot_analysis/decay_curves.{png,csv}` | Retained curves for every metric, paraphraser and round, with CIs |
| `results/sot_analysis/style_vs_content.{png,csv}` | Content vs style vs words, and the decay gap |
| `results/sot_analysis/style_blocks_and_sources.{png,csv}`, `style_components_T3.csv` | Style by family, component and source |
| `results/sot_analysis/information_curves.{png,csv}` | Information recall curves and warning rates |
| `results/sot_analysis/convergence_*.csv` | Step edit rates, exact repeats, round-1 share of the loss |
| `results/sot_analysis/translation_vs_paraphrase*.{png,csv}` | Section 5 curves, T3 gap and step changes |
| `results/sot_analysis/samer_metric_agreement.csv` | Per-pair agreement of our metrics with Samer's |
| `results/sot_analysis/floors_by_dataset.csv` | Unrelated-T0 floors per dataset |
