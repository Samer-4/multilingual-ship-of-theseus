@@ RQ1_INTERPRETATION
## Interpretation

Values are from the matched cohort (193 documents, 31 articles) and the tables above. If translators are added or
the data change, re-read the tables; this text describes the run committed with this notebook.

**1. Almost all decay happens in the first round trip.**
- Across translators, 84-94% of the total T0→T3 loss (chrF, BERTScore) is already present at T1.
- Consecutive-round change then collapses. Word edit rate falls from 0.25-0.44 (T0→T1) to 0.02-0.04 (T2→T3).
- By T3, about half of the MADLAD (50%) and NLLB (53%) chains are **exact fixed points**: the third round trip
  reproduces the second word for word. Qwen keeps rephrasing slightly (15% fixed points).
- The "ship" is mostly rebuilt once and then stays put. Repeated MT converges quickly towards a translator-specific
  attractor text rather than drifting steadily.

**2. Lexical and semantic decay tell different stories.**
- **Lexical overlap:** Qwen rewrites the most wording (BLEU 0.45 vs 0.59 MADLAD and 0.64 NLLB at T1; chrF
  0.71 / 0.78 / 0.81).
- **Meaning:** all three stay far closer to the original than an independent same-content text written by another
  author.
  - MPNet cosine 0.94-0.97 at T3, vs 0.75 for the other-author paraphrase.
  - BERTScore F1 0.955-0.968, vs 0.86.
- **Qwen's lower lexical scores reflect paraphrase, not lost content.** Its sentence coverage (0.97) and MPNet
  similarity are at or above MADLAD's, and it preserves more entities than MADLAD (0.88 vs 0.83 at T3).
- **NLLB is the most conservative on every family:** the highest lexical and semantic similarity and the best entity
  recall (0.93).

**3. Style drifts more for the LLM translator.**
- Burrows' Delta at T3 is 0.26 for Qwen vs 0.17-0.20 for the MT systems.
- Qwen's function-word and character-trigram profiles move furthest from the original.
- POS distributions barely change for any translator: Jensen-Shannon divergence ≤ 0.01, vs 0.03 for the other-author
  paraphrase.
- The surface style changes while the grammar stays the same.

**4. Information-loss warnings are frequent but mostly benign, and MADLAD is the outlier.**
- **MADLAD** has the lowest sentence coverage (0.875 at T3) and the highest rates of entity-, number- and
  sentence-omission warnings.
  - Its sentence count shrinks by about 15% by T3, because it merges sentences.
  - The largest MADLAD drop is a genuine failure: one round trip replaced the article with an unrelated,
    Wikipedia-style text ("Chicago Fire ... [1]").
- **NLLB** splits sentences increasingly (sentence count +14% by T3). In its worst case, an unrelated sentence about
  the European Parliament was inserted.
- **Inspecting the most frequently "lost" entities** shows that many are normalisations ("US", "UK", "Britain"
  rendered differently) rather than losses, so these rates are upper bounds on real information loss.

**5. Translators differ significantly, and the ranking depends on the metric family.**
- Paired T3 differences on identical documents (article-cluster CIs) rank:
  - lexical similarity: NLLB > MADLAD > Qwen;
  - stylistic drift: Qwen > MADLAD > NLLB;
  - entity, number and sentence preservation: MADLAD lowest.
- The NLLB-Qwen gaps in BERTScore and E5 are small (about 0.01) but excluded from zero.

## Limitations
- **Sample size:** 31 original articles carry 200 documents, and every interval is clustered on articles. The rarer
  failure modes rest on few cases.
- **Matched cohort:** it excludes the 7 documents with an invalid stage in some translator. Repetition and
  token-cap failures are not random, so the matched cohort slightly favours "easy" documents. The all-available
  summaries are in `summary_cumulative.csv`.
- **Segmentation confound:** MADLAD and NLLB translate sentence segments and Qwen translates whole documents.
  Sentence-count and sentence-coverage metrics partly measure this design difference.
- **Heuristic information flags:** the entity, number and sentence flags are heuristics (string presence,
  number-multiset equality, an MPNet cosine threshold of 0.6). No flag was manually validated as a factual error.
- **Metric conventions:**
  - BERTScore uses roberta-large layer 17, windowed for texts over 510 tokens (281 texts), and is reported raw and
    baseline-rescaled.
  - E5 and MPNet embed long texts as token-weighted means of sentence-packed chunks. Nothing was truncated.
- **Missing baseline:** the English-paraphrasing baseline of the original project is not part of this corpus, so RQ1
  here compares translators with each other and with the in-corpus reference pairs.
- **Single direction and settings:** only English↔Spanish, one decoding setting per translator (greedy), and one
  dataset (XSum).

@@ RQ2_INTERPRETATION
## Interpretation

Scores are macro-F1 on the matched cohort (193 documents per evaluation set, 31 articles, 5×5 grouped CV).

**1. Source attribution on originals is moderate, and partly a length effect.**
- The best model (character 2-5-gram TF-IDF) reaches macro-F1 0.43 on held-out original texts, three times chance
  (0.14).
- A length-only control already reaches 0.34. Sources differ strongly in length (median 108 words for OpenAI vs 360
  for Human), so a substantial part of "attribution" is length.
- **Distinctiveness by source:**
  - OpenAI (short, formulaic) and BigScience are the most recognisable (F1 0.71 and 0.56).
  - LLAMA and PaLM are often confused with Human: their continuations are long and news-like.

**2. Translation erodes the source signal only slightly, and only in the first round.**
- After one round trip the character model keeps most of its performance: T1 macro-F1 is 0.41 (MADLAD), 0.41
  (NLLB) and 0.37 (Qwen).
- Rounds 2 and 3 change almost nothing (T3: 0.40, 0.41, 0.38).
- The word and stylometric models lose more (0.41 → 0.32-0.35 stylometric at T3), but all stay well above chance.
- This mirrors RQ1: most change happens in round 1, and the chains then converge.

**3. Translators differ in how much source signal survives.**
- Qwen, which paraphrases most (RQ1), removes the most attribution signal from character n-grams (0.375 at T3 vs
  about 0.40-0.41).
- Sentence-level MT systems preserve more of the original surface form, and with it more of the source signal.
- The differences between translators are smaller than the bootstrap CIs (±0.05-0.06), so they are suggestive, not
  conclusive.

**4. The shared opening is not what the classifier uses.**
- Removing the measured shared opening from every text (21.6 words on average) does **not** reduce performance. T0
  character macro-F1 is 0.47 without it vs 0.43 with it, and the translated sets behave similarly.
- The opening is identical across sources, so it carries no source information and only dilutes the signal.
- In translations, the measured opening shortens for MADLAD and Qwen (median about 10-24 words), because they render
  the same opening differently in each text. NLLB keeps it identical (about 30 words).

## Limitations
- **Small test sets:** only 31 articles (about 6 per test fold) and about 28 documents per class. Confidence
  intervals are wide (±0.05-0.06), and translator differences are not individually significant.
- **Fixed models:** hyperparameters are fixed and untuned, so better tuned or neural classifiers could score higher.
  This experiment measures *relative* survival of the signal, not maximal attainable attribution.
- **Length confound:** length is a strong confound. Both the length-only control and the stylometric model use
  length-normalised rates, but text length itself survives translation almost unchanged (RQ1 length ratios
  0.92-1.0).
- **Opening ablation:** the shared-opening ablation removes a measured verbatim prefix. In LLM-translated texts,
  paraphrased remnants of the opening may remain.
- **Training data:** training uses original texts only, by design. Training on translated texts of training
  articles would answer a different question (attribution robust to translation).

@@ RQ3_INTERPRETATION
## Interpretation

Scores are macro-F1 on the matched cohort (193 documents × 3 translators × 3 rounds = 1,737 texts, 31 articles,
5×5 grouped CV). Chance is 1/K = 0.333.

**1. Translators leave a clear, stable fingerprint.**
- The character n-gram model identifies the translator with macro-F1 0.73 (accuracy 0.73) on articles it never saw.
- The fingerprint is the same at every round (T1 0.72, T2 0.73, T3 0.74) and transfers across rounds: a model
  trained on T1 scores 0.74 on T3.
- The fingerprint is set in the first round trip and then persists, consistent with the fixed-point behaviour in
  RQ1.

**2. Part of the fingerprint is typography and orthography.**
- MADLAD introduces typographic quotes and apostrophes (’ “ ”) in about 58% of its texts, Qwen in about 15%, and NLLB
  in none (0.5% of originals).
- The curly apostrophe is MADLAD's single strongest stylometric feature.
- Normalising these characters to ASCII lowers the character model from 0.73 to **0.67**. Removing them from the
  stylometric model lowers it from 0.60 to 0.55.
- The fingerprint survives both controls, well above chance. It is also lexical and syntactic:
  - Qwen prefers "stated (that)", "following" and "such as", and longer words;
  - NLLB has more subordinate markers and expletives, and a characteristic "what that" construction.

**3. It is not a length or segmentation artefact.**
- **Length:** a length and sentence-statistics control reaches 0.48, above chance but far below the text models.
- **Segmentation:** removing sentence-boundary features slightly *improves* the stylometric model (0.61 vs 0.60), so
  the fingerprint does not rely on the MT-vs-LLM segmentation difference.
- **Per-class:** Qwen is the easiest class (F1 0.78). NLLB is the most precise (0.86) but least recalled (0.58): its
  outputs are often taken for MADLAD.

**4. It generalises across original sources.**
- Training on six source labels and testing on the seventh keeps the character model between 0.55 (OpenAI) and 0.79
  (LLAMA).
- The weakest held-out source, OpenAI, has the shortest texts (median 108 words), which carry the least evidence.

## Limitations
- **Clustering:** all scores are clustered on 31 articles, and CIs are about ±0.04.
- **Three translators only:** two MT systems and one LLM. Phi-4-mini will be added automatically, and the chance
  level and class set update with it.
- **Feature importance:** the important-feature lists come from models fit on all texts. They describe the
  classifier and are not causal. Some word n-grams are content-specific to a few documents.
- **Generalisation scope:** the fingerprint is measured on XSum news via English↔Spanish only. Other genres, language
  pairs or decoding settings may change it.
- **Training material:** classifiers are trained on round-trip outputs of these specific checkpoints. Their
  robustness to post-editing or other model versions is untested.
