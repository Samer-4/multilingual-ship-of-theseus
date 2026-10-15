# Multilingual Ship of Theseus

A research project for Northeastern University's CS 6120: Natural Language Processing.

## Overview

This project investigates how repeated rewriting affects the lexical, semantic, and stylistic characteristics of text.

Building on the Ship of Theseus paraphrasing study by Tripto et al., we compare two rewriting pathways:

1. **English-only paraphrasing:** Original English text undergoes three successive paraphrasing iterations (T0 through T3).
2. **Multilingual round-trip translation:** Original English text is translated into Spanish and back into English. Future experiments will repeat this process over multiple iterations.

Our central research question is whether introducing another language into the rewriting process changes the rate at which meaning and writing style are preserved.

## Research Components

### 1. English Paraphrasing Baseline

The baseline analyzes the existing Ship of Theseus corpus across seven datasets and four paraphrasing models: ChatGPT, PaLM, Pegasus, and Dipper.

It measures:
- Lexical preservation: BLEU, ROUGE-L, word retention
- Semantic preservation: BERTScore, sentence embedding similarity
- Stylistic preservation: punctuation, function words, syntax/POS, surface features, and character n-grams

Files:
- `notebooks/sot_analysis.ipynb`
- `scripts/`
- `results/sot_analysis/`


### 2. Qwen Round-Trip Translation

This experiment uses `Qwen/Qwen3-4B-Instruct-2507`, a general-purpose instruction-tuned language model, to investigate how English-Spanish-English round-trip translation affects textual preservation.

- Dataset: XSum subset of the Ship of Theseus corpus
- Documents evaluated: 3,208 valid texts
- Pipeline: English (T0) -> Spanish -> English (T1)
- Translation strategies: Literal and free translation
- Baseline comparison: English-only ChatGPT paraphrasing

The experiment evaluates changes across lexical, semantic, and stylistic dimensions, including multilingual E5 content distance, chrF loss, bigram loss, Burrows' Delta, cosine-based stylistic distance, function-word distance, and sentence-length changes.

Selected median normalized changes from the original text (lower indicates less change):

| Metric | Qwen Free Round-Trip | ChatGPT T1 |
|---|---:|---:|
| Content distance (E5) | 0.295 | 0.373 |
| chrF loss | 0.377 | 0.546 |
| Bigram loss | 0.484 | 0.660 |
| Burrows' Delta | 0.319 | 0.522 |
| Style cosine distance | 0.260 | 0.499 |
| Function-word distance | 0.191 | 0.361 |
| Sentence-length change | 0.207 | 0.453 |

The results indicate that Qwen's free round-trip translation produced smaller measured changes than a single ChatGPT paraphrase across these dimensions. This comparison reflects the experiment's normalized evaluation framework and should not be directly compared with MADLAD's preservation scores.

Qwen experiment files:

- `experiments/qwen_translation/` - Translation, pilot, evaluation, and quality-estimation notebooks
- `results/qwen_translation/metrics_items.csv` - Item-level evaluation metrics
- `results/qwen_translation/metrics_summary.json` - Aggregated results

The original Qwen translation-chain CSV outputs are not yet included in the repository.


### 3. MADLAD-400 Round-Trip Translation

This pilot uses Google's `google/madlad400-3b-mt` specialized translation model.

- Dataset: Source-balanced XSum subset
- Documents attempted: 200
- Valid outputs: 197
- Pipeline: English -> Spanish -> English
- Hardware: NVIDIA Tesla V100, 32 GB

Selected median results:

| Metric | Score |
|---|---:|
| Multilingual E5 cosine | 0.9818 |
| BERTScore F1 | 0.9217 |
| BLEU | 0.604 |
| ROUGE-L | 0.831 |
| chrF | 0.793 |
| Word retention | 0.836 |
| Function-word cosine | 0.972 |
| Punctuation F1 | 0.853 |

The E5 and BERTScore evaluations were conducted separately from the main MADLAD translation notebook. Their summary values are reported here, but their evaluation code and document-level outputs are not yet included in the repository.

MADLAD experiment files:
- `experiments/madlad_translation/`
- `results/madlad_translation/`

## Future Work

- Expand round-trip translation to the remaining Ship of Theseus dataset domains.
- Evaluate four translation models: two general-purpose LLMs and two specialized translation models.
- Implement multiple English-Spanish-English rewriting iterations (T0 through T3).
- Investigate additional intermediate languages.
- Compare rewriting methods using consistent evaluation configurations and matched documents.

## Dataset and References

Original corpus: [Ship of Theseus Paraphrased Corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus)

Back-translation: [Sennrich et al. (2016)](https://aclanthology.org/P16-1009/)

## Project Status

Active research project. 
