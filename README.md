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

This experiment uses Qwen to investigate English-Spanish-English rewriting on the XSum subset.

The experiment includes comparisons of translation strategies and evaluations of lexical, semantic, and stylistic changes.

The Qwen implementation and results will be added separately by the team.

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

MADLAD-400: [Kudugunta et al.](https://arxiv.org/abs/2309.04662)

Back-translation: [Sennrich et al. (2016)](https://aclanthology.org/P16-1009/)

## Project Status

Active research project. The repository contains the English baseline and the preliminary MADLAD experiment. The Qwen experiment and expanded multilingual evaluations are being integrated.
