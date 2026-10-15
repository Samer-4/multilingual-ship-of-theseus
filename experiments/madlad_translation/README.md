# MADLAD-400 Round-Trip Translation Experiment

## Overview

This experiment extends the English-only Ship of Theseus
paraphrasing baseline by studying round-trip translation.

Original English documents (T0) are translated into Spanish
and then back into English (T1). The reconstructed texts are
compared with the originals to evaluate lexical, semantic,
and stylistic preservation.

## Dataset

- Corpus: Ship of Theseus
- Subset: XSum
- Sampling: Deterministic, source-balanced selection
- Documents selected: 200
- Valid reconstructed documents: 197
- Flagged documents: 3

This is a preliminary pilot, not a complete evaluation
of the entire XSum dataset.

## Model and Environment

- Model: google/madlad400-3b-mt
- Framework: Hugging Face Transformers and PyTorch
- Hardware: NVIDIA Tesla V100, 32 GB
- Compute: Northeastern Explorer HPC
- Translation: English -> Spanish -> English
- Input chunk budget: 300 tokens
- Inference batch size: 4 chunks
- Decoding: Greedy (num_beams=1)

The notebook creates an isolated Python environment to
avoid conflicts with Explorer's shared packages.

## Files

- madlad_explorer.ipynb:
  Main experiment notebook. Handles setup, dataset
  selection, translation, validation, checkpointing,
  and lexical/stylistic evaluation.

The corresponding results are stored under:

results/madlad_translation/

- chain_es_madlad4003b.csv:
  Original English texts, Spanish translations,
  reconstructed English texts, and validation flags.

- matched_item_metrics.csv:
  Document-level lexical and stylistic metrics
  for the 197 valid outputs.

## Preliminary Results

Median scores across 197 valid documents:

| Metric | Median |
|---|---:|
| BLEU | 0.604 |
| ROUGE-L | 0.831 |
| chrF | 0.793 |
| Word retention | 0.836 |
| Function-word cosine | 0.972 |
| Punctuation F1 | 0.853 |
| Multilingual E5 cosine | 0.9818 |
| BERTScore F1 | 0.9217 |

The multilingual E5 and BERTScore evaluations were
performed in a separate evaluation step and are not
implemented in the main translation notebook.

## Interpretation

The preliminary results indicate high semantic
similarity between original and reconstructed English
texts, despite differences in lexical overlap.

These metrics use different scoring definitions and
should not be interpreted as directly comparable
percentages of retained information.

## Reproduction Notes

1. Obtain access to a CUDA-enabled GPU environment.
2. Open madlad_explorer.ipynb.
3. Review its configuration and environment setup.
4. Run the cells in order.
5. Inspect the generated outputs and validation flags.

The notebook contains Explorer-specific paths and
environment configuration. Adapt these settings when
running on other systems.

The experiment uses one translation round trip.
Repeated T0-T3 translation chains are planned future work.

## Future Work

- Expand evaluation across the remaining corpus domains.
- Evaluate additional general-purpose and specialized
  translation models.
- Implement repeated round-trip translation.
- Investigate additional intermediate languages.
- Standardize evaluation for matched model comparisons.
