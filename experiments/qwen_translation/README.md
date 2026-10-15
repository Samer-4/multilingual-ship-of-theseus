# Qwen Round-Trip Translation Experiment

## Overview

This experiment investigates how round-trip translation
changes lexical, semantic, and stylistic characteristics
of text from the Ship of Theseus corpus.

The main translation pathway is:

English (T0) -> Spanish -> English (T1)

The experiment explores different translation strategies,
including literal and free translation, and compares
the resulting changes with English-only paraphrasing.

## Model

- Model: Qwen/Qwen3-4B-Instruct-2507
- Type: General-purpose instruction-tuned language model
- Intermediate language: Spanish
- Corpus: Ship of Theseus
- Initial evaluated domain: XSum

## Notebooks

- degree_zero_chain.ipynb:
  Translation chain generation and experiment configuration.

- degree_zero_spanish_pilot.ipynb:
  Initial Spanish translation experiments and prompt testing.

- degree_zero_metrics.ipynb:
  Lexical, semantic, and stylistic evaluation.

- degree_zero_qe.ipynb:
  Translation quality estimation experiments.

## Results

The following files are stored in
results/qwen_translation/:

- metrics_items.csv:
  Item-level evaluation metrics.

- metrics_summary.json:
  Aggregated evaluation results.

The initial XSum evaluation reported 3,208 valid documents.
Refer to the summary file for metric definitions and
reported statistics.

## Reproducibility

The notebooks and evaluation results are included.
However, the intermediate translation-chain CSV files
are not yet available in this repository.

Some notebooks may require external datasets, model
downloads, GPU resources, and environment-specific paths.

The current results should not be compared directly
against MADLAD scores without aligning the evaluated
document IDs and metric implementations.

## Future Work

- Recover and publish the translation-chain outputs.
- Extend the analysis to additional corpus domains.
- Evaluate repeated round-trip translation.
- Compare additional general-purpose translation models.
- Standardize evaluation across Qwen and specialized
  machine translation systems.
