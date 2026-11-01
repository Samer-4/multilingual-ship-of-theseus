# MADLAD XSum pilot: final dataset (2026-10-31)

**Final folder:** `data/generated/madlad_final/xsum/madlad-3b/` (built by `scripts/finalize_madlad.py`, never overwritten).
- `generations_merged.parquet`: a byte copy of the verified remainder export, sha `accf874f…`.
- `chains.csv`: per-item stage status with dataset, article, group and source-label ids.
- `FINAL.json`: counts, attrition, warnings and hashes.
- The audit copy of the one invalid output is in `data/generated/madlad_final/audit/`.

## Integrity (verified by `scripts/verify_madlad_pilot.py`)
- **Records:** 1,186, with unique ids and composite keys.
  - 400 Stage 2 records, unchanged against the staged sha `a636b3ed…`.
  - 786 new records.
- **Reuse:** all 516 batch-1 records and all 420 smoke records are byte-identical.
- **Lineage:** every new record consumes its valid predecessor. All new records use config `705a8bd4992e` with decoder wiring and patch `madlad-embed-decoder-v1`. No quarantined or MOCK runs are present.

## The incomplete chain: `xsum|Eleuther-AI|xsum-4729`, T2 (S2 → English)
**Reason:** `engine:generation_cap`. Segment 1 of 4 (222 input tokens) ran to the 512-token cap without EOS. The other three segments ended normally.
- The capped segment is a **decoding loop**: "the fourth segment North Britain Wind Park – 100 MW of wind power" 19 times, plus a truncated 20th. Its input says nothing of the kind.
- The whole-text repetition check did not fire, because the three clean segments dilute the 3-gram share (about 0.03 against a 0.10 threshold). The cap is what marked the record. Either check alone blocks it.

**Classification:** a genuinely broken translation, not a false positive, and not legitimate semantic loss.

**Input:** the T1 is a machine-generated (Eleuther-AI) text that turns into a long list of unpunctuated company and date fragments; it is the most loop-prone kind of input.

**Successors:** S3 and T3 were correctly never generated.

**Recovery:** not recommended.
1. Decoding is greedy and frozen, so re-running the same input with the same configuration is expected to give the same loop.
2. Changing decoding, segmentation or the cap would break the frozen configuration for one item.
3. Replacing an invalid record with a re-roll would select outputs by their outcome, against the Stage 2 decision: "preserve invalid outputs … continue later rounds only for valid records … report complete-case comparisons and attrition".

This is MADLAD attrition and is reported as such.

## Dataset summary
**Chains:** 196 of 200 items have valid S1→T3 chains: 196/197 eligible (99.5%) and 196/200 overall (98.0%).

**Attrition:**
- Round 1: 3 invalid items (BigScience xsum-4713, LLAMA xsum-4714, LLAMA xsum-4733).
- Round 2: 1 invalid T2 (Eleuther-AI xsum-4729), which blocks S3 and T3.

| Source | Items | Complete |
|---|---|---|
| BigScience | 29 | 28 |
| Eleuther-AI | 29 | 28 |
| Human | 29 | 29 |
| LLAMA | 29 | 27 |
| OpenAI | 28 | 28 |
| PaLM | 28 | 28 |
| Tsinghua | 28 | 28 |

**Nonblocking content warnings (786 new records, 326 with at least one):**

| Warning | S2 | T2 | S3 | T3 |
|---|---|---|---|---|
| sentence_drop | 50 | 41 | 28 | 27 |
| number_change | 26 | 15 | 15 | 10 |
| roundtrip_length_loss (< 0.9) | – | 22 | – | 8 |
| entity_loss | – | 36 | – | 14 |

Warnings fall with each round (convergence), and no length-ratio or truncation warnings fired.

## RQ1 readiness
- **Ready:** the MADLAD pathway, for a complete-case analysis on 196 items with article id as the grouping unit, and the attrition above reported alongside.
- **Not part of this dataset:** the English-paraphrase comparison and the other three translators.
- **Corpus artifacts:** inherited XSum encoding damage (`a'!`, `PS`) is present in T0/T1 and should be noted for the lexical metrics.
