# Remaining translation runs: NLLB, Qwen, Phi-4-mini (prepared 2026-10-31, nothing launched)

MADLAD is final (`data/generated/madlad_final/`, 196/200 chains). Llama 3.2 is not run: Kaggle no longer accepts access requests. Its config entry and `kaggle/llama/preflight/` are kept unchanged; the fourth translator is `microsoft/Phi-4-mini-instruct`.

## State
| Model | Preflight (same 3 T0 documents) | Full run (all 200 T0 documents) |
|---|---|---|
| NLLB `facebook/nllb-200-distilled-1.3B`, commit `7be3e246…` | **passed**: 18/18 valid, 56.8 s, 2.67 GB | prepared: `kaggle/nllb/full`. Estimate **39 min** (measured), 1 session, timeout 81 min, clean stop at 59 min |
| Qwen `Qwen/Qwen3-4B-Instruct-2507`, commit `cdbee75f…` | **passed**: 18/18 valid, 13 m 50 s, 8.68 GB | prepared: `kaggle/qwen3-4b/full`. Estimate **8.6 h** (measured, length-aware), 1 session expected, timeout 12 h, clean stop at 11 h 38 m, then automatic continuation |
| Phi-4-mini `microsoft/Phi-4-mini-instruct`, commit `cfbefacb…` (pinned) | prepared: `kaggle/phi4-mini/preflight` | prepared automatically after its preflight is downloaded (`full:phi4-mini`) |

**Runtime model:**
- Seconds per operation are fitted separately per direction on each model's own 18 preflight operations: Qwen en→es is about 0.170 s/word and es→en about 0.098 s/word.
- The fit is applied to every remaining document's own T0 length, carried through the measured length ratios.
- The naive 15.4 h figure for Qwen overstates the time, because the preflight mean is dominated by the 916-word document (140 s per en→es operation), while the median document has 178 words.

**Kaggle limits (assumed, not visible through the CLI):** a 12 h GPU session limit and a weekly GPU quota, usually about 30 h. The plan needs roughly 0.7 h (NLLB) + 0.3 h (Phi preflight) + 8.7 h (Qwen) + about 8 h (Phi full; unknown until its preflight). That is about 18 h, run sequentially, and needs no concurrency.

## One command
```bash
Sam/.venv/bin/python Sam/scripts/run_queue.py full:nllb Sam/kaggle/phi4-mini/preflight full:qwen3-4b full:phi4-mini \
    --keep-going --yes
```
**Order:** NLLB full, then the Phi-4-mini preflight, then Qwen full (attempts 2, 3, … are prepared and run automatically while operations remain), then Phi-4-mini full (prepared from its downloaded preflight, continued the same way).

**For each item, in turn:**
- A deployment that was already pushed is adopted: its remote status is read and it is followed or downloaded, never pushed again.
- A new one is launched; the code dataset is uploaded once, and later launches verify the remote build.
- The monitor is followed; the real remote status decides what happens next, so a local monitor crash never counts as a failure.
- The run is downloaded.

**Stopping rules:**
- `--keep-going` only passes remote failures or a model that cannot be prepared (e.g. Phi full when its preflight failed).
- An unknown or authentication status, a still-running job, or a resume attempt that made no progress always stops the queue, with nothing relaunched.

**While it runs:**
- `Sam/.venv/bin/python Sam/scripts/status_all.py` shows the combined table.
- Re-running the same queue command after an interruption resumes: finished items are skipped and running ones adopted.

## Afterwards
```bash
for m in nllb qwen3-4b phi4-mini; do Sam/.venv/bin/python Sam/scripts/finalize_madlad.py --translator $m; done
```
Each model gets `data/generated/<model>_final/xsum/<model>/`:
- `generations_merged.parquet`;
- `chains.csv` (item, dataset, article and group ids, source label, status per stage, complete);
- `FINAL.json` (hashes, counts, attrition, warnings);
- `audit/` copies of every invalid output.

Before writing, `verify_madlad_pilot.py --translator <model>` checks:
- the 400 Stage 2 records are unchanged;
- unique ids;
- S1 reads the manifest T0;
- every later stage reads the model's own valid predecessor;
- one config id and checkpoint per model;
- the preflight and earlier attempts are byte-identical.

All four final tables join on `item_id` (= dataset | source label | article id), `article_id`, `source_label` and `output_stage`.

## Comparability (see OTHER_TRANSLATORS_PREFLIGHT.md for the full table)
- **Inputs:** all four models start from the same 200 T0 documents. No model reads another model's text, and documents MADLAD failed on are not excluded.
- **Unit of translation:**
  - MADLAD: packed sentences of up to 300 tokens, batch 4;
  - NLLB: one sentence per segment, batch 8;
  - Qwen and Phi: the whole document, batch 1.
- **Prompting:** only the LLMs are prompted, with identical wording for Qwen and Phi. Decoding is plain greedy everywhere, with penalties neutralised.
- **Length caps:** the MT models have 512 tokens per segment; the LLMs have max(16, min(2·n_in + 64, 8192 − n_prompt − 8)).
- **Phi's long-context rotary scaling (LongRoPE):** it changes scaling only beyond 4096 tokens of prompt + output; the preflight records both token counts for its longest document.
- **The 916-word preflight document** was translated in full by NLLB and Qwen in all six stages:
  - every output ended on EOS;
  - Qwen used at most 1671 of a 2262-token cap;
  - the last sentence is present in S1, T1 and T3.

## Acceleration (not applied)
Each full run uses one of the two T4s; the second is idle.
- **Splitting documents across both GPUs** (two model copies, batch 1, the same decoding) would roughly halve the wall time without changing the method. It still needs a separate check that the outputs are identical before use.
- **Batching several documents per call** would change outputs through padding, so it is not acceptable for the frozen setup.
