# MADLAD five-document smoke test: audit and pilot plan (2026-10-31)

**Runs:** kernel `samahmed17/sam-sot-mt-madlad-smoke5` v3 (16 operations) plus the reused `sam-sot-mt-madlad-verify1` v2 (4 operations). Data: `data/generated/smoke5_v2/xsum/madlad-3b/` and `results/madlad_smoke5_v2/`.

## Integrity (verified)

- **Records:** 420. All 20 S2/T2/S3/T3 records are valid, and all 400 Stage 2 records are byte-identical (staged sha `a636b3ed…`).
- **Reuse:** the 4 verify1 records were reused byte-identical (same record ids, output hashes, run id and metadata). This run's delta covers only the other 4 items.
- **Consistency:** every record consumes its valid predecessor's output. Model, config (`705a8bd4992e`), wiring (`decoder`) and patch (`madlad-embed-decoder-v1`) are the same on all 20, and every segment ended with EOS.
- **Clean history:** no chain violations; no quarantined run in the table.

## Quality

| Item (source) | Chain | Information lost / changed (rounds 2–3) | Repetition / language | T2 vs T3 |
|---|---|---|---|---|
| xsum-4710 (OpenAI) | valid | S2 dropped "Tom Squires won RS:X 2017." (the "2017" was itself invented in round 1). "Came fourth … to claim silver overall" became "finished fourth in the race for the overall silver medal", a **material meaning change**. "medal.Would" lost its space. | none | identical |
| xsum-4719 (Tsinghua) | valid | Nothing; T1 = T2 = T3. Round-1 artifacts remain: stewards → "commissars", goaded → "instigated", "It's outrage." | none | identical (fixed point) |
| xsum-4720 (BigScience) | valid | All numbers kept. Lexical drift: linked → "tied" → "tied with" (reads as a draw), "under fire" → "under the attack", "for the … champions" → "by the … champions". | none | 0.99 |
| xsum-4721 (Human) | valid | S2 dropped the sentence with **Zana Marjanovic, Rade Serbedzija** and the plot (Bosnian Muslim woman / Serb man). "Visits to Bosnia for the UN" became "visits to the UN in Bosnia". | none | identical |
| xsum-4729 (LLAMA) | valid | S2 dropped the first sentence ("world's longest sub-sea interconnector"). The XSum mojibake `a'!2bn (PS1.4bn)` became "£2 billion (€1.4 billion)": probably swapped, since XSum's `PS` is usually £ and `a'!` €. Other numbers kept. | none | 0.997 |

**Pattern:**
- Most change happens in round 1; rounds 2–3 converge (T2 ≈ T3 in 5/5).
- The dominant failure is **sentence omission inside a packed segment**: 3 of 5 documents lose one sentence in S2, which is not detected by length.
- In round 1 the same warning fires on 42% of S1 steps, and the cases sampled are real omissions. This is a property of MADLAD with the frozen round-1 segmentation (300-token packing), and RQ1 should measure it.

**Drift probe xsum-4710 (OpenAI):** round-1 S1 "por **un fallo mecánico**" vs probe "por **una falla mecánica**"; the rest is identical (similarity 0.943).
- These are synonymous Spanish forms (Peninsular vs Latin American) of "a mechanical failure", so the difference is harmless decoding variation, most likely a near-tied greedy choice under fp16 on T4 / transformers 5.16.1 vs V100 / 4.46.3.
- The probe is diagnostic only; the stored round-1 S1 is unchanged. The other 4 probes are identical.

## New nonblocking warnings (new records only; Stage 2 is rebuilt with the Stage 2 rules)

| Warning | Level | Fires when |
|---|---|---|
| `check:sentence_drop=a>b` | step | output has fewer sentences than an input with ≥ 2 |
| `check:number_change=lost:…/added:…` | step | digit groups differ (separators ignored) |
| `check:roundtrip_length_loss=r` | round trip | English T_r has < 0.9 × the words of T_{r-1} |
| `check:entity_loss=…` | round trip | capitalised non-initial words (≥ 3 letters) of T_{r-1} missing from T_r |

On the smoke records they flag all three known omissions and nothing in the two clean chains. Thresholds: `experiment.yaml validation.content`.

## Pilot plan

- `data/manifests/madlad_pilot/PLAN.json` and `batch_01.csv` … `batch_08.csv`: 192 items, 768 operations.
- **Base:** Stage 2 plus the smoke export, pinned by sha256. The 5 completed items and the 3 invalid chains are excluded.
- **Batches:** consecutive in manifest order, 21–30 documents each, all 7 source labels in every batch, article and group ids preserved.
- **Runtime model** (fitted on the 20 T4 operations, MAE 1.3 s/operation): about 18–22 min of translation per batch, about 2.5 h for all 768 operations, plus about 2 min fixed per session.

Each batch is its own kernel `samahmed17/sam-sot-mt-madlad-pilot-bNN` on the same pinned base. Gates per batch: code digest, staged hashes, base export sha, tokenizer parity, a 2-item reference probe and the quality gate. A failing run writes a partial export first.

**Resuming a batch that stopped early:**
1. Download it (`--raw-only` if it errored).
2. Run `prepare_kaggle_deploy.py --preset pilot --batch N --resume-attempt 2`.
3. Launch.
