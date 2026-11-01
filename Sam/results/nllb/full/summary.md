# Kaggle run summary

- export: `nllb__mt-default`; merged records 1594 (Stage 2: 400, new: 1176)
- run status: completed; operations 1176, valid 1173, invalid 3, failed attempts 0
- model load 0.0 s; translation time 2817.0 s; 25.64 ops/min; 104.5 words/s; segments 14373
- model load before the first probe: 59.3 s
- sanity probe: 2/2 fixed sentences ok

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S1 | 197 | 0 | 2.37 | 1.104 |
| T1 | 196 | 1 | 1.79 | 0.901 |
| S2 | 196 | 0 | 2.085 | 1.1115 |
| T2 | 195 | 1 | 1.7850000000000001 | 0.906 |
| S3 | 195 | 0 | 2.05 | 1.111 |
| T3 | 194 | 1 | 1.8 | 0.908 |
- invalid: xsum|Eleuther-AI|xsum-4737 T1: check:repetition
- invalid: xsum|Eleuther-AI|xsum-4725 T2: check:repetition
- invalid: xsum|Tsinghua|xsum-4722 T3: check:repetition
