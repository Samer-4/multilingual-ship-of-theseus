# Kaggle run summary

- export: `madlad-3b__mt-default`; merged records 516 (Stage 2: 400, new: 96)
- run status: completed; operations 96, valid 96, invalid 0, failed attempts 0
- model load 0.0 s; translation time 1119.9 s; 5.16 ops/min; 22.3 words/s; segments 172
- model load before the drift probe: 90.7 s
- tokenizer parity: 400/400
- drift probe: 2/2 S1 identical to round 1

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S2 | 24 | 0 | 12.96 | 1.1030000000000002 |
| T2 | 24 | 0 | 11.885 | 0.905 |
| S3 | 24 | 0 | 11.51 | 1.096 |
| T3 | 24 | 0 | 10.61 | 0.909 |
