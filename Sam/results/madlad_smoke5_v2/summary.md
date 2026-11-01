# Kaggle run summary

- export: `madlad-3b__mt-default`; merged records 420 (Stage 2: 400, new: 16)
- run status: completed; operations 16, valid 16, invalid 0, failed attempts 0
- model load 0.0 s; translation time 177.6 s; 5.43 ops/min; 24.6 words/s; segments 30
- model load before the drift probe: 84.8 s
- tokenizer parity: 400/400
- drift probe: 4/5 S1 identical to round 1

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S2 | 4 | 0 | 13.690000000000001 | 1.0495 |
| T2 | 4 | 0 | 11.504999999999999 | 0.8935 |
| S3 | 4 | 0 | 13.63 | 1.0965 |
| T3 | 4 | 0 | 11.09 | 0.912 |
