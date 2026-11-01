# Kaggle run summary

- export: `madlad-3b__mt-default`; merged records 404 (Stage 2: 400, new: 4)
- run status: completed; operations 4, valid 4, invalid 0, failed attempts 0
- model load 0.0 s; translation time 55.2 s; 4.41 ops/min; 12.6 words/s; segments 6
- model load before the drift probe: 84.0 s
- tokenizer parity: 400/400
- drift probe: 1/1 S1 identical to round 1

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S2 | 1 | 0 | 15.78 | 1.047 |
| T2 | 1 | 0 | 11.4 | 0.915 |
| S3 | 1 | 0 | 15.89 | 1.093 |
| T3 | 1 | 0 | 11.32 | 0.91 |
