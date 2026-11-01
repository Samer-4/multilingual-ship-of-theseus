# Kaggle run summary

- export: `madlad-3b__mt-default`; merged records 1186 (Stage 2: 400, new: 670)
- run status: completed; operations 670, valid 669, invalid 1, failed attempts 0
- model load 0.0 s; translation time 7514.8 s; 5.37 ops/min; 20.4 words/s; segments 1111
- model load before the drift probe: 88.8 s
- tokenizer parity: 400/400
- drift probe: 7/7 S1 identical to round 1

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S2 | 168 | 0 | 12.73 | 1.098 |
| T2 | 167 | 1 | 11.405000000000001 | 0.89 |
| S3 | 167 | 0 | 12.36 | 1.114 |
| T3 | 167 | 0 | 11.22 | 0.895 |
- invalid: xsum|Eleuther-AI|xsum-4729 T2: engine:generation_cap
