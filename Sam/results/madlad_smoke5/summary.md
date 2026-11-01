# Kaggle run summary

- export: `madlad-3b__mt-default`; merged records 405 (Stage 2: 400, new: 5)
- run status: completed; operations 5, valid 0, invalid 5, failed attempts 0
- model load 0.0 s; translation time 141.7 s; 2.13 ops/min; 8.6 words/s; segments 8
- model load before the drift probe: 87.7 s
- tokenizer parity: 400/400
- drift probe: 0/5 S1 identical to round 1

| stage | valid | invalid | median seconds | median length ratio |
|---|---|---|---|---|
| S2 | 0 | 5 | 27.63 | 0.919 |
- invalid: xsum|OpenAI|xsum-4710 S2: engine:generation_cap;check:repetition
- invalid: xsum|Tsinghua|xsum-4719 S2: engine:generation_cap;check:repetition
- invalid: xsum|BigScience|xsum-4720 S2: engine:generation_cap;check:repetition
- invalid: xsum|Human|xsum-4721 S2: engine:generation_cap;check:repetition
- invalid: xsum|LLAMA|xsum-4729 S2: engine:generation_cap;check:repetition
