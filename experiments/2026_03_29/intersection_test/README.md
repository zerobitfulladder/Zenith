# `intersection_test/` — can the ensemble learn AND?

No writeup was kept for this experiment beyond the script's report. This
README is rebuilt from the docstring and [`results/report.md`](results/report.md).

## What it tests

Training patterns are triplets [A, B, A AND B] of 100-bit sparse vectors
(20 active bits each, so about 4 shared). 30 hypercolumns of 10 templates learn
500 triplets. At test time a new pair [X, Y, 0] is shown, the nearest stored
training triplet is found (by the joint code, or by raw input as the baseline),
and its third part is compared with the true X AND Y.

## Result

| method | Jaccard | precision | recall |
|---|---|---|---|
| raw nearest neighbour | 0.0718 | 0.1471 | 0.1171 |
| ensemble-code nearest neighbour | 0.0333 | 0.0777 | 0.0457 |

The raw baseline wins, and both are poor: the report's own caveat is that
retrieving a stored neighbour's third part can never compute AND for new
inputs. Figure: [`results/intersection.png`](results/intersection.png).

## Running it

```
.venv/bin/python experiments/2026_03_29/intersection_test/intersection_test.py
```

Pure numpy on synthetic data; writes `results/report.md` and
`results/intersection.png`.
