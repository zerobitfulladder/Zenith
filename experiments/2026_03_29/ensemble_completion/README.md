# `ensemble_completion/` — many hypercolumns side by side as one memory

No writeup was kept in this folder; the findings are in
[`EnsemblePatternCompletion.md`](EnsemblePatternCompletion.md)
("Ensemble (Multiple Zenith Nodes)"). This README points there.

## What it tests

M independent hypercolumns ("nodes" in the script) of k=10 templates each all
see the same sparse pattern; their activations, laid side by side, are the
code. A pattern is retrieved from a masked input by finding the stored full
input whose code is closest (cosine). The question is whether this holds far
more patterns than one hypercolumn has templates.

## Runs

The script was run three times with different settings; each run's report was
renamed by hand afterwards (the script itself writes `report.md`). The note
calls them `report.md`, `report2.md`, `report3.md`; on disk they are:

| report | hypercolumns M | patterns N | load N/k | sparsity | retrieval at 40% masked | at 60% | at 80% |
|---|---|---|---|---|---|---|---|
| [`report1.md`](results/report1.md) | 6 | 40 | 4 | 0.10 | 99.75% | 95.65% | 83.70% |
| [`report2.md`](results/report2.md) | 50 | 40 | 4 | 0.05 | 100.00% | 99.90% | 95.25% |
| [`report3.md`](results/report3.md) | 50 | 200 | 20 | 0.05 | 99.61% | 90.91% | 42.22% |

In the last run all 200 patterns get distinct winner codes (100% uniqueness).
The figure [`results/ensemble_completion.png`](results/ensemble_completion.png)
is from the last run.

From the note: a single hypercolumn learns almost nothing useful at load 4
(its own similarity is about -0.04), yet the joint code is discriminative.
Exact matching of all winners was too strict and was replaced by cosine on
the joint code. The note is explicit that this is a high-capacity lookup
memory, not a generaliser, which is what `generalization_test/` then tried to
measure.

## Running it

```
.venv/bin/python experiments/2026_03_29/ensemble_completion/ensemble_completion.py
```

Settings are constants at the top (currently those of `report3.md`); writes
`results/report.md` and `results/ensemble_completion.png`.
