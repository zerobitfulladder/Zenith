# `concat/` — read the digit label back out of an SRL layer trained on image + label

No writeup was kept for this experiment. This README is rebuilt from the
script's docstring, the run folders in `results/`, and the notes linked below.

## What it tests

A standalone copy of the AxonForge graph `concat.json`. During training each
28x28 MNIST image (mean-centred) gets one extra row of 28 cells stacked under
it: the digit label, written as a fixed sparse code (a hash of the label with
one active cell out of 28, also mean-centred). One layer of signed-residual
templates learns online on the 29x28 input. At test time the label row is left
at zero, the input is reconstructed from the frozen templates, and the
reconstructed label row is compared (cosine) with each digit's code; the best
match is the predicted digit. No classifier is trained.

The signed-residual rule is in [`../../2026_03_19/srl/`](../../2026_03_19/srl/README.md).
The same label-concatenation idea, later tried with top-1 templates, is written
up in [`../../2026_03_28/topk_label/TopKAndLabelConcatenation.md`](../../2026_03_28/topk_label/TopKAndLabelConcatenation.md).

## Runs

From `results/<run>/results.json` (MNIST test set, seed 42, learning rate 0.01,
one active cell per label code).

| run | templates | training images | test acc | top-3 | top-5 |
|---|---|---|---|---|---|
| `r001_digits` | 100 | 60,000 | 0.7711 | 0.9303 | 0.9707 |
| `r002_digits` | 500 | 180,000 | 0.1548 | 0.4048 | 0.6052 |

With 100 templates the reconstructed label row names the right digit 77% of
the time. With 500 templates and 180,000 presentations (three times the training set) it falls to 15.5%, barely above
chance (10%).

Each run folder holds the templates (`srl_weights.npy`), the label codes
(`digit_codebook.npy`), `classification_report.txt` and a diagnostics figure
(`concat_diagnostics.png`).

## AxonForge node code

`matrix_concat.py` and `channel_concat.py` — the AxonForge nodes that stacked the label under the image. Kept as a record; it only ran inside AxonForge.

## Running it

```
.venv/bin/python experiments/2026_03_22/concat/eval_concat_mnist.py
```

It writes to `results/r002_digits/` (edit `OUTPUT_DIR` for a new run). It does
not run as-is any more: it loads MNIST from the retired AxonForge cache
(`data/raw/...`, `data/mnist/...`, `data/mnist_data`, read with the `mnist`
package).
