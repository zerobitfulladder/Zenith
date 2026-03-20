# `feedback/` — does top-down gain feedback help a two-layer SRL stack?

No writeup was kept for this experiment. This README is rebuilt from the
script's docstring, the run folders in `results/`, and the notes linked below.

## What it tests

Two stacks of two signed-residual layers (L1: 36 templates over the 784
pixels; L2: 25 templates over L1's 36 outputs) were trained side by side in
the AxonForge graph `ax_graphs/feedback1.json`: one with top-down gain feedback
from L2 onto L1, one without (gain strength 0, no activation trail). The
training happened inside AxonForge; `eval_feedback_mnist.py` only reads the
frozen weights, encodes MNIST through each stack, fits a logistic regression on
L2's leave-one-out-inhibited activations, and also measures reconstruction
error at L1, at L2, and end to end (L2 back down to the image).

The design of the feedback is in
[`Thoughts2.md`](Thoughts2.md) (gain
field, maths, training procedure) and the discussion that led to it in
[`Thoughts1.md`](Thoughts1.md). The
single-layer rule is in [`../../2026_03_19/srl/`](../../2026_03_19/srl/README.md).

## Runs

Each run folder saves the graph settings that were read (`graph_config.json`),
so the five runs are five feedback settings of the with-feedback stack. MNIST
test set, numbers from `eval_feedback_results.json`.

| run | L1 gain strength | gain sign positive | L1 trail decay | L2 gain | test acc, feedback | test acc, no feedback | L2 recon err, feedback | L2 recon err, no feedback |
|---|---|---|---|---|---|---|---|---|
| `r001_digits` | 15.79 | no | 0.0 | 0.0 | 0.8673 | 0.8866 | 0.0913 | 0.1407 |
| `r002_digits` | 15.79 | yes | 0.0 | 0.0 | 0.8607 | 0.8833 | 0.3233 | 0.1312 |
| `r003_digits` | 3.0 | yes | 0.39 | 0.0 | 0.8660 | 0.8816 | 0.1651 | 0.1319 |
| `r004_digits` | 1.0 | no | 0.39 | 1.0 | 0.8761 | 0.8822 | 0.1347 | 0.1267 |
| `r005_digits` | 1.0 | yes | 0.39 | 1.0 | 0.8773 | 0.8794 | 0.1315 | 0.1361 |

In every run the stack without feedback reads the digit better than the stack
with feedback (by 0.2 to 2.3 points). L1 reconstruction error is about 0.024
without feedback in every run; with feedback it is 0.0215 in `r001` and
0.033-0.098 in the others. The no-feedback numbers differ slightly between
runs; why is not recorded.
All layers are fully dense (active fraction 1.0, signed activations).

Each run folder also holds `eval_feedback_diagnostics.png`.

## Running it

```
.venv/bin/python experiments/2026_03_20/feedback/eval_feedback_mnist.py
```

It writes to `results/r005_digits/` (edit `OUTPUT_DIR` for a new run). It does
not run as-is any more: it needs the retired AxonForge setup, namely the graph
`ax_graphs/feedback1.json`, the trained weights in `ax_data/`, and the old MNIST
cache (`data/mnist/...`, `data/mnist_data`).
