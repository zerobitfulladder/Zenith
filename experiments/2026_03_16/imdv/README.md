# `imdv/` — one-layer IMDV (push/pull with leave-one-out inhibition) on MNIST

No writeup was kept for this experiment. This README is rebuilt from the
scripts' docstrings, the run folders in `results/`, and the notes linked below.

## What it tests

`train_imdv_mnist.py` trains one layer of templates with GeodesicIMDVv2, one
image at a time: a pull toward the input and a push away from the
leave-one-out shadow, both as rotations on the unit sphere. It records both the
raw activations and the leave-one-out-inhibited ones (the "soma" output, which
is sparser). `eval_imdv_mnist.py` freezes the templates, encodes the images with
the inhibited activation, fits a logistic regression and writes accuracy, a
per-class report and a confusion matrix.

The rule is written up in
[`GeodesicIMDVv2.md`](GeodesicIMDVv2.md); its
history (V2 .. V13 of the "data vacuum") is in
[`ww6.md`](ww6.md). It is the
predecessor of the contrast-residual rule in [`../crg/`](../crg/README.md).

## Runs

Numbers from `results/<run>/imdv_mnist_history.json` (epoch 10) and
`eval_classification_results_loo-inhibited.json`, 100 templates each.

| run | test acc | top-3 | active fraction (test, inhibited) | recon MSE |
|---|---|---|---|---|
| `r001` | 0.5586 | 0.7475 | 0.049 | 26.26 |
| `r002` | 0.9006 | 0.9785 | 0.552 | 0.0745 |

`r001` has about 4.5 of 100 templates active per image and a very large
reconstruction error; `r002` is dense (about 56 active) and reaches 0.90.
What was changed between the two runs is not recorded.

Each run folder holds the weights (`imdv_mnist_latest.npz`), the training
history, the training diagnostics figure and the evaluation figure and json.
(The `weights_path` inside the eval json still says `experiments/r00x/...`,
the path at the time.)

## AxonForge node code

`learning2.py` — the `GeodesicIMDVv1` / `GeodesicIMDVv2` AxonForge nodes; `learning_conv.py` — `GeodesicIMDVConv` and `ReconstructConv`, the same rule over patches. Kept as a record; it only ran inside AxonForge.

## Running it

```
.venv/bin/python experiments/2026_03_16/imdv/train_imdv_mnist.py
.venv/bin/python experiments/2026_03_16/imdv/eval_imdv_mnist.py --weights experiments/2026_03_16/imdv/results/r002/imdv_mnist_latest.npz
```

The training script writes to `results/r002/` (edit `OUTPUT_DIR` for a new
run); the eval script writes next to the weights. Neither runs as-is any more:
both load MNIST from the retired AxonForge cache (`data/mnist/mnist`,
`data/mnist_data`, read with the `mnist` package).
