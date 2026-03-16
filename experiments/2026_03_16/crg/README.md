# `crg/` — one-layer contrast-residual learning on MNIST and Fashion-MNIST

No writeup was kept for this experiment. This README is rebuilt from the
scripts' docstrings, the run folders in `results/`, and the notes linked below.

## What it tests

`train_crg_mnist.py` trains one layer of templates with the
ContrastResidualGeodesic rule (the AxonForge node `learning3.py`), one image
at a time: the input is mean-centred, each template's leave-one-out residual
is its only learning signal, and templates are rotated on the unit sphere.
`eval_crg_mnist.py` freezes the templates, encodes train and test images with
the leave-one-out-inhibited activation, fits a logistic regression on those
activations, and writes accuracy, a per-class report and a confusion matrix.

The rule is written up in
[`ContrastResidualGeodesic.md`](ContrastResidualGeodesic.md);
the design notes around it are in this folder
([leave-one-out sparsity](LOOSparsityAndInactiveTemplates.md),
[activation threshold](ActivationThreshold.md),
[mean centring](MeanCenteringAndLayerCommunication.md),
[temporal integration](TemporalIntegration.md)). The
earlier attempts that led here are in [`../early_rules/`](../early_rules/)
(`ww1.md` .. `ww5.md`) and [`../imdv/ww6.md`](../imdv/ww6.md).

## Runs

Numbers from `results/<run>/crg_mnist_history.json` (last logged epoch),
`eval_classification_results_loo-inhibited.json` and the saved weights.

| run | templates | epochs | test acc | top-3 | active fraction (test) |
|---|---|---|---|---|---|
| `r001` | 100 | 10 | 0.9166 | 0.9829 | 0.475 |
| `r002` | 512 | 10 | 0.8237 | 0.9767 | 0.451 |
| `r003` | 100 | 1 | 0.9149 | 0.9813 | 0.475 |
| `r004_fashion` | 100 | 1 | 0.8255 | 0.9774 | 0.517 |
| `r005_fashion` | 1024 | 10 | 0.9018 | 0.9763 | 0.150 |
| `r006_digits` | 36 | 10 | 0.7117 | 0.9428 | 0.617 |

No template died in any run (`inh_dead_template_fraction` 0.0). The 1024-template
run is the only one where the inhibited code is clearly sparser than the raw
one (9% vs 49% active during training).

Caution on the run names: the per-class test counts in the eval json are 1000
per class (the Fashion-MNIST test set) for `r002`, `r004_fashion` and
`r006_digits`, and 980/1135/... (the MNIST test set) for `r001`, `r003` and
`r005_fashion`. So `r005_fashion` was evaluated on MNIST digits and
`r006_digits` and `r002` on Fashion-MNIST, whatever the folder name says.

Each run folder holds the weights (`crg_mnist_latest.npz`), the training
history, a training diagnostics figure (`crg_mnist_diagnostics.png`) and the
evaluation figure and json.

## AxonForge node code

`learning3.py` — the `ContrastResidualGeodesic` AxonForge node. Kept as a record; it only ran inside AxonForge.

## Running it

```
.venv/bin/python experiments/2026_03_16/crg/train_crg_mnist.py
.venv/bin/python experiments/2026_03_16/crg/eval_crg_mnist.py --weights experiments/2026_03_16/crg/results/r001/crg_mnist_latest.npz
```

The training script writes to `results/r006_<dataset>/` (edit `OUTPUT_DIR`
for a new run); the eval script writes next to the weights it is given.
Neither runs as-is any more: both load MNIST from the retired AxonForge cache
(`data/mnist/...`, `data/mnist_data`, read with the `mnist` package), which no
longer exists.
