# `srl/` — signed residual learning, one layer, density over training

No writeup was kept for this experiment. This README is rebuilt from the
script's docstring, the run folders in `results/`, and the notes linked below.

## What it tests

`train_srl_mnist.py` is a standalone copy of the SignedResidualLearning node
(AxonForge `learning4.py`): activations are fully signed (no ReLU), the
reconstruction is signed, each template's leave-one-out residual is multiplied
by the sign of its activation, and every template is rotated on the unit
sphere on every image. It trains one continuous online run and tracks how dense
the code is: the participation ratio (how many templates effectively carry the
energy) and the Gini coefficient of the inhibited activations. There is no
classifier in this experiment.

Background: [`SignedResidualLearning.md`](SignedResidualLearning.md)
(what the sign flip does),
[`SRL_vs_CRG_and_ChannelConv.md`](SRL_vs_CRG_and_ChannelConv.md)
(how it differs from the contrast-residual rule in
[`../../2026_03_16/crg/`](../../2026_03_16/crg/README.md)), and the broader
notes [`UniversalContrast.md`](UniversalContrast.md)
and [`CategoryAlignment.md`](CategoryAlignment.md).

## Runs

Template counts from the saved weights; the other columns are the last logged
row of `srl_mnist_history.json` (running averages over the logging window).

| run | templates | images seen | participation ratio | Gini | recon MSE |
|---|---|---|---|---|---|
| `r001_digits` | 36 | 30,000 | 13.6 | 0.384 | 6.3e-07 |
| `r002_digits` | 81 | 30,000 | 23.7 | 0.385 | 4.6e-07 |
| `r003_digits` | 81 | 90,000 | 21.9 | 0.386 | 4.3e-07 |
| `r004_fashion` | 81 | 30,000 | 23.3 | 0.395 | 1.7e-06 |
| `r005_fashion` | 81 | 90,000 | 27.0 | 0.383 | 1.6e-06 |
| `r006_fashion` | 1024 | 30,000 | 501.3 | 0.335 | 1.7e-06 |
| `r007_fashion` | 1024 | 90,000 | 421.8 | 0.379 | 9.1e-06 |

Each run folder holds the weights with the full participation-ratio and Gini
traces (`srl_mnist_latest.npz`), the history json and a diagnostics figure
(`srl_mnist_diagnostics.png`).

## AxonForge node code

`learning4.py` — the `SignedResidualLearning` and `ConvSRL` AxonForge nodes. Kept as a record; it only ran inside AxonForge.

## Running it

```
.venv/bin/python experiments/2026_03_19/srl/train_srl_mnist.py
```

Set `DATASET` ("digits" or "fashion") at the top; it writes to
`results/r007_<dataset>/` (edit `OUTPUT_DIR` for a new run). Unlike the other
March scripts this one reads the current `data/mnist/digits/train_images.npy` /
`data/mnist/fashion/train_images.npy`, so it should still run; it has not been
re-run since the move.
