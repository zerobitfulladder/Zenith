# 2026-03-16 — one layer of templates, two learning rules

The AxonForge era. These scripts are standalone versions of AxonForge nodes,
trained on MNIST / Fashion-MNIST and scored with a logistic regression on the
frozen activations. No writeups were kept; the READMEs are rebuilt from the
docstrings, the saved results and the design notes kept beside each experiment
(the `.md` files in `early_rules/`, `imdv/` and `crg/`).

## `early_rules/` — the first learning rules (WW1–WW5)

Activation functions alone, then rotating the winner toward the input, learning
from the residual, push-pull between templates, and functional repulsion. Each
failed in a way that motivated the next; the notes say how. AxonForge node code
and notes only.

Full writeup: [`early_rules/README.md`](early_rules/README.md).

---

## `imdv/` — push/pull with leave-one-out inhibition

GeodesicIMDVv2: templates pulled toward the input and pushed away from what the
other templates already explain. Two runs, 100 templates: 0.5586 on MNIST when
the inhibited code was very sparse (about 4.5 of 100 active), 0.9006 when it
was dense (about 56 active).

Full writeup: [`imdv/README.md`](imdv/README.md).

---

## `crg/` — contrast-residual learning

ContrastResidualGeodesic: mean-centred input, and each template's
leave-one-out residual as its only learning signal (no separate push and pull).
Six runs from 36 to 1024 templates. 100 templates reach 0.9166 on MNIST and
0.8255 on Fashion-MNIST; 1024 templates reach 0.9018 on the MNIST test set with only 15% of
templates active. Some run folder names do not match the test set that was
used (see the writeup).

Full writeup: [`crg/README.md`](crg/README.md).

---

## `shared_nodes/` — helper nodes used by every March graph

Not an experiment: the preprocessing (mean-centring), reconstruction, display
and weight-holding AxonForge nodes that the learning nodes of March relied on.

Full writeup: [`shared_nodes/README.md`](shared_nodes/README.md).
