# Many dense hypercolumns trained jointly with the label: do they vote well?

2026-08-31. The 08-30 `weighted` rule (144 templates, whole MNIST images,
centre then unit-normalise) on `[image ; label]` with the label scaled to match
the image block's energy. Many differently seeded hypercolumns trained on the
same data at the same time. Reading a label out of one is a linear map that is
exactly rotation-invariant, so if they all converge to one subspace they compute
the same classifier and voting buys nothing — that is the prediction tested.
`gate.py` loads the trained weights and tries six ways of picking the single
most confident hypercolumn instead of averaging, plus an oracle and an
anti-oracle bound.

No separate writeup was kept; the result is in the day README
([`../README.md`](../README.md), section `dense_ensemble/`): 30 hypercolumns
voting gain +2.9 points and saturate by 10, because they agree with each other
83% of the time. Gating by self-confidence never beats averaging.

Run: `.venv/bin/python experiments/2026_08_31/dense_ensemble/ensemble.py`,
then `gate.py`.

Results: `results/metrics.json`, `results/gating.json`, `results/weights.npz`,
`results/01_vote_curve.png`, `results/02_gating.png`.
