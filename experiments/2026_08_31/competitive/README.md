# Ten hypercolumns, and only the one most confident in the label learns

2026-08-31. Same shape as `../dense_ensemble` (08-30 dense `weighted` rule,
144 templates, whole MNIST, `[image ; label]` at rho=1.0) with one change: per
sample, every hypercolumn completes the blanked label, and only the one whose
completion most supports the true label takes a learning step. The control arm
has every hypercolumn learning from every sample. `reconstruct.py` lets the
winning hypercolumn rebuild a digit and draw one from a label.

No separate writeup was kept; this is the 10 x 144 row of the table in the day
README ([`../README.md`](../README.md), section `competitive/`,
`competitive40/`, `competitive200/`): 0.8698 with a fit gate, 0.1658 with a
confidence gate, 2 of 10 dead. It is the run that showed the right gate depends
on whether the experts are specialists. The follow-up with the same 1440
templates split as 40 x 36 is `../competitive40`, which has the full writeup.

Run: `.venv/bin/python experiments/2026_08_31/competitive/compete.py`, then
`reconstruct.py`.

Results: `results/metrics.json`, `results/reconstruction.json`,
`results/weights.npz`, `results/01_results.png`,
`results/02_templates_*.png`, `results/03_reconstructions.png`.
