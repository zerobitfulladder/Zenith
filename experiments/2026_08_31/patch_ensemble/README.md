# Ten dense hypercolumns on 8x8 patches (the control for `../patch_competitive`)

2026-08-31. 10 hypercolumns x 100 templates, the 08-30 dense rule, 8x8 MNIST
patches at stride 1. Every hypercolumn learns from every patch. Note the regime
change: 8x8 is 64 numbers (63 after centring) and 100 templates, so this is
overcomplete where every 08-30 dense run was undercomplete.

No separate writeup was kept; the result is written up in the day README
([`../README.md`](../README.md), section `patch_ensemble/` and
`patch_competitive/`): with all experts learning, the pairwise subspace
overlap came out at 1.0000 and the templates were 100 panels of speckle.

Run: `.venv/bin/python experiments/2026_08_31/patch_ensemble/patches.py`

Results: `results/metrics.json`, `results/weights.npz`,
`results/01_templates.png`, `results/02_hypercolumn0.png`.
