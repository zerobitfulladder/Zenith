# Competitive experts on 8x8 patches, no labels

2026-08-31. 50 experts x 8 templates. The winner is whoever rebuilds the patch
best (fit-only competition), and only the winner learns; a win-frequency
conscience keeps anyone from starving. The question: do specialists appear
without supervision? The control is `../patch_ensemble`, identical except that
every expert learns from every patch.

No separate writeup was kept; the result is in the day README
([`../README.md`](../README.md), section `patch_ensemble/` and
`patch_competitive/`): pairwise subspace overlap 0.3436 against the control's
1.0000, a 47.6-point gap between an expert's error on its own patches and on
everyone else's, and templates that are edges, corners and curves.
`results/01_claimed.png` is that vocabulary, learned with no labels at all. The
caveat recorded there: it tiles a continuum (many diagonals at gradually
rotating angles) rather than discovering discrete parts.

Run: `.venv/bin/python experiments/2026_08_31/patch_competitive/patchcomp.py`

Results: `results/metrics.json`, `results/weights.npz`,
`results/01_claimed.png`, `results/02_templates.png`, `results/03_examples.png`.
