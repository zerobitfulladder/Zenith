# 2026-09-20 — layer 1 of the choir (HelloInductor line)

Words are defined in [`../2026_09_16/VOCABULARY.md`](../2026_09_16/VOCABULARY.md).

## `choir/` — what a voice learns to measure

A restart: can a vocabulary be made out of the choir itself? Trained only to reconstruct,
layer 1 learns two-pixel differences, which are worse than random projections at the one
property the architecture needs (a shifted patch must give the same chord, rotated by a fixed amount). Adding
a shift-coherence term fixes it (coherence 0.916 against 0.873 for a hand-built Gabor bank)
and costs 10 points of reconstruction. A fixed Fourier transform does not substitute. A
vocabulary does emerge, with parts shared across all ten classes. A two-track build that
reads the label from the top choir scores 92.4% when the label is kept out of the choir
during training, against 13.9% when it is left in (it just copies itself).

Also in the folder, with no writeup: `unpool.py`, `nodecoder.py`, `generate.py`.

Full writeup: [`choir/README.md`](choir/README.md).
