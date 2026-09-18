# 2026-09-18 — angles that learn, and depth (HelloInductor line)

Words are defined in [`../2026_09_16/VOCABULARY.md`](../2026_09_16/VOCABULARY.md). The full
design record for this day (Part XVII) is kept in
[`../2026_09_16/glimpse_loop/README.md`](../2026_09_16/glimpse_loop/README.md).

## `phase/` — codes as angles, moved by prediction error

Each block becomes a continuous angle; hold out one inked cell, let the others vote for it,
and rotate everything that took part. The label is never shown. At 16 blocks it reaches
38.2%; with more blocks it keeps climbing (63.9% at 128), and from 64 blocks on the learned
code beats the count table it was approximating. Keeping more Fourier coefficients per block
made it worse at every step. [`phase/results/fourier.html`](phase/results/fourier.html) is a
tutorial on the space itself.

Full writeup: [`phase/README.md`](phase/README.md).

---

## `depth/` — does credit survive four levels of hierarchy?

A synthetic hierarchy of cards, every leaf seen but one, predict the missing one. With leaf
counts matched, depth is free: 4 levels and 81 leaves recover perfectly (1.000), and the
hierarchy beats a flat card with the same leaves (0.963). Two earlier false results came from
the setup, not the substrate.

Full writeup: [`depth/README.md`](depth/README.md).
