# 2026-09-23 — the basics again: layers, loops, bands

Most of the day was a conversation rather than code: how the same six-layer sheet becomes
visual cortex with a thick input layer and motor cortex with a thick output layer, which
layers feedback leaves from and lands on, what imagination looks like in the layers (the
feedback zones active with layer 4 quiet), what motor imagery is (the same with the tap
turned off), why every area has a thalamic loop and what a higher-order nucleus like the
pulvinar can and cannot do (sum per location, no sideways wiring), and what a Fourier
coefficient and a Gabor phase actually are, against what the angles in the September choir
were (the direction of a two-filter response, a crest position only when the two filters are
a matched wave pair).

Out of that came one testable idea: that each level of a visual hierarchy could learn from
its own band of the image, at its own scale, rather than from the level below. The
experiment below tests it against a CNN with everything else held equal. It lost, and the
idea is dropped.

`torch` was added to the project today for it (`uv add torch pyarrow`).

---

## `bands_vs_stack/` — does a layer need the layer below, or just a coarser image?

**It needs the layer below.** Grayscale CelebA at original size, 40 attributes, balanced
accuracy. Five levels that each read the image at their own scale (full to 1/16) against a
five-layer CNN with the same filters, the same pooling and an identical readout on top.
Aligned faces, 8 epochs, one seed.

With everything else equal, reading the image instead of the level below cost 5.7 points
(0.772 against 0.829), and at this budget it fell below a linear map on raw pixels (0.776).
Giving each scale three convs of its own, with more parameters than the CNN, reached 0.819
and still did not catch up. What that recovered came from depth, which the CNN has anyway,
not from the bands. The point the CNN keeps sits on small parts at a fixed place: glasses,
beard, the mouth. Undertrained at 8 epochs, one seed, aligned only; the 20-epoch and jittered
runs are scripted but were not run, since the like-for-like comparison had already gone
against the idea. The jittered condition takes away the fixed grid the bands depend on and
was expected to widen the gap, not close it.

Full writeup: [`bands_vs_stack/README.md`](bands_vs_stack/README.md).
