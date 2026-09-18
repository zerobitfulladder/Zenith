# `phase/` — angle codes that move by prediction error

Blocks become continuous angles and learn by prediction error (design record Part XVII). Verbatim from the session narrative of the original project (the rest of
it is in [`2026_09_16/glimpse_loop/README.md`](../../2026_09_16/glimpse_loop/README.md)).
`DESIGN.md` below means the design record, now kept in full at the end of that README.
Words are defined in [`2026_09_16/VOCABULARY.md`](../../2026_09_16/VOCABULARY.md).

[`results/fourier.html`](results/fourier.html) is the build-from-nothing tutorial on the
space itself (rings, the torus, Fourier). `phase.py` imports `mnist.py` from
[`../../2026_09_16/glimpse_loop/`](../../2026_09_16/glimpse_loop/) and `tally3.py` from
[`../../2026_09_17/tally/`](../../2026_09_17/tally/).

---

## 2026-09-18: angles, and the first thing that learns

Lavender's observation: this is not an MLP and there is no backprop, and the thing that plays
the part of layers is **the trajectory of the search**. So make each block a continuous angle
instead of a one-hot slot, and push prediction success or failure back along that trajectory as
a rotation. Then Lavender's objection to their own idea: *if a fingerprint is angles, how do
several go into one pile? You cannot add angles.*

You add arrows, not angles. An angle is a unit vector; binding rotates it; bundling adds the
vectors and the sum lands inside the disc, with the consensus as its direction and **the
agreement as its length**. Opposed items cancel to a stub — no opinion. A pile block is two
numbers and the second one falls out rather than being added. It is the old 41-slot block at a
different resolution: a slot histogram's first Fourier coefficient. Keep `m` harmonics and m=1
is one arrow, m=20 is the exact histogram — one knob between the two substrates.

Because binding *adds* angles, the derivative of a match with respect to every angle in the
chain is the same scalar, a sine of a phase disagreement. The backward pass is the walk run in
reverse: one number per block handed to every participant. Nothing multiplies along the chain,
so depth costs nothing — but for the same reason every link takes identical blame, and the
proposed fix (arrow length as a per-angle learning rate) is untested.

`phase.py`: no library, no chunks, no search. Hold out one inked cell, let the other inked cells
vote for what is missing through the learned relation code of their relative offset, clean up
against the vocabulary, rotate everything that took part. **The label is never shown.**
5,000 digits, 2,400 held-out cells from 300 unseen digits, ~90 s.

| | learned | frozen codes | count table | near-miss | same-class sim | diff-class |
|---|---|---|---|---|---|---|
| 32 blocks (4,640 angles) | **49.0%** | 1.5% | 53.5% | 76.6% | +0.091 | +0.056 |
| 16 blocks (2,320 angles) | 38.2% | 1.9% | 52.2% | 67.4% | +0.163 | +0.129 |
| 8 blocks (1,160 angles) | 26.5% | 1.8% | 53.0% | 56.0% | +0.189 | +0.148 |
| 16 blocks, blanks vote | 29.8% | 1.6% | 38.7% | 60.4% | +0.188 | +0.106 |
| only relations learn | 9.8% | 1.9% | — | 24.8% | +0.010 | +0.005 |
| only codes learn | 12.0% | 1.9% | — | 32.6% | +0.108 | +0.060 |

Floor (always the commonest stroke) 2.8%; the count table does the identical job with 332,800
numbers.

- **The pile is a tally, 72× smaller.** 4,640 angles land 4.5 points behind an explicit count
  table. First time in this repo the substrate pays for itself instead of being a costume over
  Python dictionaries.
- **Codes and relations must co-adapt.** 9.8% with codes frozen, 12.0% with relations frozen,
  38.2% with both moving. The halves multiply, they do not add.
- **Part III happened with no label.** Strokes of the same digit class drifted into more overlap
  than strokes of different classes, from "predict the neighbouring cell" alone. The excess over
  the all-pairs baseline is +0.03, and +0.07 when blanks vote — blanks carry the class, and
  letting them vote buys class structure at the cost of 8 points of prediction.
- **Collapse is real and the contrastive term is what stops it.** Pushing only the single wrong
  winner away: all-pairs similarity 0.00 → 0.48 in 500 digits, effective vocabulary 64 → 37.
  Pushing every alternative in proportion to belief: no collapse at any size. The fourth
  appearance of collapse in the record and the same cure each time.
- **What did not work:** arrow length as confidence is weak (0.339 on correct predictions
  against 0.311 on wrong). Unknown whether that is the m=1 loss.

Two follow-ups the same day, from Lavender's questions (*blocks are cheap, why not use lots?*
and *would keeping more Fourier coefficients fix the confidence?*):

| | learned | count table | near-miss | confident quarter (by margin) | unconfident quarter |
|---|---|---|---|---|---|
| 16 blocks, m=1 | 38.2% | 52.2% | 67.4% | 52.3% | 30.3% |
| 16 blocks, m=3 | 25.8% | 52.2% | 54.9% | 44.3% | 14.7% |
| 16 blocks, m=8 | 24.1% | 52.2% | 52.5% | 37.3% | 16.7% |
| 32 blocks | 49.0% | 53.5% | 76.6% | 73.3% | 31.7% |
| 64 blocks | 59.8% | 52.2% | 82.7% | 83.7% | 37.8% |
| 128 blocks | **63.9%** | 52.7% | 86.0% | **87.3%** | 39.3% |

- **Harmonics lost, every step.** More coefficients per block made it monotonically worse, so the
  weak confidence was *not* the one-arrow limitation. Most likely cause is optimisation: the
  gradient of harmonic k oscillates k times as fast, so the objective gets rough.
- **"Use lots of blocks" won.** 16 → 128 blocks takes 38.2% → 63.9%, and **from 64 blocks on the
  learned code beats the count table it was approximating** — it stops compressing a tally and
  starts generalising past one, which a table of exact (offset, stroke) counts cannot do.
- **Confidence was the wrong quantity, not the wrong substrate.** The margin (best score minus
  runner-up) is calibrated where the arrow length is not: at 128 blocks the most confident quarter
  is 87.3% right against 39.3% for the least confident, overall 63.9%.
- **The cost:** sharper codes share less. Same-class overlap excess falls from +0.029 at 16 blocks
  to +0.009 at 128. Prediction and Part III's pressure pull against each other along this knob.

What it changes: the class table — most of the parameters in the counting branch — dissolves
into the pile, and a card's product name becomes a differentiable function of its parts, so
correction now flows through a consolidated card into the stroke codes instead of stopping at a
random name. Structure stays discrete and memorized; codes become continuous and learned.

## How to run things

```
python experiments/2026_09_18/phase/phase.py --blocks 128                   # angle codes learned by prediction error (~5 min); add
                                                                            #   --harm, --blanks, --freeze-codes, --freeze-rel
python experiments/2026_09_18/phase/phase.py --capacity                     # bundling capacity vs harmonics, no learning
```
