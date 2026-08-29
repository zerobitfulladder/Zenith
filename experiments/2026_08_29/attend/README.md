# Attention, rung 1

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## Rung 1 of attention: the choosing works; the using has a bug

`gpu_attend.py`, `results/`, `../temporal_drone/attend_view.py` (viewer hook).

CHOOSING — VALIDATED, reproduced in every run: greedy conditional-MI
selection over the six cue blocks (quantisers free: a present block's
bump cell, a context block's L2 winner; shuffle-bias corrected). Raw
relevance credits free-riders (ctx-dx 0.216, angle 0.143 — they
correlate with position); CONDITIONED on the chosen present blocks
they collapse (ctx-dx 0.027, angle 0.024) and are refused. Selected:
pres-dx (0.287), pres-dy (0.155), nothing else — the champion's
hand-built cue, rediscovered by the architecture from its own stream.

USING — BROKEN, and provably not attention's fault: the hand-truth
control arm (shares set to the known answer, no learned attention in
the loop) fails identically. Signature: approach PERFECT (closest
0.04-0.08 m, better than the champion) and the hold dead (EVAL
0.03-0.07 vs the champion's 1.00 on a paper-identical cue). Three
principled fixes tried and refuted: notch length + champion density;
boot-from-episode-starts (fixed the approach variance — closest went
consistent — not the hold); attention-as-pruning (suppressed blocks
removed from the dimensionality). The bug lives in the rebuild
pipeline, in the last quarter-metre.

NEXT (agreed with user): stop guessing, run the FIELD PROBE — sweep a
(dx, dy) grid, read the commanded velocity of broken arm, champion
and teacher at each point, diff the three vector fields; the bug will
be visible at an address near the origin. Weights for both arms saved
(`weights_learned.npz`, `weights_handtruth.npz`, viewer-ready via
attend_view; watch the symptom: perfect aim, broken grip).

## THE DRUNK PILOT, CONVICTED (2026-08-30, 23:59)

Five refuted hypotheses (notch length/density, boot distribution,
pruning, phase scaffolding, GPU-vs-numpy parity — measured 2.5e-07
cue diff, 100% winner agreement), then the notch trap: row 239's
6765 fresh notches ALL came from r < 0.25 m (median 0.09) — clean
attribution — and the row STILL filled with "cruise right 2 m/s".
So the label->word assignment was the last suspect standing, and one
probe convicted it:

    arm bank:  label (0.00, 0.00) -> word decoding (+1.97, -0.02)

**The velocity vocabulary has no word for "stay".** The bank's 32
prototypes, booted from early spawn moments (large commands), never
grew a near-zero word; stillness gets assigned to a cruise word, so
every correctly-attributed near-goal notch lands wrong. Perfect aim,
drunk parking. (The champion's bank has the same disease in a mild
form — its zero-word carries a +0.34 m/s bias, averaged down by the
duty mixture; the arm's 2 m/s cannot be averaged away.)

FIX (tomorrow, minutes): guarantee vocabulary coverage — longer/
better bank training window, or seed a zero prototype, PLUS the
vocabulary audition as law: after bank training, assert
decode(assign(0,0)) is near zero, the way teachers are auditioned.
Nothing else in the attention rung needs to change: choosing was
right, co-adaptation was right, the çetele was honest — the failure
was one missing word in a 32-word dictionary.

## CURED: the co-adaptive loop at 1.00 (2026-08-31, small hours)

Bank curriculum extended (GO_BANKS default 240k: the vocabulary now
hears whole episodes, dwell included) + the vocabulary audition law
in code. Result: the USER'S co-adaptive attention loop — leaky
cetele, never-frozen top, self-converging mask, no phases — reaches
**EVAL 1.00, closest 0.06 m, median 265**: full champion parity, with
attention alive. Rung 1 is closed, cured, and crowned in one night.

Watch-item for later: late in the run the shares drift asymmetric
(p-dx 0.82 / p-dy 0.18) with EVAL easing 1.0 -> 0.93 — the greedy
selector credits the first-picked block's gain; consider symmetrizing
or slowing SLEW.

QUEUED NEXT (user's spec, for the fresh session): the MNIST transfer
— conv L1 (8x8, stride 1) -> conv L2 -> L3, cetele over LABELS as the
action vocabulary ("the network acts in labels"), classification =
the cetele readout; generation = TALLY INVERSION (pick a label,
invert the count table P(template|label), activate those templates,
project down the descent machinery from 2026_08_28).
Hypothesis on record: as selection improves, generation improves.
Close kin: Exp 9 labelgen (hard readout 0.9276, best gallery) — the
new elements are cetele counts instead of bound label cells, the
co-adaptive attention on the cue, and the selection->generation
coupling claim.

`gpu_attend.py` writes `results/` (console `results/attend.log`);
`gpu_coadapt.py`, the co-adaptive loop of the last section, writes
`results/coadapt/`. Both import from `../cascade/`, `../deep_outer/` and
`../chase/`.
