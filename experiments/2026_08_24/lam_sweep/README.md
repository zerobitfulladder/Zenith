# 2026-08-24 — `lam_sweep/`: F7 / F7b: the label weight lam

Part of the day log [`../README.md`](../README.md).

## F7: lambda sweep on the 8x8 champion — predictions (before running)

User position: the label half should carry EQUAL energy to the code
half (lam=1.0, a 50/50 split). Standing evidence against is from the
OLD 4x4 rig ("lam=1.0 worse everywhere — norm wasted on dims zeroed
at query"); never re-swept on the 8x8 champion. `run_lam_sweep.py`:
dense rig, lam in {0.25, 0.5, 0.75, 1.0}, measuring both directions —
image->label (hard readout) and label->image (retrieval margin of the
label-only query).

1. Hard readout declines monotonically past lam=0.5 (energy stolen
   from the matching half); lam=1.0 clearly worst.
2. Label-retrieval margin rises with lam but is ALREADY saturated at
   0.5 (10/10 with a large margin) — the gain buys nothing.
3. lam=0.25 slightly soft on retrieval margin, hard readout ~equal or
   marginally better than 0.5.

### Outcome (same night) — both mechanisms real, bracketing the
### working point; lam=0.5 stays

`results/sweep/`: hard = .9036 / **.9102** / .9092 / .8956 for
lam 0.25 / 0.5 / 0.75 / 1.0; retrieval margin rises monotonically
(0.278 -> 0.808) but consistency is 10/10 at EVERY lam (saturated —
decisiveness never converts). P3 wrong in the informative direction:
lam=0.25 LOSES 0.7pp — the user's discriminative-pressure mechanism is
real and governs the low end (separation ~6% falls inside the margin
zone; class mixing begins). The wasted-norm mechanism governs the high
end (50/50 loses 1.5pp, old-rig verdict reproduced on the champion).
Plateau 0.5-0.75. The constant is now an understood balance: label
energy must EXCEED the inter-memory margins (else mixing) and stay
far below parity (else query-invisible dims starve the matcher).

## F7b: label generation from images at lam=1.0 — predictions (before running)

User request: train at lam=1.0 and test label generation from images
directly. Note: F7's hard column IS that test (.8956 vs .9102 at 0.5).
The untested part: with double-strength stored label halves, do the
GRADED completion styles (which lost at lam=0.5) recover?
`run_lam1_labelgen.py`: one lam=1.0 train, four readers: top-1 winner,
graded completion (relu correlations x stored label halves), top-10
vote, sharp softmax.

1. top-1 reproduces ~.8956, still below lam=0.5's .9102.
2. Graded completion improves RELATIVE to its lam=0.5 self (stronger
   label halves) but stays below top-1 — the produce-by-committing
   law holds at every lam.
3. No variant at lam=1.0 beats the lam=0.5 champion .9102.

READING — the mode-switched law generalizes to the whole upward pass:
"in generous, out strict" is not just about reading memories. The
entire recognition pass IS the taking-in-evidence mode and must stay
graded; committing (hardening) is safe only where the source is a
single coherent pattern — downstream of retrieval (generation renders
from ONE memory; no noise to amplify). Sparse speech upward = hardening
noisy evidence = compounding flips. Closes the all-sparse question:
cortex-style sparse activity, if it is to work here, needs either
complementary (non-clone) speakers or stability across near-ties —
plain per-position top-k has neither. Residual competition remains the
one untested lever on this thread.

### Outcome (same night) — no recovery anywhere; doubling the label
### half makes EVERY reader worse, graded ones most of all

`results/lam1_labelgen/` (one lam=1.0 train, four readers; lam=0.5
references from F7 and the completion-readout dense row):

| reader | lam=1.0 | lam=0.5 ref | delta |
|---|---|---|---|
| top-1 winner | .8998 | .9080 / .9102 | -0.8pp |
| graded completion (relu C x stored label halves) | .6772 | (~.6894 all-200 vote) | ~flat, floor |
| top-10 raw vote | .8246 | .8468 | -2.2pp |
| sharp softmax (t=.02) | .9036 | .9080 | -0.4pp |

- P1 CONFIRMED: top-1 lands at .8998, in F7's .8956 neighbourhood and
  below the lam=0.5 champion.
- **P2 REFUTED, and in the informative direction.** The graded readers
  did not recover — they got *worse*, and the committed-vs-graded gap
  WIDENED (top-1 minus top-10 vote: 6.1pp at lam=0.5 -> 7.5pp at
  lam=1.0). Doubling the stored label energy cannot help a graded
  reader because the reader's disease was never weak label halves: it
  is flat correlations *across memories* (0.45 vs 0.40), and the label
  block is invisible at query time (zeros there), so it changes the
  correlations not at all — except to steal unit norm from the code
  half that produces them. Bigger labels = same flatness, weaker
  matching. Graded completion lands at .6772, essentially reproducing
  the all-200 vote floor (.6894): weighting by stored label halves
  instead of one-hot owners is the same flat-evidence pooling.
- P3 CONFIRMED: the best lam=1.0 reader (.9036) still loses to the
  lam=0.5 champion (.9080-.9102). Nothing at parity beats the plateau.
- Lone hint in the other direction: at lam=1.0 sharp softmax (.9036)
  edges *above* top-1 (.8998), where at lam=0.5 the two were exactly
  tied (.9080). Sub-jitter, but it is the only place stronger label
  halves paid — sharpening first, then pooling, is the one graded
  style that survives contact with a strong label block.

Reading: F7 + F7b close the lambda question in both directions. The
label half must be big enough to exceed inter-memory margins (F7's
lam=0.25 loss) and small enough to leave the code half its norm (this
result), and NO reader style rescues the high end. The
produce-by-committing law holds at every lam: the winner's max is the
sufficient statistic, and evidence pooled across memories is flat
evidence regardless of how loudly each memory states its class.

### Files

- `run_lam_sweep.py` (F7) -> `results/sweep/`
- `run_lam1_labelgen.py` (F7b) -> `results/lam1_labelgen/`
