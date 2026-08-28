# `blend/` — how should overlapping proposals blend? (Exp 2)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_blend.py` — decode-only on `../recon_ladder/results/weights.npz`.
- `results/` — metrics.json, blend galleries, run log.

## Exp 2 — run_blend.py: how should overlapping proposals blend?

User's question and proposal. Overlap handling on the way down (each
pixel is covered by up to 16 L1 windows; each L1 position by several L2
footprints) was a plain average in Exp 1. Three rules, applied at every
blending step of the descent, decode-only on Exp 1's weights:

  avg     plain average (Exp 1 baseline)
  conf    confidence-weighted average (the old renderer's soft weighting)
  winner  user's proposal — per target location, the most confident
          source writes its value alone; no blending

Confidence of a source = the max of its code segment (how well that
window matched its best template). The user correctly rejected max-
pooling the VALUES on the way to this design (brightest is not most
trustworthy); winner-by-source-confidence is the coherent hard form.
Note the symmetry with Exp 1's finding: within-window we learned to
pick the winner template; this asks whether ACROSS-window overlap wants
a winner too.

Prediction (recorded before the full run; a 100-image smoke was seen):
averaging across overlapping windows is doing real error-correction —
neighboring windows make partly independent quantization mistakes and
the average cancels them — so hard winner-per-pixel loses everywhere,
worst at the deep rungs where any one source's proposal is least
reliable; conf ~ avg with a small edge on the top-1 read.

## Results Exp 2 (5000 test images)

Verification passed: every avg arm reproduces Exp 1 exactly. Shape corr
(higher = better; MSE ordering identical):

| rung, read | avg | conf | winner |
|---|---|---|---|
| L1 graded | **0.909** | 0.904 | 0.886 |
| L1 top-1  | 0.934 | **0.936** | 0.929 |
| L2 graded | **0.892** | 0.891 | 0.838 |
| L2 top-1  | 0.895 | **0.899** | 0.845 |
| L3 graded | 0.855 | **0.855** | 0.765 |
| L3 top-1  | 0.827 | **0.828** | 0.735 |

VERDICT: the hard winner-per-location rule LOSES at every rung and
loses worse with depth (-0.09 corr at L3). The gallery (blend_L3.png)
shows the mechanism directly: rectangular seams where the writing
source switches, broken strokes, patchwork background — one confident
source is still only right about ITS patch, and its whole window gets
written into territory a neighbor knew better. Confidence-weighted
averaging is a genuine small win on the top-1 read (new best from-L1
number: 0.936 / MSE 0.0109) and a wash on graded.

LAW OF THE DAY (Exp 1 + Exp 2 together, two sides of one coin):
WHETHER TO COMPETE OR AVERAGE DEPENDS ON THE CORRELATION OF THE VOTERS.
Within a window, the 64 templates are highly correlated views of the
same patch — averaging blurs, pick the winner. Across overlapping
windows, the sources look at genuinely different patches — their
quantization errors are partly independent, averaging cancels them,
and hard selection throws the correction away. Soft confidence
weighting is safe everywhere and best where the code is sparse.
(The user's own first instinct — "hmm, that might not work when I
think about it" — was correct before the run.)

Current best ladder after Exp 2: L1 0.936 (top-1, conf), L2 0.899
(top-1, conf), L3 0.855 (graded, avg/conf).
