# `residual_learning/` — residual learning, two shapes (Exp 4)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_residual_learning.py` — the seq and loo arms.
- `results/` — metrics.json, ladders, template galleries, `weights_seq.npz` / `weights_loo.npz`
  (the seq weights are loaded by residual_full_gpu, corrections, labelgen), run log.

## Exp 4 — run_residual_learning.py: residual LEARNING, two shapes

User unshelved the idea, and it turns out they built a version of it
MONTHS ago: experiments/2026_03_16/crg/learning3.py / learning4.py are parallel
leave-one-out residual learners — every active template learns at
once, each toward what the OTHERS fail to explain (reconstruct with
everyone-except-me = the "shadow"; learn toward the unexplained part).
Notably their LOO-inhibited output IS the subtractive explaining-away
mechanism this project re-derived as the anti-freeze on 08-27 — the
user had it first, months earlier. Two arms, both retraining L1 from
scratch (55k, L2/L3 retrained on top unchanged, upward speech still
dense relu of the original window):

  seq  sequential greedy: round's winner steps toward the CURRENT
       leftover, projection subtracted, next winner competes over the
       remainder; up to R_TRAIN=4 rounds (round 1 = the old rule).
  loo  the user's parallel form, adapted (relu-active set, geodesic
       step, same eta).

Predictions (recorded before the full run; a 300-train smoke seen):
1. Each shape optimizes its own read mode — loo directly improves the
   dense graded sum (its objective IS the graded read), seq serves the
   residual/greedy read. Smoke agrees violently: loo graded-from-L1
   0.988 corr (!) but residual read collapsed to 0.73 at R=1; seq
   lifted round-2 match quality 0.47 -> 0.56 and R=6 to 0.969.
2. seq: small round-1 dip (vocabulary splits duty; smoke: 0.807 ->
   0.784) but net win on the residual read at every R.
3. loo cost prediction: its code is a TEAM code — individual templates
   stop being good solo matchers, so the skeleton-learning L2/L3 (which
   take per-position top-1 views) inherit a meaningless top-1 and the
   deep rungs degrade (smoke: L3 0.628 vs seq 0.825).

DIRECTION (user, while the run was in flight): what bothered them
about learning3/4 all along was precisely the "explaining together" —
a team code where a unit is only meaningful inside the ensemble. That
is not the vision. The vision is the SEQUENTIAL form: soloist units,
each a committed standalone template, multi-voice = one-at-a-time over
the remainder. The loo arm is hereby a CONTROL (it prices what joint
explaining buys and costs), not a candidate.

## Results Exp 4 (55k train, 5000 test)

SEQ ARM (the vision) — residual learning works on top of residual
reading; new project-best from-L1 reconstruction:

| voices | Exp 3 frozen dict | seq-trained |
|---|---|---|
| R=1 | 0.930 | 0.921 |
| R=2 | 0.943 | 0.948 |
| R=3 | 0.950 | 0.958 |
| R=4 | 0.954 | 0.964 |
| R=6 | 0.959 | **0.969** (MSE 0.00539, -24%) |

The vocabulary became leftover-shaped as designed: match quality per
round [0.771, 0.585, 0.471, 0.389, 0.331, 0.288] vs frozen [0.807,
0.468, 0.335, 0.281, 0.249, 0.224] — every later round up sharply,
round 1 paying the predicted small duty-split price. THREE VOICES NOW
BUY WHAT SIX BOUGHT THIS MORNING (seq R=3 0.958 ~ frozen R=6 0.959).
All 64 units used. Bonuses: the plain graded read ALSO improved
(0.909 -> 0.938 — leftover-shaping diversifies the palette in a way
that helps even the dense sum), and the depth cost is near-zero
(L2 0.884 vs 0.892, L3 0.847 vs 0.855) — the soloist form keeps the
skeleton view meaningful for L2/L3, unlike the loo team code.

LOO CONTROL (full 55k — the price tag on "explaining together"):
the smoke dissociation holds at scale, sharpened. At its own read the
team code is extraordinary: graded from-L1 0.989 corr / MSE 0.00222 —
2.4x lower pixel error than anything else today. At every other read
it is broken: greedy R=1 0.733; match quality FLAT ~0.31-0.36 at all
six rounds (no template individually matches anything — a team with
no soloists; one template explains ~13% of window energy vs seq's
64%); deep rungs L2 0.846 / L3 0.720 (vs seq 0.884/0.847). The
galleries are the verdict made visible: templates_L1_seq.png = clean
strokes, curves, and a visible population of corner/crumb specialists,
every unit readable alone; templates_L1_loo.png = scattered pixel
confetti, not one recognizable stroke in 64 — individually meaningless
correction terms that only sum to sense.

LAW OF EXP 4: THE LEARNING RULE AND THE READ MODE MUST BE THE SAME
SHAPE. Solo learning (sequential residual) makes a solo-readable code
that also improved the dense read (0.909 -> 0.938); team learning
(parallel LOO) makes a team-only code — supreme at its own read,
collapsed under every other, and poisonous to skeleton-learning layers
above. The user's original discomfort with learning3/4 ("they were
trying to explain together") is now a measured architectural defect,
not a taste. The 0.989 stands as the known ceiling of what a 64-unit
dense-sum code can reconstruct at L1 — a target, reached only by
giving up unit identity.
