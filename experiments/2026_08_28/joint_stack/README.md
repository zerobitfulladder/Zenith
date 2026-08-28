# `joint_stack/` — single-track joint stack (Exp 14)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_joint_stack.py` (also imported as `JS` by antifreeze, joint_cotrain).
- `results/` — metrics.json, galleries, weights_fronts.npz (loaded by antifreeze), run log.

## Exp 14 — run_joint_stack.py: single-track joint stack (user request)

One track only, [frame_t ; frame_t-1] jointly bottom to top: joint
conv L1 (K=128) -> L2 over code (skeleton, K=128) -> lag-advance top
keyed on lagged L1+L2 codes + labels. EP_TOP arms {2, 20}. Same
world/held-outs as Exp 13.

SMOKE REVERSAL AT SCALE: at T=16 the deep keys appeared to CURE the
overtraining freeze (ep20 advance 0.93 > ep2 0.81, held-out advancing
0.73). At T=32 the OPPOSITE: ep20 freezes solid (advance 0.000, match
0.783 — crisp correct digits, zero motion) and ep2 moves but mushy
(advance 0.309, match 0.434, gray ghosts). METHODOLOGY LAW: temporal
smoke runs at reduced T tell opposite stories — validate at full T
(bitten twice today).

Scorecard: (1) depth in keys did NOT lift teacher-forced (0.696 ~
joint-shallow 0.703). (2) Half-right: ep20 froze as originally
predicted; ep2's smoke-scale 0.81 didn't survive (0.309). (3) Held-out
at ep20 draws the CORRECT digit — a 2 for the never-seen (2,right) —
but static: with sharp training the label channel carries IDENTITY
across the enumeration barrier while motion freezes (T=16 had shown
the mirror image: motion composed, identity recited). Identity and
motion generalize through DIFFERENT routes (label vs dynamics), and
which survives depends on training sharpness.

VERDICT: the single-track joint stack loses to the two-track at full
scale on the only axis that matters — Exp 13 two-track is the sole
arm with BOTH crisp renders AND nonzero advance (0.362 @ match 0.79).
The crisp-vs-moving tradeoff is intrinsic to the single track here.
The freeze is now THE temporal blocker; queued: the 08-27 anti-freeze
toolbox (refractory winner, subtractive explaining-away) applied to
these tops, EP_TOP midpoints, factored top read.
