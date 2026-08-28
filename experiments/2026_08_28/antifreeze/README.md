# `antifreeze/` — the anti-freeze trio, read-time (Exp 15)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_antifreeze.py` — on `../two_pathways/results/` and `../joint_stack/results/` weights.
- `run_make_gifs.py` — animated GIFs of the Exp 15 free-runs (truth, two-track baseline,
  two-track + subtractive read incl. the held-out combos, the frozen joint stack, the joint
  stack unfrozen by adaptation). No separate writeup was kept; its docstring is the description.
- `results/` — metrics.json, antifreeze_runs.png, run logs; `results/gifs/` — the GIFs.

## Exp 15 — run_antifreeze.py: the anti-freeze trio (read-time)

Three fixes on the frozen rigs, no retraining: code-loop (feed back
retrieved cargo, self-cleaning), subtractive read (scores minus
overlap with decaying retrieved-row history, SUB=.5), adaptation
channel (per-unit fatigue, +.3/x.7). Advance / match, trained combos:

| arm | joint (frozen 0.000) | two-track (0.438 rebuilt) |
|---|---|---|
| code loop | 0.000 / 0.784 | 0.475 / 0.769 |
| + subtractive | 0.183 / 0.729 | **0.672 / 0.783** |
| + adaptation | 0.502 / 0.631 | 0.711 / 0.671 |

HEADLINE REFUTATION: the code loop — my top prediction — did NOTHING
(joint stays exactly 0.000; identical numbers to pixel loop). The
freeze survives a perfectly clean diet, so own-render drift was NOT
the binding cause. Revised diagnosis: PHASE CROWDING — at stride 2,
adjacent frames' codes are nearly identical (1px motion; the old
translation-tolerance lesson), so the self-row keeps winning
regardless of diet; plus hold-self-reinforcement.

WHAT WORKS: suppression + the arrow. Subtractive is the best
quality-preserving anti-freeze (two-track 0.438 -> 0.672 advance at
UNDIMINISHED match 0.783 — the new best temporal config); adaptation
unfreezes hardest (0.711, and the only thing that moves held-out
combos: 0.319) but pays in match. Crucially both advance FORWARD
(0.67-0.71, not 50/50): the user's skepticism of refractory was
correct for last night's arrow-less rig, and the lag-advance rows are
exactly what turn suppression from a coin flip into forward motion —
SUPPRESSION SUPPLIES MOTION, THE ARROW SUPPLIES DIRECTION.
