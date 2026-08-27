# Exp 1: run_square_completion.py — bouncing square

References: arrow forms (next-slot and lag-advance) 0.00 mean |x|
error; trail-only (no arrow) 8.12.

Instrument: signed circular phase offset of every free-run retrieval
vs the retrieved row's home phases (final-epoch learning moments),
per layer. Forward fraction is the direct measurement of the arrow.

Concessions (v1): emission deblur = brightest 16 pixels (square is
4x4, known count); free-run emission held constant between L1 speak
ticks (system emits at 1/3 input rate by design).

Predictions, registered before running:
- Claude: the offsets center at self/backward (nearest-snapshot
  symmetry: early in an era the previous era's completed snapshot is
  the closer match), emission ~ re-states the present, and the square
  crawls or freezes; forward fraction well under 0.5. The down-tint
  is era-grained and cannot pick the phase within an era.
- User's mechanism predicts: forward fraction dominates (completion
  systematically retrieves the later-written whole), and the square
  advances.

Outcome: FROZE, and the instrument caught the mechanism. Free-run
mean |x err| 10.02 (worse than trail-only 8.12): the square crawls
19->20, hits the wall, emits x=20 forever. Teacher-forced (the clean
measurement): L1 retrieves ITSELF 1.00 of the time, L2 0.82, L3 0.97;
emission sits at the present 0.95. The completion arrow never fired —
not because completion points backward, but because it was never
exercised: with K1=64 rows for 40 distinct phases (final-epoch mean
phase-span of an L1 row = 0.5 frames), the bank tiles time at MOMENT
resolution. There is always a stored row exactly AT the query's
phase, so the best match is always "yourself", never a later-written
era-whole. A memory with more units than moments is a lookup table
of moments — a map with no compass. Template galleries
(templates_L{1,2,3}.png) show it: L1 rows = single-phase square+
comet-tail snapshots; L2 rows = era-scale comet smears (span ~15
frames) but 0.82 self-matched; L3 = whole-half-canvas smears with
scattered multi-phase homes.

Next (queued, decisive for the hypothesis): starve capacity so rows
MUST span eras (fewer units than moments — also independently queued
from 08-26 as "primitives vs phase memorizers"), then re-read the
forward/backward fractions: if completion, when actually forced to
complete a half from a coarser whole, points forward, the user's
arrow is real and the failure was allocation grain, not mechanism.

---

Files: `run_square_completion.py` (the rig; the other 08-27 experiments import
`cn1`, `onehot`, `deblur`, `decode_x` and friends from it), `show_templates.py`
(draws `results/templates_L{1,2,3}.png` from the saved weights). Results in
`results/` (`generated.gif`, `homes.json`, `report.md`, `weights.npz`, galleries).

Run:

    .venv/bin/python experiments/2026_08_27/completion/run_square_completion.py
    .venv/bin/python experiments/2026_08_27/completion/show_templates.py
