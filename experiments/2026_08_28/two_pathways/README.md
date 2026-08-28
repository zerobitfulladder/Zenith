# `two_pathways/` — form + motion pathways (Exp 13)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_two_pathways.py` (also imported as `TP` by joint_stack, antifreeze, joint_cotrain).
- `results/` — metrics.json, galleries, weights_fronts.npz (loaded by antifreeze), run log.

## Exp 13 — run_two_pathways.py: form + motion pathways (user design)

Direction-preferring neurons (= lag-advance, reconfirmed from biology;
Reichardt wiring) + ventral/dorsal split: form conv track on frames,
motion conv track on SIGNED frame diffs, global lag-advance
unification [GC*present-form ; lagged form ; lagged motion ; labels];
joint-[t;t-1]-channel front as control. Digits 0-4 x {right,down,rot},
T=32 circular laps, held-out (2,right)(3,down)(4,rot). Staged: fronts
GPU B=8, top online interleaved.

Scorecard (predictions in header):
1. PARTIAL, instructive: at smoke scale (2 top-epochs) two-track
   free-runs at advance 1.00 incl. rotation; at full training (20
   epochs) advance falls to 0.362 (match 0.79 — right digit, stalling
   phase) and the JOINT control freezes SOLID (0.00). The old stall
   disease returns through the re-see loop (sharper top rows tolerate
   render-drift less — teacher-forced two-track 0.768 vs joint 0.703
   stays healthy). NEW OBSERVATION: overtraining the top HURTS
   free-run (fuzzy rows generalized to own renders; sharp ones
   don't) — an EP_TOP sweep is a real lever, plus the known
   anti-freezes (refractory, subtractive feedback).
2. CONFIRMED in kind, modest in degree: motion units motion-selective
   0.528 vs digit 0.340 (form-neutral-ish); form units 0.458/0.400.
3. CONFIRMED: held-out combos recite the nearest trained combo — the
   strips show a 3 drawn for the (2,right) query, a 0 for (3,down)
   (v5 global-top-1 law at scale). Factored READ at the top = the fix.
4. CONFIRMED both ways: joint front most specialized on both axes
   (0.559/0.430, transition-tuned) AND freezes completely in free-run
   while two-track keeps advancing — MOTION CONTEXT IS AN ANTI-FREEZE
   (the lagged diff breaks the self-match symmetry that stalls
   form-only context), incomplete but real: 0.362 vs 0.000.

Verdict: mechanism validated (two-track > joint on every axis:
teacher-forced prediction, unit economy, freeze resistance); free-run
at scale needs the anti-freeze toolbox; composition needs the
factored top read. Queued: EP_TOP sweep, refractory/subtractive top
read, factored (per-pathway partial-match) top retrieval.
