# `residual_read/` — residual coding at L1, pure read (Exp 3)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_residual.py` — pure read on `../recon_ladder/results/weights.npz`.
- `results/` — metrics.json, residual_ladder.png, run log.

## Exp 3 — run_residual.py: residual coding at L1, pure read

The reopened shelf item, scoped exactly as agreed: learning untouched,
dictionary frozen (this morning's weights); at encode time each window
gets extra READ rounds — subtract what the winner explained, match the
leftover against the same bank, add that voice; up to R=6 rounds.
Window estimate = mean + norm x (c1*w1 + ... + cR*wR), each c the raw
projection of the current leftover. Confidence-weighted blending
(Exp 2's winner). From-L1 rung only — residual rounds need the real
window to subtract from; deeper rungs reconstruct without the input.
Bridge arm: R=1 in the old renormalized-direction form must reproduce
Exp 2's best from-L1 number (0.936).

Predictions (recorded before the full run; 100-image smoke seen):
1. Round-2+ winners match their leftovers far worse than round-1
   winners match windows — the vocabulary is whole-window-shaped,
   leftovers are off-distribution (the discussed caveat, quantified).
2. Still, real gains: R=2 beats the Exp 2 best; diminishing returns
   by R>=4.
3. REFUTED IN SMOKE ALREADY: I predicted projection scaling would beat
   the renormalized form at R=1 (per-window math says its MSE is
   smaller). Wrong once blending is involved: projection-scaled
   windows carry understated contrast (norm c1 < 1, faded strokes)
   and R=1-projection (0.931) lost to R=1-renorm (0.937); the rounds
   refill the missing energy and overtake by R=2.

## Results Exp 3 (5000 test images)

Bridge verified: R=1 renormalized = 0.0109 / 0.936, Exp 2's number
exactly. The residual arms (projection form, conf blending):

| rounds | MSE | corr | leftover norm | round's match quality |
|---|---|---|---|---|
| R=1 | 0.01366 | 0.930 | 0.542 | 0.807 |
| R=2 | 0.00995 | 0.943 | 0.466 | 0.468 |
| R=3 | 0.00868 | 0.950 | 0.432 | 0.335 |
| R=4 | 0.00795 | 0.954 | 0.411 | 0.281 |
| R=5 | 0.00747 | 0.957 | 0.396 | 0.249 |
| R=6 | 0.00711 | 0.959 | 0.384 | 0.224 |

VERDICT: residual coding WORKS as pure read — the from-L1 rung goes
0.936 -> 0.959, pixel MSE 0.0109 -> 0.0071 (and 0.0157 -> 0.0071 vs
where the morning started: -55%). One extra voice (R=2) already beats
the day's previous best. Gains diminish (+7/+7/+4/+3/+2 thousandths)
but are NOT exhausted at R=6; no window ever stopped early (floor
0.02 unreached, best-dot always positive).

Prediction scorecard: (1) CORRECT and quantified — round-2 winners
match their leftovers at 0.47 vs 0.81 for round-1 winners on windows,
sliding to 0.22 by round 6: the vocabulary is whole-window-shaped and
leftovers are off-distribution to it, yet even mediocre second voices
buy real error because each round only adds what is still missing.
(2) CORRECT. (3) REFUTED (see smoke note above): renormalized beats
projection at R=1 because faded contrast costs more than direction
error; rounds refill the energy and overtake by R=2.

Reading the curves: round 1 explains ~71% of window energy
(1 - 0.542^2); six rounds reach only ~85% — late rounds grab ~2-3%
each. That inefficiency is exactly the case for residual LEARNING
(step two, still shelved): a leftover-shaped vocabulary would lift the
0.47/0.33/0.28 match qualities and get the same error in fewer voices.

Ladder after Exp 3: L1 0.959 / L2 0.899 / L3 0.855. L1's rung moved
+2.3 points while the deeper rungs stood still — the expansion step
(the downward mapping through L2/L3) is now unambiguously the binding
constraint on deep reconstruction. Open beyond that: if L2 consumed a
residual-coded L1 code, its input distribution changes = retraining
question, out of today's pure-read scope.
