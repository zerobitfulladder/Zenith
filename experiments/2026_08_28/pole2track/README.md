# `pole2track/` — deep two-track swing-up (Exp 17)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_pole2track.py` — the rig and DAgger run (imports the pole world from
  `experiments/2026_08_26`).
- `run_pole_controls.py` — MLP and 1-NN controls on the same codes.
- `pole2track_viewer.py` — pupil / MLP / oracle viewer, loads `results/`.
- `results/` — the 87/100 run: config_best.json, weights_best.npz, expert.json, mlp.joblib,
  metrics.json, logs (`run_pole_regen.log` is a rerun log of the same folder).
- `results/k4096/` — a variant with a 4096-row top, flat arm only; no writeup was kept.
  Its metrics.json records a best of 2/100 (`run_pole_k4096.log`).

## Exp 17 — run_pole2track.py: deep two-track swing-up (user spec) — SOLVED

The run_pole_tracks retry (parked 0/100 on 08-26). Sensory = foveated
360-deg bump ring x lagged slices [now, -4, -16] (direction-selective
pattern; lag-1 measured ALIASED: 0.996 corr between opposite swings at
catch speeds — delay baselines must match the speeds to resolve, the
two-hands law as delay lines) + efference echo. Motor = bipolar
thermometer -> 32-unit bank. Top K=2048, ACTION BESIDE THE METRIC
(rows sensory-only; the action a tally beside — action-in-metric
measured -11 pts alone), correction-rich balance tails, LQR TEACHER
(discrete-Riccati catch + energy-shaping pump vs this exact simulator;
50/50 validation; pump sign audition — first version pumped energy
OUT, 8/50). MLP/1-NN controls: 0.931/0.905 vs bank 0.5 — proved the
encoding fine and localized every gap. DAgger x4, arms:

| arm | best closed-loop (100 eps, swing-up-from-anywhere gate) |
|---|---|
| flat (no banks) | 15 (mean-decode: boundary mush -> zero force) |
| ident (motor bank) | 69 (top-1 motor unit = MODE read beats mean) |
| **full (both banks)** | **87** (r0 41 -> r1 87) |

THE DEEP TWO-TRACK WORKS — 0/100 -> 87/100, and the banks HELP
(full > ident > flat; the sensory dictionary is the best part, not the
breaker). Notable: TF agreement stays ~0.6 while closed-loop hits 87 —
per-step mimicry is not control competence (many force sequences solve
the task). MODE-VS-MEAN law: at decision boundaries decode the modal
action (top-1 motor unit), never the mean (mean of opposing commands
= zero force = the observed release-at-top). Viewer:
pole2track_viewer.py with pupil/MLP/oracle dropdown. NEXT: planar
drone (6-D state, two coupled thrusters — the factored-policy
question), inheriting this recipe.
