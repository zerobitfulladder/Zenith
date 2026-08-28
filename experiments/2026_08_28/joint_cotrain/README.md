# `joint_cotrain/` — everything learns together (Exp 16)

2026-08-28. Moved verbatim from the day README.

## Files

- `run_joint_cotrain.py`
- `results/` — metrics.json, galleries, weights.npz, run log.

## Exp 16 — run_joint_cotrain.py: everything learns together (user)

One stream, all layers plastic at once (L1 joint 2ch, L2 skeleton,
lag-advance top on carried lagged codes, interleaved, 20 epochs).
All three predictions held: NO monopoly anywhere (128/128/512 units
used, top max win share 4.2% — online-interleaved co-training is
safe, unlike the 08-27 minibatch-on-forming-code case); teacher-forced
pays ~6 points vs staged (0.637 vs 0.696, held 0.36 ~ 0.379); the
freeze is untouched (0.000 both loops — read-side property,
independent of training schedule; code==pixel again, corroborating
Exp 15). VERDICT: joint bottom-to-top training is viable at modest
cost; combine with the subtractive read next.
