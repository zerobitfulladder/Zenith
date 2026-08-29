# Reward refinement on the cascade champion

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## Reward refinement v2: the frontier appears (2026-08-31, very late)

On the champion base, teacher OFF, two fixes from the afternoon's
refutation: TERMINAL PROTECTION (value-tilt only on rows whose habit
says "move"; hover rows read pure habit) and DIRECTED EXPLORATION
(20% of moving decisions try the fastest same-direction word the row
knows). Result — the first controllable RL dial of the project:

    beta 0.0   0.98 / 270      beta 0.2   0.84 / 234
    beta 0.1   1.00 / 273      beta 0.5   0.70 / 206

At beta 0.5 the policy flies at the MEASURED VOCABULARY CEILING
(206 vs 200) — the value tables found exactly the slack the interface
decomposition priced — but along a speed-success trade, not a
dominating point. The afternoon's monotone collapse is gone. Next
lever if pursued: per-row tilt spent only where V says time is being
wasted. (scratchpad outer_td2.py)

### Longer RL (200k steps, user's call: "you didn't train it long"):
### THE FIRST DOMINATING REWARD IMPROVEMENT

    beta 0.1:  success 0.98, median 246  (base: 0.98, 270)

Same reliability, 9% faster, teacher off, pure reward on own flights.
The user was right — 60k steps was undertrained; at 200k the value
tables sharpen enough that the tilt finds genuinely wasted time
instead of noise. beta 0.5 went nonmonotone (|W| grew to 4.3;
exp-tilt saturates) — normalize or cap W next. Toward the 200-tick
vocabulary ceiling, ~46 ticks of slack remain.

## The closing experiment: raw-motor tabula rasa — REFUTED, thesis complete

No reflex, raw (L,R) levels at 50 Hz, pure reward, 400k decisions:
indistinguishable from random (upright 20% vs 17%, survival 81 vs 70
ticks). With the innate reflex the identical machinery hit 0.63 in
fewer decisions. The cascade decomposition is a LEARNABILITY
requirement, not an optimisation: reward cannot erect the reflex
floor while the plant tumbles faster than credit can flow. Brainstems
come innate; dopamine learns upstairs. Full story + computations in
[`ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).

## What is here

The scripts for these runs were kept in the session scratchpad and are not in
the repo. What was kept, in `results/`: `checkpoint_rl.npz` (the champion
plus the reward-refined value tilt, viewer-ready through `cascade_view`,
which loads `value_tilt.npz` from here), `pure_rl.npz` (viewer-ready through
`purerl_view`; log `pure_rl.log`), `raw_rl.npz`, and `rl_resave.log`.
