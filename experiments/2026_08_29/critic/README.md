# The critic — value only at the selector

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## The RL design, settled (user, 2026-08-30) — value only at the selector

Three-part law, decided in discussion before building:

1. **Template content is a record, everywhere.** Sensory tracks, motor
   machinery, top layer alike learn by rotation toward what actually
   happened — never rotated away, never reward-modulated. Outcome
   signals must not edit memory ("rotating away could rotate anywhere").
2. **Value is a per-template SCALAR, and only on the top layer.** Like
   the `wins` counter: template-attached, outside the normalized row
   (cells would drag content on every TD update and leak value into
   the similarity metric). The top layer is the action picker, so the
   credit lands there — "my choice to pick up a cup serves drinking;
   the credit goes to the picker, not the grasp controller." Motor
   layers learn sensor-motor association from whatever is executed.
3. **Goals reach the lower layers only through the behavior
   distribution.** Selection bias changes what is practiced; practice
   changes the associative diet. (The same mechanism that made
   tight-envelope DAgger work.)

Staging: (1) saturation run lands; (2) critic only — V[j] on L3,
SARSA-style TD with eligibility traces along frozen-policy flights,
sanity gates: V high in basin / low mid-tumble / correlated with
time-to-balance, plus the near-tie V-spread check (if near-ties carry
equal value the selector has nothing to choose with); (3) selection
bias — top-k by similarity, winner by similarity + beta*V, content
learning untouched, teacher off. Watchlist for stage 3: the picker can
only pick what the repertoire contains — exploration must put rare
primitives into the stream before association can record them.

## Stage 2 in flight: the critic (`run_td_critic.py`, `results/`)

Saturation settled first: the 600k dagger3 run COMPLETED before the
session died — final strict EVAL **0.72**, bal_frac 0.86, end-speed
~0.2-0.4 m/s, plateauing 0.60-0.72 over its last 100k. That checkpoint
is the frozen base; the replication rerun was terminated (user call).

Critic: V[j] per L3 template (a scalar beside the row, like `wins`),
SARSA TD along the frozen pupil's own winner chain, r = 1 iff
balanced, gamma 0.98 (V = discounted balanced-time, 0..50), lambda
0.9 traces cut at episode boundaries. Predictions, before the run:

1. Gate A (separation): basin >> tumble mean V. (Smoke, 8k ticks:
   0.51 vs 0.04 — already ordered.)
2. Gate B (honesty): corr(V[winner], empirical return) > 0.6 at 160k.
3. Gate C (leverage): the decisive unknown — spread of V among top-8
   similarity candidates, and how often argmax-V disagrees with
   argmax-similarity. High spread + frequent disagreement = stage 3
   (selection bias) has room; near-zero spread = the tiling does not
   separate good from bad actions and stage 3 is dead on arrival.

## Critic gates passed; naive value-biased selection REFUTED

Gates at 160k ticks: A separation basin/mid/tumble mean V = 8.55 /
3.81 / 1.33; B honesty corr(V[winner], measured return) = 0.712;
C leverage: top-8 V spread 4.3, argmax-V vs argmax-sim disagreement
0.83, mean +4.4 V over the similarity winner.

Then the beta sweep (read-only, margin 0.05, 50 eps/arm, beta=0
reproduces the 0.72 baseline exactly — instrument validated):

    beta   0     0.005  0.01  0.02  0.05  0.1
    EVAL   0.72  0.68   0.42  0.30  0.26  0.22

MONOTONE HARM. The 83% "leverage" was optimism, not opportunity: V is
a state value, so among near-ties the higher-V template remembers a
NICER situation, and picking it executes the command for a calmer
world than the drone is in — systematic under-reaction, the glide->
hover retrieval disease resurfacing through the value channel. Root
cause: each template's V was learned only in its own home states —
no near-tie was ever TRIED where its rival won, so within-set V
differences measure state quality, not action advantage. The value of
the road not taken cannot be read from data that never took it.

Fix in flight (`results/explore/`): phase A repeats with
eps=0.25 exploration among near-ties, TD credit going to the EXECUTED
candidate (the old rig's recorded credit bug, honoured). Prediction:
if exploration makes near-tie V action-informative, some beta > 0
beats 0.72; if the sweep still degrades monotonically, a per-template
scalar cannot express action advantage at this tiling and the next
move is a state-baseline-corrected advantage or true actor learning.

`run_td_critic.py` writes `results/critic.npz` (console `results/critic.log`;
the exploration rerun `results/explore/`, `results/critic_explore.log`;
smoke test `results/smoke/`). `run_td_select.py` is the read-only beta
sweep; it writes `results/select_sweep.json`. Both start from
`../balance/results/dagger3/weights_best.npz`.
