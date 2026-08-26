# `pole_tracks/` — the two-track pole architecture (open, parked)

Script: [`run_pole_tracks.py`](run_pole_tracks.py). Results: [`results/`](results/) — `report.md`, `weights.npz`.

### Two-track faithful architecture (run_pole_tracks.py) — OPEN, parked

The user's full design (perception track -> code; motor track -> code;
unification over [enc ; mot]; inference = angle up the enc track,
motor-half reprojects through the motor track): currently FAILS,
0/100 in every DAgger round across three configurations, vs 58/100
for the flat single-bank collapse on identical demos/harness.

Established facts (measured): motor track round-trips perfectly
(9/9); the founding convention (per-pathway center+normalize before
concat, then global — the MNIST top's H->cn->concat->cn pattern) was
initially violated, restoring it lifted teacher-forced 0.21 -> 0.35
(still far from the flat rig's 0.92); widening the perception bank
KE 128 -> 512 did not fix closed-loop. Something in the perception
recoding still loses what the unification needs; not yet identified.

Next-session bisect plan: (a) teacher-forced at KE=512; (b) fixed
1-NN level-predictability from enc codes (first attempt had an
empty-slice bug); (c) the decisive bisect — encoder=identity + real
motor track (isolates which track's recoding breaks the chain).
Convention reminder banked at cost: EVERY consumer centers+normalizes
its input, per pathway first, then the concat — always, everywhere.
