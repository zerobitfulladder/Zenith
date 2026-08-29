# td_bound — the bound form on sensor tracks (2026-08-29, late)

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

`../temporal_drone/td_bound.py`, `run_td_bound.py`, `results/`. The user's calls:
no tally — all learning and mapping inside the templates/hypercolumns;
no motor-history tracks (thrusters have no dynamics and the teacher is
a pure function of state, so past commands carry nothing the sensor
windows don't); fixed template counts.

A layer-3 template is one 480-cell vector:

    [ dx L2 code | dy L2 code | tilt L2 code | collective | differential ]
       128           128          128            48 fov       48 fov

Training is the ordinary rule — full vector competes, winner rotates.
Inference is the curve rig's partial-cue read — sensor cells written,
command cells empty, masked norm-corrected scores (floor 0.25), top-1
winner, each command channel read as one graded bump, sharpened ONCE
on its foveated axis. Energy: 25% per sensor track, 12.5% per command
channel (`TB_CMD`). Tracks unchanged from td_tracks (5 taps x 24 cells,
K1=64; 7 codes, K2=128; graded top-8 up). K3=8192, same as the failed
run on purpose. Staged freeze: L1s learn to 15k ticks, L2s to 30k,
then L3 alone (stored codes must keep their meaning).

Predictions, recorded before the full run:

1. Steering on-path corr clears the 0.62 alphabet cap (dedicated cells
   dissolve it); ceiling ≈ the d~5 neighbour bound, ~0.84.
2. Collective without the motor echo is the real test of whether the
   track windows resolve vertical velocity: ≥0.8 says yes; ~0.63 says
   no and queues the Aug-28 speedometer fix (a velocity channel of its
   own, never a difference of foveated position codes).
3. EVAL may stay 0.00 even with good reads — pure cloning still
   compounds its own drift closed-loop; if so the next lever is
   DAgger-style correction data, not capacity or layers.
4. Fragmentation watch: command-in-the-metric is the mechanism that
   lost the Aug-28 bound-vs-tally A/B 61-to-3. If the read decays as
   L3 fills, that is it showing up; the dial is `TB_CMD`.

Smoke (15k ticks, stages 3k/6k): mechanics sound, and with only ~1200
templates the read already beat the tracks rig's full training —
collective corr 0.92 / steering 0.58 vs 0.63 / 0.34.

## Results (300k ticks, 5 min, `results/bound.log`)

On-path read at the end: **collective corr 0.898 / SNR 2.24, steering
corr 0.586 / SNR 1.11, agreement 0.672** (tracks rig: 0.63 / 0.34 /
0.28). EVAL 0.00 at every checkpoint. Prediction scorecard:

1. **Steering did NOT clear 0.62 — it reached it.** Plateau 0.53–0.59.
   The full pipeline now scores what yesterday's rig could only reach
   with layer 3 bypassed and the true code decoded graded (0.62) — the
   decode losses are gone, and what binds now is the CUE: picking the
   right moment out of a d~5 state space. The remaining gap to the
   ~0.84 neighbour floor is selection, not reconstruction.
   `stored_commands.png` confirms it from the other side: the
   templates' stored command histograms match the oracle's command
   distribution closely on both channels — coverage and resolution are
   fine; which template wins is what's noisy.
2. **CONFIRMED: the windows resolve vertical velocity.** Collective
   0.89–0.94 with no motor echo — above the 0.82 the lagged-motor cue
   bought. No speedometer needed on this axis.
3. **CONFIRMED: EVAL 0.00 with good reads — but look at
   `episodes.png`.** The trajectories genuinely bend toward the target
   (closest 1.3–2.5 m on half the episodes); the drone approaches,
   cannot brake and hold inside 0.25 m, sails past, and exits. Real
   closed-loop competence with no terminal precision — exactly the
   compounding signature. The lever is correction data on the pupil's
   own states, or the stabiliser-plus-residual split.
4. **Fragmentation: mild, then stable.** A sag after the fill
   (0.94 → 0.86 collective) recovering to 0.90 by 300k, ~4000 distinct
   winners per window throughout — nothing like the 0.68 → 0.39 decay
   that killed the Aug-28 bound arm. At 25% command share the bound
   form holds.

`read_traces.png` shows the texture of the remaining error: the pupil
tracks the collective closely, follows the slow shape of the steering,
and misses the oracle's fast one-tick corrective spikes entirely; the
worst reads are the first ~1.5 s of an episode, where the track
windows are still part-empty.

Viewing:

    VIEW_CKPT=experiments/2026_08_29/bound/results/weights_best.npz \
      .venv/bin/python viewer.py        # pick "pupil" in the dropdown

`weights_best` = the final state (EVAL tied at 0.0 all run, so best is
by on-path read; the sidecar json says so). `run_td_bound.py` now
tie-breaks best-tracking on agreement for future runs.

Figures: `bound_viz.py` writes `training_curves` / `read_scatter` /
`read_traces` / `stored_commands` / `track_vocab` / `episodes` into
`results/`.

## The handover/noise probe — what is actually missing (measured)

Fifty fresh episodes per arm, the EVAL gate, on `weights_best`:

| arm | success | closest (mean) |
|---|---|---|
| A. pupil end-to-end | 0.00 | 3.38 m |
| B. oracle flies the first 1 s, pupil the rest | 0.02 | **0.60 m** |
| C. oracle to 1.5 m & slow, pupil holds | 0.08 | **0.16 m** |
| N1. oracle + pupil-sd noise, fresh each tick | **1.00** | 0.05 m |
| N3. same noise, held 8 ticks | **1.00** | 0.04 m |

(Pupil's measured on-path error: coll sd 0.267 N, diff sd 0.148 N,
winner holds 2.5 ticks.)

1. **Precision is NOT the binding constraint.** The oracle wearing the
   pupil's full error magnitude — even held 8 ticks — succeeds 1.00.
   Closed-loop feedback eats unbiased error of this size. What it
   cannot eat is the pupil's error being a fixed function of state
   (the same wrong answer at every revisit — bias, not noise). The
   d~5 story needs revising again: at this task gate, selection noise
   per se is survivable.
2. **The blind start dominates end-to-end failure.** Covering just the
   first second (the windows' warm-up, during which the pupil can only
   hover while any_init tumbles it) moves closest approach 3.38 m ->
   0.60 m. The previously REJECTED blind-period hypothesis was
   rejected while the read was broken; with the read fixed it is the
   largest single factor. A faithful fix exists: act on partial
   windows (L1 needs 13 ticks, not 50 — the masked partial-cue read
   already handles short L2 windows), shrinking blindness ~4x.
3. **Hold data does not exist.** Handed the drone at 1.5 m and slow,
   the pupil gets INSIDE the 0.25 m ball (closest 0.16 m) and still
   fails 92% of the time — because training episodes end at the FIRST
   at_goal tick (inherited from run_tracks), so the demonstrations
   contain approach only, never the ten-tick hold the gate demands.
   The memory was never shown "at the goal, slow, level -> stay".
   Fix: let training episodes run through the hold before resetting.

Queue, in order: dwell-at-goal training episodes; partial-window
bootstrap reads; then DAgger-style correction rounds for what remains.

The 15k-tick smoke run is in `results/smoke/`.
