# Two tracks into one memory: the drone flown from a (situation, thrusts) memory

2026-08-30. No writeup was kept for this experiment; this page is written from
the scripts' docstrings and `results/metrics.json`.

## What it tests

Can a drone be flown by a memory that stores whole "in this situation, these
thrusts" pairs, learned from the PID oracle?

    sensory track   dx, dy, angle          -> one hypercolumn
    motor track     the two thrust levels  -> one hypercolumn
    layer two       [sensory code ; motor code] concatenated, one
                    hypercolumn, so a template is a whole
                    "in this situation, these thrusts" pair

When flying, the motor half is left empty. The sensory half alone picks a
winner (scored only over the cells that are actually written), and the
winner's own motor half is read back out and flown. Each value is written as
a bump of neighbouring cells; the channels get separate blocks so no two
channels share a cell. The sensory track is not given velocity or angular
rate, although the oracle uses both, so some of any gap to the oracle is
built in, not a training failure.

## What came out (`results/metrics.json`)

900 oracle episodes, 155,462 ticks, 3 epochs; 1024 sensory templates, 169
motor templates, 4096 layer-two templates; 295 s.

| policy | success |
|---|---|
| PID oracle (the teacher) | 1.0 |
| two-track memory, graded read | 0.0 |
| two-track memory, top-1 read | 0.0 |
| hover, no control | 0.0 |

The memory did not reach the target in either read mode. `probe_td.py` was
written to split the question: is the read path broken (shown a state the
oracle visited, does it return the oracle's thrusts?) or is (dx, dy, angle)
simply not enough (how many different commands did the oracle give for one
sensory state?). Its output was not kept.

## Files

| | |
|---|---|
| `two_track.py` | the two tracks, layer two, the partial-cue read |
| `run_td_tracks.py` | oracle rollouts, training, benchmark, figures |
| `probe_td.py` | open-loop read check and ambiguity count (needs the rollout cache in an old scratchpad) |
| `results/flights.png`, `results/commands.png` | flights, and what comes out of the motor half that was never written |
| `results/metrics.json` | the table above |
| `results/two_track.npz`, `results/two_track.json` | the trained memory, loadable in the viewer |

    .venv/bin/python experiments/2026_08_30/td_tracks/run_td_tracks.py
    VIEW_CKPT=experiments/2026_08_30/td_tracks/results/two_track.npz .venv/bin/python viewer.py

Uses `sl_drone` from `experiments/2026_08_28/single_layer_drone` and
`fast_oracle` from `experiments/2026_08_29/temporal_drone`.
