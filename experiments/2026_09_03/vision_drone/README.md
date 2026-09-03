# The drone that sees: image in, tally out

2026-09-03. The earlier drone rigs were handed dx and dy. This one gets an
image and has to build its own state from it, through the stack that came out
of today's MNIST/Fashion work, and the tally maps that state to the PID
teacher's thrust levels. No gradient anywhere; frozen when it flies.

## The world (`box_world.py`)

The 2-D quadrotor of `2026_08_28/single_layer_drone` (x, y, vx, vy, tilt,
tilt rate; two thrust levels of 13) inside a box of half-size 5. The drone
starts at the centre at rest and level; the target is uniform inside a
margin of 1. Egocentric vision: 48x48 pixels covering +-6 units around the
drone, the drone always at the centre as a bar rotated by its tilt with a
rotor blob at each end, the target a bright blob. No walls drawn. Teacher:
the PID tuned 2026-08-30 (viewer.py). `results/sample.png` shows an episode
and what the drone sees.

## Data (`collect.py`)

400 PID episodes from the centre: 43,304 ticks, success 1.00, median 109
ticks. `flight.npz` = states, targets, levels, episode, tick; `frames.npy` =
the 48x48 frames rendered from the stored states. 80/20 split by episode.

## The rig (`vrig.py`, `train.py`, `vision_pupil.py`)

```
frames  (1 channel, or 2: the frame and the frame minus the previous frame)
L1      400 strokes, 9x9 (x channels), top-1, cnt/n, no pressure, unsupervised on 6000 frames
pool    contrast message (match - mean over templates, ReLU) max-pooled into 4x4 -> 6400, standardised
L2      400 object templates, purity step, label 0.25 + belief 0.5, hire on error, 10 epochs on 20k frames
tally   one table over the joint command (left x right level, 169 columns),
        read as the joint argmax, the two marginal argmaxes, or the count-weighted MEAN level per motor
```

`vision_pupil.load_policy` renders the frame itself, so the viewer (repo
root, patched: the drone's eye is drawn live top-right, and it defaults to
`results/pupil.npz`) only hands it the state.

## First results

```
pupil                     within 1 level (both)   mean |level err|   flight success   median final dist
single frame                      0.696                 1.31              0.02              7.67
two frames, offset bug            0.711                 1.20              0.00              7.61
PID                                                                       1.00       (109 ticks)
```

- **The templates are right** (`objects_mean_2ch.png`: each object template
  drawn as the mean of the frames it wins). Every object is a definite scene
  -- target up-right of a tilted drone, target below, target on the drone --
  and the tally reads them as a pilot would: target above -> 11/12, below ->
  0/2, up-left with the drone tilted -> 11/2, on the drone -> hover. 380/400
  objects live. (`templates_L2_*.png` is a misleading per-cell mosaic; ignore
  it.) `templates_L1_*.png`: ~100 of 400 strokes learned (blobs, bars at
  tilts, blob edges); the rest never won, since 98% of every frame is black.
- **The flights fail the cloning way.** Right within a level 70% of the
  time, off by two or more 30% of the time, and the thrust scale is steep
  around hover, so every wrong tick is a kick; a few kicks put the drone in
  states the teacher never visited, the tally reads nothing useful there,
  and it ends pinned to the box edge (final distance 7.6 > any start).
- **The first two-frame run carried no motion**: the difference channel was
  drawn on a mid-grey background and the joint mean-centring of the
  two-channel patch turned it into a constant block (visible in
  `templates_L1_2ch.png`). Fixed: signed difference on a zero background,
  gated on `cfg["diff"] == "signed"` so old checkpoints stay consistent.

## Second pass: what the flights taught

Every arm below is scored on held-out teacher episodes (within 1 level on both
motors, and the error split into collective (l+r)/2 and differential (l-r)/2)
and by one probe flight tick by tick against the teacher (`probe_flight.py`).

```
pupil                                        within-1   |coll|   |diff|   flight
frame + track -> L2 objects -> tally (mean)   0.780     --       --      0.04 success; approach then overshoot
  + DAgger rounds (L2 retrained, hire wipes)  0.642     --       --      0.00; 23k hires/round: churn
  + DAgger, hire free-only, tol 1             0.712     --       --      0.00; offline falls as pupil states join
strokes + track -> TWO tallies (L/R)          0.760     0.74     0.50    0.00; no differential at all
strokes + track -> ONE joint tally (13x13)    0.852     0.56     0.29    first command = teacher's; reaches 0.9 units, then overshoots
  + counting DAgger (6 rounds, 15 s each)     0.803     0.65     0.36    0.00, but final distance 2.4 instead of 7
  12x12 cells                                 0.857     0.57     0.30    same flight, same tick
  track weight x3 / x6 at read               0.844/0.825                worse; at rest the track votes roll the wrong way
```

- **Distribution shift was the first diagnosis and it is not the whole story.**
  DAgger, in three forms, made the offline score worse every round: the
  pupil's states, once counted, blur the rows the teacher's states built.
- **The failure is the brake.** In every probe flight the pupil climbs with the
  teacher and does not cut thrust when the target is close and the drone is
  rising fast (teacher 1/3 or 0/2; pupil 10/10). The braking labels exist in
  the teacher's data. An additive read cannot use them, because "rising fast"
  co-occurs with "target above" in the teacher's flights and its row says
  climb; the conditional rule "target above AND rising fast -> cut" is a
  conjunction, and a sum of per-feature votes has no place for it. Weighting
  the track more just lets its rest-state bias win at tick 0.
- **The joint pair tally was the one real gain** (+9 points offline, the
  differential error halved): which left level goes with which right level
  is itself a conjunction, and one table over the pair holds it.
- **Resolution is not the limit** (12x12 = 6x6). The old dx/dy rig worked
  because it was a dense lookup over the whole state (4096 minicolumns over
  8 channels), i.e. conjunctions everywhere, at foveated resolution.

What remains, still with no object layer: a tally indexed by the motion
context (the image rows read against the table for the current velocity bin),
so the conjunction lives in the index rather than in a sum. Not run.

## Files

| | |
|---|---|
| `box_world.py` | world, egocentric render, PID, episode |
| `collect.py` | flights + frames |
| `vrig.py` | the rig: patches, L1, contrast message, L2, tables, reads |
| `train.py` | train, score offline, fly. `--frames 1|2 --k2 --tag --episodes-test` |
| `vision_pupil.py` | `load_policy` for the viewer |
| `templates.py`, `objects.py` | draw L1 / L2 templates; L2 as mean won frames |
| `coverage.py` | stroke coverage of patches; command spread inside object templates |
| `l1_tally.py`, `l1_joint.py` | no object layer: strokes + track -> two tallies / one joint tally. `--grid`, `--trackw` |
| `l1_dagger.py` | counting DAgger on the tally pupils. `--joint --wide` |
| `dagger.py` | DAgger with the object layer retrained per round |
| `probe_flight.py` | one pure-pupil flight, tick by tick against the teacher |
| `results/pupil_*.npz/.json` | checkpoints per tag; `pupil.npz` = latest |

    uv run python collect.py                          # ~20 s
    uv run python train.py --frames 2 --tag f2signed  # ~8 min with the flights
    uv run python objects.py; uv run python templates.py
    uv run python ../../../viewer.py                  # from the repo root: loads results/pupil.npz
