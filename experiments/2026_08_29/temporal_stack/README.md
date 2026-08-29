# Two layers over time — the drone

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

The single-layer rig bound one sensor snapshot to one thrust command,
and its recorded failure was not forgetting: retrieval quality stayed
flat at 0.54 while commanded thrust walked from hover to full power.
A snapshot cannot say which way things are going — "hovering at
dx = +1.2" looks the same whether the drone is converging on the target
or sailing past it — so the memory can only answer with whatever it
stored at the nearest-looking state.

Here the convolution is over **time** instead of over x, and replay
gives the memory the dataset a purely streaming learner never has.

## What goes in

Five signals per tick, and deliberately no more:

| signal | why |
|---|---|
| `dx, dy` | offset to the target. Target-relative, so position and goal are one thing rather than four channels the layer would have to subtract. |
| `tilt` | the attitude. |
| `mL, mR` | the thrusts commanded **one tick ago**. Tells the policy what it just did; lagged so the command being predicted never appears in its own input. |

**Velocity, angular rate and acceleration are not encoded.** They are the
slope and curvature of these five across the window, so a temporal
convolution already carries them — encoding them separately would be
duplicating what the shape says. This is the user's call and it is the
substantive difference from the single-layer rig, which fed vx, vy, ax,
ay and gyro explicitly.

## The stack

    level 0   the tick stream: 5 signals, normalised to [-1, 1].
    layer 1   ONE shared hypercolumn over a 5-tap window spaced 3 ticks
              apart (240 ms), applied at every tick. Each signal is
              written relative to its own mean over the window, so a
              template is a SHAPE of motion — pitching over, thrust
              ramping, drifting right — free of where it happened.
              128 templates; emits a graded top-6 code.
    layer 2   a 7-code window spaced 3 ticks apart (540 ms of history),
              OR'ed with the absolute state now (dx, dy, tilt) and the
              two thrusts to command. 4096 templates.

Inference is the ordinary partial read: write the code history and the
absolute state, leave the thrust cells empty, take the winner, read its
own thrust cells.

Layer one is **frozen** when the warm-up ends. Its vocabulary is set by
then, and freezing keeps the codes in the reservoir meaning what they
meant when they were stored.

Only one layer-one code is computed per tick — the layer-two window
needs seven, but six were computed on earlier ticks and are kept in a
ring. That is the difference between 11 ms and 1.7 ms per tick.

## Replay

A competitive memory has no dataset; it tracks the recent stream, and
the pupil's stream is mostly emergencies. The reservoir keeps a uniform
sample of **every moment ever lived** (reservoir sampling: fill `cap`
slots, then for moment k overwrite a random slot with probability
`cap/k`) and re-presents two per tick alongside the live one. A plain
ring buffer of recent ticks would not do — it would be just as dominated
by the latest emergencies as the memory already is.

Note what this does and does not fix: it removes *recency* bias, not
*distribution* bias. If most moments ever lived were emergencies, a
uniform reservoir is mostly emergencies too. The stronger version bins
moments and keeps the buffer balanced across bins; that is the next
thing to try if the drift statistics say the uniform one is not enough.

## The oracle

`fast_oracle.py`, not the original PD. Its velocity gain is 1.0 instead
of an implicit 0.5, and collective is divided by `cos(tilt)`: 85 ticks
to reach the target instead of 160, 172 total instead of 212, success
1.00 either way. A faster teacher means faster demonstrations to imitate.

## What is measured

Every block reports the two numbers that told the single-layer failure
apart from a memory failure:

- **retrieval** — the winning match score. It stayed flat while the old
  rig decayed, which is how we know the memory was still finding the
  right neighbourhood.
- **thrust_pupil vs thrust_oracle** and **spread_pupil vs
  spread_oracle** — commanded collective and |L−R| against the oracle's
  on the same ticks. This is what actually drifted: 6.5 → 8.0 collective
  and 0.54 → 1.7 spread as success went 1.00 → 0.00.

## Files

    ../temporal_drone/td_stack.py    signals, both layers, the reservoir, the policy
    ../temporal_drone/fast_oracle.py the retuned autopilot (runnable: prints its benchmark)
    run_td.py      the training run

    .venv/bin/python experiments/2026_08_29/temporal_stack/run_td.py

Env: `TD_TICKS TD_WARM TD_REPLAY TD_K1 TD_K2 TD_REPORT TD_ORACLE TD_TAG`.

To watch it fly, use the shared viewer at the repo root — this rig does
not have its own:

    VIEW_CKPT=experiments/2026_08_29/temporal_stack/results/two_layer/weights_best.npz \
      .venv/bin/python viewer.py

then pick **pupil** in the dropdown. The checkpoint's sidecar json says
`{"kind": "module", "module": "td_stack", ...}`, and the viewer imports
`td_stack.load_policy` to bring it to life.

---

# Results, 2026-08-29 — it does not fly, and why

**Nothing here flies yet.** Frozen evaluation is 0.00 for every
configuration tried: 2-layer, 3-layer, with and without replay, with the
original motor encoding and with the corrected one, on up to 60 000
ticks of pure oracle demonstration.

## First: the training metric was a lie

The success rate printed during training is measured on a trajectory the
network is being trained on at every tick. Ticks are 20 ms apart, so
consecutive states are near-duplicates — the template written at tick t
is still the best match at tick t+1 and hands back the oracle's command
from a tick earlier. The memory becomes a one-tick-delayed oracle
repeater. It looks like flying.

| | training says | frozen, fresh episodes |
|---|---|---|
| 3-layer @ 40k | 0.79 | **0.00** |
| 3-layer @ 50k | 0.61 | **0.00** |

The evaluation harness was validated by running the oracle through it:
1.00. `run_td3.py` now runs a real frozen evaluation (`EVAL`) at every
report and `weights_best` tracks that, not the leaky number.

**The single-layer rig has the same learn-before-act ordering.** Its
recorded 0.80 has not been re-checked frozen. It should be.

## Two fixes that were right but not sufficient

**Group energy.** Layer two's input vector was 99.2 % layer-1 code
history, 0.5 % absolute state, 0.3 % motor command — codes carry raw
match scores (up to ~27), bumps are capped at 1.0. With the command at
0.3 % of the vector, the winner during learning is chosen almost without
regard to what was commanded, so one template absorbs moments wanting
opposite tilts and rotates to their average. Now normalised to
50/25/25.

**Motor encoding.** The steering command was never read — it was the
difference of two independently read rotors, so their errors added, and
it spanned ~2.9 cells of a 48-cell axis. Now encoded as collective +
differential on foveated axes (0.007 N per cell near zero, against 0.167
linear). Exact-level agreement 0.381 -> 0.455; differential correlation
0.333 -> 0.410. Signal-to-noise stayed at ~1.

## The actual wall: dimensionality

A prototype memory answers with its nearest stored example, so its error
is how much the answer changes over the distance to that neighbour —
which shrinks as **n^(-1/d)**, with d the number of independently varying
quantities. Measured on oracle flight (neighbours drawn from *other*
episodes, and against the continuous command rather than the snapped one):

| stored moments | 400 | 800 | 1600 | 3200 | 6000 |
|---|---|---|---|---|---|
| nn error | 0.0761 | 0.0674 | 0.0588 | 0.0511 | 0.0503 |
| improvement | — | 1.13x | 1.15x | 1.15x | 1.02x |

**1.15x per doubling = 2^(1/5), so d is about 5.** Halving the error
needs 32x more templates; 10x better needs ~100 000x. With 4096
templates that is ~5 distinguishable levels per axis.

The curve experiment had **d = 1**: 256 templates gave 256 levels along
x and error fell as n^(-1), 2x per doubling. Same machinery, same
learning rule — the difference in outcome is entirely dimensionality.

Encoding choices do not touch this. Measured directly: adding gyro, or
velocity, or the whole single-layer state set, or removing the layer-1/2
codes entirely, all give the same bound (ratio 2.17-2.22).

**Consequence: adding layers or capacity cannot fix this. Reducing d
is the only lever that pays exponentially** — which is what the relative
encoding did in the curve rig, collapsing "this shape at any height"
into one template.

## One more thing worth knowing

The oracle's commanded differential has sd 0.148, but after snapping to
the 13 thrust levels it has sd 0.245 — **quantisation dither of sd
0.139**. The steering signal being imitated is roughly half snapping
noise. An averaging memory recovers the smooth part and discards the
dither, which is defensible, but it means exact-level agreement was
never going to look good.

## Untried, in the order I would try them

1. **Factor the memory by what it controls.** Vertical depends on
   (dy, vy, tilt); steering on (dx, vx, tilt, gyro). Two memories at
   d~3 and d~4 give ~16 and ~8 levels per axis instead of ~5. Fits the
   measurement: collective is already well determined (11:1), only the
   differential is starved (2.2:1).
2. **Mirror symmetry** — the law is odd in x, so encoding |dx| with a
   sign halves the space for free.
3. Only then distribution shift, replay, and layer count.

## Tools left behind

    probe.py    frozen read precision on the oracle's own trajectory —
                the number to watch long before episode success moves

## Where the results are

`results/two_layer/` is `run_td.py` (console: `results/run.log`);
`results/three_layer/` is `run_td3.py` (`results/run3.log`); the
`three_layer_clone` and `three_layer_clone2` runs are `results/clone.log` and
`results/clone2.log`; `*_smoke/` are short smoke tests.
`td3_stack.py` (the three-layer stack) is in `../temporal_drone/`.
