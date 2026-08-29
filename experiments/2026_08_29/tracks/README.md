# Per-signal tracks and the decomposition probes

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## Per-signal tracks (`td_tracks.py`, `run_tracks.py`) — also 0.00

The user's design: every quantity gets its own two-layer hierarchy —
dx, dy, tilt, collective, differential — and layer three binds the five
tracks' codes and nothing else. The thrusts are tracks like any sensor;
at generation their slots go in empty and the winner's own are read back,
decoded down through the motor track (layer-2 template -> its last
layer-1 code -> that template's last tap). Tracks keep ABSOLUTE values,
since here position is what says which way to tilt.

Chosen on evidence — measured on the nearest-neighbour bound with no
training: tracks 0.0336 against raw channels' 0.0420, and restricting
the collective to the vertical tracks gave 0.2028 against 0.2816. It is
the best representation we measured at the capacity available. It is
also capped: 8x more per-track capacity (48 -> 384) moved the floor from
0.0305 to 0.0303.

**Result: EVAL 0.000 at all twelve checkpoints over 300k ticks of pure
oracle demonstration.** Agreement on the oracle's own path plateaus at
~0.28 once layer three saturates (8192 templates, reached at 75k).

Do not be fooled by `closest_approach` in the log — it appears to
improve 4.05 -> 2.37 around 75k. It does not. The evaluation uses a
fixed seed, and the values 4.05 and 2.37 recur EXACTLY at later
checkpoints, which means the median episode's outcome is unchanged and
the number is only flipping between which episode sits at the median.
A 25-episode median is far too coarse to read a trend from; report a
mean over more episodes, or the distribution.

## Standing summary of the drone work

Every architecture tried scores EVAL 0.00 frozen: single hypercolumn,
2-layer temporal, 3-layer temporal, per-signal tracks; with and without
replay; both motor encodings.

What is established:

* The in-training success rate is fiction (0.79 reported, 0.00 frozen).
* Two real bugs found and fixed — the 99.2/0.5/0.3 energy imbalance, and
  the steering command read as a difference of two noisy rotors. Both
  helped; neither was sufficient.
* The task has ~5 effective dimensions; error falls only as n^(-1/5).
  The wave rig had d=1 and n^(-1). That difference is the whole story.
* The oracle's own differential is ~half quantisation dither (sd 0.139
  of 0.245), so exact-level agreement was never going to look good.
* Read precision tops out at signal-to-noise ~1 and determinability
  ratio ~5, and nothing tried moves it: not layers, not capacity, not
  which signals are encoded, not per-signal tracks.

Hypotheses tested and REJECTED (all mine, all wrong): tilt-axis
resolution, missing gyro/velocity channels, the blind period at episode
start, cosine-vs-uncorrelated scoring, and per-track capacity.

If this is picked up again, the two directions with an actual argument
behind them:

1. **Make the memory a correction, not the controller.** Put a simple
   stabiliser underneath and have the memory supply a residual. A wrong
   residual does not compound the way a wrong absolute command does,
   which is the mechanism killing every version here.
2. **Reduce d, or change task.** Nothing else has paid. The wave rig
   won outright at d=1; this is d=5 and nothing has crossed it.

---

# The decomposition probes — where the tracks rig actually loses it

Three read-only probes against the trained `results/` checkpoint
(scripts in the session scratchpad), splitting the frozen pipeline into
stages on the oracle's own trajectory. The summary above said "read
precision tops out at SNR ~1 and nothing moves it" — that conclusion
was premature. The stack was NOT at the dimensionality wall; two
fixable losses sat in front of it.

Steering (oracle sd 0.062, all corr against the oracle's command):

| stage | corr |
|---|---|
| the motor track's own L1, best template decoded | 0.89 |
| the TRUE L2 winner, argmax-chain decode | 0.43 |
| the true L2 code, graded decode | 0.62 |
| full pipeline as `act()` does it | 0.34 |
| same L3 winner, graded decode within its block | 0.56 |
| graded across 16 L3 winners | 0.49 (collective: worse, 0.63 → 0.51) |

1. **The argmax chain.** Generation decoded through identities —
   argmax of the winner's motor block → that L2 template → argmax of
   its last tap → that L1 template. Every argmax discards graded
   structure; projecting the block down in value space and sharpening
   once is worth 0.34 → 0.56 on steering. Competing stays right at the
   top (averaging 16 L3 winners loses — rival moments must be chosen
   between); averaging is right within the winner (near-tie evidence
   about one value). The read law: choose between answers, average
   within an answer.
2. **The present command was buried inside the track's code.** One tap
   of five at L1, one code of seven at L2 — ~3% of the evidence that
   picks the motor track's L2 template. Even perfect L3 pointing plus
   graded decode caps steering at 0.62 (the 128-identity alphabet
   cannot say the present precisely). And because the code contains
   the answer, the motor tracks could not join the cue at all —
   cueing with the one-tick-lagged motor code lifts collective
   0.63 → 0.82 (a smoothness prior; self-confirming in closed loop).
3. The NN floor for steering was measured at ~0.84 corr. The stack sat
   at 0.34–0.56. The gap was (1) + (2), not d.

`run_tracks.py` writes `results/`; console output in `results/tracks.log`.
