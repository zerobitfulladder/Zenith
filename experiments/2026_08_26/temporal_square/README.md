# `temporal_square/` — temporal v1: a bouncing square replayed from two trails

Script: [`run_temporal_square.py`](run_temporal_square.py) (also imported by `../square_nonext/`, `../hier_time/`, `../pure_form/`, `../temporal_compose/` and by later days). Results: [`results/`](results/) — [`filmstrip.png`](results/filmstrip.png), [`input.gif`](results/input.gif), [`generated.gif`](results/generated.gif), [`units.png`](results/units.png), `report.md`.

## Temporal v1: bouncing square (run_temporal_square.py)

User's mechanism, minimal test. Two EMA traces per state (fast 0.5,
slow 0.06) = two clock hands: at any interior position the
instantaneous frame is direction-ambiguous (same pixels leftbound and
rightbound), only the trail disambiguates. Learning vector =
[fast ; 0.5*slow ; 0.5*NEXT frame], standard top-1 geodesic learning,
NO labels — the label-concat pattern aimed at time. Playback: prime
traces 60 frames, then query with next-half empty, emit the winner's
stored next-half, feed the emission back into the traces — the user's
auto-advance loop (each timestep's joint state is discriminative, so
the state advances itself).

Predictions (stated before running): free-run continues the loop
including wall reversals; watch-item = blur accumulating as soft
emissions feed back.

### Outcome — PERFECT on first run

100 free-run frames after the prime: mean |x error| 0.00, max 0, all
reversals executed, emitted squares crisp (filmstrip.png; input.gif /
generated.gif). The predicted blur accumulation did NOT occur, and the
reason is structural: the emission is the winner's STORED next-half —
a committed memory, not a re-rendered echo — so retrieval snaps the
state back onto the learned trajectory every step. The loop is
self-cleaning: an attractor chain, where the visual bounce was an
unanchored echo chamber. Auto-advance via two-timescale traces is
DEMONSTRATED: sequence replay with direction disambiguation from one
local learning rule, no labels, no clock, no positional axis.

Honest scope note: easy regime — one deterministic 40-phase loop,
K=64 > phases, so each phase can own a unit. Real tests queued:
multiple label-conditioned sequences, longer periods vs trace
capacity, noise robustness, generalization, and the figure-8 (true
self-intersection) with a fast-only ablation arm.
