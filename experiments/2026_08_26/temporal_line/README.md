# `temporal_line/` — temporal v2: a rotating line, spatial L1 + temporal L2

Script: [`run_temporal_line.py`](run_temporal_line.py) (imported by `../temporal_movies/` and `../hier_time/`). Results: [`results/`](results/) — [`filmstrip.png`](results/filmstrip.png), [`l1_templates.png`](results/l1_templates.png), [`l2_units.png`](results/l2_units.png), [`input.gif`](results/input.gif), [`generated.gif`](results/generated.gif), `report.md`.

## Temporal v2: rotating line, 2 layers (run_temporal_line.py)

Harder stimulus (user): 16x16 canvas, anti-aliased center line rotating
4 deg/frame (45-frame loop; theta == theta+180). Two layers with a
division of labor: L1 SPATIAL (8x8 s4 windows, K1=64, standard
pixel-window learning, trained first then frozen), L2 TEMPORAL (the
two-trace mechanism over L1's normalized code, [fast ; 0.5*slow ;
0.5*next code], K2=96). Playback advances in CODE space; each
predicted code renders to pixels through L1.

### Outcome

Free-run 120 frames: mean angle error 3.3 deg (< one frame step),
max 13.6 (transient wobble, recovered). Predicted-and-confirmed: L1
invented ORIENTATION-TUNED templates — the gallery is a V1 simple-cell
sheet (oriented segments at all angles), grown from a rotating line
and one local rule. L2 units pair a fast-trail orientation with a
slightly advanced emitted next-code — each unit is a phase of the
rotation. Slight kinks in some generated frames = per-position top-1
disagreement across the 9 render windows (the coordination issue at
miniature scale), not enough to destabilize the loop since traces
update from the committed stored code. The auto-advance mechanism
scales to a 2-layer spatial+temporal split.
