# `temporal_movies/` — temporal v3: three labeled movies in one network

Script: [`run_temporal_movies.py`](run_temporal_movies.py) (imported by `../hier_time/`). Results: [`results/`](results/) — [`filmstrip.png`](results/filmstrip.png), `input_*.gif`, `gen_*.gif`, the two-label runs `gen_pair_*.gif`, `report.md`.

## Temporal v3: three labeled movies (run_temporal_movies.py)

Movies on one 16x16 canvas: rot (rotating line, period 45), tb
(horizontal line scanning down, period 16), lr (vertical line scanning
right, period 16). Shared frames across movies (rot@90 ~ lr mid,
rot@0 ~ tb mid). Shared spatial L1; temporal L2 joint =
[fast ; 0.5*slow ; 0.5*onehot(3) ; 0.5*next code], K2=160. Playback =
COLD START from the label alone (zero traces, no priming).

Two failures on the way, both instructive:
1. Stride-4 code too translation-tolerant for 1px/frame scans —
   adjacent phases near-identical, crowding. Fix: L1 stride 2.
2. BOOTSTRAP STARVATION (the decisive one, caught by the
   distinct-units diagnostic): sequential per-movie episodes let the
   first movie's frames adopt ALL K2 units at bootstrap; tb and lr
   then collapsed onto 1 unit each — the whole movie averaged into a
   superposition that was its own fixed point (frozen playback,
   purity still 1.00). METHODOLOGY LESSON, general: with
   adopt-until-full bootstrap, curriculum order IS allocation; feed
   interleaved streams whenever multiple sources must share a bank.

### Outcome (after interleaving)

Cold start from label alone: purity 1.00 on all three; advance
tb/lr 1.00 (35/36 distinct units), rot 0.72 by the strict
nearest-frame metric (53 units; visually clean rotation — render
kinks jitter the matcher). Three movies, one network, label-selected
replay with no priming: the label IS the cue. Files in
`results/`.

### Two-label superposition (moved here from the digit-carousel section of the day log)

Two-label cue on the movie rig (retrained, single-label results
reproduced): TOTAL LOCK — every pair plays one movie 100/0 with zero
switches. No blends (WTA cannot superpose) and no crossing-wormholes:
the trail disambiguates shared frames BY DESIGN, so the mechanism's
success is what closes the wormholes. Tie-breaks follow within-tick
interleave order (rot > tb > lr): allocation order surfaces at ties
even when interleaved. Composition would need sampling at the top,
weaker trail gain, or composite training stimuli.
