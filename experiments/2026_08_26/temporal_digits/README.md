# `temporal_digits/` — temporal v4: the MNIST digit carousel

Script: [`run_temporal_digits.py`](run_temporal_digits.py) (its `SeqDict` and `cn` are imported by most later scripts of the day). Results: [`results/`](results/) — [`cycles.png`](results/cycles.png), [`nearest.png`](results/nearest.png), [`input.gif`](results/input.gif), [`generated.gif`](results/generated.gif), `report.md`, `run.npz`.

## Temporal v4: digit carousel (run_temporal_digits.py)

User's abstraction test: MNIST 0..9 looping, HOLD=4 frames per digit,
FRESH exemplars every cycle (1,200 cycles, 12,000 digits). Transition
structure stable, pixels never repeat. Next-label part of the
emission; cold start from label 0 alone.

Run 1 (alpha_f=.35): counted 0-3 too fast (2-frame holds) then FROZE
on 4 forever. NEW ARCHITECTURAL BOUNDARY, named: input-driven clocks
stop when the input stops moving — under a HELD frame the EMA state
goes stationary; the only dwell clock is the previous digit's decaying
residue (.65^4=.18, too faint), and "keep holding" is self-reinforcing
(stationary state -> hold unit -> same emission -> stationary).
Biology's fix = adaptation (a response that decays under constant
input = a dwell timer); queued as the principled repair.

Run 2 (alpha_f=.2, residue .41 at hold end): carousel LOOPS through
all ten digits indefinitely (3+ full cycles in 200 frames), rhythm
irregular (holds 1-6 frames; 3/7/9 rushed, 2/5 lingered).

VERDICT on memorize-vs-abstract: NEITHER pole — it invents CANONICAL
DRAWINGS. Emissions are crisp digit-grade drawings (nothing like the
blurry class pixel-means), yet max corr to any training exemplar is
only ~.77-.87 (a near-copy would be .95+; only the 1 hits .956, and
1s are all alike) while corr to class mean is ~.72-.84. Each is a
tight-cluster average — a plausible NEW digit — and cross-cycle corr
= 1.000: the SAME drawing every loop (the user's "same drawing when
it outputs 2" hypothesis, confirmed). Class 0's .43 is a cold-start
transient (cycle-1 zero differs; later cycles stable). Prototype
crispness came free because exemplar context in the fast trail splits
units into small same-style clusters (~6 units/phase): average of a
tight cluster = crisp canonical form, not blur. Abstraction lives at
the transitions; style lives in the clusters.

### Provenance audit (nearest.png) + two-label superposition

Audit method: emission vs nearest of ALL 1200 shown exemplars,
compared against that exemplar's own render self-fidelity (the score
a true copy would get). Verdict: 2/4/5/6/8 clearly below ceiling
(.84-.89 vs .91-.95) — genuinely NEW drawings, visibly unlike their
three nearest relatives; 0 and 1 AT ceiling — indistinguishable from
copies. Pattern: COPY WHERE THE CLASS IS TIGHT, INVENT WHERE IT IS
VARIED — the instance/prototype spectrum measured within one layer,
with intra-class variety as the knob. Emission crispness comes free:
tight-cluster averages are crisp; class-wide means are mush.

(The two-label cue on the movie rig, reported in this same section of the day log, was moved to [`../temporal_movies/README.md`](../temporal_movies/README.md).)
