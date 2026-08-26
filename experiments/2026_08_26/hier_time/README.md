# `hier_time/` — hierarchical time: the slow hand comes from the layer above

Scripts: [`run_square_hier.py`](run_square_hier.py), [`run_movies_hier.py`](run_movies_hier.py), [`run_digits_hier.py`](run_digits_hier.py). Results: [`results/square/`](results/square/) ([`generated_hier.gif`](results/square/generated_hier.gif)), [`results/movies/`](results/movies/) ([`filmstrip.png`](results/movies/filmstrip.png), `input_*.gif`, `gen_*.gif`), [`results/digits/`](results/digits/) ([`cycles.png`](results/digits/cycles.png), `input.gif`, `generated.gif`, `weights.npz`). The split-vocabulary carousel mentioned below lives in [`../digits_bridge/`](../digits_bridge/).

### Hierarchical time (run_square_hier.py) — the user's ORIGINAL design,
### finally built, VALIDATED

The user's founding conception, never previously implemented: L1's
query = [own leaky-integrated input ; THE LAYER ABOVE'S DOWNWARD
PROJECTION]; the layer above samples MORE and outputs LESS (temporal
stride k=5) so the timescale separation is STRUCTURAL — a stack gives
clocks at dt, k*dt, k^2*dt for free (log-spaced bank by architecture,
not by tuned gammas). Ladder: down-channel born of empty-window noise
(bootstrap bug, the zero-symbol law again) = 8.34; fixed = 1.07/max 3
— the hierarchy works as the slow hand; residual traced to L2's
projection describing the era just ENDED -> ARROWS ARE RECURSIVE:
each layer needs its own arrow at its own timescale; L2 given an
era-level next-slot (associate this window with the COMING window,
project the prediction down) = **0.95/max 1** — replay within one
pixel, constant one-frame offset (cosmetic).

Verdict: the two-timescale disambiguation can span LAYERS — fast hand
local, slow hand descending from above — exactly as the user
originally specified; the local-slow-trail rigs were a legitimate
flattening, and the hierarchical form now stands as the scalable one
(deeper stacks = wider time horizons at constant per-layer cost).
Note the trail-only ablation's law is untouched: even here, every
layer carries its arrow; what the hierarchy replaces is the slow
POSITION hand, not the arrow.

### Movies retest under hierarchical time (run_movies_hier.py)

Full v3 regime — three labeled movies, shared spatial vocabulary,
cold start from label alone — rebuilt hierarchically (temporal L1
query = [leaky-summed code ; label ; L2 down-projection ; next-slot];
L2 = 5-tick temporal stride with era-arrows, predicted-next-era
projected down). All artifacts regenerated (input_*.gif, gen_*.gif,
filmstrip.png).

Result: purity 1.00/1.00/1.00 (= flat), advance tb 1.00, lr 1.00
(= flat); rot 0.39 vs flat 0.72 — the rotation visibly advances in
the filmstrip (diagonal sweeping through horizontal, softer near era
boundaries) but at roughly half the strict-metric rate: the
finest-grained movie pays an era-boundary stickiness cost (the down
channel is constant within an era, slightly flattening within-era
discrimination). The hierarchy's scalability is bought at a small
fine-timescale tax; levers noted (smaller KSTRIDE for fine content,
gain rebalance) but not pursued. Cold-start label selection fully
preserved under the hierarchical form.

### Label-free hierarchical carousel (run_digits_hier.py) — the label's
### job, measured

User's claim: the original carousel's label channel + next-label cargo
hand-fed sequence-position context that a slower layer should supply
structurally. Rig: NO labels anywhere; L1 = [leaky code (gamma .8,
dwell clock) ; L2 down ; next-code arrow]; L2 = one era per digit
exposure (KSTRIDE=HOLD), era-arrows, predicted-next-era down. External
judge reads emissions for measurement only. Also: split-vocabulary
pixel-loop carousel (run_digits_bridge) completed ONE full clean 0-9
lap through re-perceiving its own handwriting across disjoint eye/hand
vocabularies, then froze on the 4 (the honest loop works; its dwell
clock needs the adaptation fix).

Three-run ladder, each step buying more cycle:
- gamma .5 (dead dwell clock): total freeze from frame one (the
  held-input disease, third appearance — trail decay must match dwell
  time; portability tax argues for the queued adaptation channel).
- gamma .8, weak era hand (K2T=32, GD=.5): stable 5-7-9 orbit —
  self-sustained rhythmic sequencing, coarse eras shortcut the loop.
- strong era hand (K2T=64, GD=1.0): 1-2-3-4-5-7 in order, correct
  dwells, CRISP canonical emissions — then stalls at the 7-era.

VERDICT: the mechanism works in kind — label-free multi-digit
sequencing with canonical handwriting, position from the two-layer
clock alone — and era-context fidelity is the quantified limiting
resource; each strengthening extends the run. The label had been
worth roughly "the last four digits of reliability." Levers queued:
era memory depth, slow-leaky era trails (vs window resets), a third
layer (eras of eras), the adaptation channel for stalls.
