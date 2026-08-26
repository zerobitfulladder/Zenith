# `base/` — the 3-layer base rig, widened to 1024 templates

Script: [`run_base.py`](run_base.py). Results: [`results/`](results/) — [`generation.png`](results/generation.png), [`templates_L1.png`](results/templates_L1.png), [`templates_L2.png`](results/templates_L2.png), [`templates_L3.png`](results/templates_L3.png), `report.md`, `weights.npz` (loaded by `../precision/` and `../bounce/`).

## Base rig (run_base.py)

Today's base: the best 3-layer network, standalone and widened.

- L1 8x8 stride 1, K1=1024 — dense pixel-window learning
- L2 3x3 stride 2 over L1's code, K2=1024 — skeleton learning targets
  (standard)
- L3 top over [whole L2 code ; 0.5*onehot label], KTOP=200 — dense targets
- Dense relu communication everywhere; top-1 (argmax) plasticity in every
  bank; generation = label-only query -> winning L3 unit -> hardened top-1
  reverse reconstruction to pixels.

Rebuilt from run_wide3level.py's dense arm (validated at K=512: probe
.9660, hard .8972); widening 512 -> 1024 at both banks is the one new
variable. Deliverables: generation.png (label-in, image-out) and
per-layer template galleries (templates_L1/L2/L3.png, most-rehearsed
first).

### Outcome

Train 74s. Probe L2 .9570, hard readout **.9112** (ties the project
record at 2 epochs; +1.4pp over the 512-wide reference), consistent
10/10. Generation: all ten digits recognizable (6 weakest). Galleries:
L1 = rich pen-like patches; L2 = single soft strokes, barely more
structure than L1; L3 = varied whole-digit instances.

Reading from the galleries: L2's pixel footprint is only 10x10 (3x3
window over the heavily-overlapping stride-1 L1 grid) vs L1's 8x8 —
there is almost no compositional step between L1 and L2; the whole
scale jump happens at L3. Widening the palette raised matching (hard)
but not composition (probe dipped .9660 -> .9570). Template count =
vocabulary; footprint geometry = composition.
