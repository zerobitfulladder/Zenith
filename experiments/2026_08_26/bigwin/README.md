# `bigwin/` — a bigger L2 window (8x8 over L1) and the generation-mush diagnostic

Scripts: [`run_bigwin.py`](run_bigwin.py) (training), [`run_rerender.py`](run_rerender.py) (sparse-position re-render from the saved weights). Results: [`results/`](results/) — [`generation.png`](results/generation.png), [`generation_sparse_positions.png`](results/generation_sparse_positions.png), template galleries, `report.md`, `weights.npz`.

## Big-window L2 (run_bigwin.py)

One footprint change vs the base: L2 window 3x3 -> 8x8 (stride 2) over
the stride-1 L1 grid — pixel footprint 10x10 -> 15x15, the geometric
midpoint between L1's 8x8 and the 28x28 image (the half-digit scale) —
with K2 1024 -> 2048 and KTOP 200 -> 400 (user: more templates at L2
and L3 as well). B=64 (8x bigger L2 windows vs base; 6GB GPU).

Predictions (before running):
1. L2 templates show multi-stroke structure — curve pairs, half-digits
   — instead of the base's single soft strokes.
2. Probe L2 recovers to >= .966 (a real mid-scale should beat the
   near-L1 re-coding of the base).
3. Generation crisper than the base (L3 assembles from meaningful
   halves); hard readout at least holds ~.91 with KTOP=400 helping
   nearest-memory matching.

Engineering note: the 8x8 L2 windows (65,536 dims) OOM the 6GB GPU
through the shared level_pass at B=64 and even B=32 (skeleton buffers +
the update's full scatter matrix). run_bigwin.py carries its own lean
L2 pass — position-chunked window/skeleton buffers, scatter sums
accumulated in place, geodesic step in row chunks — same per-batch
math, peak ~2GB, B back at 64.

### Outcome

Train 379s. Probe L2 .9572, hard .9132, consistent 10/10.

P1 PARTIAL: L2 templates are visibly larger-scale — hooks, arcs,
curls, S-turns, a few bar-over-curve compounds — clearly beyond the
base's single soft strokes, but still mostly one curved stroke each,
not half-digits. P2 REFUTED: probe flat (.9570 -> .9572); the
mid-scale footprint bought no linear readability. P3 REVERSED on
generation: hard held (.9112 -> .9132, within jitter) but generation
got visibly WORSE — mushy, ghosted digits (the 2 a swirl, the 3
malformed). Likely mechanism: 49 heavily-overlapping 15x15 patches in
the overlap-add — each hardened position paints a big soft patch, and
their disagreement blurs; the base's 100 small 10x10 patches disagree
less per pixel. L3 units also grainier (400 units on the same data =
half the rehearsal each). CONFOUND, noted: three variables moved at
once (footprint, K2, KTOP) — attribution between them is not clean.

Verdict: composition at L2 improved the *parts* but nothing downstream;
the rig's recognition lives at the top and its generation prefers many
small confident patches over few big soft ones. Footprint growth is
not obviously a win on MNIST at this depth.

### Generation-mush diagnostic (run_rerender.py) — first hypothesis
### refuted, mechanism found

Sparse-position re-render (top 49/25/12/6 L2 positions by confidence):
sparsifying does NOT sharpen — top-25 no better, top-12/6 lose coverage
and fragment. The mush is not disagreement-averaging between
overlapping hypotheses; each voice is individually soft.

Peakiness check on the saved weights (winner's share of per-position
mass): base L2 0.215, big-window L2 **0.035** — 6x flatter. A 15x15
footprint wins far more diverse content per template; averaging skeleton
targets over that diversity flattens every per-position profile. Same
disease the peakiness metric was built to catch, driven by footprint
instead of dense targets. Recognition is immune (dense graded matching
averages over the flatness); generation reads the stored profile as a
drawing instruction and paints the flatness directly.

Implication: template DECISIVENESS must scale with footprint. Candidate
levers, untested today: K2 scaling faster than footprint;
margin-weighted plasticity at L2 (the validated update_blend rule);
finer stride-1 L2 tiling. Big parts are viable only if each stays a
committed drawing.
