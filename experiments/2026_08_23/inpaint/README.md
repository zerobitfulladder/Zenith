# `inpaint/` — inpainting by grafting a retrieved memory into the masked region

(2026-08-23, from the day log.) Script: `run_inpaint.py` (loads `../gpu_minibatch/results/stride1/weights.npz`, no training).
Results: `results/inpaint.png`, `results/inpaint_nolabel.png` (`GF_NOLABEL=1`).

## Inpainting via constellation grafting (2026-08-23, zero training)

`run_inpaint.py` on the saved stride-1 MNIST flagship: mask a region,
encode the remainder (masked windows silent via the contrast floor),
retrieve with [partial code ; label], graft the winner-memory's stored
entries into the masked grid positions, render the merged constellation.

- Retrieval from partial evidence + label: 8/8 correct, style-matched in
  places (the open-4). Associative completion confirmed through the
  deep rig.
- Grafting initially failed silent — unit-scale mismatch (memory rows
  are normalized; entries ~2% of encoded-correlation scale, squelched at
  render). Fixed by rescaling the graft to the visible code's mean
  active value. LESSON: stored codes and live codes differ in scale;
  any cross-source constellation surgery must volume-match first.
- Result: 6/8 clean fills, 2 partial (seam doubling where prototype
  strokes disagree with instance strokes at the boundary). The
  "zero-mask means empty, not unknown" ambiguity did not impair
  retrieval — the label carried it.
- LABEL-FREE variant (inpaint_nolabel.png): 6/8 correct retrieval from
  the fragment alone — including a 5 from four disconnected corners.
  The two misses are principled ambiguities (bottom-hook of a 1 read as
  7; chevron-top of a 4 read as 5), each completed as a coherent wrong
  digit. Measures the label's worth (8/8 -> 6/8) and produces the
  searchlight's exact target cases in the wild — with a refinement:
  attention only helps where hypothesis disagreement overlaps VISIBLE
  evidence (disagreement inside the mask is unobservable; correct
  behavior there is uncertainty).
