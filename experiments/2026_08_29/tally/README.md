# The cetele on the balance rig

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## The cetele (user's design): proposer/chooser split, run_td_tally.py

The user's realisation, re-grounded in the record: in the winning
Aug-28 cascade the LAYER was never the chooser — rows recognised the
situation, a count matrix beside them (`ty[row, action-unit]`, built
by adding the executed action's code to the winning row's line) held
the action menu, and the MODE of that line chose. Cortex proposes,
the tally disposes — and the tally entries are exactly where dopamine
acts in biology, though RL never touched them in the record.

On the balance rig (`results/`): beside the frozen 0.72 top
layer, per template j: C[j, 169] counts over executed (left, right)
level pairs — jointly, so a chosen action is always a pair that was
actually flown — plus W[j, 169] advantage values and V[j] as the
state baseline (the scalar critic's right job; as a chooser it was
refuted twice — its per-template conditioning cannot express action
advantage, and no per-tick baseline can reorder a candidate set).

Predictions, before the run:
1. Gate N: the pure mode read over the notched cetele reproduces
   ~0.72 — the record must carry the policy before values may tilt.
2. Phase V (teacher OFF, epsilon exploration among supported actions
   plus +-1 neighbour jitter, TD error credited to the EXECUTED
   entry): some beta > 0 beats beta = 0 in the final sweep — the
   first improvement earned from the drone's own experience alone.
   Honestly ~even odds after the day's two refutations, but the
   conditioning is per-(situation, action) this time, which is what
   both refutations demanded.
3. Watch: fraction of templates with support; unsupported-tick
   fallbacks during eval; |W| scale on supported entries.

### Gate N, first attempt: MODE READ REFUTED for regulation — 0.02

Full support (100% of templates notched, 15 fallback ticks) and yet
the mode read scores 0.02 vs the 0.72 baseline. The finding: within a
template's watch the action distribution is unimodal-but-skewed
(mostly hover, a minority of corrections) and the DUTY CYCLE is the
control signal — the teacher regulates by flickering between adjacent
levels, and the mode collapses that to the majority action: hover
everywhere, fall. The mode-vs-mean law is now properly bounded: MODE
at genuine decision boundaries (bimodal rivals — the pole), MEAN for
regulation (unimodal-skewed — the skew is the law). The cetele read
is now the EXPECTED action under value-tilted duty weights
(w = C * exp(beta*W), count-weighted mean in (coll, diff), snapped
once); beta = 0 is the pure count mean and must reproduce ~0.72.
Rerun in flight.

`run_td_tally.py` writes `results/tally.npz` and `results/metrics.json`;
console output `results/tally.log`, smoke test `results/smoke/`.
