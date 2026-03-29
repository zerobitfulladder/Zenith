# `zenith_node/` — the Zenith unit as an AxonForge node

The unit the other experiments of this day use: Pearson correlation between
input and templates, top-1 (only the winner learns), templates rotated on the
unit sphere. The standalone experiments of the day copy this rule into their own
scripts. It was also carried forward into the August rig
([`../../2026_08_23/rig/gain_feedback.py`](../../2026_08_23/rig/gain_feedback.py)).

## Code

- `learning6.py` — the `Zenith` node.
- `random_sparse_pattern.py` — `RandomSparsePattern`: cycles through a bank of
  random sparse binary vectors, the synthetic input used on this day.

The node code only ran inside AxonForge (it imports `axonforge` and a shared `utilities` module), so it is kept as a record and does not run on its own.
