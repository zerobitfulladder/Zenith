# `shared_nodes/` — AxonForge nodes used by every March experiment

Not an experiment: the helper nodes the March graphs were built from.

- `lgn.py` — preprocessing. `LGN` mean-centres the input; `ConvLGN` does it per
  patch, with `ConvConfig` setting the patch size. Used by almost every graph.
- `color_opponent.py` — preprocessing, colour-opponent channels.
- `reconstruction.py` — `Reconstruction` / `ConvReconstruction`: rebuild the
  input from the templates and their activations, for display.
- `cortex.py` — `Weights`: makes and holds a matrix of unit-length templates.
- `utilities.py` — display helpers (`to_display_grid`, `scale_to_bwr`) and a
  softmax, imported by most of the learning nodes.

The node code only ran inside AxonForge (it imports `axonforge` and a shared `utilities` module), so it is kept as a record and does not run on its own.
