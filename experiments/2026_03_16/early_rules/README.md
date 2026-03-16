# `early_rules/` — the first learning rules (WW1–WW5)

The rules tried before IMDV, as AxonForge nodes, with the write-up of each
attempt. No training script or saved results were kept for these; what was
tried and why it was dropped is in the notes.

| Note | What was tried |
|---|---|
| [`ww1.md`](ww1.md) | activation functions only: subtractive, divisive (shunting) and softmax inhibition |
| [`ww2.md`](ww2.md) | rotate the winning template toward the input |
| [`ww3.md`](ww3.md) | learn from the residual, with inhibition in input space |
| [`ww4.md`](ww4.md) | GeodesicPushPull: pull toward the input, push templates apart |
| [`ww5.md`](ww5.md) | functional instead of structural repulsion, softmax push-pull |

WW6 (IMDV) continues in [`../imdv/`](../imdv/).

## Code

- `activations.py` — the WW1 inhibition nodes: `LateralInhibition`,
  `Hyperpolarization` (+V2), `Shunting` (+V2), `SoftmaxActivation`.
- `learning.py` — the WW2–WW5 learning nodes: `GeodesicHebbian`,
  `GeodesicHebbianResidual`, `GeodesicPushPull`, `GeodesicFunctionalPushPull`,
  `GeodesicSoftmaxPushPull`, plus the shared rotation helpers.

The node code only ran inside AxonForge (it imports `axonforge` and a shared `utilities` module), so it is kept as a record and does not run on its own.
