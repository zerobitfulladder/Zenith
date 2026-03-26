# `cshl/` — Contrast-Structure Hypothesis Learning

Each template is a hypothesis about the whole input; templates in a hypercolumn
compete, and competition was computed first by iterative inhibition and then
directly as partial correlations. There is no training script or saved result
for this line: it lived only as an AxonForge graph. What came out of it is
summed up at the start of
[`../../2026_03_28/topk_label/TopKAndLabelConcatenation.md`](../../2026_03_28/topk_label/TopKAndLabelConcatenation.md):
the partial correlations diversified the templates but made reconstruction poor,
which led to top-1.

## Notes

- [`ContrastStructureHypothesisLearning.md`](ContrastStructureHypothesisLearning.md) — the idea, the design decisions and the open questions
- [`InhibitionAndLearningDecoupling.md`](InhibitionAndLearningDecoupling.md) — brainstorm from 2026-03-25
- [`partial_correlations.md`](partial_correlations.md) — from iterative inhibition to direct partial correlations
- [`process_diagram.svg`](process_diagram.svg) — diagram of the process

## Code

- `learning5.py` — the `CSHL` and `ConvCSHL` (shared weights over patches) nodes.

The node code only ran inside AxonForge (it imports `axonforge` and a shared `utilities` module), so it is kept as a record and does not run on its own.
