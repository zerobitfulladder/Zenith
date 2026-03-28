# `topk_label/` — top-1 selection and the label as extra input

The step from partial correlations to plain top-1: only the template with the
highest correlation learns. The expected monopoly did not appear, because with
mean-centred input and templates there is no shared direction to camp on. The
same notes cover adding the label as extra input next to the image. There is no
training script or saved result; the write-up is
[`TopKAndLabelConcatenation.md`](TopKAndLabelConcatenation.md).

## Code

- `top_k.py` — `TopK`: keeps only the k largest activations.
- `random_unit_vector.py` — `RandomUnitVector`: a random mean-centred unit
  vector each tick.

The node code only ran inside AxonForge (it imports `axonforge` and a shared `utilities` module), so it is kept as a record and does not run on its own.
