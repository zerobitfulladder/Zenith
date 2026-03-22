# 2026-03-22

## `concat/` — the label as an extra input row

MNIST images trained with the digit's sparse label code stacked under them as
one extra row; at test time the label row is blank and the digit is read from
how the layer fills it in. 100 templates: 0.7711. 500 templates and
180,000 presentations: 0.1548, near chance.

Full writeup: [`concat/README.md`](concat/README.md).
