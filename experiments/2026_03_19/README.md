# 2026-03-19

## `srl/` — signed residual learning, one layer

The contrast-residual rule with the ReLU removed: every template takes part
with its signed similarity, and templates on the far side of the sphere learn
too. Seven online runs (36, 81 and 1024 templates; MNIST and Fashion-MNIST;
30,000 or 90,000 images) that track how dense the code is. No classifier was
run. With 81 templates about 22-27 templates effectively carry each image;
with 1024 templates about 420-500 do.

Full writeup: [`srl/README.md`](srl/README.md).
