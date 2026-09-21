# 2026-09-21 — networks made of angles (HelloInductor line)

Words are defined in [`../2026_09_16/VOCABULARY.md`](../2026_09_16/VOCABULARY.md). These
scripts use torch.

## `nice_flow/` — an invertible torus network

Every pixel an angle, every layer an exact bijection of the torus. It works exactly
(bijective to float32 precision) and classifies at 96.2%, with every class chord decoding
backwards into a readable prototype digit. Treating pixel intensity as a true circle costs 9
points. A decode bug cost most of the day and produced three negative findings that were not
real.

Full writeup: [`nice_flow/README.md`](nice_flow/README.md).

---

## `catmap/` — is the torus wrap enough nonlinearity to classify?

No. Linear shears with wraps between them reach 97.3%, and the same stack without the wraps,
a single linear map, reaches 97.1% (logistic regression 91.4%). The nonlinearity comes from
the cosine readout. Making the shear read its input through sine and cosine gives 98.3%.

Full writeup: [`catmap/README.md`](catmap/README.md).

---

## `trigmlp/` — ReLU against sine-and-cosine in a plain MLP

No difference on MNIST (0.9769 ReLU vs 0.9768 cos+sin with half the weights). On
Fashion-MNIST ReLU is ahead by 0.3–0.4 points, under one standard deviation, and the trig
arms overfit more.

Full writeup: [`trigmlp/README.md`](trigmlp/README.md).

---

## `trigflow/` — the capped sine/cosine shear, run backwards

Invertible, classifies at 0.979. Run backwards from a class chord plus noise it gives
readable but speckled digits; adding a term that pulls images onto their chord makes every
chord decode to a clean digit and test accuracy goes to 0.981. Moving one latent angle
changes about one pixel, not a style.

Full writeup: [`trigflow/README.md`](trigflow/README.md).

---

## `grok/` — does a network made of angles have anything to grok?

On (a + b) mod 97, a ReLU MLP memorises by step 100 and generalises only near step 9900. A
network built of angles, addition and a cosine cleanup has no plateau: train and test rise
together, with 32 times fewer parameters, and it learns one frequency per angle as Nanda et al.
found. But it is slow (final test 0.992).

Full writeup: [`grok/README.md`](grok/README.md).
