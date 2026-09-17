# `product_names/` — a card's name as its parts walked together

Product names (design record Part XVI). Verbatim from the session narrative of the original project (the rest of
it is in [`2026_09_16/glimpse_loop/README.md`](../../2026_09_16/glimpse_loop/README.md)).
`DESIGN.md` below means the design record, now kept in full at the end of that README.
Words are defined in [`2026_09_16/VOCABULARY.md`](../../2026_09_16/VOCABULARY.md).

`product.py` imports `sdr.py` from [`../../2026_09_16/glimpse_loop/`](../../2026_09_16/glimpse_loop/).

---

## Later on 2026-09-17: product names, and the torus explained

After the as-if test Lavender asked whether a card's fingerprint could come from the parts
themselves rather than from a random draw: A walked by B walked by C is the same as A walked
by C walked by B, so a name built by walking is order-free. `product.py` tests it in the exact
toy setting:

| test | result |
|---|---|
| same 5 parts integrated in 20 orders | 1 name |
| subtract one part of a 2-part card, clean up the rest against 960 candidates | 100% |
| subtract parts one by one from cards of 3, 5, 8, 12, 20 parts | 100% at every size |
| whole = sub-card ⊗ sub-card; subtracting one yields the other's name directly | 100% |
| move a 6-leaf whole by t | equals walking its name by t six times |
| two wholes sharing 4 of 5 parts, product names | overlap 0.58 of 16 blocks (strangers 0.35) |
| the same two as bundles of parts | 62 of 80 slots shared |

What this buys: names derived from content (two routes to the same whole meet by arithmetic),
subtraction to any depth ("what remains to be seen, and where" in one operation), and a
hierarchy navigable without expanding to leaves. What it costs: product names have no graded
similarity, so recognition from partial evidence must go through the residual, not overlap;
the practical answer is two fingerprints per card, product for identity, bundle for "looks
like". Two facts fell out: a card carries its leaf count (a K-leaf card placed at offset t is
walked by K·t), and because 41 is prime that K-fold walk can be divided out again (3 × 14 =
42 ≡ 1, so walking by 14 undoes tripling); a card with more than 40 leaves cannot be moved
unambiguously at this block size.

Graded codes and products: keep the product exact on canonical stroke ids and let the graded
codes act only in cleanup (accept a look-alike, subtract the expected part). A fully graded
product exists (circular convolution, as in holographic reduced representations) but bump
widths add with every binding and subtraction becomes approximate and shallow.

Explanations given along the way, recorded because they are the vocabulary of the next
steps: a block is a ring of 41 spots and a fingerprint marks one per block; the space is a
torus (16 independent rings); binding is walking, poses add, and walking by a random pose
decorrelates only because the ring wraps; in the complex picture binding is elementwise
phase multiplication, so it is linear after all; "holographic" means information spread over
the whole vector rather than placed (cut a hologram in half and the whole image remains,
dimmer); "reduced" means the bound vector keeps the original size; the pile is a hologram
with several exposures and a pose is the reference beam. Torus properties not yet used:
coarse-to-fine matching (squint on 4 blocks, then 16), continuous positions (a half-cell
shift is a half-step walk; the tally can store the average walked fingerprint per class and
stroke, whose angle is the mean position and whose strength is the inverse variance),
ordered rings (neighbouring spots hold similar strokes, so one notch is a meaningful change),
and Fourier on the ring (a kernel is invertible exactly when none of its wave amplitudes is
zero: single marks always, narrow bumps in principle, wide bumps not).

Lavender's summary of graded binding, which is correct: the pose acts as a circular kernel
that smears the code, and whether that can be undone depends on the kernel alone.

## How to run things

```
python experiments/2026_09_17/product_names/product.py                              # product names: order, subtraction, hierarchy (seconds)
```
