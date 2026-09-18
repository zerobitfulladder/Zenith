# `depth/` — does credit survive four levels of hierarchy?

Depth tested on a synthetic hierarchy of cards (design record Part XVII.5). Verbatim from the session narrative of the original project (the rest of
it is in [`2026_09_16/glimpse_loop/README.md`](../../2026_09_16/glimpse_loop/README.md)).
`DESIGN.md` below means the design record, now kept in full at the end of that README.
Words are defined in [`2026_09_16/VOCABULARY.md`](../../2026_09_16/VOCABULARY.md).

`depth.py` needs only numpy.

---

## Depth: the idea that started the day, tested

Lavender's originating claim was that this architecture has no layers, and what plays the part of
depth is **the trajectory of the search** — so prediction error should propagate back along it as a
rotation on the angles. `phase.py` tested the rotation rule on a chain one bind and one bundle
deep. `depth.py` tests depth: a hierarchy of cards, names derived by walking, every leaf observed
but one, predict the missing one.

Two false results were reported along the way, both from Claude's setup rather than the substrate.
The first sweep showed 1.000 → 0.052 across depths 1–4 and read as "depth destroys credit" — but
the leaf count triples per level, so depth was confounded with pile size; and the pose update
carried a `1/len(path)` factor that trained deep paths at a smaller step, which the correct
gradient does not have. With matched leaf counts and the poses updated exactly as the leaf codes
are:

| leaves | flat (depth 1) | deep (3 children per card) |
|---|---|---|
| 3 | 1.000 | 1.000 (depth 1) |
| 9 | 1.000 | 1.000 (depth 2) |
| 27 | 1.000 | 1.000 (depth 3) |
| 81 | 0.963 | **1.000** (depth 4) |

120 leaves, 32 blocks, chance 0.008.

- **Depth is free.** Four levels, 81 leaves, perfect recovery. Nothing multiplies along a chain of
  additions, so there is no vanishing term. The claim holds in its strong form.
- **Composition beats flatness.** At 81 leaves the hierarchy wins (1.000 vs 0.963) while moving its
  codes a fifth as far. A flat card needs 81 distinct poses; a 4-deep card reaches the same leaves
  from 3 poses per level combined along paths. The design assumed hierarchy was *forced* by
  capacity; this says it also pays directly.
- **Learned codes beat the random-code capacity bound.** Random codes give ~0.11 for 32 items at 16
  blocks; here 80 items at 32 blocks recover 0.963–1.000. The `√N` law is a floor for random
  allocation, not a ceiling for a learned representation — Part II's thesis as a capacity result.
- **Re-sparsification is untested, not refuted.** The cleanup arm snapped walked codes to an
  unwalked codebook — the wrong comparison — so its numbers say nothing. Depth 4 works without any
  cleanup, so it is not needed at this scale.

## How to run things

```
python experiments/2026_09_18/depth/depth.py --sweep                        # does credit survive four levels of hierarchy
```
