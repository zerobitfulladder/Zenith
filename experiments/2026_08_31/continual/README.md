# Split-MNIST: learn 0-4, then 5-9. INCONCLUSIVE.

2026-08-31. Experts are hired rather than pre-allocated: competition on FIT
of the joined `[image ; label]` vector, and when nothing explains an input
below `theta` it is buffered until 2K samples accumulate and a new expert is
seeded from their SVD. Existing experts are only updated by inputs they win,
so learning 5-9 should not overwrite 0-4.

## The baselines forget catastrophically. That part is solid.

| | 0-4 after phase 1 | 0-4 after phase 2 | 5-9 | forgot |
|---|---|---|---|---|
| logistic (SGD) | 0.9487 | **0.0094** | 0.8949 | −93.9 pts |
| MLP 256 (SGD) | 0.9765 | **0.0000** | 0.9673 | −97.7 pts |

The MLP is erased, not degraded. Textbook split-MNIST.

(These baselines are deliberately naive — plain sequential SGD, no replay, no
regularisation toward old weights. Real continual-learning methods do far
better. Beating them is not beating the state of the art.)

## The hiring model does not learn well enough for the comparison to mean anything

| theta | experts | 0-4 after phase 1 | 0-4 after phase 2 | 5-9 | forgot |
|---|---|---|---|---|---|
| 0.20 | 200 (capped) | 0.4479 | 0.4287 | 0.0335 | +0.019 |
| 0.25 | 102 → 200 | **0.5211** | 0.5020 | 0.3406 | +0.019 |
| 0.30 | 56 → 120 | 0.4941 | 0.4742 | 0.4174 | +0.020 |
| 0.35 | 27 → 59 | 0.4800 | 0.4589 | 0.3990 | +0.021 |

**Phase-one accuracy tops out at 0.52 against the MLP's 0.98.** A model that
never learned 0-4 cannot demonstrate that it does not forget 0-4, so the
+0.02 columns prove nothing on their own. Low forgetting and low learning
are correlated for trivial reasons.

The one mildly interesting thing: forgetting is **+0.019 to +0.021 at every
theta**, across expert counts from 27 to 200. That consistency is what a
structural property looks like rather than a tuned one. It is weak evidence
and it is not a result.

## Diagnosis: the competition rule was changed and that broke it

`../competitive40` reaches **0.9392 on all ten digits** with 40 experts. The
difference is not hiring, it is what the experts compete on:

* `competitive40` competes on **label confidence** — which expert best
  predicts the TRUE label. That forces class purity directly, and measured
  purity was 1.00.
* this run competes on **fit of the joined vector**, chosen because it gives
  a natural hiring trigger (a new digit's label block is orthogonal to
  everything, so nothing explains it). But an expert seeded by SVD from a
  mixed buffer spans several classes, `wins.argmax` then assigns it one
  label, and it fires for the others too.

So the hiring mechanism is probably fine and the competition rule is the
regression. The fix to try first is to keep **label-confidence competition**
for learning and use **fit** only as the hiring trigger — the two roles do
not have to share a criterion.

## Two bugs found and fixed along the way

1. `theta` was set at 0.70-0.88 when the generalist's error distribution runs
   0.19-0.44, so nothing ever exceeded it and exactly **one** expert was ever
   hired. Calibrate `theta` against the measured error distribution, not
   intuition — it belongs near the 20th-40th percentile.
2. Learning was gated on `theta`, so every poorly-explained sample was
   buffered and **never trained anyone**; once the expert cap was reached
   those samples were discarded entirely. The winner must learn regardless:
   being the best available explanation and being a good one are different.

## Files

| | |
|---|---|
| `continual.py` | baselines, the hiring model, the theta sweep |
| `results/metrics.json` | every number above |
| `results/models.npz` | trained experts per theta |
| `results/01_forgetting.png` | the bars |

    python continual.py     # ~4.5 min

## Status

Parked. The claim "this architecture does not forget" is **untested**, not
supported. Retry with label-confidence competition once the accuracy is in
the same league as the baseline.
