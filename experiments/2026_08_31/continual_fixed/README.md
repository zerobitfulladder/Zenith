# Split-MNIST on the fixed architecture: it does not forget

2026-08-31. `../competitive40` unchanged — 40 experts x 36 templates,
pre-allocated, label-confidence competition, fit gate at test. **No hiring,
no threshold, no conscience.** Phase 1 is digits 0-4, phase 2 is 5-9.

## The result

| | 0-4 after phase 1 | 0-4 after phase 2 | 5-9 | forgot |
|---|---|---|---|---|
| logistic (SGD) | 0.9487 | **0.0094** | 0.8949 | −93.9 pts |
| MLP 256 (SGD) | 0.9765 | **0.0000** | 0.9673 | −97.7 pts |
| **this** | **0.9757** | **0.9526** | 0.9043 | **−2.3 pts** |

Phase-one accuracy matches the MLP (0.9757 vs 0.9765), so unlike the
`../continual` attempt this comparison is valid — the model genuinely
learned 0-4 before being asked whether it kept them.

**And the −2.3 is not forgetting.** Against the right control:

| | 0-4 | 5-9 | all |
|---|---|---|---|
| trained jointly on all ten at once | 0.9558 | 0.9215 | 0.9390 |
| trained sequentially, 0-4 then 5-9 | 0.9526 | 0.9043 | 0.9290 |
| **cost of sequence** | **−0.0031** | −0.0172 | −0.0100 |

A model that saw all ten digits together also scores only 0.9558 on 0-4 —
the drop is the price of ten-way rather than five-way discrimination, paid by
everyone. **Learning them in sequence costs three tenths of a point.**

## The mechanism, measured

    after phase 1:  14 live experts, 26 dead
    phase-2 samples claimed by previously-LIVE experts:     0.0%
    phase-2 samples claimed by previously-DEAD experts:   100.0%
    phase-1 experts whose claimed digit changed:            0 / 14

**Every 5-9 sample went to spare capacity. Not one 0-4 specialist was
touched.**

Why, and this is the part worth keeping: **specialisation protects itself.**
The competition is on confidence in the true label, and a trained "3" expert
has a label block pointing hard at 3 — so its score for label 7 is near zero
or negative. It *actively rejects* 7s. An untrained expert's label block is
merely uncommitted noise, which beats an active rejection. So new classes are
routed to unused experts by the ordinary competition, with no mechanism added
for the purpose.

This is why gradient descent cannot do it. There, every weight is shared by
every class, so learning 7s necessarily moves the weights encoding 3s. Here
only the winner is updated, and a committed expert stops winning things it is
not committed to.

## Three caveats, all real

**The baselines are naive.** Plain sequential SGD, no replay buffer, no
regularisation toward old weights, no rehearsal. Proper continual-learning
methods do far better than 0.0000. This shows the architecture *does not have
the problem*, which is a different claim from beating the state of the art.

**It works because there was slack.** 26 of 40 experts were free when 5-9
arrived. Had phase 1 consumed all 40, phase 2 would have had to poach and
this result would look different. So the protection is real but
capacity-bounded — which is precisely the argument for adaptive hiring: not a
correctness fix, a *scaling* fix.

**Two phases, one dataset.** Nothing here says what happens over ten
sequential tasks, or on data where classes overlap more than digits do.

## Files

| | |
|---|---|
| `split.py` | the two-phase protocol, the mechanism counters, the baselines |
| `results/metrics.json`, `results/state.npz` | numbers and final experts |

    python split.py     # ~50 s
