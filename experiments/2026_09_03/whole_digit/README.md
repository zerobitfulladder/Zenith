# Whole-digit receptive field: what "drift is not damage" actually depended on

2026-09-03. The boundary-condition test for
[`2026_09_01/stack/forget.py`](../../2026_09_01/stack/forget.py).

That run found the champion rig's representation forgot nothing across
split-MNIST: a full recount after sequential training gave 0.9690 against
joint's 0.9683, even though **54% of old-data winners had changed hands**. The
reading recorded at the time was that 5x5 stroke features are class-agnostic, so
phase B's rewrite was *lateral* — old digits stayed describable in the new
vocabulary, and the only casualty was the index-to-class table going stale.

The obvious objection: that may be a statement about tiny generic features
rather than about representations. So this removes one thing — the small
receptive field — and changes nothing else.

    patch rig    400 templates over 5x5 px, 576 positions, T[t, 6x6 cell, class]
    this         400 templates over 28x28 px, 1 position,  T[t, class]

Same competition rule `1-(V·W)²`, same table bias at beta 1.5, same winner-only
learning with the eta floor at 0.02, same log-ratio table, same five final
tables, same diagnostics, same 12k/3k split, same 3 seeds.

**The prediction was that whole-digit templates, being class-specific, would
make churn destructive.** That turned out to be right, and to have a condition
attached that the original run could not see.

---

## The grid

Everything below is 10-class test accuracy, mean of seeds 7/8/9.
`damage = joint − oracle` (what sequential training did to the *representation*,
measured after a full recount, so bookkeeping is excluded).
`staleness = oracle − seq` (what the un-recounted table costs on top).

```
receptive field   beta   churn   drift    joint   oracle      seq   damage  staleness
5x5 patch          1.5   0.542   0.026   0.9683   0.9690   0.8482  -0.0007    +0.1208
5x5 patch          0.0   0.107   0.004   0.9736   0.9742   0.9742  -0.0007    +0.0000
28x28 whole        1.5   0.211   0.053   0.9040   0.8982   0.8920  +0.0058    +0.0062
28x28 whole        0.0   0.582   0.189   0.8944   0.7492   0.7024  +0.1452    +0.0468
```

Phase-A accuracy on 0-4 before phase B: patch 0.9843, whole 0.9832 — both rigs
genuinely learned the old classes, which is the first way this comparison could
have been void.

The second is free capacity. It is not uniform across the grid, and the detail
matters:

```
                 dead after A   share of phase-B wins landing on...
                                 dead   light-used   heavily-used
5x5 patch  1.5        0/400      0.00      0.31          0.69
5x5 patch  0.0       65/400      0.00      0.09          0.91
28x28 whole 1.5       0/400      0.00      0.49          0.51
28x28 whole 0.0     274/400      0.06      0.42          0.52
```

At beta 1.5 nothing is dead and phase B has no choice but to poach. At beta 0
there is slack — 274 free templates in the whole-digit cell — and **phase B
poaches anyway**, sending 94% of its wins into templates already committed to
0-4. Nothing was forced by capacity pressure. A trained 0-4 prototype is simply
geometrically closer to a 5-9 digit than an untrained one is, and with no class
pressure there is nothing to say "leave that one alone".

So poaching happens in every cell. What differs is what it costs.

## 1. The original claim survives, with the receptive field as its condition

On 5x5 patches, drift is free at both settings — 54% churn and 11% churn both
cost **−0.0007**. On whole digits it is not: at beta 0, 58% churn costs
**14.5 points**.

So "a rewritten vocabulary is a renaming, not a loss" is a property of *small
generic features*, not of counted tables or of winner-take-all learning. A 5x5
stroke that stops being template 37 and becomes template 112 still describes the
same ink. A whole-digit prototype that was a 3 and is dragged to an 8 does not
describe a 3 any more, and no recount brings it back.

## 2. There are TWO protections, and the original run confounded them

The whole-digit rig at beta 1.5 does not forget either (+0.0058) — but for a
different reason than the patch rig. It is not that drift is harmless; it is
that **drift barely happens**:

```
                       churn   drift of heavy-in-A templates
whole, beta 1.5        0.211   0.053
whole, beta 0.0        0.582   0.189
```

The class pressure makes templates class-committed, and a committed "3" is no
longer the argmin for any 5-9 image, so phase B cannot drag it. That is the
`continual_fixed` mechanism — *specialisation protects itself* — reproduced on
the champion rig as a single knob, with no dead experts involved.

The two are independent, and separable only because whole digits break one of
them:

| | patches | whole digits |
|---|---|---|
| genericity protection | works (damage −0.0007 at churn 0.542) | absent |
| pressure protection | unnecessary, and *raises* churn 0.107 → 0.542 | rescues 14.5 pts |

On patches the pressure does nothing for forgetting and actively costs accuracy
(0.9736 → 0.9683), because it re-assigns patches by class belief instead of
geometry. Two mechanisms that look like one result whenever both are present.

## 3. Correction: the 12-point staleness was caused by beta, not by counting

This is the part that revises 2026-09-01. That run's headline included a large
table-staleness cost and the prescription *freeze the vocabulary after task A,
stream only the counts, because a moving vocabulary costs 26 points* (on old
classes; +0.1208 on all-class accuracy, the column above).

**Switch the pressure off and the staleness is +0.0000 — to four decimal
places.** The stale table is exactly as good as the full recount.

Because the pressure feeds the table's beliefs back into which template wins,
phase B's table — whose 0-4 rows are stale — actively reorganises the winner
assignment, and that is what invalidates phase A's counts. Without the feedback
loop, assignment is purely geometric, drift moves winners between templates that
mean the same thing, and old counts stay valid. The staleness was a cost of the
pressure, not a cost of counted tables.

## 4. "Freeze the vocabulary and stream counts" does not port

Accuracy of the frozen phase-A vocabulary on the classes it never saw:

```
5x5 patch    beta 1.5   0.9634        28x28 whole   beta 1.5   0.5019
5x5 patch    beta 0.0   0.9704        28x28 whole   beta 0.0   0.4185
```

400 whole-digit prototypes fitted to 0-4 cannot describe 5-9 at all, and the
recipe collapses from joint−0.004 to joint−0.25. It is not a capacity limit —
the same 400 templates reach 0.9040 when trained jointly on all ten. It is
transfer. The recipe is only available where features are generic enough to be
reused by classes that were never seen, which is again the receptive field.

## 5. Redundancy is real, but it is a better readout — not a protection

The templates after phase B hold many examples of each digit, so converting some
0-slots into 6-slots should leave other 0-slots standing. If so, a top-1 read
throws that away: when the single nearest template happens to be a converted
one, the image is lost. `topk.py` tests it by changing ONLY the read — learning
stays winner-only, the tally stays top-1, and the score becomes the sum of the
table rows of the k nearest templates.

It helps, and the optimum is small:

```
                 k=1      best k              gain
beta 1.5      0.8957   0.9137 at k=3        +0.0180
beta 0.0      0.7263   0.7737 at k=5        +0.0474
```

Past k≈5 it collapses (beta 1.5 falls to 0.4630 at k=100) — with 400 templates
over 10 classes, summing 50 of them averages over an eighth of the vocabulary.

**But it does not reduce the forgetting.** Old-class accuracy lost across phase
B, recount tally:

```
beta 0.0   k=1    0.8106 -> 0.7119    -9.9 pts
beta 0.0   k=10   0.8637 -> 0.7690    -9.5 pts
```

Top-k lifts the whole curve by about five points and phase B then removes the
same amount from the higher starting position. The redundancy it collects was
already there before phase B; it is not surviving-template coverage standing in
for converted ones. Reading more templates is worth doing, and it is not an
answer to forgetting. The gap to joint (0.8944) stays at ~12 points.

**On the tally a live system actually holds, it backfires.** With the online
accumulated counts rather than a recount, top-10 costs 28 points on the new
classes at beta 1.5 (0.7426 -> 0.4613), and deepens the old-class loss at beta 0
from 26.5 to 41.0 points. The online table carries phase-A mass on templates
that have since moved, so summing more rows sums more stale mass. Top-k and an
un-recounted table are actively bad together.

Not tested: a similarity THRESHOLD instead of a fixed k, so the number of
templates read varies per image. That adapts where fixed k cannot, and is the
obvious follow-up.

## Caveats

**Whole digits are a weak model.** Joint accuracy 0.9040 against the patch rig's
0.9683 — 400-prototype vector quantisation, essentially. It learned 0-4 to
0.9832 so the forgetting comparison is valid, but nothing here says the
whole-digit rig is worth using.

**Two forced deviations,** both from there being one position per image instead
of 576, both recorded in the output json. `MIN_S` 4 → 1: a template must win
MIN_S items in a batch to learn, and where the patch rig's batch gave each
template ~106 wins this gives under one, so the threshold would have frozen
learning outright. Batch 128 → 512 and epochs 3 → 40, because the patch rig got
576 winner-updates per image per epoch and this gets 1; `saturation.py` and the
logged phase-A curve show the plateau.

**The two betas do not have equal effective capacity.** Phase A leaves 0/400
dead at beta 1.5 and 274/400 at beta 0, so the pressure is also spreading data
over more templates. Damage is measured against each condition's own joint
baseline so the comparison stays internally valid, but "beta protects the
representation" and "beta uses more of the vocabulary" are not separated here.
Equalising live-template count across betas is the control this run lacks.

**One dataset, two phases, three seeds.** Nothing here says what happens over ten
sequential tasks, on data where classes overlap more than digits do, or at
receptive fields between 5 and 28 pixels — and that intermediate sweep is the
obvious next run, because it should show damage rising continuously with
receptive field and would turn a 2x2 into a curve.

**The beta-0 patch cell has low churn (0.107)**, so it does not independently
show that high churn is safe on patches. The cell that does is patch/beta 1.5:
churn 0.542, damage −0.0007.

## Files

| | |
|---|---|
| `whole.py` | the rig and the split protocol. `--beta 0`, `--epochs N`, `--k N`, `--smoke` |
| `saturation.py` | K sweep: where phase A stops leaving free capacity |
| `patch_beta0.py` | the fourth cell — 09-01's `forget.py` unmodified with `G.BETA = 0` |
| `results/grid.json` | the 2x2 above |
| `results/whole_digit_mnist{,_beta0}.json` `.log` | per-seed numbers and diagnostics |
| `results/patch_rig_beta0.json` `.log` | |
| `results/saturation.json` `.log` | |
| `pictures.py` | the figures below |
| `results/templates_random.png` | **start here** — random live templates, before and after phase B, both betas |
| `results/templates_drift.png` | the same for the 12 most-moved templates (the tail) |
| `results/templates_patch.png` | the 400 patch templates, for contrast — not one is a digit |
| `results/drift_hist.png` | how far templates moved, both betas |
| `results/templates_all_*.png` | all 400, after A and after B, each beta |
| `topk.py` | the top-k read, and the accuracy curve through both phases |
| `results/topk_curve.png` | accuracy on 0-4 and 5-9 across training, both tallies, both betas |
| `results/topk_sweep.png` `topk.json` | accuracy vs k after sequential training |

    uv run python saturation.py        # ~25 s
    uv run python whole.py             # ~30 s
    uv run python whole.py --beta 0    # ~30 s
    uv run python patch_beta0.py       # ~30 s
    uv run python pictures.py          # ~30 s
    uv run python topk.py              # ~15 s
