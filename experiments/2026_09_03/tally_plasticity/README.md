# Plasticity set by the tally alone — full 0-9

2026-09-03. The champion's step size is blind: `eta = clip(cnt/n, 0.02, 1)` is
the same rule for a template carrying the classifier and one that is noise, and
it never reaches zero, so templates drift forever. This replaces it outright.
How far a template moves depends on nothing but its worth to the tally:

```
imp(t) = SUM over cells,classes  P(t,c,y) * T[t,c,y]      its share of the information
                                                          the code carries about the label
r(t)   = fraction of templates with STRICTLY lower imp     ties -> everyone at rank 0
eta_t  = ETA_MAX * (1 - r(t))
```

One knob. Two properties come free: an empty table makes every `T` entry 0, so
every importance is 0 and everything is fully plastic; and a template that
becomes informative and then stops winning keeps its counts, keeps its
importance, and stays frozen — a fossil, which is the memory you want.

Everything else is the champion untouched — 5x5 patches, 576 positions, 400
templates, `T[t, 6x6 cell, class]`, beta 1.5, read by summing one lookup per
position. Full 0-9, no phases: 12,000 train / 3,000 test, 3 epochs, seed 7.

## The grid

```
                 recount   online     gap   frozen  win-share-frozen
base              0.9703   0.9133  +0.0570    0/400        0.000
frozen_all        0.9613   0.8967  +0.0647  400/400        1.000

flat 0.1          0.9697   0.6070  +0.3627    0/400        0.000
flat 0.2          0.9697   0.5270  +0.4427    0/400        0.000
flat 0.3          0.9687   0.5383  +0.4303    0/400        0.000
flat 0.5          0.9703   0.4780  +0.4923    0/400        0.000
flat 1.0          0.9697   0.3143  +0.6553    0/400        0.000

tally 0.1         0.9673   0.7633  +0.2040   79/400        0.453
tally 0.2         0.9687   0.7690  +0.1997   39/400        0.252
tally 0.3         0.9713   0.6560  +0.3153   26/400        0.179
tally 0.5         0.9703   0.6243  +0.3460   15/400        0.145
tally 1.0         0.9703   0.5663  +0.4040    7/400        0.057

shuffled 0.1      0.9680   0.7250  +0.2430   79/400        0.130
shuffled 0.2      0.9667   0.5833  +0.3833   39/400        0.025
shuffled 0.3      0.9707   0.6850  +0.2857   26/400        0.037
shuffled 0.5      0.9720   0.6200  +0.3520   15/400        0.034
shuffled 1.0      0.9700   0.5417  +0.4283    7/400        0.014
```

`base` = the champion. `frozen_all` = eta 0, templates never leave their random
init, only the table learns. `flat` = constant eta for everyone: no annealing,
no gating, which isolates what removing `cnt/n` costs on its own. `shuffled` =
the identical multiset of plasticities pushed through a fixed random permutation
of template indices — the same templates frozen, just the wrong ones.

## 1. The mechanism works exactly as designed

`templates_time.png` shows it directly: the highest-importance templates lock in
place by step 60 and are unchanged at step 282, while the lowest-importance ones
keep churning to the end. And the importance measure really is finding the
templates that carry the load — under `tally 0.1` the frozen templates absorb
**45.3%** of all wins, against **13.0%** when the same number are frozen at
random. Importance and usage are tightly coupled, and the measure detects it.

## 2. It changes nothing

```
ETA_MAX      0.1      0.2      0.3      0.5      1.0
tally     0.9673   0.9687   0.9713   0.9703   0.9703
shuffled  0.9680   0.9667   0.9707   0.9720   0.9700
```

Dead even at every setting, and both indistinguishable from `base` (0.9703).
The control does its job: **the ordering contributes nothing.** Freezing the
templates the tally values most is worth the same as freezing an equal number
chosen at random.

The black hole is real — 45% of the data lands on templates that learn nothing
from it — and it costs zero. Which is itself the answer to why the ordering does
not matter.

## 3. The predicted benefit went the other way

The hypothesis was that the online-recount gap would close, because useful
templates would settle instead of drifting forever. It opened: base +0.0570,
every tally arm +0.20 to +0.40.

`flat` says why. Removing `cnt/n` and holding eta constant leaves recount
untouched (0.9687-0.9703) and destroys the online tally (0.3143-0.6070). The
annealing was never there for the templates' benefit — it is what keeps the
running counts describing templates that still exist. Tally-gating recovers part
of that by freezing the busiest templates (gap 0.20 vs flat's 0.36 at the same
eta), but it never gets near what it replaced.

## 4. Why nothing moved: the templates are barely worth anything here

The control that reframes the run. `frozen_all` never moves a template — the 400
filters stay at their random initialisation, only the table learns:

```
random 5x5 filters, table only    0.9613
the champion's trained vocabulary 0.9703
```

**Training the vocabulary is worth 0.7 points.** `learned_vs_random.png` puts the
two side by side: pure noise on the left, clean oriented edges on the right, and
seven tenths of a point between them.

So no scheme for allocating plasticity could have won much. There was under a
point on the table, and the whole question of which templates deserve to move is
nearly inert at this receptive field.

## What this does and does not say

It does not refute the idea. The proposal was aimed at a stream, and this is one
stationary distribution, where there is no interference to prevent — that was
stated before the run. What it establishes is a ceiling: at 5x5, protecting
templates cannot be worth more than the 0.7 points templates are worth at all.

And it lands on the same boundary as `../whole_digit`. Small patches: templates
are nearly interchangeable, learning buys 0.7 points, drift is free, and it makes
no difference which ones you freeze. Whole digits: templates carry class identity,
and losing them costs 14.5 points. The same receptive-field split explains both
results.

**Which is where this mechanism should be tested.** It needs templates that are
worth protecting, and 5x5 patches are not. The run worth doing is this gate at a
receptive field where drift actually costs something — 11x11 or larger, on the
split — because there the 45% of data flowing into frozen templates is protecting
something instead of nothing.

## Caveats

One seed. The tally-vs-shuffled differences are 0.1-0.5 points on a 3,000-image
test set, well inside seed noise, so the claim is "no measurable difference",
not "exactly equal". MNIST only; random 5x5 filters over 576 positions with a
counted table is a strong random-feature baseline, and Fashion may not be as
forgiving.

## Files

| | |
|---|---|
| `tally.py` | the rig, all five arms, the sweep. `--smoke`, `--seeds` |
| `compare_init.py` | draws random vs trained templates against their two accuracies |
| `results/tally.json` | every number above, plus the per-probe curves |
| `results/curves.png` | recount, staleness and frozen count across training, all arms |
| `results/learned_vs_random.png` | **the one to look at** — what template learning buys |
| `results/templates_time.png` | most/median/least important templates through training |
| `results/templates_by_importance.png` | the 16 frozen vs the 16 most plastic |
| `results/templates_final_*.png` | all 400, sorted by importance |

    uv run python tally.py           # ~5 min, 17 runs
    uv run python compare_init.py    # ~40 s
