# Where templates matter, and whether the tally can protect them

2026-09-03. Two runs. The first finds the receptive field at which templates are
worth anything at all; the second tests tally-driven plasticity there.

## 1. `rf_sweep.py` — what is template learning worth?

Two arms at each patch size, everything else the champion: templates learn
normally (`eta = clip(cnt/n, 0.02, 1)`), or they never move at all and stay at
their random initialisation while only the table counts. The gap between them is
what learning the vocabulary buys. 2 seeds, MNIST 0-9, 400 templates.

```
patch    positions   trained            never learned      learning is worth
 5x5        576      0.9677 +-0.0013    0.9612 +-0.0002        +0.0065
 9x9        400      0.9677 +-0.0007    0.9567 +-0.0013        +0.0110
13x13       256      0.9542 +-0.0002    0.9458 +-0.0035        +0.0083
17x17       144      0.9233 +-0.0003    0.9248 +-0.0018        -0.0015
21x21        64      0.8958 +-0.0005    0.9072 +-0.0005        -0.0113
25x25        16      0.8852 +-0.0012    0.8340 +-0.0047        +0.0512
28x28         1      0.8055 +-0.0072    0.4220 +-0.0167        +0.3835
```

**There is no gradual middle.** From 5 to 21 pixels, learning the vocabulary is
worth about a point — and at 17 and 21 it is *negative*, learned templates being
slightly worse than random noise. It only becomes decisive at 25x25 and 28x28. So no
intermediate patch size would have worked either — 11x11 sits in the middle of
the flat stretch.

The `positions` column is the reason. While templates are shared across dozens or
hundreds of positions, a random filter bank plus position-indexed counting is
already a strong code; learning buys a little sharpening. At 28x28 there is
exactly one position, nothing is shared, each template has to *be* a digit — and
random filters collapse to 0.4220 while learned ones reach 0.8055.

**And `rf_templates.png` is the part worth keeping.** At 17x17 and 21x21 the
learned vocabulary contains plainly readable digits — 7s, 1s, 9s, 0s — and
learning it is worth −0.15 and −1.13 points against random noise. Templates that
*look* meaningful are not templates that *matter*. Same lesson as
`crisp-rebuilds-are-not-evidence`, now on the input side.

This is why `../tally_plasticity` at 5x5 could not differentiate anything: there
was 0.65 of a point available to any rule about which templates may move.

## 2. `gated_split.py` — the gate where the stakes are real

28x28, 400 templates, phases 0-4 -> 5-9 -> 0-9, 20 epochs each.

```
eta_t = ETA_MAX * (1 - rank of importance(t))
importance(t) = SUM over cells,classes  P(t,c,y) * T[t,c,y]
```

At beta 0 drift is unopposed, so the gate acts alone — `../whole_digit` measured
the hole it is meant to fill: 14.5 points lost, 57% of templates converted into
other digits. At beta 1.5 the routing already protects, so there is less room.

```
                         recount 0-4   all ten   online all   frozen   wins into frozen
beta 0   base               0.8670     0.8397     0.8130         0          0.000
         tally 0.05         0.7737     0.6670     0.6727       155          1.000
         shuffled 0.05      0.8549     0.7707     0.7557       154          0.000
         tally 0.1          0.7911     0.7233     0.7057        79          1.000
         shuffled 0.1       0.8402     0.7973     0.7947        79          0.000
         tally 0.3          0.8173     0.8117     0.8003        26          0.518
         shuffled 0.3       0.8576     0.7983     0.7910        26          0.000

beta 1.5 base               0.9389     0.9127     0.8933         0          0.000
         tally 0.1          0.9422     0.9237     0.9240        79          0.416
         shuffled 0.1       0.9449     0.9307     0.9297        79          0.268
         tally 0.3          0.9436     0.9233     0.8923        26          0.189
         shuffled 0.3       0.9402     0.9293     0.9197        26          0.074
```

**The control wins.** At beta 0, `shuffled` beats `tally` on the old classes at
every setting — 0.8549 vs 0.7737, 0.8402 vs 0.7911, 0.8576 vs 0.8173 — and the
champion's blind rule beats both. At beta 1.5 the two are level and both are a
touch above base, but `shuffled` still matches or beats `tally`. **The ordering
never helps, anywhere.**

## Why it fails, and it is not a tuning problem

Look at the last column. At beta 0, `tally` sends **100%** of the data into
frozen templates.

Importance is a proxy for usage — the templates worth something are the ones that
win. At 28x28 there is one winner per image, so a small set of templates takes
essentially every image. Freeze the top of the importance ranking and you have
frozen exactly the templates the entire data stream flows through. Nothing
learns. The remaining 245-321 templates never win at all, so whatever plasticity
they are granted is irrelevant.

`shuffled` avoids this by accident: the randomly chosen templates are almost all
ones that never win anyway (0.000 win share), so the control is close to a no-op
— which is precisely why it does better.

**The general point.** Freezing a template stops it from *changing*; it does not
stop it from *winning*. A frozen template that keeps winning is a sink: it
absorbs data and learns nothing from it, and the data is not available to anyone
else either. Protection has to redirect the data, not just halt the update.

Which is exactly what beta already does — it routes patches toward templates
already committed to the believed class, so committed templates take their own
class and decline everything else. `../whole_digit` measured that protection:
churn 0.582 -> 0.211, 14.5 points saved. Routing is the right lever for this job
and plasticity is the wrong one, for a structural reason rather than a tuning one.

## What the failure points at: a conscience, not a freeze

The problem was never that important templates got protected. It is that
protecting them by halting their updates left them still collecting every image.
So move the intervention from the step size to the competition:

```
winner = argmin( err_h  -  beta * (agreement with believed class)  +  gamma * importance(h) )
```

An important template now pays a toll to win. It keeps its shape because nothing
displaces it, and the images it declines go to a template that can still use
them. **Frozen and unused is the fossil the proposal wanted. Frozen and busy is a
sink.** One line in `compete`.

Note this is not the same lever as beta, and they compose. Beta biases *toward*
templates that agree with the believed class — class-specific routing. A
conscience biases *away* from templates already carrying load — load-spreading.

**But it is not a free win, and the project has already found the reason.** A
conscience pushes usage toward uniformity, and templates are supposed to follow
data density — dense regions of the input deserve more templates, not an equal
share. That tension is recorded elsewhere in these notes and is unresolved. The
one thing this run adds: keying the toll to *importance* rather than to raw usage
is a different quantity in principle, but at 28x28 the two turned out to be
almost the same thing (which is the whole finding above), so the distinction may
not buy an escape.

## Caveats

Seed 7 only for the split (the sweep has 2 seeds). Differences at beta 0 are
large enough (5-10 points) that seed noise is not the explanation; the beta 1.5
differences are 0.5-1 point and are not resolved. Freezing by a *rank* threshold is
one choice among many, and the conscience variant above is untested.

## Files

| | |
|---|---|
| `rf_sweep.py` | trained vs never-learned at 7 patch sizes |
| `results/rf_sweep.png` | accuracy and the gap, vs receptive field |
| `results/rf_templates.png` | **36 templates at each size, learned above random** |
| `gated_split.py` | the gate at 28x28 across three phases, two betas |
| `results/gated_split_b*.png` | old-class and overall accuracy through the phases |
| `results/one_template_b*.png` | one template's life across 0-4 -> 5-9 -> 0-9 |
| `results/*.json` | every number above |

    uv run python rf_sweep.py        # ~2 min
    uv run python gated_split.py     # ~3 min
