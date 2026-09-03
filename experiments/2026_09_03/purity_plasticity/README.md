# The tally sets the step: purity-annealed plasticity

2026-09-03. Follow-up to `../plasticity_rf`, which froze templates by importance
*rank* and lost because the freeze fired on noise and the frozen templates kept
winning everything. This replaces the rank with the absolute number the tally
already holds, and removes the floor, the annealing and any schedule.

## The rule

A template's step is the uncertainty of its own class row, read from the row as
it stands before the batch is counted:

```
P(y|t)  = (N[t,y] + 1) / (sum_y N[t,y] + NL)        smoothed, as the table is
eta_t   = (1 - max_y P(y|t)) / (1 - 1/NL)            empty row -> 1, pure row -> 0
```

For a row of n wins all of one class that is 9/(n+10): the champion's 1/n
annealing, except that it is earned by *commitment* rather than by age. A
template that has won a thousand mixed images stays fully plastic; one that has
won three hundred of a single class is nearly still. Nothing is clamped.

The prediction going in: at beta 0 a committed template still wins the intruders
from the new classes, refuses to move for them, but *counts* them, so its purity
erodes and it unlocks. So the competition was also asked to consult the same
row, with the label in place of the belief:

```
none     win = argmin(err)
belief   win = argmin(err - beta * q . T[t,c,:])       the champion's routing
label    win = argmin(err - beta * T[t,c,y])           q = one-hot(label)
```

Two step rules x three competitions, 28x28 whole-digit templates, 400 of them,
split 0-4 -> 5-9 -> 0-9, 20 epochs per phase, 3 seeds. Through the 5-9 phase the
templates *committed* at the end of 0-4 (row purity > 0.9) are followed: their
purity, how far their shape moves, and what share of the wins they take.

## The grid, beta 1.5

```
                after 0-4   after 5-9: old/new   final old/new/all   online   #com   purity B  drift B  wins B  dead
cntn_none        0.7907     0.7687 / 0.7470    0.8919/0.8253/0.8583  0.8316   110    0.708    0.129    0.72    254
cntn_belief      0.8032     0.9272 / 0.8500    0.9463/0.8840/0.9149  0.8989   366    0.876    0.029    0.64      0
cntn_label       0.7482     0.8140 / 0.7379    0.8146/0.7320/0.7730  0.7539     7    0.999    0.000    0.00    250
purity_none      0.8034     0.7291 / 0.7820    0.8979/0.8626/0.8801  0.7942   115    0.674    0.299    0.73    254
purity_belief    0.7925     0.9098 / 0.9029    0.9389/0.8789/0.9087  0.8596   372    0.819    0.168    0.81      0
purity_label     0.7345     0.8265 / 0.7556    0.8209/0.7582/0.7893  0.7661     8    0.999    0.000    0.00    250
```

`cntn` = the champion's `clip(cnt/n, 0.02, 1)`. `#com` = templates committed at
the end of 0-4; `purity B` and `drift B` (1 - |cos| from their 0-4 shape) are
those templates at the end of 5-9; `wins B` is their share of the last batch's
wins. `online` is the un-recounted table, all ten classes.

## 1. Erosion happens, and it feeds on itself

With geometry-only competition the committed templates take three quarters of
the 5-9 wins. Their purity falls from 0.97 to 0.67 across the phase, their step
rises as it falls, and they drift **0.30** against the champion's **0.13** —
the floorless rule ends up *more* mobile than the floored one, because the
champion's floor is 0.02 and an eroded row gives 0.3. `committed_templates.png`
shows it plainly: the purity rule's committed 0s become 6s, its 3s become 5s,
its 4s become 9s and 7s. Refuse to move but count, and you eventually move.

And yet on the final recount, after the all-ten phase, `purity_none` finishes
**2.2 points above** `cntn_none` (0.8801 vs 0.8583). The plasticity that
destroys the old classes in phase B learns the new ones faster (0.782 vs 0.747)
and re-learns everything faster in phase C. So the step rule is better at
learning; what it lacks is anything stopping the intruders.

## 2. Under belief routing it is level, and still leakier

The belief route sends most 5-9 images to templates that agree with the
network's own guess, which keeps most intruders off the committed set. There
`purity_belief` matches `cntn_belief` on the final recount (0.9087 vs 0.9149,
inside the seed spread: 0.918/0.901/0.908 vs 0.913/0.918/0.914) but drifts six
times as far (0.168 vs 0.029) and the online table is four points worse
(0.860 vs 0.899). Whatever the belief lets through, the floorless step answers.

## 3. Label routing: perfect protection, and the strength matters

At beta 1.5 the label route protects perfectly — purity 0.999, drift 0.000,
**zero** intruders — and collapses: 7 or 8 templates carry all the data. The
label bonus is a log-ratio up to about 2, times 1.5, against geometric
differences of 0.1-0.3, so the first template to commit to a class takes the
whole class. The belief route survives 1.5 because its belief is soft (about half
the mass on the top class), which halves the effective bonus. So `label_beta.py`
sweeps the strength:

```
label routing     after 0-4   after 5-9: old/new   final old/new/all   online   #com   drift B  dead
cntn   beta 0.1    0.7954     0.9593 / 0.7604    0.9651/0.7726/0.8681  0.8663   102    0.001    253
purity beta 0.1    0.7965     0.9646 / 0.8056    0.9669/0.8092/0.8874  0.8864    98    0.000    253
cntn   beta 0.25   0.8171     0.9387 / 0.8339    0.9476/0.8319/0.8893  0.8871    55    0.000    254
purity beta 0.25   0.8017     0.9499 / 0.8498    0.9528/0.8566/0.9043  0.8998    57    0.000    254
cntn   beta 0.5    0.7679     0.8637 / 0.8151    0.8697/0.8246/0.8470  0.8350    18    0.000    254
purity beta 0.5    0.7994     0.8858 / 0.8385    0.8961/0.8418/0.8688  0.8478    23    0.000    254
cntn   beta 1.0    0.7732     0.8247 / 0.7342    0.8211/0.7377/0.7791  0.7570    10    0.000    254
purity beta 1.0    0.7515     0.8287 / 0.7779    0.8330/0.7801/0.8063  0.7723    11    0.000    254
```

**`purity` + label at beta 0.25 is the arm to look at.** Old classes after the
5-9 phase: **0.9499**, the best of any arm anywhere (belief routing: 0.9272).
Drift of the committed templates: 0.000. And the online table reads **0.8998
against a recount of 0.9043** — a staleness of 0.4 points, where the champion
with belief routing pays 1.6 and `purity_none` pays 8.6. The counts stay valid
because the templates they describe never moved. That was the stated goal.

The price is on the final all-ten recount: 0.9043 against 0.9149 for the
champion with belief routing, about a point. See §5 for what that point is made of.

## 4. The step rule wins wherever the intruders are kept out

`purity` beats `cntn` at every label beta (+1.9, +1.5, +2.2, +2.7 points on the
final recount), and by +2.2 with no routing at all. Only under belief routing is
it level or slightly behind (-0.6, inside noise). The step rule is a small,
consistent gain; the routing decides whether it is allowed to be one.

Also worth knowing what "still" means here. Only 5-15 templates ever have a step
below 0.02, so the committed templates are *not* clamped; their steps sit around
0.03-0.1. They stand still under label routing (drift 0.000) because everything
they win is their own class, so every step moves them toward where they already
are. Stillness comes from consistent data, not from a zero step.

## 5. The confound: label routing recruits nobody

Every label-routed arm, at every beta, ends with ~254 dead templates. So does
geometry-only (244-269). Belief routing ends with **0**. So the one-point gap
between `purity_label_0.25` and `cntn_belief` is measured with roughly 150
working templates against 400. The label bias protects; it does not spread. What
the belief route does that the label route does not — sending an image to a
template of the believed class *whichever* template that is, thereby waking
fresh ones — is a recruitment effect, and it is the thing to add. The two
biases are not exclusive; they can be summed.

## 6. Label protects, belief recruits: sum them

`combined.py`. Same table entry, two weights:

```
win = argmin( err  -  beta_L * T[t,c,y]  -  beta_B * q . T[t,c,:] )
```

Purity step throughout, 3 seeds. `conv` = share of the templates committed
after 0-4 whose row now names a 5-9 class; `drift kept` = drift of the rest.
Under belief routing nearly every template is committed after 0-4, so the new
classes have to convert some; the question is how many, and whether the rest
stood still.

```
                   after 5-9: old/new   final old/new/all   online   #com   purity B  drift   conv   drift kept  dead
champion (cnt/n, B1.5)  0.9272 / 0.8500  0.9454/0.8835/0.9142  0.8984   366    0.876    0.029   0.12    0.010       0
belief 1.5              0.9098 / 0.9029  0.9373/0.8712/0.9040  0.8619   372    0.819    0.168   0.19    0.098       0
label 0.25              0.9499 / 0.8498  0.9528/0.8566/0.9043  0.8998    57    0.988    0.000   0.00    0.000     254
L0.25 + B0.5            0.9664 / 0.8959  0.9702/0.8946/0.9321  0.9290   321    0.949    0.013   0.03    0.006       0
L0.25 + B1.0            0.9628 / 0.8714  0.9698/0.8720/0.9206  0.9092   386    0.920    0.069   0.14    0.011       0
L0.25 + B1.5            0.9566 / 0.8765  0.9687/0.8749/0.9214  0.9073   383    0.907    0.081   0.15    0.018       0
L0.1  + B1.5            0.9342 / 0.8941  0.9519/0.8901/0.9208  0.8894   375    0.870    0.118   0.16    0.052       0
L0.5  + B1.5            0.9626 / 0.8427  0.9700/0.8632/0.9162  0.9113   389    0.939    0.050   0.11    0.006       0
```

**`L0.25 + B0.5` beats the champion on every column.** Old classes after the
5-9 phase 0.9664 against 0.9272; new classes 0.8959 against 0.8500; final
recount **0.9321 against 0.9142**, with all three seeds (0.934 / 0.930 / 0.932)
above the champion's best (0.918); the online table 0.9290, three tenths of a
point from the recount; no dead templates. Of the templates committed to 0-4,
**3%** were converted to a new class and the other 97% moved 0.006 — against 12%
converted and 0.010 under the champion, and 19% and 0.098 under belief routing
with the purity step alone.

Every summed arm beats every single-bias arm on the final recount. The label
term is what stops the leak: at the same belief 1.5, raising the label weight
from 0.1 to 0.25 to 0.5 takes the drift of kept templates from 0.052 to 0.018
to 0.006. The belief term is what fills the vocabulary: any belief at all takes
dead templates from 254 to 0.

**And the weaker belief is the better one.** At belief 0.5 only 321 templates
are committed after 0-4, leaving about 80 uncommitted; the 5-9 classes go into
those and convert almost nobody. At belief 1.0-1.5 all but a dozen are committed
after 0-4, so 5-9 has to convert 14-15% of them, and the old-class accuracy is
still 3 points above the champion's because the label term picks the *least
pure* ones to convert. So the belief's job is to wake fresh templates, not to
route; the label routes. Read that way, the whole thing is: the tally's row for
a template says who may move it (the step), who may not enter it (the label
term), and whether anyone has claimed it yet (the belief term's smoothing
bonus, which is what actually recruits — see §5 and the caveat below).

## 7. With nothing to forget: joint 0-9, no phases

`joint.py`. All ten classes at once, the same 960 updates as the split (40
epochs), both step rules crossed with the three competitions, 3 seeds.

```
                          recount    (seeds)              online    gap    dead   still   purity
cnt/n   belief 1.5         0.9003   0.899/0.898/0.903    0.8953  0.0050   135      0     0.870     the champion
purity  belief 1.5         0.8766   0.877/0.881/0.872    0.8190  0.0576   212      0     0.839
cnt/n   L0.25 + B0.5       0.9191   0.916/0.921/0.920    0.9156  0.0036   185      0     0.887
purity  L0.25 + B0.5       0.9271   0.928/0.925/0.928    0.9197  0.0074   175    103     0.880
cnt/n   L0.25 + B1.5       0.9248   0.927/0.923/0.924    0.9234  0.0013     1      0     0.909
purity  L0.25 + B1.5       0.9314   0.931/0.935/0.928    0.9218  0.0097    77      8     0.890
purity  label 0.25         0.9157   0.916/0.919/0.912    0.9137  0.0020   255     93     0.807
purity  none               0.8920   0.878/0.897/0.901    0.8776  0.0144   255      4     0.771
cnt/n   none               0.8934   0.892/0.898/0.891    0.8909  0.0026   255      0     0.857
```

**It beats the champion by 2.7-3.1 points with nothing to forget**, every seed
of the summed arms above every seed of the champion. And it gets there fast:
at 240 batches (10 epochs) `purity L0.25+B0.5` reads 0.9160, already above the
champion's final 0.9003, which sits at 0.8683 at that point.

**Who gets the credit.** Swapping the champion's competition for the summed
bias, keeping its cnt/n step, is worth +1.9 (B0.5) to +2.5 (B1.5). Swapping the
step to purity on top of that adds another +0.7-0.8. So about three quarters of
the gain is the label term in the competition and a quarter is the step rule.

**The step rule alone is harmful.** `purity belief 1.5` is 2.4 points *below*
the champion, with a staleness gap of 5.8 points and 212 dead templates. The
same thing the split showed in §2: with only a soft belief to keep foreign
images out, the floorless step follows every one that gets in. The step rule
is a gain only once the label term is there to make rows pure, and then it is a
consistent one (+0.8 under both summed biases, +2 in §4).

`still` is worth a glance: under `purity L0.25+B0.5` 103 templates have a step
below 0.02 at the end, which is the rule doing what it says on a stationary
distribution — templates that have won several hundred images of one class
have stopped. Under cnt/n the count is always 0, because of the floor.

Two things the joint run adds to the caveats. The champion leaves 135 dead
templates here and the summed arms 77-185, so recruitment is incomplete for
everyone except `cnt/n L0.25+B1.5` (1 dead); the smoothing-artefact bonus
(§5, caveats) is not a reliable recruiter. And `purity L0.25+B1.5` edges
`L0.25+B0.5` here (0.9314 vs 0.9271) where the split preferred B0.5 (0.9214 vs
0.9321): the belief weight trades recruitment on a stationary stream against
spare capacity on a shifting one, and there is no single right value yet.

## 8. The middle ground: 5, 9, 13, 17 pixels

`patch_sizes.py`. The bet: between generic strokes and whole digits there is a
size where templates are specific enough for the tally to guide them, and a
template winning at a *position* is itself a class fact. For this the step rule
was changed to per-cell purity, usage-weighted over the 6x6 cells (identical at
28x28, so nothing above changes). Batch 128, 3 epochs per phase, 2 seeds, as in
`../plasticity_rf/rf_sweep.py`.

```
JOINT 0-9                     recount   online    gap    still  purity
 5   champion                  0.9673   0.9230   0.044      0   0.866
 5   cnt/n  L0.25+B0.5         0.9695   0.8933   0.076      0   0.966
 5   purity L0.25+B0.5         0.9685   0.8732   0.095     38   0.966
 5   purity B1.5               0.9695   0.7037   0.266      0   0.770
 9   champion                  0.9680   0.9368   0.031      0   0.892
 9   cnt/n  L0.25+B0.5         0.9692   0.9458   0.023      0   0.978
 9   purity L0.25+B0.5         0.9692   0.9370   0.032    184   0.978
 9   purity B1.5               0.9687   0.5938   0.375      0   0.751
13   champion                  0.9537   0.8965   0.057      0   0.862
13   cnt/n  L0.25+B0.5         0.9558   0.9285   0.027      0   0.972
13   purity L0.25+B0.5         0.9528   0.9047   0.048     39   0.971
13   purity B1.5               0.9498   0.5488   0.401      0   0.690
17   champion                  0.9272   0.8530   0.074      0   0.823
17   cnt/n  L0.25+B0.5         0.9260   0.8958   0.030      0   0.948
17   purity L0.25+B0.5         0.9282   0.8442   0.084      1   0.945
17   purity B1.5               0.9048   0.5652   0.340      0   0.681

SPLIT                         after 5-9: old/new   final   online   #com   drift   conv
 9   champion                  0.9728 / 0.9629    0.9688   0.7015   290   0.068   0.28
 9   cnt/n  L0.25+B0.5         0.9721 / 0.9659    0.9668   0.5070   399   0.102   0.58
 9   purity L0.25+B0.5         0.9715 / 0.9682    0.9695   0.4950   400   0.497   0.60
 9   purity B1.5               0.9708 / 0.9629    0.9657   0.1187   203   0.717   0.14
13   champion                  0.9594 / 0.9474    0.9522   0.7247   352   0.111   0.34
13   cnt/n  L0.25+B0.5         0.9661 / 0.9484    0.9550   0.6298   400   0.147   0.62
13   purity L0.25+B0.5         0.9570 / 0.9477    0.9545   0.5910   400   0.535   0.65
13   purity B1.5               0.9533 / 0.9434    0.9485   0.3007   306   0.736   0.30
```

**No accuracy gain at any size.** Joint recount is within 0.2 points across the
three sensible arms at 5, 9, 13 and 17. The label term does what it does at
28x28 — mean per-cell purity rises from 0.87 to 0.97, the templates become
class-pure at each position — and it buys nothing, because a position-indexed
table over generic strokes classifies exactly as well without it. This is
`rf_sweep` again: where learning the vocabulary is worth under a point, no rule
about the vocabulary can be worth more.

**Nothing to protect either.** On the 9x9 split all four arms hold the old
classes at 0.971-0.973. At 13x13 `cnt/n + summed` gains 0.7 on the champion and
the purity step loses 0.2. Drift is free on patches, as `../whole_digit` found,
so protection has no value.

**The purity step is wrong for generic features, in principle.** On the split
it drifts 0.50-0.54 against the champion's 0.07-0.11 and its online table
collapses (0.495 at 9x9 vs 0.70). The reason is the one the whole-digit run
predicted, with nowhere to hide: the label term commits all 400 templates in
0-4, the 5-9 patches must then land on committed templates at every cell, the
per-cell rows go mixed, and the floorless step re-mobilises everything. A stroke
*should* be shared across classes. Purity measures commitment to a class, and a
generic feature is supposed to have none, so purity calls every good stroke
template uncommitted and keeps it moving. Per-cell purity does not rescue this:
the 0.97 the label term manufactures is an assignment artefact that a
distribution shift dissolves in one phase.

**The one thing that does carry over: the label term reduces staleness in joint
training** — online table 0.9458 vs 0.9368 at 9x9, 0.9285 vs 0.8965 at 13x13,
0.8958 vs 0.8530 at 17x17, with the champion's step. On the split it does the
opposite (0.51 vs 0.70 at 9x9), so even that is not a general property.

**Where the boundary is.** Together with `rf_sweep` (learning worth: 5-21px
under a point or negative, 25px +5, 28px +38), the mechanism pays only where a
template is an *object*: 25x25 and up on MNIST. There is no intermediate size
where it starts to help gradually. The regime is "templates are things" versus
"templates are parts", and the parts side is the champion's.

## 9. Dead templates chase the data

`dead_chase.py`. A template that has never won moves toward the patch it is
nearest to (one dead template per patch, sign-aligned, uncounted and unread)
until it wins a real competition. `chase` = every patch, step 0.5; `err` = the
same weighted by the live winner's error (1 - cos^2), so covered patches barely
pull. 28x28, 20 epochs per phase, 2 seeds.

```
                    JOINT recount  online   dead  |  SPLIT after 5-9 old/new    final   online  dead
champion                  0.8800   0.8797   241  |
champion + chase          0.9182   0.9162     0  |
champion + err            0.9120   0.9118     0  |
best (purity, L.25+B.5)   0.9228   0.9158   226  |      0.9681 / 0.8921     0.9320   0.9285     0
best + chase              0.9402   0.9342     0  |      0.9698 / 0.8527     0.9125   0.9050     0
best + err                0.9400   0.9353     0  |      0.9718 / 0.8511     0.9145   0.9118     0
```

**Joint: the largest gain of the day.** Champion +3.8, best +1.7 (0.9402, both
seeds 0.938 / 0.942), every dead template recruited, and the online table
follows. The losers with nothing found something, as proposed.

**Split: it costs 2 points, and the reason is the reserve.** With everyone
recruited during 0-4, the new classes learn at 0.853 instead of 0.892. §6 found
the best arm won *because* it left ~80 templates unused after 0-4, and the
reserve is not about shapes: a template with an EMPTY row is the one a new
class can enter, because its label term for an unseen class is neutral where a
used template's is negative (smoothed P(y|t) = 1/(n_t+10)). Recruiting a
template spends its empty row.

**Error weighting does not distinguish novelty at whole-digit scale.** 400
prototypes explain a digit at cosine ~0.75, so every patch is "poorly
explained", the weight is always large, and `err` recruits everyone just as
`chase` does. The signal that would keep a reserve — this patch is worse
explained than patches usually are — needs a reference level, and every
threshold-free form tried here collapses to "recruit everything".

**So at fixed capacity this is a real trade**, not a bug: templates spent now
buy 1.7 points on a stationary set and cost 2 on a stream that arrives later.
The rig as it stands leaves the choice to the belief weight (§6: 0.5 keeps a
reserve, 1.5 does not). An untested resolution is to let a *used but
uncommitted* template be re-entered by a new class — give the label term the
row's uncertainty rather than its count — so the reserve is rebuilt from the
least committed templates instead of the unused ones.

## 10. Hire on error, best effort

`hire.py`. The symmetric branch the rig never had: what happens when the winner
does NOT match the label. Now, when the tally's own prediction for an image is
wrong and the winner is committed (row purity >= 0.5 — it knows what it is and
this is not it), the image is handed to the **cheapest** template: the one
whose row contributes least information to the read (dead = 0). Its row is
wiped, so it learns the image at full step and its old role is forgotten on
purpose. One hire per class per batch. When the winner is uncommitted or the
prediction is right, nothing changes. No dead template is required: at fixed
capacity the tally decides what to forget. 28x28, 20 epochs/phase, 2 seeds.

```
                    JOINT recount  online  dead  |  SPLIT after 5-9 old/new    final   online  dead
best (§6)                 0.9228   0.9158   226  |      0.9681 / 0.8921     0.9320   0.9285     0
best + hire               0.9458   0.9438     0  |      0.8687 / 0.9315     0.9470   0.9472     0
champion                  0.8800   0.8797   241  |
champion + hire           0.9290   0.9292     0  |
```

**The best whole-digit numbers of the day, joint and split.** Joint 0.9458
(seeds 0.943 / 0.948) against 0.9402 for the chase and 0.9228 without either;
the champion gains 4.9 points. Split final 0.9470 against 0.9320, new classes
after 5-9 at 0.9315 against 0.8921, online table equal to the recount, no dead
templates anywhere. Misclassification is a novelty signal that the recon error
was not (§9): it is absolute, it is the tally's own verdict, and it stops by
itself once a class is read correctly.

**And the trade is now visible where it belongs.** Old classes after the 5-9
phase fall to 0.8687 from 0.9681. With every template in use after 0-4, each
hire during 5-9 sacrifices a live 0-4 template, and hires keep firing on
confusions *within* the new classes (sandal vs sneaker), not only on the first
boot. Ten points of old-class accuracy are spent during the new phase and
recovered in the all-ten phase. That is the limited-capacity behaviour asked
for — forget the least useful thing to learn the new one — and it is a choice
the tally makes, not damage it suffers. Whether ten points is the right price
is a question about the importance measure (usage x information, from
`../plasticity_rf`) and about hiring on new-vs-new confusions at all; neither
was varied here, and the number of hires per phase was not logged.

## 11. Hire on error on parts, and the record

`patch_hire.py`. On parts a misread image does not mean any patch was wrong,
so the offending patches are those whose winner is committed at that cell to
evidence against the label; each goes to the nearest template uncommitted at
that cell. Joint 0-9, 3 epochs, 2 seeds, cnt/n step, at 12k/3k and at 40k/10k.

```
                       12k/3k, 9x9        40k/10k, 9x9       40k/10k, 5x5
top-1, no pressure     0.9710  (16 dead)  0.9739  (16 dead)  0.9708  (6 dead)
none + hire            0.9708  (15)       0.9740  (12)       0.9709  (6)
summed + hire          0.9680  ( 0)       0.9717  ( 0)       0.9671  (0)
```

**Nothing.** Hire changes the patch rig by 0.0001 with both seeds agreeing,
and the summed bias costs 0.2-0.4 as before. The novelty branch has nothing to
do on parts: a misread MNIST digit is an ambiguous digit, not a missing stroke,
and the vocabulary is complete at 400 templates.

**On the record.** The project's 0.9869 (`2026_09_01/stack/percell.py`)
is not this readout: it fits a linear map over the per-cell evidence (36 cells
x 10 classes) on 40k images. The counting table's argmax on the same data and
templates sits at 0.974, which is what every arm above reaches. Since none of
today's mechanisms changes the code on parts (same dead count, same purity,
same accuracy), the per-cell probe on these codes would return the same 0.987
with or without them. The record is a readout-and-data result, and the
tally-guided machinery neither helps nor hurts it.

Where the machinery pays is a layer whose templates are objects (§6-10). The
run that could move the record is therefore a second rung: a whole-image layer
over the pooled 9x9 code, with hire-on-error, read by the table — the August
two-rung stack (0.9492 with k-means) with today's competition and recruiter in
the upper layer. Not run.

## 12. SoftHebb's two terms on this rig

`soft_push.py`. After reading Journé et al. 2023 ("Hebbian Deep Learning
Without Feedback"): losers move by their softmax activity, toward the input
(`soft`, winner-take-most) or away from it (`anti`, their soft anti-Hebbian
term), each gated by its own purity step; counting stays top-1. On top of the
best rig (§10). 28x28 MNIST, 2 seeds, two temperatures.

```
                    JOINT recount  online  |  SPLIT after 5-9 old/new    final   online
best (§10)                0.9458   0.9438  |      0.8687 / 0.9315     0.9470   0.9472
soft  tau 0.05            0.9395   0.9373  |      0.7871 / 0.8021     0.9187   0.9133
soft  tau 0.2             0.9120   0.9122  |      0.5682 / 0.7108     0.8893   0.8882
anti  tau 0.05            0.9415   0.9408  |      0.6528 / 0.8703     0.9417   0.9398
anti  tau 0.2             0.8533   0.8420  |      0.5631 / 0.7402     0.8773   0.8775
```

**Neither term helps a counted read, and both cost on the stream.** Losers
moving toward the input blur (winner-take-most is the all-learn collapse in
slow motion: -0.6 joint at the cold temperature, -3.4 warm). The push is level
joint at the cold temperature (-0.4, inside noise) and -9 warm, and on the
split it takes the old classes after 5-9 from 0.87 to 0.65 even though it is
gated by purity and never touches a committed template's own inputs — a
pushed-off loser is still a moved template with stale counts.

Read against the paper: their gain from the push is measured with a fitted
linear head over a dense contrast code, where a feature that separates two
neighbouring clusters is worth more than one that averages them. A counted
table over identities never reads contrast; it reads who won where, and for
that a prototype is the right feature. Same rule, two readouts, opposite
verdicts. The paper's other ingredient, the mean-centred rectified message
upward, is the one that could matter here, and it needs the second rung
(`../two_rung_fashion`), untested.

## 13. Winner by partial correlation: the least replaceable template

`loo_split.py`. Score(t) = the input's correlation with template t after
everything the other 399 templates explain is removed (Gram inverse, ridge
1e-3). Best score wins, learns the WHOLE input with the purity step, is
counted; the read uses the same rule. Everything else the best rig. 28x28 (784
dims > 400 templates, so the leftover exists), 2 seeds.

```
                       JOINT recount  online  purity  |  SPLIT after 5-9 old/new    final
cosine winner                0.9458   0.9438   0.894  |      0.8687 / 0.9315     0.9470
partial-corr winner          0.6043   0.6030   0.860  |      0.5497 / 0.6291     0.5753
```

**-34 points.** `loo_templates.png` shows the mechanism. The templates are
still digits (the winner learns the whole input, so they stay prototypes), but
the 36 most-used templates under the partial-correlation rule are almost all
thin 1s and a few 8s. A template with many near-twins (the 0s, the 6s) has a
large diagonal in the Gram inverse and its partial correlation is deflated for
every input; a template with no twin — a thin 1, which overlaps little with
anything — is "least replaceable" for inputs of every class and wins them. The
rule rewards distinctiveness of the *template*, not fit to the *input*, and
for a counted read those are different things: the winner's row goes mixed
(purity 0.86) and the read collapses. With 400 nearly collinear templates the
Gram inverse is also ill-conditioned, so the scores are noisy on top.

This is the CSHL notes' conclusion ("templates are whole-input hypotheses;
cosine is right") reached from the selection side: uniqueness among templates
is not evidence about the input. The local form (leave out only the nearest
few) was not run.

## Caveats

Three seeds; arm differences under a point are not resolved. The whole-digit rig
is a weak model (joint accuracy about 0.91), used because it is where templates
are worth protecting (`../plasticity_rf`, §1); nothing here is measured on
patches. Label routing uses the label at training time, as a second input
stream; test-time reading is pure geometry in every arm. "Committed" is a
threshold (purity > 0.9) chosen for the diagnostic only; the rule itself has no
threshold. The row is summed over cells, which is exact at 28x28 (one cell) and
an approximation on patches.

**Recruitment is a smoothing artefact.** An unused template's table entry is
`log((n_c + K) / (n_cy + K))`, about +1.4 after one phase, because its
conditional count is divided by a smaller total than its marginal. That bonus,
times the belief weight, is what wakes fresh templates; it is not designed, and
it is worth making explicit before this is relied on. Also: the champion's
`cnt/n` step was not paired with the summed bias, so "purity step + summed
bias" is compared against "cnt/n + belief", not against "cnt/n + summed bias".
The step rule's own contribution under the summed bias is unmeasured here.

## Files

| | |
|---|---|
| `purity_split.py` | the rule, the three competitions, the split. `--smoke` |
| `label_beta.py` | label routing at beta 0.1 / 0.25 / 0.5 / 1.0, both step rules |
| `combined.py` | label + belief summed, five weightings, against the three single-bias references |
| `joint.py` | all ten classes, no phases: both step rules x three competitions |
| `patch_sizes.py` | 5 / 9 / 13 / 17 px, joint and split, four arms |
| `dead_chase.py` | dead templates chase the data, three modes, 28x28 joint and split |
| `hire.py` | hire on error: misread image with a committed winner -> cheapest template, row wiped |
| `patch_hire.py` | hire on error on parts, 9x9 / 5x5, 12k and 40k. `--ps`, `--ntrain`, `--ntest` |
| `soft_push.py` | SoftHebb's soft / anti-Hebbian loser plasticity on the best rig, two temperatures |
| `loo_split.py` | winner by partial correlation (Gram inverse) vs cosine; saves and draws templates |
| `results/loo_templates.png`, `loo_templates_{cos,pc}.npy` | the 28x28 templates, both rules |
| `results/patch_sizes.png` | recount, staleness, and old-class retention vs patch side |
| `results/joint.png` | recount, online table and row purity through joint training |
| `results/combined.png` | the board for §6: accuracy, committed purity, drift, converted share |
| `results/purity_split.png` | **the board** — accuracy, committed purity, drift, win share through the phases |
| `results/committed_templates.png` | the 12 most committed templates after 0-4 and the same after 5-9, every arm |
| `results/label_beta.png` | the beta sweep |
| `results/*.json`, `*.log` | every number above, per seed, per probe |

    uv run python purity_split.py    # ~3 min
    uv run python label_beta.py      # ~3 min
    uv run python combined.py        # ~6 min
    uv run python joint.py           # ~6 min
    uv run python patch_sizes.py     # ~10 min
    uv run python dead_chase.py      # ~3 min;  --err-only, --unc-only for the variants
    uv run python hire.py            # ~2.5 min
    uv run python patch_hire.py --ps 9 --ntrain 40000 --ntest 10000   # ~2 min
    uv run python soft_push.py       # ~5 min
    uv run python loo_split.py       # ~4 min
