# 2026-09-03

## `whole_digit/` — what "drift is not damage" depended on

Re-ran 09-01's split-MNIST forgetting test with the whole digit as the receptive
field instead of 5x5 patches, everything else identical, then filled in the 2x2
with the class pressure switched off in both rigs.

```
receptive field   beta   churn    joint   oracle   damage  staleness
5x5 patch          1.5   0.542   0.9683   0.9690  -0.0007    +0.1208
5x5 patch          0.0   0.107   0.9736   0.9742  -0.0007    +0.0000
28x28 whole        1.5   0.211   0.9040   0.8982  +0.0058    +0.0062
28x28 whole        0.0   0.582   0.8944   0.7492  +0.1452    +0.0468
```

Three things came out of it:

1. **The 09-01 claim survives with a condition.** Rewriting the vocabulary is
   free on 5x5 patches (54% churn, −0.0007) and expensive on whole digits (58%
   churn, +14.5 points). The receptive field is what decides.
2. **Two separate protections were confounded.** Small generic features make
   drift *harmless*; the class pressure stops drift *happening* (churn 0.582 →
   0.211 on whole digits, rescuing 14.5 points). On patches the pressure is
   unnecessary and costs accuracy.
3. **The table-staleness cost was caused by the pressure.** Patch rig at beta 0
   has staleness **+0.0000**. "Freeze the vocabulary" was a prescription for a
   problem the belief feedback loop created — and it does not port to whole
   digits anyway, where a frozen phase-A vocabulary scores 0.42-0.50 on unseen
   classes against the patch rig's 0.96-0.97.

You can see all of this directly, because a whole-digit template is 784 numbers
and can just be looked at: [`results/templates_random.png`](whole_digit/results/templates_random.png)
shows random templates before and after phase B. At beta 1.5 most are untouched
(22% change class); at beta 0 you can watch a 3 become an 8 and a 0 become a 6
(57% change). The 5x5 patch templates, by contrast, are edges and bars —
[`results/templates_patch.png`](whole_digit/results/templates_patch.png) — and not
one of them is a digit, which is the whole reason rewriting them is free.

A fourth result came from asking whether the population's redundancy could be
collected by reading the k nearest templates instead of the single winner
([`topk.py`](whole_digit/topk.py)). It helps as a readout — +1.8 points at beta
1.5, +4.7 at beta 0, optimum at k=3-5 — but it does not reduce forgetting: the
old-class loss across phase B is 9.9 points at k=1 and 9.5 at k=10. And with the
online (un-recounted) tally it backfires badly, because summing more rows sums
more stale mass.

Full writeup, caveats and the two forced deviations: [`whole_digit/README.md`](whole_digit/README.md).

---

## `tally_plasticity/` — letting the tally decide which templates may move

The champion's step size is blind to whether a template matters. This replaces it
with a plasticity set only by each template's share of the information the code
carries about the label: important templates freeze, unimportant ones stay fully
plastic, and an empty table leaves everything plastic for free.

**The mechanism works and buys nothing.** The high-importance templates visibly
lock in place by step 60 while the rest churn, and the frozen ones really are the
busy ones — they absorb 45% of all wins against 13% when an equal number are
frozen at random. But recount accuracy is identical across every arm: tally
0.9673-0.9713, shuffled 0.9667-0.9720, champion 0.9703. Freezing the templates
the tally values is worth the same as freezing random ones.

And the control that explains why: **templates that never learn at all — 400
random 5x5 filters, table only — score 0.9613 against the champion's 0.9703.**
Training the vocabulary is worth 0.7 points, so no plasticity scheme could have
won much.

Same boundary as `whole_digit/`: at 5x5 the templates are nearly interchangeable,
so drift is free and allocation is inert; at 28x28 they carry class identity and
losing them costs 14.5 points. The gate should be retested where templates are
worth protecting.

Full writeup: [`tally_plasticity/README.md`](tally_plasticity/README.md).

---

## `plasticity_rf/` — where templates matter, and the gate tested there

`tally_plasticity/` was run at 5x5, which could not differentiate anything. This
first measures where templates are worth anything, then retests the gate there.

**Template learning is worth about a point everywhere except at whole-image
scale.** Trained minus never-learned: +0.0065 at 5x5, +0.0110 at 9x9, +0.0083 at
13x13, **−0.0015 at 17x17, −0.0113 at 21x21**, +0.0512 at 25x25, **+0.3835 at
28x28**. While templates are shared across many positions a random filter bank
plus position-indexed counting is already a strong code. And at 17x17 and 21x21
the learned vocabulary contains plainly readable digits while being *worse* than
random noise — templates that look meaningful are not templates that matter.

**At 28x28, on the split, the gate is refuted by its own control.** At beta 0,
`shuffled` beats `tally` on old classes at every setting (0.8549 vs 0.7737,
0.8402 vs 0.7911, 0.8576 vs 0.8173), and the champion's blind rule beats both.

The reason is structural. Importance is a proxy for usage, so freezing the most
important templates freezes exactly the ones the data flows through — **100% of
wins land on frozen templates**, and nothing learns. Freezing stops a template
changing; it does not stop it winning, and a frozen template that keeps winning
is a sink. Protection has to redirect data, not halt updates — which is what beta
already does.

Full writeup: [`plasticity_rf/README.md`](plasticity_rf/README.md).

---

## `purity_plasticity/` — the tally sets the step

Follow-up to `plasticity_rf/`: replace the importance rank with the absolute
number the tally already holds. A template's step is the uncertainty of its own
class row (an empty row moves fully, a pure row barely moves), with no floor, no
annealing and no schedule. On whole-digit templates, split 0-4 -> 5-9 -> 0-9:

- Refuse to move but keep counting, and a committed template's purity erodes
  and it moves anyway (drift 0.30 against the champion's 0.13).
- Routing by the label protects old classes perfectly but recruits nobody;
  routing by the belief recruits. Summing them (label 0.25 + belief 0.5) gives
  0.9321 final on the split against the champion's 0.9142.
- Adding hire-on-error (a misread with a committed winner hands the image to the
  cheapest template) gives the best whole-digit numbers of the day: joint
  0.9458, split final 0.9470, no dead templates, online table equal to the
  recount. The price: old classes fall to 0.8687 during the new phase and
  recover afterwards.
- On parts (5x5-17px patches) none of it changes anything; hire moves the patch
  rig by 0.0001. The machinery pays only where templates are objects.

Full writeup (13 sections, including SoftHebb's two terms and partial-correlation
winners): [`purity_plasticity/README.md`](purity_plasticity/README.md).

---

## `fashion_tally/` — the tally-guided rig on Fashion-MNIST

The same rig on Fashion. At 28x28 the whole-image champion cannot learn the new
classes (0.6333 after phase B, 0.7243 final); the summed label+belief bias
learns them (0.8252) and, with the purity step, finishes at 0.8167 — level with
the 9x9 patch rig (0.8048). At 9x9 there is nothing to protect: all four arms
finish at 0.804-0.806, and any belief pressure only makes the online table
stale (0.4385 against 0.8052 with no pressure).

Full writeup: [`fashion_tally/README.md`](fashion_tally/README.md).

---

## `or_readout/` — OR-ing the winners into one position-free code

Collapse the winners into one 400-bit presence vector, no position, and count on
that. Position is worth 21 points at 5x5 and 8 at 9x9 / 13x13, but the OR code
is what shift tolerance looks like: at a 3-pixel shift the champion is near
chance (0.24-0.37) while the OR code loses only 5-7 points. As a dial
(`grid_sweep.py`), 9x9 templates in 2x2 cells is the practical position-light
code: 0.957 unshifted, 0.80 at 3 px.

Full writeup: [`or_readout/README.md`](or_readout/README.md).

---

## `residual_chain/` — winner, then residual winner

After the winner learns, compete again on what it left over and let that
winner learn from the residual. At 9x9 the chain recruits every template (dead
16 -> 0) but costs accuracy (-1.2 points, -0.5 with a surprise gate); reading
the two nearest templates hurts too. Top-1 with no pressure is the patch-layer
stream rule: 0.9710 joint, online table equal to the recount. The 5x5 half was
killed as uninformative and slow. Also `xor.py`: XOR with templates + tally,
solved 20/20 with a constant channel plus hire-on-surprise.

Full writeup: [`residual_chain/README.md`](residual_chain/README.md).

---

## `two_rung_fashion/` — a stroke layer read by an object layer

A frozen 9x9 stroke layer, pooled into 4x4 cells, read by an object layer of
400 templates with the day's competition and hire-on-error. The object layer
beats reading the strokes directly by a point (0.8133 against 0.8027). Sending
up SoftHebb's contrast code (centre, then ReLU) lifts it to 0.846, and width to
0.861, saturating there. A fitted linear probe on the contrast message gives
0.894, while a probe on the object layer's output reads 0.860 — no better than
counting it: a good classifier, a poor feature layer.

Full writeup: [`two_rung_fashion/README.md`](two_rung_fashion/README.md).

---

## `importance_map/` — an exact importance map from the tally read

The read is a sum over positions of table rows, so each position's share of the
decision margin is its exact contribution — no gradient, no approximation.
Painted over the digits (`results/importance.png`), the decisive regions are
where the class is (the top bar of the 7, the loop of the 9), and misreads show
their argument (a 6 read as 1 is carried by its long vertical stroke against its
own loop).

Full writeup: [`importance_map/README.md`](importance_map/README.md).

---

## `vision_drone/` — the drone that sees: image in, tally out

The drone now gets a 48x48 egocentric image instead of dx and dy, builds its own
state through the day's stack, and a tally maps that state to the PID teacher's
thrust levels. The object templates are right (each a definite scene the tally
reads as a pilot would), and one joint tally over the pair of motor levels was
the one real gain (0.852 within one level offline). But the flights fail: the
pupil climbs with the teacher and never brakes, because "target above AND
rising fast -> cut" is a conjunction and a sum of per-feature votes has no place
for it. DAgger made the offline score worse every round. The viewer at the repo
root flies it from `results/pupil.npz`.

Full writeup: [`vision_drone/README.md`](vision_drone/README.md).
