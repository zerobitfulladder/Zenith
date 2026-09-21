# 2026-09-21 — An invertible torus network

A detour off the choir, prompted by Nanda et al. (2023): a network that learns modular
addition turns out to use a quadrature embedding, angle addition and a cosine cleanup —
the same three moves the choir is built from. The question here is the converse. Rather
than waiting for a network to *discover* that structure, **build it in and see what it
costs.**

Script: `torusflow.py`. Two arms, `--span tau` and `--span pi`. Figures and checkpoints (`torusflow_*.pt`) in `results/`.

**The one-line result.** The construction works exactly — bijective to float32 precision —
and classifies at **96.2%**, past a linear readout, with every class chord decoding
backwards into a **readable prototype digit**. Treating pixel intensity as a genuine
circular coordinate costs **9 points**, the price of a symmetry the data does not have.
Dropping the `(label, residual)` split and putting the label on all 784 coordinates is
both simpler and slightly better. **A decode bug cost most of a day**: angles a hair below
zero were rendered as pure white instead of black, which made every generated image look
like salt-and-pepper noise and produced three negative findings that were not real (§IV.1).

---

## I. The construction

Classification is many-to-one — six thousand images of "3" all become "3" — so no bijection
does image → label. The way to have both is to keep what classification would discard:

```
forward   image -> T^784 -> flow -> (head: T^16 , residual: T^768)
          read the class off the head, ignore the residual
backward  pick a class, sample a residual, run the flow in reverse -> an image
```

Not *the* image, but *an* image of that class — a generative model rather than a classifier.
See [`arm-pi/architecture.png`](results/arm-pi/architecture.png).

Every pixel is an angle. Every layer is a bijection of `T^784`, built from two unimodular
(`|det| = 1`) pieces:

- **permutation** — a fixed random shuffle. An integer matrix with `det = ±1`, i.e. the
  benign member of the unimodular family. A cat-map shear is equally legal but large
  integer entries multiply the angle, which aliases and destroys the gradient.
- **coupling** — `θ_B += f(θ_A) mod 2π`, half the voices passing through untouched.
  Invertible by subtraction *whatever `f` is*, so `f` can be an arbitrary MLP. Addition
  mod 2π is the torus's own group operation, so this is more native here than in `R^n`.

Because `|det J| = 1` exactly, `log p(x) = log p_base(f(x))` with **no log-det term**.
Training is therefore exact maximum likelihood and the base carries the whole story:

```
base = vonMises(head ; chord_y , κ=8)      ten fixed chords, one per class
       × vonMises(residual ; μ , κ)        learned, per coordinate
```

Classification falls out as Bayes: the residual factor does not depend on `y`, so `argmax_y`
reduces to scoring the head against the ten chords — a cleanup, exactly as in the choir.

60,000 MNIST / 5,000 test, 6 coupling blocks, hidden 1024, Adam, 60 epochs, ~170 s.

---

## II. It is exactly invertible [MEASURED]

| arm | max `\|f⁻¹(f(x)) − x\|` | mean |
|---|---|---|
| span = 2π | 7.6e-06 rad | 1.4e-07 |
| span = π | 1.0e-05 rad | 2.7e-07 |

Float32 machine epsilon is ~1e-07, so this is numerical noise and nothing else. The
bijection is not approximate, learned, or regularised toward — it is structural.
[`arm-pi/roundtrip.png`](results/arm-pi/roundtrip.png).

Worth stating plainly because it is the one thing that worked without qualification.

---

## III. It classifies, and the seam is expensive [MEASURED]

| arm | head K | span | train | **test** | −log p (nats) |
|---|---|---|---|---|---|
| seam | 16 | 2π | 0.962 | **0.865** | 269 |
| no seam | 16 | π | 0.995 | **0.958** | −124 |
| **no seam, label is the whole chord** | 784 | π | 0.993 | **0.962** | 426 |
| logistic regression on raw pixels | — | — | — | 0.902 | — |
| uniform on `T^784` | — | — | — | 0.100 | 1441 |

**[CONFOUND] The −log p column is not comparable across spans.** Squeezing the data into
half the circle multiplies the density by 2 per dimension — `784·log 2 = 543` nats of
improvement before any modelling happens. On a common footing:

| arm | reported | minus the change-of-variable |
|---|---|---|
| tau | 269 | 269 |
| pi | −124 | **419** |
| full | 426 | **969** |

So the seamless arm models the data *worse*, not better; it only looked better because the
coordinate changed underneath it. The **accuracy** column is unaffected — it is computed in
latent space and involves no density. Caught in the §VII.1 audit, not during the runs.

- **The seamless arm beats a linear readout**, 0.958 against 0.902. Not a strong baseline,
  but the angular constraint is not simply costing accuracy either.
- **[NEGATIVE] The seam costs 9 points.** `θ = 2πx` puts intensity 0 and intensity 255 at
  the *same angle* — black is glued to white. [`arm-tau/seam.png`](results/arm-tau/seam.png) measures the damage
  directly: real MNIST has a 14:1 asymmetry between black and white pixels, and the model's
  samples come back as a **symmetric U**. On a circle that asymmetry is not representable.
- **This is the general caution, measured.** Intensity is not a circular variable. Wrapping
  it imposes a symmetry the data does not have, and the 9 points are what that costs. The
  same argument predicts an angular bias should not help on any task without genuine
  periodicity, which is the case Nanda's network had for free and this one does not.
- Half-circle is not free either — see §V, 41% of sampled pixels land in the empty half.
  There is a real tension: to be a *genuine* torus coordinate, wraparound has to mean
  something, and for intensity it does not.

---

## IV. The label as a whole chord, and no residual at all [MEASURED]

**[Lavender]** If the point is classification, the `(label, residual)` split is not needed.
Put the label on **all 784 coordinates** — ten fixed random points of `T^784`, one per class
— and drop the residual term. Collapse is impossible for a bijection, but *collapse was
never the requirement*: an image only has to land nearer its own chord than the other nine.

| | value |
|---|---|
| cos to **own** chord | **0.891** |
| cos to the other nine | **0.012** |
| learned κ_head | 6.56 |
| test accuracy | **0.962** |

- **Separation without collapse, exactly as predicted.** 0.891 against 0.012 is almost the
  whole way to its own chord and orthogonal to every other. The 6,000 threes stay 6,000
  distinct points — conserved volume demands it — and classify anyway.
- **Marginally the best arm**, 0.962 against 0.958, with a simpler objective and no base
  to fit. The cleanup now votes over 784 coordinates rather than 16.
- **[POSITIVE] Every chord decodes backwards into a readable digit of its class.**
  [`arm-full/chords.png`](results/arm-full/chords.png), top row. `f⁻¹(chord_y)` is a single point — with no residual
  the inverse is a *function*, not a sample — so this is not generation. It is the model's
  one canonical image per class, and it is legible for all ten. Adding angular noise around
  the chord degrades it smoothly (rows 2–4), so the chord sits in a neighbourhood of
  digit-like points rather than on an isolated spike.
- These figures were regenerated after the decode fix; the first version of this section
  described them as heavily speckled, which was the bug of §IV.1 and not the model.
- **[CONFOUND]** Test accuracy peaked at 0.971 near epoch 50 and drifted to 0.962 by 60
  while the loss rose from 379 to 426 nats. The arm is mildly unstable late; the table
  reports epoch 60, not the peak. Not swept.

### IV.1 [CORRECTION] A decode bug, and the three findings it invented

**[Lavender]** *"why is it pure black or white, and gradients only on the digits? black and
white are 180° apart — I'd understand a hard transition, but that isn't what this is."*

Right, and it was a bug in the decode, not the model. At span = π the data occupies the arc
`[0, π]`. An angle *below* zero — say 6.27, i.e. 0.013 rad short of black — should render as
black, the nearest point of the arc. `clip(θ/π, 0, 1)` instead sent it to 1.996 → **1.0,
pure white**, the far end. The background sits exactly on the wrap point, so half of it
rendered black and half pure white, 0.02 rad apart.

```
to_x:   clip(θ/SPAN, 0, 1)                    ->   project θ to the NEAREST point of [0, SPAN]
```

| on the ten prototypes | before | after |
|---|---|---|
| pixels rendered pure white | **37.6%** | **0.0%** |
| mean pixel value | 0.496 | 0.120 (real MNIST ≈ 0.13) |

[`analysis/decode_fix.png`](results/analysis/decode_fix.png). The ten prototypes are **clean, readable digits** and always were.

**The metric that hid it.** "37.6% of angles out of band" is true, and useless as stated:
the median overshoot is **0.145 rad, 4.6% of the arc**, and only **3.6%** of those are
nearer white than black. They are black pixels a hair too black. A fraction-outside count
said nothing; the distance outside was the number that mattered.

**Three findings this invented, now withdrawn:**

1. **"Averaging plateaus at 0.746 — bias, not variance."** Withdrawn. Rerun (`averaging.py`):

   | | bugged | corrected |
   |---|---|---|
   | corr between two single samples | 0.319 | **0.882** |
   | corr(single sample, prototype) | 0.458 | **0.934** |
   | corr to prototype, converged | 0.746 | **0.993** |
   | per-pixel std across samples | 0.327 | 0.068 |

   A *single* sample is already 0.934 correlated with the prototype. Samples are one image
   plus mild noise — the naive picture — not "genuinely different images each draw". The
   whole discussion of robust temporal filters (MOG and friends) was answering a question
   that did not exist.

2. **"Swapping the head destroys the image; head and residual are entangled."** Withdrawn as
   stated. The perturbation is **0.050**, not 0.701, and [`arm-pi/transfer.png`](results/arm-pi/transfer.png) shows clean
   digits keeping their identity across every swap. The real finding is the *opposite* and
   more interesting: swapping those 16 coordinates does **almost nothing**, because the
   residual already carries the class. A failure of factorisation still, from the other side.

3. **"The chord is badly placed; optimise the point to fix the speckle."** Withdrawn — the
   premise was the bug. `project.py` ran a real optimisation (37.6% → 3.2% out-of-band at
   λ=3000, all ten still classifying) but it was correcting a rendering artifact. The script
   is kept because the *method* — freeze the weights, optimise the point — is sound and may
   be wanted later; its conclusion is not.

**What survives untouched**, because it never passes through a decode: invertibility (§II),
every accuracy number (§III, §IV), the 9-point seam result, and the 0.891/0.012 separation.
Those are measured in angle space.

---

## V. Generation: better than reported, still not digits [MEASURED]

Rerun with the corrected decode.

- **Sampling the residual** ([`arm-pi/samples.png`](results/arm-pi/samples.png)) gives a black background with stroke-like
  ink in the right region of the frame — not salt-and-pepper. But it is texture, not an
  organised digit. The conclusion stands; the evidence is far weaker than the first version
  claimed.
- **Interpolating between two real images** ([`arm-full/interpolate.png`](results/arm-full/interpolate.png)) is much better than
  reported: endpoints clean, and the 5→6 row morphs smoothly through recognisable shapes.
  The middle of a long jump still degrades.
- **The prototypes are clean** (§IV.1). `f⁻¹(chord_y)` is a legible digit for all ten.

So the honest split: the model reproduces and *interpolates* far better than a bijection
trained only forwards has any right to, and it does not synthesise new digits from noise.

---

## VI. Why generation is limited [MEASURED]

The structural argument is unaffected by the decode bug, because it is about measure.

Under a volume-preserving map, `E[−log p_model] = h(X) + KL(pushforward ‖ base)` — the
differential entropy is exactly preserved, so the NLL floor is the data's own entropy and
the whole generative gap is that KL.

- **Sampled angles land outside the arc 41.1% of the time, median 0.259 rad past the end**
  (arm-pi) — a quarter of a radian on a 3.14 rad arc, unlike the prototypes' 0.145. The
  pushforward is genuinely not the base.
- **The circular mean of real latents is 35.2% out of band.** Means in this space are not
  data.
- **The residual carries the class** (§IV.1 item 2), so the factorised base was never fitted.

`|det| = 1` was introduced as the torus's advantage: free log-det, none of the Jacobian
bookkeeping flows are built around. The honest reading is that **it is free because it is
not doing anything.** A volume-preserving map can move probability mass but cannot
concentrate it, and concentrating it is the whole job of a generative flow.

---

## VII. Corrections

Recorded because each was asserted before being checked.

### VII.1 Audit of the training code

Run after the decode bug, on the reasonable suspicion that there were others.

**Found:**

- **The NLL is not comparable across spans** (§III). 543 of the pi arm's 393-nat "advantage"
  is pure change of variable; corrected, it is the worse model. A knob that moved two things
  at once.

**Checked and clean:**

- **`|det J| = 1` holds exactly.** Explicit 784×784 Jacobian, double precision, three
  samples: `log|det J| = −3.0e-14, +8.7e-15, −6.4e-15`, sign `+1`. The claim §VI rests on is
  verified, not assumed.
- **The dequantisation noise is drawn once rather than per epoch**, which could have been
  memorised. It was not: 403.7 nats on the training draw against 403.8 on a fresh one.
- **The logistic-regression reference is converged**, not undertrained — 0.9022 at 12 epochs,
  0.9076 at 60, 0.9018 at 200. The comparison in §III is fair.
- **No label leakage.** Train is `train`, test is `t10k`; at test time the readout sees only
  `flow(image)` scored against fixed chords.
- The `% TAU` in the coupling is cosmetic — every downstream use is periodic (`cos`), so the
  wrap affects neither loss nor gradient. The conditioner is fed `cos/sin`, so it has no wrap
  discontinuity either.

### VII.2 Asserted before checking

1. **The decode bug and its three fabricated findings** — §IV.1. The largest error of the
   day, caught by Lavender from the *pictures*, not from any metric: pure black and pure
   white with gradients only on the strokes is not what a wrapped coordinate should look
   like. No number in the run flagged it.
2. **"The torus is a great substrate for a flow because log-det is free"** — backwards. The
   free log-det is the *absence* of the mechanism that makes flows work (§VI).
3. **[CORRECTION, Lavender] "Asking a volume-preserving map to collapse is impossible, so
   this will not work."** True about collapse, wrong about the consequence — raised as an
   objection to §IV before running it. Classification needs *separation*, not collapse, and
   no two digits are the same anyway. The arm went on to be the best one.
4. **"The out-of-band penalty landscape is flat; the point cannot be moved."** Asserted from
   a λ sweep that topped out at 30 with lr 0.05 — the optimisation was too weak, not the
   landscape. lr 0.2 and 1000 steps moved it an order of magnitude further.
5. **"Adding noise to the label target would make it learn a region."** It is a provable
   no-op: for a von Mises target, `E_ε[−κ cos(θ − c − ε)] = −κ·E[cos ε]·cos(θ − c)`, i.e.
   exactly a rescaling of κ — which is a learned parameter and compensates. The useful
   version of the idea is to train the *reverse* direction, which is untried (§VIII).
6. **The first run was called a result at 71.4% test** (20k images, 3 layers, 12 epochs).
   Undertrained; the loss was still falling steeply. Not reported above.

---

## VIII. Open

1. **Give up `|det| = 1`.** Circular splines or Möbius transforms on each coordinate are
   non-volume-preserving and would supply exactly the missing capacity. Rezende et al.
   (2020) did flows on tori and spheres; worth reading before building. The log-det stops
   being free, which per §V is the point.
2. **Stabilise §IV** — it peaked at 0.971 and drifted down. A learning-rate schedule
   is the obvious first thing, and the peak may simply be the answer.
3. **Train the reverse direction** — **[Lavender]** sample `θ = chord_y + ε` during training
   and require `f⁻¹(θ)` to decode to something valid. That is §IV.1's inference-time search
   baked into the objective, and it is the one idea from today that has not been tried. Note
   the caution: in-band is necessary, not sufficient.
4. **Force the factorisation** rather than hoping for it — an adversarial or
   mutual-information penalty making the residual class-independent. Cheap to try, and it
   would separate "the base is a bad fit" from "the split is wrong in principle".
5. **The prediction that has not been tested.** An angular bias should *win* where the task
   is genuinely periodic. Rotated MNIST with rotation angle as the target is the clean case.
   Everything above is the no-periodicity control.
6. **Does any of this come back to the choir?** The cleanup readout is shared, and §V's
   interpolation behaviour is the polychord question in another costume. The invertibility
   is not — the choir's quantiser is deliberately many-to-one.
