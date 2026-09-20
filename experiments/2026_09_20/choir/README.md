# 2026-09-20 — Layer 1: what a voice learns to measure

A restart. The k-means vocabulary is dropped; the question is whether a vocabulary can be
made out of the choir itself. Terms are in [`VOCABULARY.md`](../../2026_09_16/VOCABULARY.md) and are not repeated here.

Scripts: `layer1.py`, `shift.py`, `coherent.py`, `fourier.py`, `recur.py`. Figures in `results/`.

**The one-line result.** Reconstruction alone makes layer 1 learn two-pixel differences,
which are *worse than random projections* at the only property the architecture needs.
Adding a shift-coherence term fixes it, beats a hand-built Gabor bank, and costs 10 points
of reconstruction. A fixed Fourier transform does not substitute for the learning. A
vocabulary does emerge from the result, with parts shared across all ten classes.

---

## I. The architecture, as decided

`patch (49) --linear--> 2B --atan2--> B angles`. **`atan2` is the only nonlinearity**; no ReLU,
deliberately, because `atan2(s,a)` is unchanged when both scale, so the chord is contrast
invariant and a ReLU would destroy that.

Each voice is a **quadrature pair** of projections: `θ_b = atan2(w_b^s·x, w_b^c·x)`,
`r_b = |z_b|`. Two projections because one number cannot give an angle.

Four decisions and their reasons:

1. **The patch is atomic.** One patch → one chord. Composition happens *above* the patch —
   chords transposed by cell offset and summed into a polychord — not inside it. The
   alternative (patch ≈ sum of few atoms) would make layer 1 emit polychords and is what
   classical sparse coding assumes; it was considered and rejected.
2. **Quantize before summing.** Summing arrows *averages*: notes at 7 and 9 give 8, which is
   also what 6 and 10 give. Summing notes or clusters *lists*: two peaks survive. Only the
   second supports subtracting one part and keeping the rest.
3. **Greedy layerwise training.** The quantizer is non-differentiable, so no gradient crosses
   it. Each layer reconstructs its own input. This keeps everything local and online, which
   was the stated constraint anyway.
4. **Mean-centre each patch.** Otherwise every patch shares a brightness direction and
   unrelated patches score 0.098 chord similarity instead of 0.

---

## II. Things that are free, and therefore not what learning is for [MEASURED]

**II.1 Random projections already preserve similarity.** 172,827 inked 7×7 patches, random
quadrature pairs, no learning:

| B | pixel-sim <0 | <0.3 | <0.6 | <0.9 | ≈1.0 | correlation |
|---|---|---|---|---|---|---|
| 8 | 0.181 | 0.390 | 0.516 | 0.686 | 0.882 | 0.496 |
| 32 | 0.098 | 0.327 | 0.476 | 0.672 | 0.893 | 0.710 |
| 64 | 0.074 | 0.321 | 0.477 | 0.683 | 0.897 | 0.803 |

Monotone throughout; Johnson–Lindenstrauss. **"Similar strokes get similar chords" is free at
initialisation.** Learning is therefore not for producing similarity — it is for *reshaping*
it, or for a different property entirely, which turned out to be the case.

**II.2 `atan2` gives normalisation, not sparsity.** Every voice is always active. One-hot
notation makes the *bit vector* sparse (16 of 656) while all 16 voices participate: that is a
notation, not a property. What `atan2` guarantees is that no voice can dominate in scale.

**II.3 Small magnitude destabilises the note.** Jitter a patch by 0.05 and measure the angle
shift, split by that voice's `|z|`:

| `|z|` quartile | mean angle shift |
|---|---|
| lowest | 19.30° |
| 2nd | 6.25° |
| 3rd | 3.71° |
| highest | 2.12° |

Nine times worse in the bottom quartile. The magnitude is not bookkeeping: it says when to
trust the note. It is also where the `1/r²` factor in the encoder gradient blows up.

**II.4 Neighbour prediction is confounded by patch overlap.** At stride 2, adjacent cells
share 5 of 7 columns:

| offset | shared columns | pixel sim | chord sim |
|---|---|---|---|
| 1 cell | 5 of 7 | 0.364 | 0.485 |
| 2 cells | 3 of 7 | −0.007 | 0.281 |
| 4 cells | none | 0.018 | 0.233 |
| different digits | none | 0.003 | **0.225** |

Beyond two cells, same-digit patches are no more similar than patches from different digits.
`phase.py` (XVII) ran at `ps=5, stride=2` — 3 of 5 columns shared — so its numbers carry this
contamination. Note "not similar" ≠ "not predictable"; this kills the trivial version only.

---

## III. Reconstruction alone learns something useless [MEASURED]

`layer1.py`. 129,431 mean-centred 7×7 patches, B=64, phase-only decoder, Adam.

| arm | R² | quantised to 41 notes | top-1 pixel energy |
|---|---|---|---|
| PCA, 32 components | 0.981 | — | — |
| random, frozen | 0.913 | 0.912 | 13.2% |
| Gabor, frozen | 0.814 | 0.813 | 12.4% |
| **random, learned** | **0.980** | 0.980 | **51.8%** |
| Gabor, learned | 0.977 | 0.977 | 42.2% |
| sparse (group-L1 on r) | 0.994 | — | 37.6%, 30.1% voices active |

- **Learning works and reaches the linear ceiling.** 0.913 → 0.980, against PCA-32 at 0.981.
  Since a chord keeps B numbers and PCA-32 keeps 32, phase-only costs about 2× in code size —
  exactly the magnitudes being discarded.
- **Quantisation is free.** 0.980 → 0.980. Rounding each note onto the 41-slot gamut costs one
  thousandth. The non-differentiable quantizer is a non-issue at layer 1 *for reconstruction*.
- **[NEGATIVE] The filters are two-pixel differences.** Top-1 pixel holds 51.8% of a filter's
  energy against 13.2% for random. See [`layer1_filters.png`](results/layer1_filters.png): one red pixel beside one blue
  pixel. A finite-difference stencil at the smallest possible scale — a basis with no parts in
  it, which composes into nothing.
- **[NEGATIVE] Sparsity did not fix it** at λ=0.04: top-1 37.6%, and visually the same pixel
  pairs, merely fewer and pushed to the patch border. 30% of voices active is not sparse.
  λ was never swept, because §V made the point moot.

**Why this was predictable.** Reconstruction error is invariant under any invertible rotation
of the code (`x̂ = Dc = (DR⁻¹)(Rc)`), so it fixes the *span* and cannot prefer a basis. Given
freedom it picked the most trivial one available.

**[CONFOUND, mine] Two numbers above are not comparable.** The sparse arm uses a magnitude
decoder, keeping 2B=128 numbers against the phase-only arms' 64 — its 0.994 should be ignored.
And the frozen-Gabor arm is *broken*: the bank was parameterised badly for a 7×7 window (every
`w_cos` is the same central blob), so 0.814 says nothing about Gabors. `shift.py` carries a
corrected bank.

---

## IV. The property the architecture actually needs [MEASURED]

Pooling transposes each chord by its cell offset. Transposition is a **lookup**: offset → one
fixed vector of intervals, the same vector whatever the stroke. That asserts something
falsifiable:

> Moving *any* patch by t advances voice b by the *same* interval.

**Coherence** measures it: collect Δθ_b over many patch pairs and take the circular resultant
length `|mean exp(iΔ)|`. 1.0 = a fixed interval. 0.0 = a different one each time.

`shift.py`, 73,053 real patch pairs:

| shift | random | Gabor (corrected) | learned (pixel pairs) |
|---|---|---|---|
| 1 px | 0.635 | **0.873** | 0.561 |
| 2 px | 0.302 | **0.660** | 0.318 |
| 3 px | 0.176 | **0.446** | 0.166 |

**Training for reconstruction made transposability worse than the random projections it started
from.** Per-voice distribution:

| | best | median | worst |
|---|---|---|---|
| Gabor | 0.923 | 0.888 | **0.691** |
| learned | 0.805 | 0.687 | **0.053** |
| random | 0.780 | 0.617 | 0.490 |

The learned bank's *typical* voice (0.687) beats random (0.617); what sinks its mean is a
**tail of dead voices** at 0.053 — filters that landed on patch corners that are nearly always
blank. Gabor is the only bank where every voice is usable: its worst beats the learned median.

**Mechanism.** A two-pixel difference reads pixels (3,3) and (3,4); shift one pixel and those
positions see entirely different content, so nothing connects the new angle to the old. A
filter spanning many pixels with a smooth wave has its phase advance by an amount set by its
own frequency — a property of the filter, not the patch.

**Coherence and separation are different conditions, and neither implies the other.** A
*constant* encoder has coherence 1.0 (Δθ = 0 always) and holds no information. Separation —
distinct (identity, pose) pairs must not collide, i.e. no ghosts — is the anti-collapse
condition and is what reconstruction supplies. Both terms are load-bearing.

---

## V. The coherence objective works [MEASURED]

`coherent.py`:

```
E  =  ‖x̂ − x‖²  +  μ · Σ_b (1 − cos(Δθ_b − ρ_b[t]))
```

ρ is a **learned parameter**, so the same term shapes the basis *and* estimates the
transposition vectors pooling needs. Four shifts: (0,1), (1,0), (0,2), (2,0). 68,405 patch
tuples. Coherence scored on held-out pairs.

| μ | R² | top-1 pixel | coherence mean | worst voice | all-pairs sim |
|---|---|---|---|---|---|
| 0.0 | 0.972 | 43.1% | 0.565 | 0.047 | 0.051 |
| **0.05** | **0.871** | **16.0%** | **0.916** | **0.809** | 0.013 |
| 0.1 | 0.829 | 14.1% | 0.919 | 0.813 | 0.008 |
| 0.2 | 0.785 | 12.5% | 0.922 | 0.836 | 0.006 |
| 0.3 | 0.756 | 11.9% | 0.923 | 0.843 | 0.011 |
| 0.6 | 0.691 | 11.4% | 0.924 | 0.874 | 0.008 |
| Gabor | — | 17.0% | 0.873 | 0.691 | 0.017 |
| random | — | 13.2% | 0.638 | 0.493 | 0.003 |

- **It beats the hand-built Gabor bank**: 0.916 against 0.873, and the dead-voice tail is gone
  (worst 0.047 → 0.809, past Gabor's worst *and* its median).
- **The pixel-pair failure is cured**: top-1 43.1% → 16.0%.
- **No collapse.** All-pairs chord similarity stays at 0.006–0.013 — and μ=0 is the *least*
  diverse arm at 0.051. Adding coherence made codes more distinct, not less.
- **The knee is μ=0.05.** It reaches 99% of the coherence ceiling; μ=0.6 buys +0.008 more for
  −0.18 of R². Below 0.05 is untested.
- **Cost: 10 points of R²** (0.972 → 0.871).

**μ is the locality–frequency dial**, and this is the clean reading of the whole day:

```
reconstruction alone  →  pixel basis     maximally local,  worst coherence
coherence alone       →  Fourier basis   maximally global, best coherence
Gabor                 →  the compromise  local AND oriented
```

The classical time–frequency trade as a hyperparameter, rather than a guess about filter
shapes. See [`coherent_filters.png`](results/coherent_filters.png): at μ ≥ 0.1 the voices are full-patch oriented gratings.

---

## VI. A fixed Fourier transform does not substitute [MEASURED]

`fourier.py`. If the learned filters look like gratings, why learn them?

| bank | voices | coh 1px | coh 2px | phase-only R² |
|---|---|---|---|---|
| learned, μ=0.05 | 64 | **0.916** | — | 0.871 |
| Gabor | 64 | 0.873 | 0.660 | 0.784 |
| **fixed DFT** | 24 | **0.653** | 0.523 | 0.746 |
| random | 64 | 0.637 | 0.315 | 0.914 |

Barely above random. **Because our shift is not circular** — content moves through the window,
new pixels enter one edge and old ones leave, which the DFT shift theorem does not cover:

```
freq (0,1)   predicted −51.4°    measured −33.6°    coherence 0.901
freq (0,2)   predicted −102.9°   measured −47.9°    coherence 0.649
freq (1,2)   predicted −102.9°   measured −65.3°    coherence 0.675
```

Split by frequency along the shift axis:

| v (freq along shift) | voices | mean coherence |
|---|---|---|
| 0–1 | 7 | **0.894** |
| 2–3 | 8 | 0.541 |
| 4–6 | 9 | 0.566 |

The basis is not wrong; the **allocation** is. The DFT spends 17 of 24 voices on frequencies
that alias under a 1px shift. Its 7 usable voices score 0.894, level with Gabor.

**So learning buys two identifiable things:** (1) the *empirical* ρ rather than the theoretical
one — −33.6° where the formula says −51.4°, on a voice that is perfectly coherent at 0.901;
(2) putting all 64 voices in the coherent band instead of 7 of 24. The learned solution has a
compact description: **a band-limited Fourier-like basis with empirically measured intervals.**

Partly a 7×7 artifact: on a larger patch more of the frequency range stays coherent under a 1px
shift, so patch size and shift size interact. Untested.

---

## VII. Cluster: the differentiable note [MEASURED]

A **cluster** is not top-k. Top-k *selects* slots and is non-differentiable; a cluster *paints*
a bump of fixed shape centred at the continuous angle, so all 41 slots get a value and the
bump's position is a continuous parameter. Sliding θ from 7.0 to 8.0:

| θ | cluster @ 6,7,8,9 | change | one-hot @ 7,8 | change |
|---|---|---|---|---|
| 7.0 | .235 .332 .235 .083 | | 1 0 | |
| 7.2 | .202 .328 .266 .108 | 0.129 | 1 0 | **0** |
| 7.4 | .168 .314 .293 .137 | 0.135 | 1 0 | **0** |
| 7.6 | .137 .293 .314 .168 | 0.137 | 0 1 | **2** |
| 7.8 | .108 .266 .328 .202 | 0.135 | 0 1 | **0** |

The one-hot is the cluster in the limit as width → 0.

**Width is squeezed from both sides, and the constraint is tighter than expected.** Two notes
at separation 2 stay bimodal only for `w ≲ 0.8`; at `w ≥ 1.0` they merge into one peak, which
is the arrow failure reintroduced. Near-field sensitivity also *rises* as width falls (1.538 at
w=0.4 vs 0.332 at w=2.4). Only far-field reach wants width. **Use w ≈ 0.6–0.8; annealing is
probably unnecessary.**

---

## VIII. Corrections made during the day

Recorded because each was asserted before being checked.

1. **"Two levels of pooling flatten the hierarchy"** — wrong. `(A+B)+(C+D) = A+B+C+D` holds
   only for *pure bundling*. With a re-encoder at each level, `f(A+B)` is a new name and
   grouping survives. The real cost of re-encoding is losing exact subtraction and the ability
   to query "what is at offset t" at upper levels — a trade with product names, not a flaw.
2. **"Two bumps stay two bumps"** — only for `w ≲ 0.8` (§VII).
3. **"Narrow bumps have weak gradients"** — backwards in the near field.
4. **"The learned bank is at or below random on coherence"** — true of the mean only; its
   median is better and the mean is sunk by a dead-voice tail (§IV).
5. **The frozen-Gabor reference arm in §III was broken** (bad parameters for a 7×7 window).
6. Squashed dot products were ruled out before building: they mix contrast into the angle
   (216°/250°/347° for one shape at three brightnesses, against a flat −43.81° for quadrature),
   they have a seam (`w·x = −8` → 0.12°, `+8` → 359.88°, maximally opposite inputs 0.24° apart),
   and they discard the magnitude entirely.

---

## IX. Does a vocabulary emerge? [MEASURED]

`recur.py`. Reconstruction and coherence are both satisfied by a smooth embedding in which no
two patches ever land in the same place. A vocabulary needs the opposite. 49,948 inked
patches, K=128 regions by spherical k-means on the chord vectors.

| encoder | tightness | class entropy | classes/region | top 10% share |
|---|---|---|---|---|
| **learned (coherent, mu=0.05)** | **0.969** | 0.954 | 9.6 | 15.2% |
| Gabor | 0.909 | 0.955 | 9.6 | 15.1% |
| raw pixels | 0.869 | **0.916** | 9.1 | 15.1% |
| random | 0.827 | 0.934 | 9.4 | 14.6% |
| learned (recon only) | 0.765 | 0.931 | 9.4 | 13.4% |

*tightness* = mean chord similarity to own centre; *class entropy* normalised so 1.0 = all ten
classes equally present.

- **Concentration: yes, and learning earns it.** 0.969 against 0.827 (random) and 0.869
  (pixels). With all-pairs similarity at 0.013 (§V), this is genuine clustering and not
  collapse. [`vocabulary_regions.png`](results/vocabulary_regions.png) shows the region means are **interpretable strokes** —
  diagonal bars, verticals, horizontals, corners — not arbitrary slices.
- **Reuse: total.** 9.6 of 10 digit classes visit the average region; class entropy 0.954.
- **The learned encoder is *less* class-specific than raw pixels** (0.954 vs 0.916).
  Coherence training discards class information relative to using the pixels directly, which
  follows from what was asked of it: contrast invariance and shift covariance both throw away
  exactly the cues that make a patch class-diagnostic.
- **[CORRECTION, Lavender] Generic parts are not a failure here, and the first draft of this
  section wrongly called them one.** The encoder saw *no label signal at any point* —
  reconstruction and coherence only — so class-specific parts were never something it could be
  expected to produce. The old record's complaint (XV) was sharper than "the library is
  generic": it was that the library measurably **cost points on the label**, a harm not
  observed here. Generic is what "part" *means* — a diagonal segment should appear in 3s, 7s
  and 9s alike — and the class must then live in the *configuration* of parts, which is what
  pooling and layer 2 exist to test. What survives as a finding is only the directional one
  above: coherence training discarded class-correlated information relative to raw pixels.
  Whether that matters is unknown until configuration is tried.
- **[OPEN] The regions are redundant.** Nine of the twelve largest coherent-encoder regions are
  diagonal bars at nearly the same orientation, differing mainly in position — vocabulary spent
  on phase rather than on distinct shapes, as a Fourier-like basis would. Whether that is
  wasteful or exactly right depends on whether pooling factors position back out via
  transposition. Untested.

**Consequence for the label plan.** Patch codes carry little class information, so a label
gradient will lean even harder on the blank mask (§X). The control is mandatory.

---

## X. Two tracks into one choir: the label task [MEASURED]

Label prediction by joint reconstruction, with the label as **one more cell** in the polychord
at a reserved offset, rather than as a second input channel.

- **Two designs, and only one of them needs masking.**
  *Label as target only* — a decoder head off the top chord; the encoder never sees the label,
  so there is nothing to copy, and no masking is needed. **This is the one to build first.**
  *Label as input and target* (inside the polychord) — the shortest path from input to target
  does not pass through the image, so the loss is solved by copying and nothing about
  image->label is learned. Masking is then not a refinement but the thing that creates the
  task. It earns its complexity only if one uniform "complete whatever is missing" mechanism
  is wanted, covering absent image cells too.
- **Blanks carry the class, and the control is how the result gets read.** About 77 of 144
  cells are blank. **[CORRECTION] A blank-only classifier was never measured**: XV reports a
  *full* table at 93% and attributes it "largely to the shape of the blanks" (DESIGN.md:1202),
  which is an interpretation, not a number. An earlier draft of this section asserted the mask
  alone reached 93%; it does not say that.
  Two things are true at once. Joint reconstruction *does* protect the code — it must also
  rebuild the image, so stroke content cannot be discarded. But that guarantees the strokes are
  **in** the code, not that the label head **uses** them; the decoder may read the class off
  silhouette-correlated directions and ignore the rest. And if pooling sums over inked cells
  only, the set of contributing offsets *is* the blank mask — it is the support of the sum, not
  a separate feature the model might happen to notice.
  So the control is not there to prevent a problem. It is the only way to interpret the
  result: blank-only 90% against full 91% means the strokes bought one point; against 97% it
  means seven. **Measure it in the same run** — unlike XV, we would then actually have it.
- Label chords should be **fixed and random**, not learned: learned ones collapse, and fixed
  ones make readback an exact cleanup against ten candidates. Prior art: whole-image bundles
  with `label (x) class` bound in read back at 100% by unbinding, 86% by nearest neighbour (XIV).
### X.1 Built and run — `twotrack.py`

```
image  49-px patch -> L1 (frozen, coherence-trained) -> 121 chords
                   -> pool 3x3, transposed by rho    -> 25 histograms
                   -> L2 -> 25 chords
                   -> pool global, transposed by rho -> 1 histogram
                   -> L3 -> h_img
label  one-hot(10) -> M1 -> M2 -> M3 -> h_lab        (no transposition)
top    Z = cluster(h_img) + cluster(h_lab)           one choir
read   score all ten label chords against Z; at test Z = cluster(h_img) alone
```

10,000 train / 2,000 test digits, B=64, 15 epochs, clusters at w=0.7 so gradients flow
through both poolings. L1 frozen (no gradient crosses a hard quantiser).

| mask p | train | **test** |
|---|---|---|
| 0.0 — label always in the choir | 0.139 | **0.139** |
| 0.5 | 0.834 | 0.838 |
| **1.0 — label never in the choir** | 0.980 | **0.924** |
| linear classifier on the same L1 features | — | 0.911 |

- **[CONFIRMED] The copying shortcut is the whole story at p=0.** 13.9%, barely above the 10%
  chance line. With the label present in the top choir it is reconstructed from itself, the
  image is never consulted, and removing the label track at test time leaves nothing. This was
  predicted before the run and is the single largest effect in the table.
- **Masking helps monotonically; full masking wins.** p=1.0 *is* the target-only design — the
  label track never enters Z, and the label chords act only as scoring candidates.
- **92.4% beats a linear readout of the same L1 features (91.1%).** The two pooling stages plus
  L2/L3 contribute something a linear map on L1 cannot. Modest, and not guaranteed in advance.
- **The transposition algebra was used, not re-fitted.** Pool-1's intervals are *composed* from
  the pixel intervals of §V: `rho(dy,dx) = dy*rho(2,0) + dx*rho(0,2)`, cell step being 2 px.
  Poses adding, in anger. Only pool-2's rho is a free parameter.
- **[VACUOUS] The 3-layer label track adds nothing** and was built as specified anyway. `M1` on
  a one-hot yields ten arbitrary chords; `M2`, `M3` are deterministic functions of those, so the
  composition is still just ten chords. Depth on a one-hot input cannot add capacity.
- Label chords are **learned**, not fixed as §X originally advised. Cross-entropy over the ten
  candidates is contrastive, so nothing collapsed; the earlier caution was unnecessary here.

### X.2 What this does not show [OPEN]

- **Whether the class comes from configuration or from silhouette.** The blank-only control was
  deliberately skipped, so 92.4% cannot be split into "the parts and their arrangement" versus
  "the shape of the inked region". The §IX question — *can configuration carry the class where
  generic parts cannot* — is therefore still open, and this number does not answer it.
- **92.4% is well below what MNIST affords** (a small CNN exceeds 99%). Train 0.980 against test
  0.924 at p=1.0 also shows real overfitting at 10k digits.
- L1 was frozen throughout. Whether the coherence-trained bank is the right front end for the
  label task, as opposed to one trained jointly, is untested.

---

## XI. Open, in the order the evidence points

1. **[ANSWERED in §IX]** A vocabulary does emerge, learning sharpens it, and its parts are
   shared across all ten classes — which is what parts are, given no label signal was used.
   The open successor: can *configuration* carry the class? That is the claim pooling and
   layer 2 exist to test, and the one the counting branch failed at from the other direction.
2. **The (identity, offset) recovery test.** Given a patch shifted by an unknown offset, score
   every candidate pair and ask how often *both* come back right. It measures coherence and
   separation at once, and it is what the architecture will do at runtime. Designed, not run.
3. **Does μ=0.05 survive quantisation and pooling?** Quantisation is free for reconstruction
   (§III) but was never checked for *coherence*, and pooling was never built.
4. **Layer 2.** Nothing above it exists. The histogram input is `B × 41` = 2,624 numbers at
   B=64, about 50× wider than layer 1's input — size it deliberately.
5. **Patch size.** 7×7 may be too small for anything but derivatives; it also caps the DFT's
   coherent band (§VI). 11×11 changes both and is a one-line change.
6. **μ below 0.05**, and whether the R² cost can be reduced by letting r into the decoder.

---

## Later scripts of the day, with no writeup

Three more scripts sit in this folder that the findings above do not mention; no writeup was
kept for them. What each asks, from its own docstring:

- `unpool.py` — can the parts be read back out of a pooled chord? With n cells pooled, how
  often does a cell's true part come back, swept against the number of cells and how the
  parts are drawn. Prints only.
- `nodecoder.py` — up and back down with no learned decoder: pool the parts, then un-pool
  them by un-shifting and matching against the vocabulary, and put back each part's stored
  mean patch. Reports how good that can be at best, the round trip, and the gap. Figure:
  [`results/nodecoder.png`](results/nodecoder.png).
- `generate.py` — can a whole image be rebuilt from the single top-level chord (64 angles),
  no labels, against PCA at the same budget? Needs torch; writes `results/generate.png`,
  which is not here (no run was kept).

Files the scripts share: `results/W_coherent.npy` and `results/rho_coherent.npy` (written by
`coherent.py` / `recur.py`, read by `generate.py`, `nodecoder.py`, `twotrack.py`,
`unpool.py`), `results/W_learned.npy` (read by `shift.py`, `recur.py`).

Run any of them from the repo root, e.g. `python experiments/2026_09_20/choir/layer1.py`.
