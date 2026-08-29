# The Hypercolumn — reference

What one unit is, how it learns, how an answer is projected out of it,
and what a layer may say to the layer above. Every claim marked
*measured* comes from `experiments/2026_08_29/`.

Terminology, fixed: a **template** (or **minicolumn**) is one stored
hypothesis. A **hypercolumn** is the group of templates that compete
over the same input. A **layer** is one or more hypercolumns.

---

## 1. The code

Everything a unit ever sees is one sparse vector `x ∈ R^D`.

**A value becomes a patch of cells.** A channel `c` owns `NB` cells with
centres `μ_c[0..NB-1]`. A value `v` lights cell `i` with weight

    w_i = 0.5 · (1 + cos(π · d_i / r))    for d_i < r,  else 0
    d_i = | μ_c[i] − clip(v, μ_c[0], μ_c[NB-1]) |
    r   = halfw · (μ_c[1] − μ_c[0])           (fall-off radius, in value units)

The radius is measured in **value units, not cell indices**, so the
pattern slides continuously as `v` moves: a value halfway between two
cells lights both, and a small change in `v` is a small change in the
code. This is the property everything else rests on.

**Channels share one array.** Each channel owns `NB` positions inside
`R^D`, and the positions of all channels are **disjoint** — carved out
of a single permutation of `0..D-1`, not drawn independently per
channel (§7.1). Channels combine by element-wise max:

    x[p] = max over channels writing to position p

This is the OR. One vector therefore states a whole moment: *these
inputs co-occurred with these answers.* Typical scale: 2 channels ×
96 cells in `D = 8192`, ≈ 40 positions nonzero (0.5 % density).

There is no separate input space and output space. Which part of the
vector is a question and which is an answer is decided at read time
(§4), not at build time.

---

## 2. Template state and matching

The hypercolumn holds `W ∈ R^{K×D}`. Every row is kept

    Σ_j W_ij = 0        (mean-centred)
    ‖W_i‖ = 1           (unit length)

**Match is Pearson correlation.** With `m = Σ_j x_j / D` and
`n = ‖x − m·1‖`, the correlation between template `i` and the code is

    c_i = ⟨W_i, x − m·1⟩ / n  =  ⟨W_i, x⟩ / n

The mean subtraction is free because `W_i` is already centred, so the
score touches only the ~40 nonzero positions of `x`:

    ⟨W_i, x⟩ = Σ_{j ∈ active} W_ij · x_j
    n² = Σ_j x_j² − D·m²

`n` is the same for every template, so `argmax_i c_i = argmax_i ⟨W_i, x⟩`.
A full-vector competition costs one `K × |active|` gather-and-multiply.

**Why unit length is not housekeeping.** It is the only way to get
sharp templates *and* diverse usage at the same time. Unnormalised
matching with argmax gives a usage monopoly (entropy 0.32); replacing
argmax with sampled selection restores diversity but blurs every
template, because sampling is graded learning in expectation. Only the
normalised core gives both.

---

## 3. Learning

Exactly one template updates per input.

**Bootstrap.** While fewer than `K` templates exist, adopt the code
itself as a new template:

    w = x − mean(x) + ε,    ε ~ N(0, (0.02/√D)²) per element
    w ← w − mean(w)
    W_new = w / ‖w‖

So the memory starts from real examples, not noise.

**Geodesic rotation.** Otherwise, with `i* = argmax_i ⟨W_i, x⟩` and
`c = ⟨W_{i*}, x⟩ / n`:

    skip if c ≤ 0 or c ≥ 1
    θ = η · c                              (η = 0.05 default)
    t = √(1 − c²)
    a = cos θ − c·sin θ / t
    b = sin θ / t
    W_{i*} ← a·W_{i*} + b·x̂                where x̂ = (x − m·1)/n

This is the exact rotation of `W_{i*}` by angle `θ` toward `x̂` inside
the plane they span: writing `u = (x̂ − c·W)/t` for the orthogonal
component, `W' = cos θ·W + sin θ·u = a·W + b·x̂`. Both invariants are
preserved exactly — `W'` stays unit length and stays centred.

On the sparse code it expands to a scale, a scalar shift and a short
add, with no dense vector ever built:

    W_{i*} *= a
    W_{i*} -= b·m/n                        (uniform shift, keeps it centred)
    W_{i*}[active] += (b/n) · x[active]

Re-centre and re-normalise every 128 updates to absorb float drift.

**The step is proportional to how well it already matched** (`θ = η·c`).
A template that nearly recognised the input moves a lot; one that
barely recognised it barely moves. Optional consolidation divides `θ`
by `1 + wins_i / consol`, making a template plastic when young and
stiff when experienced.

**Only one template learns.** Top-5 learning degraded accuracy and
probe scores in 8/8 seeds (p < 0.0001). Softmax-sampled selection
blurred identically, for the same reason as above. The failure is
invisible to cosine-style metrics — the memory still *matches* fine
while its templates go flat — so it has to be watched directly by
measuring template peakiness.

---

## 4. Projection — getting an answer out

Write the channels you know. Leave the rest of the array empty. Compete.
Read the winner's own cells where the emptiness was.

    1. build the partial code x from the known channels only
    2. score every template over the written positions (§4.1)
    3. i* = argmax
    4. for each unwritten channel c:  p = W_{i*}[pos_c]        ∈ R^NB
    5. sharpen p to a value (§4.2)

Nothing else is involved: no decoder, no output weights, no second
pass. The answer is literally part of the winning template.

### 4.1 Scoring a partial code

A bare dot product is **wrong** here, and silently so. It assumes every
template carries the same norm inside the written positions. It does
not — and a template that has become *vague about the missing part*
keeps its whole norm in the written part and outbids the templates that
actually match. Winning more queries makes it vaguer still, so it
compounds. Use

    s_i = ⟨W_i|_S , x|_S⟩ / max( ‖W_i|_S‖ , floor · max_k ‖W_k|_S‖ )

with `S` the written positions. This is the same cosine rule, applied
to the cells that are actually present.

`floor` (default 0.25) is a contrast floor. Without it, a template
holding almost nothing in `S` gets a near-zero overlap divided by a
near-zero norm and bids absurdly high. `floor = 0` is the bare masked
cosine; `floor = 1` makes the denominator constant and reproduces the
plain dot product exactly.

*Measured*, error on unseen inputs as templates are added:

| K | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| bare dot product | 0.102 | 0.107 | 0.396 | 0.546 |
| scored on the written part | 0.092 | 0.105 | 0.098 | 0.094 |
| templates actually used, bare | 109 | 166 | 164 | 130 |
| templates actually used, corrected | 123 | 236 | 377 | 471 |

With the bare rule, **adding templates makes the unit worse** and it
collapses onto 13 % of itself. The floor changes none of these numbers
between 0 and 0.5.

### 4.2 Sharpening the profile

The read is a profile `p ∈ R^NB`, not a number — a graded assertion over
the channel's cells. Three ways to use it:

- **peak** — `μ_c[argmax p]`. What generation uses. Resolution is one cell.
- **centroid** — centre of mass of the contiguous run around `argmax p`
  where `p > 0.5·max(p)`, weighted by `p − 0.5·max(p)`. Recovers
  sub-cell precision.
- **as-is** — read the shape. When the unit is torn between two answers
  the profile is visibly bimodal, which the peak discards. That is real
  information about the unit's own uncertainty, available for free.

The store is rich; **the reader picks the nonlinearity.** Recognition
reads graded profiles, generation hardens to one answer. The memory
does not decide this.

---

## 5. What a layer says to the next layer

The message is **all positive correlations, magnitudes intact,
sparsified to the best few** — not the winner, not everything.

**Top-1 upward is the trap.** If a layer reports its winner, the layer
above receives one lit cell out of `K`. That is a one-hot code, and a
one-hot code has no overlap — which is the exact property the whole
scheme depends on. Template #237 and #238 may stand for nearly the same
thing; as messages they share nothing, so the layer above cannot see
they are related. The overlap that made the layer work is destroyed at
its own output.

Keep the distinction sharp: **top-1 is the learning rule, not the
message format.** Different channels, opposite requirements.

**Everything upward is the other trap.** Sending every positive match
washes the differences out.

*Measured*, same unit, only the read changed:

| read | winner only | best 4 | best 16 | best 64 | every positive |
|---|---|---|---|---|---|
| error | 0.094 | 0.065 | **0.046** | 0.103 | 0.488 |

*Measured*, with a real second layer stacked on top and asked to
reconstruct a blanked stretch of input whose shape it has seen:

| message | winner (one-hot) | graded top-16 |
|---|---|---|
| error, problem A | 0.638 | **0.143** |
| error, problem B | 0.763 | **0.141** |

Graded speech is 4.5× better than one-hot at plain reconstruction. Also
note the failed message costs *more* to send: the one-hot code is over
`K = 1024` cells per tap, the same as graded, and carries less.

### 5.1 Overlap is necessary but not sufficient

There is a second constraint, independent of the first, and it decides
what the layer above can represent at all.

A message assembled from **template identities** can only say *which
templates fired*. If each template binds an input to an answer at one
specific place, then no message built from those identities has a word
for *"this shape, at whatever height"*. The invariance is not degraded
in transmission — it is unrepresentable in that vocabulary. Restoring
graded overlap does not help, because the problem is the alphabet.

*Measured*: a second layer storing windows of curve, filling a gap the
training data never covered. Identical machinery; only what each tap of
the window says differs.

| what a tap says | gap, non-repeating signal | gap, strictly repeating signal |
|---|---|---|
| value, relative to a known reference tap | **0.033** | 0.102 |
| value, absolute | 3.221 | **0.063** |
| the winning template (one-hot) | 4.317 | 0.427 |
| graded top-16 | 2.783 | 1.594 |

Both value codes overlap identically, and on the non-repeating signal
one gets 0.033 and the other 3.221. The difference is only that the
relative code can express the shape independently of its height. The
repeating-signal column is the control that proves the mechanism is
sound: make the exact absolute pattern occur elsewhere in the data and
the absolute code becomes the best interface.

**Design consequence.** If a layer should learn structure that recurs
at different offsets, something in the message must be measured
relative to a reference the receiver also holds. Neither activation
interface can be made relative even in principle.

---

## 6. The commitments

1. **Similar inputs are written similarly.** Values are overlapping
   patches of cells. Generalisation is by overlap; if the code has none,
   nothing downstream recovers it.
2. **One vector holds the whole moment.** Inputs and answers are the
   same object. Prediction is filling in a blanked part of it.
3. **Matching is Pearson correlation.** Mean-centred, unit length, both
   sides — otherwise the loudest template wins rather than the best.
4. **Exactly one template learns**, by geodesic rotation, `θ = η·c`,
   bootstrap-adopted. Any softer rule blurs every template toward an
   average.
5. **Speech is graded and sparse.** Best few matches, magnitudes intact.
   Never the winner alone, never everything.
6. **The reader chooses the sharpness.** Stored codes are rich;
   recognition reads graded, generation hardens.

Two further rules for code-level layers, from earlier work, not
re-measured here: mid/upper dictionaries **learn** from the
per-position top-1 skeleton of their input window while **matching** on
the dense view (dense learning targets share a large common mass and
collapse template content — peakiness 0.23 → 0.53–0.72 when fixed); and
the upward message is never per-position normalised, because magnitude
*is* the matching signal (bare softmax collapsed matching to chance,
0.11 vs 0.75).

---

## 7. Choices, and what the alternative cost

### 7.1 Disjoint positions, not independent scatter

If two channels draw their positions independently, some cells collide.
One shared position puts a phantom peak into the **answer** row of
every template whose input patch covers it, and the top-1 read emits
that phantom confidently across a whole band of inputs.

*Measured*: 3 collisions in `D = 4096` → error 0.675 vs 0.160 disjoint;
2 collisions in `D = 8192` → 0.157 vs 0.107. These were the only
catastrophic errors the rig produced. **The single-layer drone has 31 of
its 256 motor cells sharing a position with a sensory cell** and reads
its motors out exactly this way — unfixed.

### 7.2 Everything else

| decision | alternative | why the alternative lost |
|---|---|---|
| overlapping patches | one cell per value | unseen values match nothing; 3.5× worse error on sparse data (0.576 → 0.163) |
| normalised matching | raw overlap | usage monopoly, entropy 0.32 |
| one template learns | top-5, or sampled learner | both blur every template; sampling ≡ graded learning in expectation |
| geodesic rotation | additive update + renormalise | rotation preserves both invariants exactly and costs a scale, a shift and a short add |
| score on the written part | bare dot product | vague templates outbid matching ones; worsens with capacity |
| contrast floor on that score | none | near-empty templates divide by ~0 and bid absurdly high |
| graded sparse speech | winner, or everything | one-hot has no overlap for the layer above; everything washes it out |

### 7.3 Choosing the blur width

There is a genuine optimum and it is not sharp. *Measured*, 60 training
pairs, only the fall-off radius changed (cells, of 96):

| half-width | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| error | 0.576 | 0.347 | **0.163** | 0.168 | 0.217 | 0.447 | 0.824 |

Too sharp and an unseen value means nothing. Too blurry and different
values stop being distinguishable — past ~16 cells the hypercolumn
collapses onto a handful of templates. Note this only shows up when the
data is thin: with 800 dense training pairs a one-cell code memorises
perfectly well, and the blur appears to cost nothing. Test the code's
generalisation where examples are sparse, or the measurement lies.

---

## 8. Known limits

**It answers near what it has seen.** An unseen input lands on top of
seen ones and retrieves their answer, degrading gracefully.

**One layer cannot continue a pattern through a gap.** Inside a hole in
the training data the answer is whatever was stored at the nearest rim —
*measured*: 0.817 in the hole against 0.819 for a rule that just holds
the rim value. This is structural, not a tuning problem: one winner can
emit only one stored answer, and a single input-answer pair carries no
information about which way the underlying function is going.

**Two layers can**, when the second layer's templates are *shapes* — a
window of several of the first layer's answers, so a template stores a
piece of the pattern rather than a point. *Measured* on the same gap:
0.033, against 0.817 for one layer and 0.149 for a fitted MLP on the
same data, out of identical machinery. The condition is §5.1: the
message reaching it must be one a shape can be recovered from.

---

## 9. Cost

Per input, with `K` templates, dimension `D`, and `A` active positions
(`A ≈ 40`, `D = 8192`, `A/D ≈ 0.5 %`):

- competition: one `K × A` gather and multiply-accumulate
- learning: one row scaled, one scalar shift, one `A`-element add
- memory: `K × D` floats for the templates

Nothing scales with `D` except storage. A 256-template, 8192-dimension
hypercolumn trains on 32 000 inputs in 3.4 s on CPU.
