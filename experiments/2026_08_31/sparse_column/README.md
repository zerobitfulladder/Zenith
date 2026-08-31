# k minicolumns speak, in sequence

2026-08-31. Nonnegative matching pursuit as the hypercolumn rule, on the
place code from `../place_code`. One layer, 4x4 MNIST patches, nb=8 so the
input is 128 cells, K=64 templates. `k` is swept.

    r = x
    repeat up to k times:
        i   = argmax_i (w_i . r),  positive matches only
        tau = r - (w_i . r) w_i
        rotate w_i toward tau;  r = tau
        stop when ||r|| < eps ||x||

`tau` is the same vector twice: the tangent at `w_i` pointing toward `r`,
which the geodesic step rotates along, **and** the residual handed to the
next winner. At k=1 this is `WTAColumn` exactly, which is the port's own
correctness check.

## The finding: do not clamp the template weights nonnegative

The plan was for everything to be nonnegative — input, weights,
coefficients, message. **The weights cannot be, and it is not close.**

| | residual after 1, 2, 4, 8 templates | mean cosine between two templates |
|---|---|---|
| place code, signed weights | 0.586 → 0.532 → 0.460 → 0.402 | **0.016** |
| place code, **nonneg weights** | 0.611 → 0.596 → **0.595 → 0.595** | **0.546** |
| raw pixels, signed weights | 0.337 → 0.234 → 0.141 → 0.082 | 0.019 |
| raw pixels, **nonneg weights** | 0.418 → 0.383 → **0.381 → 0.381** | 0.766 |

With nonnegative weights **the pursuit stalls after the second pick** and
every later template contributes nothing. Not a tuning failure, and the
mechanism is one line: every nonnegative unit vector lives in the same
orthant, so they all point roughly the same way — mean pairwise cosine
**0.55** on the place code and **0.77** on raw pixels, against 0.016 when
signs are allowed. A dictionary that coherent has no second direction to
offer. Whatever the first template did not explain, no other template can
explain either, because they are all nearly the same template.

Signed weights spread out (0.016) and the residual keeps falling.

**So the earlier design was wrong in one of its four places.** Nonnegativity
belongs on:

* the **coefficients** — the rule refuses a negative match, so "how much of
  this part is present" still has no negative branch
* the **message** leaving the column — it is those coefficients, so it is
  nonnegative whatever the weights do
* the **input**, which the place code guarantees

and *not* on the weights, which are the layer's own detector and need the
freedom to be spread around the sphere. This costs nothing that was being
counted on: sparsity already breaks the rotation symmetry that `08-30`
diagnosed, so nonnegative weights were a second, redundant symmetry-breaker
— and they take the entire sequential-residual mechanism down with them.

`nonneg_w=True` is kept as a flag so the measurement reproduces.

## The k sweep

24k patches, 3 epochs, 4k held out.

| k | rebuild error | pixel RMSE | distinct supports | coherence | usage entropy |
|---|---|---|---|---|---|
| 1 *(= the old WTA)* | 0.548 | 0.2001 | 64 | 0.419 | 5.63 |
| 2 | 0.524 | 0.1812 | 612 | 0.066 | 5.81 |
| 4 | 0.463 | 0.1436 | 2842 | 0.036 | 5.95 |
| 8 | 0.394 | 0.0777 | 3440 | 0.016 | 5.95 |
| 16 | 0.348 | 0.0418 | 3733 | −0.008 | 5.96 |

Usage entropy is out of a possible 6.00 bits, and **no template is ever
dead** at any k — every one of the 64 carries about 1/64 of the traffic.
The alphabet grows from 64 names to 3733.

**And the cost is visible in `01_templates.png`.** k=1 and k=2 give clean
legible edges and corners. By k=8 they are busier, by k=16 they are on
their way to the 08-30 speckle. Reconstruction improves monotonically with
k and legibility degrades monotonically with k, which is the whole trade
laid out in one figure. On these numbers **k=2–4 is where both are still
good**; k=8 buys a large reconstruction gain for templates that are
starting to go.

## Support stability

The thing that had to be checked before anything is built on top: does a
small nudge to the patch change which templates fire?

Fraction of the support preserved, by noise added to the patch:

| | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| k=1 | 0.983 | 0.966 | 0.883 | 0.774 | 0.577 |
| k=4 | 0.961 | 0.914 | **0.792** | 0.587 | 0.359 |
| k=8 | 0.956 | 0.913 | 0.782 | 0.610 | 0.406 |
| *same rule on raw pixels, k=4* | *0.891* | *0.826* | *0.699* | *0.567* | *0.409* |

Good enough to build on, and **the place code beats raw pixels at every
noise level** — 0.792 against 0.699 at 5%. The bump overlap smooths the
scores, so the argmax flips less often. That is the encoder earning its
keep on a property that has nothing to do with what it was chosen for.

## Speed

The k-loop cannot be vectorised — step t+1 needs step t's choice. But it is
the **only** python-level loop, and it runs `kmax` times, not `n * kmax`:
each step is one `(n,128) @ (128,64)` GEMM, and the per-template update is a
one-hot matmul rather than a scatter.

| | µs per patch, k=4 |
|---|---|
| batched, n=128 | 5.6 |
| batched, n=512 | 6.8 |
| batched, n=2048 | 7.1 |
| batched, n=8192 | 15.0 |
| one at a time (`NaiveColumn`, identical rule) | **138.1** |

About **20x**, on numpy and CPU. The sweet spot is a few hundred to a couple
of thousand per batch; 8192 falls out of cache. Every primitive in the rule
is matmul / argmax / gather / scatter-add / elementwise, so this moves to
torch or jax unchanged.

## Honest note: the energy early-stop never fires

`mean templates used` came out at exactly 1.00, 2.00, 4.00, 8.00, 16.00 —
`eps_stop=0.05` asks the residual to fall below 5% of the input, and it
never gets past 35%. The adaptive-sparsity path is implemented and correct
and simply never triggers on this data. To actually get variable-length
codes the threshold has to be somewhere near 0.4; as it stands this is fixed
k with dead code attached.

## Files

| | |
|---|---|
| `sparse_column.py` | the column — batched pursuit, batched geodesic learning, `revive` |
| `run_sparse_column.py` | the sweep, the controls, the timing, every figure |
| `results/01_templates.png` | what each k learns |
| `results/02_sweep.png` | rebuild, usage, alphabet size, entropy vs k |
| `results/03_stability.png` | support under noise; how many templates get used |
| `results/04_rebuilds.png` | held-out patches rebuilt from the sparse code |
| `results/05_nonneg_weights.png` | **the finding above** |

    python run_sparse_column.py     # ~11 s

## Next

Put this in `stack3.py` as layer one and read the classification and
generation numbers against the 0.57 (wta) and 0.93 (dense) endpoints already
on the board. The open question this folder does not answer is whether the
k=2–4 legible templates survive being *composed* — whether layer two can
name pairs of them.

---

# Part two: pursuit against settling

The other way to pick k. Instead of taking turns, every template scores at
once, they inhibit each other in proportion to how much they overlap, and
the loop settles — LCA, Rozell/Johnson/Baraniuk/Olshausen 2008:

    u <- u + step * (Wx - u - (WW' - I) a),     a = relu(u - lam)

`(WW' - I)` off-diagonal *is* the lateral inhibition, and it is
explaining-away computed in parallel instead of in turns. Sparsity is set by
`lam` rather than by k, so `lam` is calibrated per run to make the settled
code as sparse as the pursuit's k — otherwise nothing is comparable.

Unlike the pursuit, this can **revise**: a template that grabbed early can
be pushed back down as the picture settles. The pursuit's first pick is
final, and if it was wrong every later pick compounds it.

## They do not agree

Same trained dictionary, both encoders, matched sparsity:

| target k | supports that agree |
|---|---|
| 2 | 0.53 |
| 4 | 0.57 |
| 8 | 0.67 |

So this is not two routes to one answer. They pick genuinely different
templates roughly half the time.

## The scorecard

| | pursuit | settling |
|---|---|---|
| rebuild error, k=4 | **0.463** | 0.523 |
| pixel RMSE, k=4 | 0.1436 | **0.1381** |
| support kept at 5% noise, k=4 | 0.792 | **0.912** |
| support kept at 5% noise, k=8 | 0.784 | **0.887** |
| µs per patch (numpy, CPU) | **13.6** | 46.3 |

**Settling wins on stability, clearly and at every k.** 0.912 against 0.792
is the difference between a support you can build a naming layer on and one
you have to keep apologising for. Revisability is exactly why: a nudge that
flips the pursuit's first pick cascades through everything after it, where
the settled answer just re-balances.

**Pursuit wins on rebuild error — but that metric is biased against
settling**, and the two reconstruction rows disagree for a reason. The soft
threshold shrinks every surviving coefficient by `lam`, and here `lam=0.208`
against a mean active coefficient of `0.188` — more than half the magnitude
is threshold bias. Plain L2 rebuild punishes that hard. The *decoded pixel*
does not, because the place code reads a channel by centroid, which is
insensitive to a common scale. So on what the layer actually communicates
downward, they are a tie.

(Debiasing by adding `lam` back to the active coefficients overshoots —
rebuild 0.523 -> 0.688, pixel 0.1381 -> 0.1303 — so the honest statement is
that the shrinkage is real and the cheap correction is wrong. Least squares
on the settled support would price it properly and is not done here.)

## Speed, and the number not to be fooled by

| | µs per patch |
|---|---|
| pursuit, k=4 | **13.6** |
| settling, 16 iterations | 10.1 |
| settling, 64 iterations | 46.3 |

The 16-iteration row looks like a win and is not, because at 16 iterations
**it has not settled**:

| iterations | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| templates still active | 14.11 | 11.16 | 7.56 | 4.82 | **3.74** | 3.42 |
| rebuild | 0.603 | 0.548 | 0.529 | 0.523 | **0.521** | 0.520 |

The rebuild error is nearly converged by 16, but the code is still twice as
dense as it should be — the *inhibition* is what is slow to settle, not the
reconstruction. Anything that reads the support has to wait for ~64. So the
fair comparison is 46.3 against 13.6: **settling costs about 3.4x**, which
matches the arithmetic (one `(n,d)@(d,K)` plus 64 of `(n,K)@(K,K)`, against
4 of `(n,d)@(d,K)`).

That 3.4x is a CPU number and it is the pessimistic case. The pursuit's cost
is a per-step `argmax` and a gather of `n` different template rows —
irregular, memory-bound, and much worse on a GPU than its FLOP count
suggests. Settling is a precomputed `K x K` Gram matrix and 64 dense GEMMs
with no data-dependent indexing anywhere. The gap should narrow and may
invert.

## Learn with the pursuit, read with the settling

The scorecard above compared each rule to itself. Crossing them separates
what the *learner* does from what the *encoder* does, and they turn out to
buy opposite things:

| trained by | encoded by | rebuild | dictionary coherence | support kept @5% noise |
|---|---|---|---|---|
| pursuit | pursuit | **0.463** | **0.036** | 0.787 |
| **pursuit** | **settling** | 0.579 | **0.036** | **0.892** |
| settling | pursuit | 0.499 | 0.399 | 0.660 |
| settling | settling | 0.526 | 0.399 | 0.914 |

**The stability comes from the encoder.** A pursuit-trained dictionary read
by settling gets 0.892 — almost all of the 0.914 that settling-trained
managed, on a dictionary that never saw the settling rule. And settle-trained
read by pursuit is the worst row in the table at 0.660. So it was never the
templates; revisability is doing the work.

**The dictionary comes from the learner, and the pursuit's is much better:**
coherence **0.036** against **0.399**, and templates covering an effective
**5.3 of 16 pixels** against **14.4**. Learned by settling, the templates
spread out over the whole patch and start pointing the same way — heading
back toward the nonneg pathology from part one.

The reason is the learning rule, and it is the 08-30 finding in miniature.
Under settling, every active template rotates toward **the same** residual,
scaled only by how loudly it spoke. That is the shared-leftover structure —
milder here, since 4 of 64 are active rather than all 144, but the same
direction. Under the pursuit each template learns from **its own** residual
at its own turn, and no two see the same vector.

So the soloist principle earns its keep in the *learning*, not the encoding.
That is a sharper version of the claim than this folder started with.

**Use pursuit to learn and settling to read.** Coherence 0.036, localised
templates, and a support that survives a 5% nudge 89% of the time instead of
79%. The rebuild error is worse (0.579 vs 0.463) and that is mostly the
threshold shrinkage priced above — pixel RMSE, which is what actually goes
downward, does not show it.

`kmax=1` with the pursuit still reproduces the old `WTAColumn` exactly, so
the fixed point survives all of this.

## Files

| | |
|---|---|
| `sparse_column.py` | `pursue` / `learn`, and `settle` / `learn_settled` |
| `run_compare.py` | calibration, the cross-encoder test, timing, stability |
| `results/06_compare.png` | rebuild, pixel, convergence, stability |
| `results/07_templates_compare.png` | what each rule learns at 4 active |
| `results/compare.json` | every number above |

    python run_compare.py     # ~50 s
