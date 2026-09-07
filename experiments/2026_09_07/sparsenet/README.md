# Olshausen & Field 1996 on 8x8 MNIST patches

Paper 31 in `~/Breadcrumbs/pdfs` -- *Emergence of simple-cell receptive field
properties by learning a sparse code for natural images*, Nature 381:607-609.
Their cost, their settling, their learning rule, on our patches, to see what
their bases look like next to our templates.

    E = SUM_xy [ I(x,y) - SUM_i a_i phi_i(x,y) ]^2  +  lambda SUM_i S(a_i/sigma)   (2-4)
    adot_i = b_i - SUM_j C_ij a_j - (lambda/sigma) S'(a_i/sigma)                   (5)
    dphi_i = eta < a_i [ I - Ihat ] >                                              (6)

`sparsenet.py` is the paper as written. `unitnorm.py` is the same objective on
our conventions -- unit-length patches, unit-length templates, rotation-only
learning -- plus two extra per-template terms that did not work.

2000 MNIST images, 8x8 patches sampled 24 per image, DC removed, a contrast
gate dropping the blank ones (92% of sites survive whitened, 84% raw), 1000
updates of 100 presentations.

## What was found

**1. The paper as written kills most of its own templates.** 46 of 64 (and 109
of 128) end with vector length exactly 0.000 and never fire. The gain rule in
their Fig. 4 caption is a death spiral for an under-used template: low
coefficient variance shrinks its length, which shrinks its drive phi.I, so it
is used less still -- and eq. (6) hands a template with no coefficient exactly
zero update, so death is absorbing. The 17 survivors then fire on ~90% of
patches each. The code is dense.

**2. Their Fig. 4d hides this completely.** The pooled coefficient histogram
reads kurtosis 18.8 for that arm and looks convincingly sparse, because 46
spikes at zero plus 17 broad tents pools to something heavy-tailed. Per
template the kurtosis is 2.6. Always plot the per-template histograms; the
pooled one cannot distinguish a sparse code from a dead one.
(`firing_*.png` is the honest figure, `basis_*.png` panel 3 the misleading one.)

**3. S is the lever, not lambda and not K.** Their text says the three forms of
S they tried "all yield qualitatively similar results". On MNIST that is false
at the per-template level. At matched reconstruction (~8%):

| S            | residual | active/64 | per-template kurtosis | exact zeros |
|--------------|----------|-----------|-----------------------|-------------|
| log(1+x^2)   |  8.2%    | 51.7      |   2.3                 |  0%         |
| \|x\|        |  7.6%    | 13.8      | 103.9                 | 76%         |

log(1+x^2) never reaches zero, so every template keeps a shoulder of small
coefficients; |x|, solved by its exact proximal step (soft-thresholding), puts
real zeros in the code. |x| also gives more localised templates (18.1 vs 23.3
effective pixels of energy) and visibly cleaner oriented filters. Sweeping
lambda from 0.14 to 4 and doubling K to 128 both leave per-template kurtosis
pinned near 2 under log(1+x^2).

**4. The templates are general-purpose filters, not digit parts, and that is
the result rather than a failure.** Reconstruction alone cannot pick them:
W^T W x is rotation-invariant, so any rotation of a basis reconstructs
identically -- which is why the PCA panel of the same patches gives
checkerboards (their Fig. 1, reproduced here on MNIST). The sparseness term is
the entire reason a particular set of axes gets chosen. Reconstruction fixes
the subspace; sparseness fixes the axes. A template only becomes a digit *part*
when it must explain the patch alone, which this objective never asks.

**5. Unit-length templates remove the death mode structurally.** With
||phi_i|| = 1 the length is not a free parameter, so nothing can shrink out of
the competition; learning is the reconstruction gradient projected tangential
to the sphere. 0 dead templates in every arm, with no re-seeding anywhere.
Re-seeding dead templates does work but breeds near-duplicates (mean coherence
0.028 -> 0.137, max pair 0.91), so it is not used.

**6. Unit-length INPUTS matter more than expected.** Whitened MNIST patch
contrast spans 4.9x from the 10th to the 90th percentile, and a *random* unit
direction already shows excess kurtosis 0.77 on unnormalised patches -- free
sparseness that is contrast heterogeneity, not structure. Normalising takes it
to -0.27. It also fixes lambda: the error term scales as ||I||^2 and the
sparseness term as ||I||, so on unnormalised patches the *effective* lambda
varies ~5x across the dataset and every per-template histogram is a mixture
over that range. Unit-length patches make lambda mean one thing everywhere,
and make the coefficients cosines -- the same quantity our own unit competes
on, so the comparison is finally apples to apples.

**7. Adding sparsity to the LEARNING rule buys nothing (negative result).**
Their eq. (6) carries no sparseness term, and not by oversight: at the
equilibrium of eq. (5) the envelope theorem makes the reconstruction gradient
the whole gradient, since S(a_i) depends on phi only through a. A per-template
*lifetime* term was tried anyway -- minimise E[S(p_i/sig_i)] over the raw
projection p_i = phi_i.I, with sig_i a running RMS (without it the template
rotates to the lowest-variance direction and goes quiet instead of getting
sparse) -- together with a magnitude-decorrelation term penalising templates
for firing together. At matched residual, both lose to the plain rule:

| arm                    | residual | per-template kurt | coherence | pairwise MI |
|------------------------|----------|-------------------|-----------|-------------|
| random dictionary      |  40.2%   |   3.9             |  --       | 1.60 m-bits |
| recon (paper, unit)    |   6.3%   |  18.3             | 0.120     | 6.72 m-bits |
| + sparsity beta 0.3    |   6.9%   |  15.7             | 0.073     | 7.10 m-bits |
| + sparsity beta 1.0    |   9.7%   |  15.6             | 0.069     | 16.7 m-bits |
| + independence gam 0.3 |   6.7%   |  16.9             | 0.077     | 6.93 m-bits |
| + both                 |   7.9%   |  19.1             | 0.055     | 10.1 m-bits |

**8. Independence is not a target you can chase on its own.** A random
unit-norm dictionary has the LOWEST pairwise coefficient MI of anything here --
four times more independent than the learned code -- and reconstructs at 40%
error. Generic directions cannot co-occur with structure, so MI is minimised by
not learning. It is only readable at matched reconstruction. This is why the
objective has to be `minimise SUM_i H(a_i) subject to low reconstruction
error`: the constraint carries the whole thing.

**9. Coherence and coefficient independence are different quantities.** The
gamma term cut mean template coherence 0.120 -> 0.055 while coefficient MI went
UP. Pushing templates apart geometrically did not make their coefficients
independent. The likely cause is a fixable mistake: the penalty reads the RAW
projections, but the dependence lives in the SETTLED coefficients, which are
shaped by the lateral inhibition of eq. (5). Raw projections were chosen
because a settled coefficient is zero most of the time and that gating is what
kills templates -- so the fix is to penalise co-firing on the settled code and
keep a small raw-projection term as an anti-death floor only.

## Files

    sparsenet.py    the paper as written; arms {raw,white} x K {64,128} x S {log,abs}
    unitnorm.py     unit-length patches and templates, rotation-only, +beta/+gamma terms
    rebuild.py      rebuild whole digits from templates learned on raw patches
    per_patch.py    how many templates one patch uses (loads rebuild.py's templates)
    results/
      basis_*.png     learned templates | PCA of the same patches | pooled histogram
      firing_*.png    every template above its OWN coefficient histogram (the honest one)
      usage_*.png     per-template fire rate and lifetime kurtosis
      phi_*.npy       the templates            (K x 64)
      coeff_*.npy     their firing             (8000 patches x K)
      summary.json    lambda sweeps, curves, per-template stats
      as_written/     the first run, before the death spiral was diagnosed
      unitnorm/       the rotation-only rig
      rebuild/        rebuild.py and per_patch.py: digits rebuilt, per-patch usage

## Next

- Independence penalty on the settled code rather than the raw projections (9).
- Matching pursuit inference -- each template explains alone, in sequence --
  with the same rotation learning, to test whether the soloist form reaches
  this sparseness without the joint explanation.
- The comparison this was all for: these templates against our winner-take-all
  unit on the identical unit-length patches.
