# 2026-08-24 — `top_skeleton/`: Top-skeleton: does skeleton learning at the top memory help?

Part of the day log [`../README.md`](../README.md).

## Top-skeleton (archive) — predictions (before running, 2026-08-24)

Motivation, measured this morning on the saved s1 flagship weights:
pairwise template similarity is near-orthogonal at the skeleton-learned
levels (L2 mean |cos| 0.052, L3 0.034) but the top's 200 memories share
a common positive component (mean cos +0.115) — the L2 content-collapse
mechanism operating one level up, in mild form. The top is the one
dictionary above pixels that still learns from DENSE graded codes.

`run_top_skeleton.py`: twin-top controlled comparison. One lower stack
(L1-L3, skeleton learning as standard) trained once; TWO top
dictionaries learn side by side from the identical code stream:

- **dense arm** (control): rotates toward [center_norm(code) ; lam*onehot],
  exactly the current standard;
- **skeleton arm**: the L3 code map gets per-position top-1 over its 100
  channels first (at most 100 nonzeros of 10,000), then the same
  center/norm/concat path. Bootstrap adopts skeleton views too.

Recognition queries stay DENSE for both arms (crisp templates, dense
evidence — the L2/L3 lesson). Lower-stack probes are untouched by
construction and not re-measured.

1. THE MECHANISM CHECK: memory mean pairwise cos drops from ~+0.11
   (dense arm) to below +0.05 in the skeleton arm (code-half cosine
   toward ~0). If it does not drop, the shared mass is not coming from
   dense rotation targets and the story needs rethinking.
2. Memories stay individuals: within-class code-half similarity drops,
   and the top-rehearsed gallery is visibly more varied in the skeleton
   arm (the wide-3 homogeneity mechanism, treated). Top-position
   peakiness (per-position winner share of the stored code) clearly
   higher in the skeleton arm.
3. Generation equal or crisper: the hardened read is nearly a no-op on
   an already-committed memory. Label-only retrieval stays 10/10
   consistent in both arms.
4. THE RISK: hard readout (nearest-memory, dense query) within +/-0.02
   of the dense arm. The contrast-floor precedent says deleting
   weak-but-real runner-up signal can cost nearest-neighbor matching a
   few points; the L2/L3 precedent says crisp-templates-dense-evidence
   improves matching. Genuinely uncertain — this is the trade being
   measured.
5. NOT expected to be fixed: nonstationarity (memories adopted against
   still-wet dictionaries) — different disease, different queued fix.

Stakes: if P1-P3 hold at acceptable P4, the unit becomes fully uniform —
every layer above pixels learns strict and speaks graded, no exception
for the top — and commitment #5 sharpens to "richness lives in the live
code, not the stored row." The motor rig inherits whichever version wins.

### Outcome (same day) — mechanism confirmed, proposal REJECTED; the
### shared mass at the top is functional

`results/`, train 13.8s (both tops in one pass):

| arm | hard | consistent | mean cos | code cos | within-class | top peak |
|---|---|---|---|---|---|---|
| dense | .8932 | 10/10 | +0.189 | +0.220 | +0.379 | 0.109 |
| skel  | .7462 | 10/10 | +0.046 | +0.031 | +0.077 | 0.899 |

- P1 ✓✓ (mechanism): skeleton targets erase the shared component
  (+0.189 -> +0.046; peakiness 0.109 -> 0.899). The dense-target
  accumulation story is CORRECT.
- P2 ✓ on metrics (within-class similarity 0.379 -> 0.077) — but the
  diversity buys nothing visible: dense top-rehearsed memories were
  already distinct legible digits (MNIST's homogeneity is mild).
- P3 ✗ half: consistency 10/10 both, but skeleton generation is
  slightly WORSE (3, 5, 8 degrade; 8 renders 2-like). Hardened read of
  a dense memory beats storing the skeleton outright.
- P4 ✗✗ decisively: hard readout .8932 -> .7462, MINUS 14.7pp — seven
  times outside the +/-2pp band. The contrast-floor precedent scaled up.
- P5 ✓: median/low-rehearsal memories are stroke-jumbles in BOTH arms —
  the nonstationarity disease reproduced on MNIST, not face-specific.

READING — the question was wrong, productively. The top's shared mass
is not the L2 disease one level up: at L2 the shared component of dense
targets is task-IRRELEVANT stroke-family haze, so absorbing it flattens
parts. At the top the shared component is mostly WITHIN-CLASS (+0.379)
prototype mass — the common part of many 3s IS 3-ness, exactly what
nearest-memory recognition runs on (and what makes dense memories
render clean under the hardened read). Deleting it cost recognition
14.7pp and helped nothing. General law, now with three measured
regimes: the shared component of learning targets accumulates in every
template; whether that is poison or nutrition depends on whether it
carries the layer's task signal. L1: shared mass ~0 after centering ->
dense learning fine. L2/L3: large + irrelevant -> collapse, skeleton
required. Top: large + class-relevant -> prototype formation, dense
required. Commitment #5 ("stored codes are rich") is now EVIDENCED,
not just designed; commitment #3 stays scoped to code-level layers
below the memory.

Caveat on cross-run numbers: dense-arm hard .8932 vs the flagship's
.849 reflects this run's GPU reluall eval pipeline vs the CPU one —
only the within-run twin comparison is claimed.

### Diagnostic addendum — the shared ingredient, measured per layer

Norm of the mean of each layer's centered+normalized learning targets
(0 = directions spread symmetrically, 1 = all targets identical); this
mean is exactly what rotation-learning accumulates, everything else
cancels. Saved s1 flagship, 1000 images:

| targets | L1 dense | L2 dense | L2 skel | L3 dense | L3 skel | L4 dense | L4 skel |
|---|---|---|---|---|---|---|---|
| shared size | 0.046 | 0.103 | 0.050 | 0.117 | 0.056 | **0.409** | 0.168 |

Two kinds of sharing distinguished: pairwise overlap/crowding (L1
templates: mean |cos| 0.348 but SIGNED mean -0.015 — 36 templates
crowded into a 15-dim space, overlaps symmetric, cancel in learning)
vs one common ingredient in all targets (the mean above). L1 has much
of the first, almost none of the second — that is why it learns dense
safely. L4's common ingredient is 4x L2/L3's and is class signal;
L2/L3's is smaller but meaningless, and collapse there also rides the
per-family haze that the skeleton removes beyond the global mean.
