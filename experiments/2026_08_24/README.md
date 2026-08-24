# 2026-08-24 — sparse speech, the top memory, and feedback

Continuation of the log in `../2026_08_23/README.md`
(the consolidated-unit rebuild, through the wide-3 faces showcase).
Rig of record: the stride-1 MNIST flagship (4x4s1/36 -> 3x3s2/64 ->
3x3s1/100 -> concat top 200, lam=0.5), GPU f32 mini-batch trainer.

The shared 8x8 rig module `run_rich_palette_8x8.py` lives in
[`rich_palette_8x8/`](rich_palette_8x8/README.md); most experiments below import it.
The last experiment (`ambiguity/`) was run on 2026-08-25 but was logged here.

---

## `top_skeleton/` — should the top memory learn from skeleton codes too?

The top's 200 memories share a common positive component (mean cos +0.115).
Two top layers were trained side by side on the same lower stack: one learning
from dense codes (standard), one from per-position top-1 codes. The skeleton arm
erased the shared part (+0.189 -> +0.046) but the hard readout fell from .8932 to
.7462. The shared part at the top is mostly within-class "3-ness", which is what
nearest-memory recognition runs on: dense learning stays at the top.

Full writeup: [`top_skeleton/README.md`](top_skeleton/README.md).

---

## `rich_palette/` — does a bigger L1 palette make sparse output work? (4x4 windows)

K1 in {36, 512} x L1 output in {dense, top3, top1}. The bet reversed: sparse
output got worse as the palette grew (dense-top1 hard gap 7.3pp -> 17.2pp),
because a 4x4 window has only 15 dimensions and 512 templates become near-ties
(top1-2 margin 0.111 -> 0.017). Dense output at K1=512 set a hard-readout record,
.9066.

Full writeup: [`rich_palette/README.md`](rich_palette/README.md).

---

## `rich_palette_8x8/` — the same question with 8x8 windows

With 63 usable dimensions the dense-top1 gap at K1=512 shrank from 17.2pp to
5.0pp, probes were unhurt by sparse output, top1 generation was clean, and dense
at 512 set a new record, .9094. The remaining ~5pp looked like near-clone
speakers. This folder also holds the shared 8x8 rig module and the L1 palette
figures.

Full writeup: [`rich_palette_8x8/README.md`](rich_palette_8x8/README.md).

---

## `completion_readout/` — classify by letting memories vote

Instead of copying the nearest memory's label, memories vote their label weighted
by match. All four predictions lost: raw voting cost 5-10pp everywhere, sharpened
voting only tied the top-1 readout (dense .9080). On a sharp matcher the winner's
match is all the signal there is; the sparse arms' ~5pp deficit is not recoverable
at the readout.

Full writeup: [`completion_readout/README.md`](completion_readout/README.md).

---

## `allsparse/` — sparse output at every layer

Sparse output at L1, L2 and L3. Probes held within ~1-3pp, but the hard readout
compounded: all-layer top3 .7596 (vs .8642 with sparse L1 only), top1 with rich
palettes .5982. Generation stayed clean in every arm, so recognition by lookup
and generation came apart completely.

Full writeup: [`allsparse/README.md`](allsparse/README.md).

---

## `residual/` — residual competition in the output

The winner speaks, subtracts what it explained, and the others compete over the
rest (up to 3 speakers). Only predictions were written up; the run's report gives
hard .8502 with residual output at L1 only and .7898 at all layers.

Full writeup: [`residual/README.md`](residual/README.md).

---

## `resid_learning/` — residual competition in learning

Runner-up templates learn from what the winner left over. The near-duplicate
templates vanished (pairs above cos 0.8: ~0.01%), but the margin only rose
0.038 -> 0.052 and residual output got worse (.8030 vs .8502). Near-ties come
from finely tiling a continuous stroke space, not from duplicates. The day's
sparse-output line closes here: dense graded output stays.

Full writeup: [`resid_learning/README.md`](resid_learning/README.md).

---

## `wide3level/` — are 3 wide layers enough for MNIST?

For the probes yes (.966, same as 4 layers); for the lookup almost (.8972 at
K2=512, -1.2pp); generation got softer. Sparse output at 3 layers (.7616) cost
the same as at 4 (.7596): what matters is whether the code stored in the memory
is dense. The 4-layer 8x8/512 dense rig stays champion.

Full writeup: [`wide3level/README.md`](wide3level/README.md).

---

## `feedback_program/` — top-down feedback on the spatial rig (F1-F3)

F1, label as a small channel at L2/L3: free on dense, catastrophic on top1
(.7268 -> .4646), because 8x8 parts carry no class. F2, class gain on the top
message: inert on dense, mildly harmful on top1. F3, inference searchlight
(re-score close calls by comparing rendered candidates with the pixels): inert on
dense, +9.2pp on the ambiguous subset of the top1 rig. Feedback pays in
proportion to what the forward pass threw away.

Full writeup: [`feedback_program/README.md`](feedback_program/README.md).

---

## `sample_commit/` — F4: sample a candidate, verify it by rendering, let it learn

On close calls, sample one of the top-3 memories, keep it only if its rendering
explains the input as well as the winner's, and let it learn the instance. Dense
rig: inert (.9112 both). Top1 rig: +0.5pp unsupervised, +1.8pp with the label
check. Real but modest; same law as the searchlight.

Full writeup: [`sample_commit/README.md`](sample_commit/README.md).

---

## `exposure/` — F5: show the classes the network gets wrong more often

Per-class error averages steer which classes appear in training batches. The
curriculum found the 4/9 confusion by itself. Dense .9030 -> .9058 (jitter
level), top1 .6920 -> .7194 with 4->9 confusions down 40%.

Full writeup: [`exposure/README.md`](exposure/README.md).

---

## `lvq/` — F6: corrective memory updates

When the answer is wrong, the best true-class memory steps toward the instance and
the wrong winner steps away. Dense .9132 -> .9182 (best dense number of the day),
top1 .7414 -> .7554 with 4->9 thefts cut 55%. Attraction alone hurt the top1 rig
(.7138). Generation and label retrieval stayed intact.

Full writeup: [`lvq/README.md`](lvq/README.md).

---

## `lam_sweep/` — F7 / F7b: how strong should the label half be?

Hard readout for lam 0.25 / 0.5 / 0.75 / 1.0: .9036 / .9102 / .9092 / .8956.
Too small and classes mix, too large and it steals norm from the matching half;
lam=0.5 stays. F7b trained at lam=1.0 and tried four label readers: every one got
worse than at lam=0.5, the graded ones most.

Full writeup: [`lam_sweep/README.md`](lam_sweep/README.md).

---

## `ambiguity/` — F8: what should a near-tie do to learning? (run 2026-08-25)

Sampling the winner from a softmax made near-duplicate templates (margin
.038 -> .015 at t=0.1) and lowered accuracy. Weighting each learning target by
how decisive the win was improved the L1 palette over 5 seeds (margin +20%,
downstream peakiness +15%) but hard readout only +0.41pp (p=0.19). Adopted as the
default L1 rule anyway, since it is free.

Full writeup: [`ambiguity/README.md`](ambiguity/README.md).
