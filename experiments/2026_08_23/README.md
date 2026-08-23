# 2026-08-23 — from top-down feedback to the consolidated unit

One long session (it ran into the small hours of 08-24). It starts by testing top-down gain
feedback, closes that line, rebuilds the rig from patches, and ends with the settled unit
([`rig/ConsolidatedUnit.md`](rig/ConsolidatedUnit.md)) trained on the GPU and used on CelebA faces.
The code the experiments share (`gain_feedback.py`, `run_4layer_topk.py`, `run_gpu_minibatch.py`) is
in [`rig/`](rig/README.md).

---

## `gamma_sweep/` — top-down gain feedback, Variants A and B

Does a label at the top shape the lower layer through feedback alone? Variant A multiplies L1's
input by a gain field projected down from the class template. It does not work: accuracy falls as
the gain grows (0.70 -> 0.51 at gamma=2), and the downward projections collapse to one shared
blob; a contrast source only softens this. Variant B puts the gain on the winner's learning step
instead. Perception is protected (top-1 correlation 0.784 -> 0.767), generation stays
digit-shaped, and with the contrast source the confusable-pair margins and the L1 probe rise a
little (0.8708 -> 0.8774), on one seed.

Full writeup: [`gamma_sweep/README.md`](gamma_sweep/README.md).

---

## `ab_test/` — does the Variant B effect replicate?

Fashion-MNIST, 8 seeds, {top-1, top-5 learning} x {feedback off, on}. The effect did not
replicate (pair margin +0.0024, 5/8 seeds, p=0.085). Feedback reliably spreads template usage
and costs a little fit; top-5 learning is reliably harmful on its own (probe -0.018, 0/8).

Full writeup: [`ab_test/README.md`](ab_test/README.md).

---

## `concat_generation/` — the label as evidence in a free top layer

The top layer competes freely over [L1 code ; lam x label]. Label-only queries retrieve the right
class 100% of the time and render readable digits; classification without the label reaches
0.727, above the wired class-template pipeline's 0.699.

Full writeup: [`concat_generation/README.md`](concat_generation/README.md).

---

## `warmup_test/` — feedback switched on only after the templates settle

Warm-up is no better than always-on or off (best: pair margin +0.0026, 6/8, p=0.058).
Training-time feedback on this rig is closed.

Full writeup: [`warmup_test/README.md`](warmup_test/README.md).

---

## `rebuild/` — 49 patch Hypercolumns + concat top, then the capacity sweep

L1 becomes a 7x7 grid of 4x4-patch Hypercolumns. Templates are strokes, label-only generation is
compositional for the first time, and the code probes 0.9170 (whole-image rig 0.871). Looking up
the nearest top-layer template reads far worse than the probe; batch-1 fixes and growing the top
layer 8x (best readout 0.672 -> 0.764) do not close the gap: prototype lookup saturates in the
mid-0.7s.

Full writeup: [`rebuild/README.md`](rebuild/README.md).

---

## `batch2/` — one shared stroke dictionary, overlapping windows

One 36-template dictionary swept over 169 overlapping positions. Probe 0.9466 (validated over
8 seeds: 0.9460 ± 0.0034), hard readout 0.743, smooth generation, 80/80 label-only retrievals.
On Fashion-MNIST: probe 0.8550, garments drawn as outlines.

Full writeup: [`batch2/README.md`](batch2/README.md).

---

## `four_layer/` — stacking dictionaries, and what the messages between them should be

A naive 3-dictionary stack loses information at each level (probe 0.943 -> 0.919 -> 0.855).
Sending several winners up (top-3/2/1) shrinks the loss; sending graded values (relu of all)
turns it around (.953/.960/.953, hard readout 0.86) but spoils generation, which is recovered by
reading the same weights hardened (dual mode) or sampled at the top. Signed messages hurt
(probes down 3-5 points). Learning the upper dictionaries from the top-1 skeleton of their input
gives the consolidated unit: .953/.959/.959, hard readout 0.8748.

Full writeup: [`four_layer/README.md`](four_layer/README.md).

---

## `detail/` — reporting at stride 1

The batch-2 rig with a stride-1 code (learning still at stride 2): much smoother generation,
hard readout 0.8008, probe 0.9516 (single seed).

Full writeup: [`detail/README.md`](detail/README.md).

---

## `unnormalized_l1/` — can L1 drop mean-centring and unit length?

Softmax-sampled learning keeps usage spread without normalisation (entropy 0.99) but blurs the
templates, and softmax messages kill the hard readout (0.11-0.14). Argmax learning without
normalisation brings the monopoly back (entropy 0.318). Only the normalised unit is both sharp and
diverse.

Full writeup: [`unnormalized_l1/README.md`](unnormalized_l1/README.md).

---

## `gpu_minibatch/` — the consolidated unit on the GPU

The mini-batch trainer matches the online CPU run (probes within ±0.010, hard .8738 vs .8748) at
114x the speed (10.5 s). At stride 1: probes .954/.963/.9658, rising with depth, and the best
generation so far, in 27 s.

Full writeup: [`gpu_minibatch/README.md`](gpu_minibatch/README.md).

---

## `celeba_faces/` — CelebA faces and composition

The stride-1 GPU rig on 39,910 48x48 faces with 40 attributes. After three retrieval fixes
(including a birth-noise bug), attribute portraits are 10/10 distinct real faces, and adding a
mustache direction to a female memory draws a woman with a mustache, which the data never
contains.

Full writeup: [`celeba_faces/README.md`](celeba_faces/README.md).

---

## `celeba_big/` — bigger faces run, zero-count attribute pairs

Bigger dictionaries and 1000 memories: portraits 10/10 distinct and attribute-legible; two
combinations that never occur in the data compose cleanly, and bald-on-woman fails because
baldness has to remove structure.

Full writeup: [`celeba_big/README.md`](celeba_big/README.md).

---

## `pyramid/` — 8 levels with a luminance channel

~1.46M parameters. Levels settle bottom-up, so the early stop never fired; the upper levels were
still moving and their Male probes sag (0.878 -> 0.759). Level-5 units are shaded face parts, but
the round trip through all seven levels fails.

Full writeup: [`pyramid/README.md`](pyramid/README.md).

---

## `inpaint/` — filling a masked region from a retrieved memory

On the saved stride-1 MNIST weights, no training: 8/8 correct retrievals with the label, 6/8 clean
fills; 6/8 correct from the fragment alone.

Full writeup: [`inpaint/README.md`](inpaint/README.md).

---

## `wide3/` — wide-shallow faces rig and showcase

Three wide levels (64/256/256). A float32 trainer made it 6x faster. Male probe at L2 0.886 beats
the pyramid's best level; round trips are shaded faces. Attribute probes, low-rehearsal memories
and morphs are in the showcase.

Full writeup: [`wide3/README.md`](wide3/README.md).
