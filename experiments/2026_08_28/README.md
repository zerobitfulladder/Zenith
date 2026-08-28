# 2026-08-28 — reconstruction error becomes THE objective

Reframing (user, this morning): what this project does is at core online
dictionary learning, and its key, irreducible property — especially once
convolutional and deep — is the QUALITY OF THE RECONSTRUCTION. Until now
reconstruction was a sanity check measured out of the corner of the eye
(static3's roundtrip 0.651, hardened read, 50 images, never optimized).
From today it is the target.

Two standing decisions from the discussion:
- The geodesic learning is untouched. Matching and learning still operate on
  mean-centered, normalized windows.
- The per-window mean and contrast that centering deletes are carried as
  SIDE CHANNELS (pure bookkeeping at L1, re-applied at decode), so the
  reconstruction lives in real pixel space and pixel MSE is honest. Shape
  correlation (vs the centered/normalized input) is reported alongside.

## `recon_ladder/` — the reconstruction ladder (Exp 1)

A three-layer stack trained without labels, then scored by how well each layer's code rebuilds the image. From L1 0.934, from L2 0.895, from L3 0.855 shape correlation (the old instrument gave 0.651). Top-1 reads win near the bottom, graded reads win at L3. `run_recon3.py` here is also the shared module almost every other experiment of the day imports.

Full writeup: [`recon_ladder/README.md`](recon_ladder/README.md).

---

## `blend/` — how overlapping windows should be combined (Exp 2)

Plain average, confidence-weighted average, or let the most confident window write alone. Letting one window win loses at every layer (−0.09 at L3); confidence weighting is a small win (from-L1 0.936).

Full writeup: [`blend/README.md`](blend/README.md).

---

## `residual_read/` — residual coding at L1, pure read (Exp 3)

Extra read rounds: subtract what the winner explained, let a second template cover the rest. From-L1 goes 0.936 → 0.959 at six rounds, with no retraining.

Full writeup: [`residual_read/README.md`](residual_read/README.md).

---

## `residual_learning/` — residual learning, two shapes (Exp 4)

Learning toward the leftover. The sequential form (one template at a time) reaches 0.969 from L1 and keeps every template readable on its own; the parallel leave-one-out form reaches 0.989 on its own read but collapses under every other read and at depth. The leave-one-out form is kept only as a control.

Full writeup: [`residual_learning/README.md`](residual_learning/README.md).

---

## `residual_full_gpu/` — full stack on GPU, and the batch-size sweep (Exps 5 and 10)

Residual reads at L2/L3 lift the deep layers (L2 0.904, L3 0.857). Mini-batch training dulls the residual reads; the batch-size sweep shows batch size 2 matches CPU-online training (L1 0.969 / L2 0.908 / L3 0.868) in 602 s.

Full writeup: [`residual_full_gpu/README.md`](residual_full_gpu/README.md).

---

## `corrections/` — corrections across layers, pure read (Exp 6)

Each layer codes only the gap left by the layer above. L2 corrections add about 3 points and stop after two voices; the layered description wins at low storage, flat L1 coding wins at high storage.

Full writeup: [`corrections/README.md`](corrections/README.md).

---

## `k2span/` — is L2's ceiling its vocabulary size? (Exp 7)

More L2 templates raise L2's residual read (0.846 → 0.876) but blur its dense read and starve L3 (0.786 → 0.690).

Full writeup: [`k2span/README.md`](k2span/README.md).

---

## `fullseq_cpu/` — CPU-online full-stack residual learning (Exp 8)

The same rule that failed under mini-batch in Exp 5 works online: ladder L1 0.969 / L2 0.909 / L3 0.867.

Full writeup: [`fullseq_cpu/README.md`](fullseq_cpu/README.md).

---

## `labelgen/` — label-conditioned generation (Exp 9)

A label-bearing top on the L2 code: 0.9276 readout (project record at the time) and clean generated digits for all ten labels. The per-position (conv) arm produces mush, as predicted.

Full writeup: [`labelgen/README.md`](labelgen/README.md).

---

## `resolve_read/` — closed-form coefficients after greedy selection (Exp 11)

Re-solving the chosen templates' strengths together: free gain at L1 (0.969 → 0.972), nothing at L2/L3.

Full writeup: [`resolve_read/README.md`](resolve_read/README.md).

---

## `committee_learn/` — committee leave-one-out learning (Exp 12)

Sequential selection, exact blame among the chosen templates. Best learning rule of the day: 0.974 / 0.921 / 0.899; templates become compact, local stroke pieces.

Full writeup: [`committee_learn/README.md`](committee_learn/README.md).

---

## `two_pathways/` — form + motion pathways (Exp 13)

Separate form and motion tracks feeding one top. Better than the joint control on every axis; free-run advance 0.362 vs 0.000 for the joint control, which freezes.

Full writeup: [`two_pathways/README.md`](two_pathways/README.md).

---

## `joint_stack/` — single-track joint stack (Exp 14)

One track, current and previous frame together. At full length it either freezes (advance 0.000) or moves as mush; loses to two tracks.

Full writeup: [`joint_stack/README.md`](joint_stack/README.md).

---

## `antifreeze/` — the anti-freeze trio at read time (Exp 15)

Feeding back the clean code does nothing; the subtractive read lifts the two-track advance 0.438 → 0.672 at undiminished match 0.783; adaptation unfreezes hardest but costs match. Also holds `run_make_gifs.py` and the GIFs of these free-runs.

Full writeup: [`antifreeze/README.md`](antifreeze/README.md).

---

## `joint_cotrain/` — every layer learns at once (Exp 16)

All layers plastic together: no monopoly, about 6 points lost on teacher-forced prediction, freeze unchanged.

Full writeup: [`joint_cotrain/README.md`](joint_cotrain/README.md).

---

## `pole2track/` — deep two-track cart-pole swing-up (Exp 17)

The pole task parked at 0/100 on 08-26 is solved: 87/100 with both sensory and motor banks, reading the top-1 motor template rather than the mean.

Full writeup: [`pole2track/README.md`](pole2track/README.md).

---

## `drone/` — planar drone, unified and cascaded (Exps 18-19)

One 6-D top: 0/100. Splitting into an attitude loop and a position loop, plus a separate velocity channel, makes it fly (20/100); letting memory grow with the data climbs to 61/100. Gluing the command into the template (bound) loses: 0/100, and 3/100 with a top-1 read.

Full writeup: [`drone/README.md`](drone/README.md).

---

## `drone_continual/` — the continual drone (Exp 20)

One continuous life with a sliced read and fixed-size hypercolumns. No writeup was kept, and the saved record stops inside the oracle warm-up, so there is no result.

Full writeup: [`drone_continual/README.md`](drone_continual/README.md).

---

## `single_layer_drone/` — single-layer sparse-OR drone

One hypercolumn, one sparse vector holding both the sensed state and the oracle's commands. Peaks (0.80 in the smoke run, 0.89 with more templates) then always decays to zero. Not forgetting: the pupil's commands drift toward emergency responses because most of what it learns comes from bad states. Replay is the proposed fix; the RL rule waits behind it. `sl_drone.py` here is the drone rig other folders import.

Full writeup: [`single_layer_drone/README.md`](single_layer_drone/README.md).

---

## Day summary

DAY SUMMARY — reconstruction objective, first day. Ladder went
0.651 (yesterday's instrument) -> 0.855 deep / 0.969 from-L1.
Standing recipe: sequential-residual learning at L1 (R_TRAIN=4),
residual read at L1 (R~3-6), graded read + avg/conf blending deeper,
side channels for pixel-true output. Open queue, priority order:
(1) the expansion step (deep rungs frozen all day at ~0.85-0.89 —
now the whole gap); (2) L2/L3 residual learning at their own windows;
(3) L3 footprint / true mid-scale level; (4) GPU port if iteration
speed starts binding; (5) does the residual-trained L1 dictionary
change the classification probes? (untested today — reconstruction
objective only).
