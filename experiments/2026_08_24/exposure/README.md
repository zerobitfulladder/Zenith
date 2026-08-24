# 2026-08-24 — `exposure/`: F5: error-driven exposure

Part of the day log [`../README.md`](../README.md).

## F5: error-driven exposure (boosting curriculum) — predictions (before running, 2026-08-24, night)

User proposal: no architecture feedback at all — the data feed is the
teacher. Live loop: the network answers each training batch (hard
readout, zero label, BEFORE the update); per-class error EMAs steer
batch composition (wrong -> that class appears more; right -> less;
floor keeps every class present; total exposure equal). Premise: what
is exposed gets allocated more and sharpens (expertise = finer tiling
of the practiced manifold). Class-neutrality caveat: L1-L3 parts are
shared across classes, so the sharpening should land at L4.

`run_exposure.py`: {dense, top1} x {balanced baseline, adaptive}, 3
epochs, adaptivity from epoch 2.

1. The curriculum DISCOVERS the confusion classes without being told:
   sampling weights concentrate on 4/9/7 (dense), 7/9/4/8 (top1).
2. Hard readout up modestly on both rigs (+0.5-1.5pp), driven by the
   boosted classes' errors dropping (some reallocation cost
   elsewhere allowed).
3. LOCUS TEST: probe L3 ~unchanged — exposure sharpens the memory,
   not the dictionaries (class-neutral parts don't benefit from
   class-targeted practice). If the probe moves, the premise
   "exposure sharpens representation" holds below L4 after all.
4. Risk: oscillation (error drops -> weight drops -> error returns);
   EMA damping should hold it.

### Outcome F5 (same night) — the curriculum finds the confusions by
### itself; gains where the code is lossy; the user's critique confirmed

`results/`: dense base .9030 / adapt .9058 (+0.3, jitter-level;
probe flat .967 — locus at the memory as predicted); top1 base .6920 /
adapt **.7194** (+2.7; probe also +0.9 — on a lossy code exposure
sharpens representation too; 4->9 confusion 274 -> 163, -40%). Error
EMAs discovered the confusion classes unsupervised (dense ranking:
4=.20, 8=.18, 9=.14 — the 4/9 axis exactly). No oscillation. The
dense ceiling is the user's point: positive-only exposure sharpens the
innocent but never relocates the guilty memory — hence F6.
