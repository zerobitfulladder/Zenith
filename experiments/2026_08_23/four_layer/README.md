# `four_layer/` — stacking the dictionaries: naive stack, top-k and graded messages, read modes

(2026-08-23, from the day log.) Scripts: `run_4layer.py` (naive stack), `../rig/run_4layer_topk.py` (all other arms, picked by
`GF_REPORT=topk|reluall|contrast|signed|sparselearn`), `run_dualmode_render.py` and
`run_sampled_generation.py` (read modes on saved weights, no training).
Results: `results/naive/`, `results/topk/`, `results/reluall/` (also holds the dual-mode and
sampled figures), `results/contrast/`, `results/signed/`, `results/sparselearn/`; each has
`weights.npz` and `report.md`.

## 4-layer hierarchy — predictions (before running, 2026-08-23)

`run_4layer.py` (MNIST): three stacked shared dictionaries + concat top.
L1: 4x4 px windows, stride 2, K=36 (grid 13x13 — the batch-2 layer).
L2: 3x3 windows over L1's grid, stride 2, K=64 (grid 6x6, sees 8x8 px).
L3: 3x3 windows over L2's grid, stride 1, K=100 (grid 4x4, sees 16x16 px).
Top: K=200 concat memory over [L3 code (1600 dims) ; lam=0.5 onehot].
All levels learn simultaneously, online, no feedback, no pooling — this
is the NAIVE-stack arm of the depth question. Weights saved to disk this
time (weights.npz) for later generation play.

1. The depth trajectory (the real question): probe(L1) ~0.94; the user
   predicts sag at L2/L3 (unsupervised re-encodings shed task info).
   Counter-story: locality+overlap may hold it. Committed guess: L2
   holds within a few points, L3 drops harder (16 active positions is a
   savage bottleneck).
2. Mid-level dictionaries render as recognizable compositions: L2 units
   = stroke combinations (curves/junctions), L3 units = digit parts.
3. Label-only generation survives three expansions (each with squelch),
   at some cost in crispness vs batch 2 — soft expansion compounds.
4. Retrieval stays >= 9/10; top readouts from the 1600-dim L3 code vs
   batch 2's hard 0.743 / top10 0.782: open, no commitment.
5. Full round-trip reconstruction (pixels -> L3 -> pixels) stays
   recognizable — the harshest compression test yet (three
   quantizations each way).

### Outcome (same day) — the user's sag prediction confirmed

- P1: **probe trajectory 0.943 -> 0.919 -> 0.855.** Each unlabeled
  re-encoding sheds class information (-2.4pp at L2, -6.4pp more at L3).
  The naive-stack arm of the depth question is settled: depth without
  pooling/invariance design costs, exactly as predicted. The pooling
  rescue arm remains untested.
- P2 partial fail: L3 "part" units render as barely-enlarged stroke
  elements, not digit chunks — composition is weak. Top-1-per-position
  sparse codes seem to let one sub-element dominate each higher template
  instead of binding multi-element configurations. Hierarchical binding
  needs something more (multi-winner upward codes, larger windows, or
  pooling).
- P3 ✓: label-only generation survives three expansions — all ten
  digits legible, though rougher than batch 2.
- P4: top readouts collapsed (hard 0.463, top10 0.553 vs batch 2's
  0.743/0.782) — 16 active positions is too brittle for constellation
  matching. Retrieval still 10/10.
- P5 partial: round-trip reconstruction pixels->L3->pixels keeps ~5/8
  digits recognizable; several break. Instance fidelity does not
  survive the 16-position bottleneck.

Net: **depth as naively stacked costs on every metric except retrieval
and generation legibility; batch 2 remains the champion rig.** For the
generation-detail goal, the levers are elsewhere: stride-1 dense
constellations, larger alphabets, texture/surface channel, higher-res
data. Weights saved (results/naive/weights.npz).

## Top-k reporting depth fix — predictions (before running, 2026-08-23)

`../rig/run_4layer_topk.py`: identical to run_4layer.py except the upward CODE
carries multiple winners per window — k_out = 3 (L1), 2 (L2), 1 (L3),
the user's sparsity gradient (denser low, sparser high, matching rising
selectivity in cortex). LEARNING stays top-1 everywhere (top-k learning
is refuted, 8 seeds, p<0.0001). One-variable comparison against the
naive stack (probes 0.943/0.919/0.855, hard 0.463, top10 0.553, weak
L3 parts, ~5/8 round-trips).

1. The sag shrinks: probe L2 > 0.919 and probe L3 >= 0.88 (binding
   starvation was the leak; richer sentences plug it).
2. L3 part units render as multi-element compositions, not lone strokes.
3. Top readouts improve over 0.463 / 0.553.
4. Round-trip reconstruction >= 6/8 recognizable.
5. L1 probe holds ~0.94 (top-3 code carries at least top-1's info).

Risk acknowledged: denser upward codes weaken the "sparsity forces
allocation" pressure at mid dictionaries (March lesson) — watch for
blurrier L2/L3 templates.

### Outcome (same day) — the sparsity gradient works; all predictions met

| | naive stack | top-k reporting |
|---|---|---|
| probe L1 | 0.943 | **0.9626** (new project high — beats stride-1's 0.9516) |
| probe L2 | 0.919 | **0.9482** |
| probe L3 | 0.855 | **0.8914** |
| hard / top10 | 0.463 / 0.553 | 0.626 / 0.679 |

- P1 ✓ sag shrinks at every level (L1->L3 deficit 8.8pp -> 7.1pp, all
  absolute values up). P5 exceeded: top-3 L1 reporting alone out-probes
  the stride-1 dense code at a quarter of the dimension — a second,
  cheaper densification axis.
- P2 ✓ binding is visibly repaired: L3 units now render as extended
  multi-element parts — hooks, elbows, long curves, T-junctions —
  instead of lone strokes.
- P3 ✓ readouts +16pp/+13pp, though still below the 2-layer rigs (the
  16-position top bottleneck remains the depth tax).
- P4 ~✓ round-trip ~6/8 recognizable (the 8 still breaks).
- Risk (blurred mid dictionaries) did not materialize.

Two validated densification axes now exist: WHERE (stride-1 reporting)
and HOW MANY (top-k reporting), both under strict top-1 learning.
Natural follow-ups: combine them; run the graded-reporting sweep
(relu-all vs softmax-T vs top-3) on this 4-layer testbed.

## Graded-reporting sweep — predictions (before running, 2026-08-23)

`GF_REPORT=reluall|contrast python ../rig/run_4layer_topk.py`, learning top-1
in all arms. Arms: top-3/2/1 (done: 0.9626/0.9482/0.8914, hard 0.626),
relu-all (density, no output inhibition), contrast relu(c - mean(c))
(graded margins over the field — the user's "same inhibition shapes the
output" principle, knob-free, magnitude-preserving; softmax rejected for
destroying magnitude).

1. If density is the whole story: relu-all >= contrast >= top-k probes.
2. If output competition matters (March sparsity lesson: dense shared
   tails blur top-layer allocation): contrast beats relu-all,
   especially on readouts/allocation. Committed bet: contrast wins or
   ties everywhere and beats relu-all on readouts.
3. Both graded arms >= top-k on probe L1.
4. Conceptual grounding recorded: one competition, two thresholds —
   plasticity reads the inhibited field steeply (top-1), output reads it
   gradually. Biology-consistent (graded firing, thresholded
   plasticity); both halves measured today (top-5 learning blurs 8/8;
   top-1 reporting starves binding).

### Outcome (same day) — density wins recognition, sparsity owns generation

| | top-3/2/1 | relu-all | contrast |
|---|---|---|---|
| probe L1/L2/L3 | .963/.948/.891 | .953/.960/.953 | .953/.961/.954 |
| hard / top10 | .626/.679 | **.860**/.770 | **.862**/.768 |
| generation | legible digits | mush | mush |

- Graded reporting REVERSES the depth sag (L2 out-probes L1 — first
  time depth gains) and produces the project's best classifier (hard
  0.86, beating the 2-layer detail rig's 0.80) — but destroys
  generation: dense memories expand to porridge; the squelch has no
  signal/noise split to gate on.
- Both prediction-bets failed instructively. The user's relu-all was
  best-recognizer AND worst-generator at once. The contrast bet
  (explicit output inhibition beats raw density) tied everywhere —
  because every downstream consumer mean-centers its input window, the
  field average is ALREADY subtracted where it matters; explicit output
  inhibition was redundant. The user's "same inhibition shapes output"
  principle is satisfied implicitly by the architecture.
- Standing synthesis — **VALIDATED, user's design, zero retraining**
  (`run_dualmode_render.py`, figures in results/reluall/
  dualmode_*.png): DUAL MODES on one set of weights. Recognition reads
  the field graded (keeps probes ~0.95 and hard 0.86); generation reads
  the SAME stored memories hardened — top-1 per position at the memory
  read and after every expansion. Result: the squelch-mode porridge
  becomes clean bold digits, and the round-trip is the best of the
  project (~7/8, including the first-ever recognizable 8). The dense
  memories' per-position peaks survived the blur, so read-time
  hardening recovers the crisp constellation without storing a second
  code. Final form of the day's pattern — one competition, mode-picked
  thresholds per consumer: plasticity top-1, recognition graded,
  generation hardened. (Biological gloss: perception vs imagery as
  dynamical modes of one circuit.)

## Sampled generation — predictions (before running, 2026-08-23)

`run_sampled_generation.py`: third read mode on the SAME reluall weights
(zero retraining) — per position, sample one template with prob
proportional to stored-coefficient^(1/T); resample after each expansion.
T->0 = argmax (deterministic prototype), T=1 = proportional. The user's
idea: profiles as distributions, sampled -> many variations per label.
(Softmax-at-training not needed for this and would cost the magnitude
signal; sampling normalizes at read time instead.)

1. Samples are distinct, complete, legible digits — real per-ask
   variation of the same label (generation-ladder level 3 achieved).
2. Temperature behaves as a style dial: low T ~ prototype with small
   variations; T=1 noticeably diverse but noisier; too-hot degrades.
3. Argmax column remains the crispest single image.

### Outcome (same day) — sampled variety achieved, with one lesson

- Sampling at ALL levels (first cut): real variety but frequent
  incoherent scribbles — independent draws from per-position marginals
  ignore the joint structure, and three levels of it compound
  (sampled_variety.png).
- **Sample-at-top + hardened rendering below: works.** Structural
  choice is stochastic (the 16 part-level slots), rendering is
  deterministic argmax — six distinct, coherent digits per label per
  ask (sampled_variety_top.png). Generation-ladder level 3 done, on
  stored weights, zero training. Occasional class-drift at T=0.5
  (a 2 wandering 7-ward) — the temperature trade-off, tunable.
- Recorded principle: sample where choices are structural, harden where
  rendering needs coherence — stochasticity belongs at the top of the
  downward pass, determinism at the bottom.

## Signed-output arm — predictions (before running, 2026-08-23)

`GF_REPORT=signed python ../rig/run_4layer_topk.py`: the full raw correlation
vector as the upward message, negatives included (the user's point:
anti-correlation is information). Compare against reluall
(.953/.960/.953, hard .860). Generation compared under the top-1
hardened read on saved weights.

1. Committed bet (antipodal coverage): signed ~= reluall within ±0.01
   on probes and hard — the dictionary already carries both polarities
   of common strokes, so negatives are re-expressed as other templates'
   positives. If signed clearly wins, K=36 lacks antipodes and the
   negative half was genuinely unique information.
2. Retrieval and top-1 generation comparable to reluall's dual-mode
   quality (hardened read clips negatives; stored negatives harmless).

### Outcome (same day) — both bets lost; relu keeps its job, with evidence

signed: probes .898/.919/.922 (vs reluall .953/.960/.953 — DOWN 3-5pp
at every level), hard .869 (~flat), retrieval 10/10.

Neither the tie (antipodal-redundancy bet) nor the signed win (unique-
information bet): including negatives actively HURT the representation.
Reading: the negative half is largely redundant (antipodal coverage)
AND it dilutes — doubling effective feature mass with mostly-redundant
entries worsens the probe at fixed data/regularization, and shifts the
mid-level dictionaries' competitive geometry for no gain. Positive-only
output is now an evidenced choice, not a default. (Opponent-channel
variant [relu(c); relu(-c)] remains untested and is now unmotivated.)

## Learn-sparse-view fix — predictions (before running, 2026-08-23)

Diagnosis refined by measurement first: the reluall mid-dictionary
collapse is NOT visible in template cosines (raw ~0.03, hardened-skeleton
~0.01 — orthogonality via graded tails) but IS visible in **profile
peakiness** (winner's share of its position's mass): reluall 0.26/0.23
(d2/d3) vs topk 0.37/0.41, and the hardened-render gallery stays blobby
— flat low-contrast templates whose renderable content is mutually
similar. `GF_REPORT=sparselearn`: communicate exactly like reluall;
L2/L3 LEARN from the per-position-top-1 skeleton of their window.

1. Peakiness d2/d3 rises to >= 0.35 (topk territory) and the L3 gallery
   becomes diverse in both render modes.
2. Probes and hard readout stay within ~2pp of reluall
   (.953/.960/.953, hard .860) — the message path is untouched.
3. Dual-mode generation stays crisp or improves (peaked templates render
   cleaner constellations).

### Outcome (same day) — all predictions exceeded; the consolidated unit

| | reluall | sparselearn |
|---|---|---|
| probes L1/L2/L3 | .953/.960/.953 | .953/.959/**.959** (best-ever L3; sag ~gone) |
| hard readout | .860 | **.8748** (new project record) |
| peakiness d2/d3 | .26/.23 | **.72/.53** (target was .35; topk was .37/.41) |
| L3 parts gallery | identical blobs | best parts ever: hooks, elbows, bars, curves |

Learning from the skeleton view cost nothing anywhere and improved
everything: information flow untouched (dense messages), differentiation
pressure restored (peaked, near-disjoint learning targets), and the
sharper templates even improved memory matching. The day's split-view
pattern is now installed at ALL THREE interfaces of the unit —
input (learn sparse view / match dense), output (learn top-1 / report
graded), memory (store graded / read hardened or sampled) — and every
installation was independently validated.

**The consolidated unit design** (new standard): normalized Pearson core,
argmax geodesic learning on skeleton views, dense relu communication
with magnitudes, dual-mode/sampled reads. Queue: GPU mini-batch trainer
verified against THIS configuration; stride-1 densification; 8-seed
validation; CelebA composition.
