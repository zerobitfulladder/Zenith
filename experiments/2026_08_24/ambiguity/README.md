# 2026-08-24 — `ambiguity/`: F8: what should ambiguity do to plasticity?

Part of the day log [`../README.md`](../README.md).

## F8: what should ambiguity do to plasticity? — predictions (before running, 2026-08-25)

One question, three opposed answers. Today argmax picks exactly one
winner and steps it by `theta = eta * c_max` whether the position had a
single obvious owner or a forty-way tie: the ambiguity is computed and
thrown away. `run_ambiguity.py` on the 8x8/K1=512 dense champion
(2 epochs, SEED 42), plasticity rule applied at L1 (GF_SCOPE=all also
available):

- **base** — argmax, step from `c_max`. Today's champion.
- **samp t** — winner SAMPLED from softmax(C/t) via Gumbel-max, step
  from the sampled winner's correlation. t in {0.01, 0.02, 0.05, 0.1}.
  The user's proposal. At L1's median margin 0.038 these give the
  winner roughly 98% / 87% / 68% / 59% of the runner-up's odds — from
  "argmax with rare deferrals" to "quite random".
- **margin** — argmax winner, step from `c_max - c_2nd`. The opposite
  answer: ambiguity SUPPRESSES the update. Untested.
- **margin_x** — argmax winner, step from `c_max * (c_max - c_2nd)`,
  keeping the energy term alongside decisiveness.

Methodological control: every arm's step sizes are rescaled per batch
to the same mean as `c_max`, so the arms differ in HOW learning is
distributed across samples, not in how much total learning happens.
Without this the margin arms would merely be "learn ~12x slower" and
lose for a trivial reason.

Standing evidence: Commitment #2 of the consolidated unit says argmax
is the only non-blurring rule, refuted twice — top-5 learning lost in
8/8 seeds (p<0.0001) and softmax-sampled selection blurred identically
because each template's expected update is its softmax-weighted blend.
BUT both refutations ran on the UNNORMALIZED rig, where sampling was
doing duty as the anti-monopoly mechanism (hot). Sampling has never
been swept at low temperature on the normalized champion, where
bootstrap + normalization already keep usage diverse. That corner is
what arm 2 closes.

1. samp degrades monotonically with t (hard, peakiness, generation
   blobbiness); t=0.01 lands within jitter of base. The expectation
   argument leaves it little room — any t > 0 moves the expected update
   toward the graded blend in proportion to t.
2. Sampling does NOT de-clone: L1 crowd flat or UP, top1-2 margin not
   improved. Near-twins receive the same patch cloud in expectation and
   converge together, where argmax at least partitions it between them.
   (This directly opposes the mechanism hoped for — that random winners
   would push the layer to differentiate.)
3. Usage entropy stays ~flat and dead stays 0 across all arms — the
   diversity that sampling buys on a no-norm rig is already bought here
   by normalization + bootstrap, so it is a benefit with nothing to do.
4. **margin is the live arm**: peakiness up, generation crisper, hard
   flat to +1pp. Called at even odds. Mechanism: the near-tie at 0.90
   vs 0.89 is currently the LOUDEST teacher in the rig while the
   decisive 0.40 vs 0.10 barely teaches — plausibly backwards.
5. margin_x ~ margin, marginally better if the energy term matters.

(No user prediction recorded for this one — asked, not stated.)

### Outcome (2026-08-25) — sampling refuted with its mechanism caught in
### the act; the margin arm was a NULL RUN until fixed, then it improved
### the dictionary reproducibly and the accuracy still did not follow

`results/`. Jitter first, because it governs every reading
below: base ran .9090 / .9108 / .9020 on three IDENTICAL configs (GPU
atomics reorder the scatter-add float sums), so single-run hard-readout
deltas under ~1pp mean nothing on this rig.

**Arm 2 — sampled winners (scope l1, single seed):**

| arm | hard | peak2 | L1 crowd | clones% | top1-2 margin | usage H | dead |
|---|---|---|---|---|---|---|---|
| base | .9108 | .231 | .257 | 0.00 | .038 | .989 | 0 |
| samp t=0.01 | .9050 | .206 | .256 | 0.01 | .036 | .992 | 0 |
| samp t=0.02 | .9088 | .185 | .259 | 0.03 | .031 | .996 | 0 |
| samp t=0.05 | .9046 | .173 | .283 | 0.19 | .019 | 1.000 | 0 |
| samp t=0.1 | .8954 | .209 | .321 | 0.38 | .015 | 1.000 | 0 |

- P1 CONFIRMED: hard declines with t (t=0.1 costs 1.5pp); t=0.01 sits
  inside the jitter band. The expectation argument holds on the
  NORMALIZED rig too, closing the corner the original refutations left
  open (they ran unnormalized, with sampling doing anti-monopoly duty).
- **P2 CONFIRMED, emphatically, and it is the finding.** Sampling does
  not de-clone — it MANUFACTURES clones. Crowd .257 -> .321, clone
  pairs 0.00% -> 0.38%, and the top1-2 margin more than HALVES
  (.038 -> .015). The homogenization mechanism is exactly as argued:
  near-twins each receive the whole shared patch cloud in expectation
  and converge onto one point, where argmax at least partitions it
  between them. peak2 falls .231 -> .173 as the blur propagates to L2.
  The hoped-for mechanism (random winners pressure the layer to
  differentiate) runs precisely backwards.
- P3 CONFIRMED: usage entropy rises .989 -> 1.000 and dead stays 0 in
  every arm — but base was already .989 with 0 dead. Sampling's one
  genuine benefit has nothing left to fix here.
- Generation: softer, fatter strokes and a more mottled ground. Visible,
  not dramatic — the quantitative collapse signature is the real
  evidence, not the gallery.

**Arm 3 as first written — a NULL RUN, logged as a methodology lesson.**
`margin`/`margin_x` came back inert (hard .9094/.9088, every structural
metric flat). Cause found by instrumenting: `Dict.update` feeds `cvals`
ONLY into `Csum` -> theta, the step SIZE, never the target direction —
and `theta = eta*Csum` is clipped at THETA_CAP=0.3 for **94.9% of
template updates** (median eta*Csum = 1.336, 4.5x the cap). The rule
modulated a quantity that is discarded. **Standing consequence: at L1
the step size is effectively a constant 0.3 rad per template per batch,
ETA1 is not a live parameter at B=128, and any future rule that works
by modulating cvals at L1 is dead on arrival.**

**Arm 3 done correctly — weight the target BLEND, not the step.**
`update_blend`: each target contributes to its template's mean in
proportion to its weight; theta stays exactly base, so the arms differ
only in WHERE a template moves. Five seeds (training-order permutation),
paired t-tests vs base:

| metric | base | dir_cmax (control) | dir_margin | dir_marginx |
|---|---|---|---|---|
| hard | .8990 | .8983 (p=.81) | **.9031 (+0.41pp, 4W/0L/1T, p=.19)** | .9003 (p=.50) |
| peak2 | .2186 | .2132 (p=.03, down) | **.2524 (+15%, 5/5, p=.0002)** | .2518 (5/5, p=.0006) |
| L1 crowd | .2506 | .2520 (p=.02) | **.2318 (5/5, p<.0001)** | .2328 (5/5, p<.0001) |
| top1-2 margin | .0392 | .0382 (p=.09) | **.0470 (+20%, 5/5, p=.0001)** | .0464 (5/5, p=.0002) |
| usage H | .9866 | .9892 | .9686 (5/5, p<.0001) | .9712 (5/5) |
| dead | 0 | 0 | 0 | 0 |

- The **dir_cmax control is null on everything** — weighting the blend
  by match STRENGTH does nothing. The effect belongs specifically to
  DECISIVENESS. Clean isolation.
- The dictionary genuinely improves, and reproducibly: palette less
  crowded, winner margin +20%, downstream profiles +15% peakier, all
  5/5 seeds at p<=0.0002, zero dead templates. Usage becomes slightly
  less uniform (.987 -> .969) — expected, since templates that only ever
  won ambiguously now earn less mass.
- **And it does not convert.** Hard readout +0.41pp, 4 wins / 1 tie /
  0 losses but t=1.59, p=0.19. Suggestive, not established.

**SYNTHESIS — the law, now three refutations deep.** Improving the L1
dictionary's geometric organization does not improve the lookup:
(1) residual learning de-cloned the 512 palette perfectly at zero probe
cost, sparse speech still got worse (.8030); (2) the 8x8 dimensionality
repair closed most of the sparse gap but left ~5pp untouched by any
readout variant; (3) now margin-weighted plasticity de-crowds the
palette and widens margins 20% across 5/5 seeds for +0.4pp of nothing.
The lookup is not limited by how well-separated the dictionary is. This
is the strongest support yet for the standing theory — dictionary by
observation, memory and generation in the concat top, **task skill in a
learned readout**, which remains unbuilt and is now the only lever with
an unexplored ceiling.

**Adoption call:** keep `dir_margin` as the default L1 plasticity rule
anyway. It is free (identical step size, ~1s/epoch after replacing the
partition with mask-and-max), it improves the project's collapse
detector by 15%, it de-crowds the palette, and it trends positive on
accuracy. A strictly better-organized dictionary at no cost.
Applied at L1 only; peak3 unchanged as expected. Untested: scope=all,
and whether 10 seeds would move the +0.41pp over the line.
