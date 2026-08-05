# 2026-08-05 — Zenith: the hidden permutation arc (08-05/06)

Nine experiments on the hidden-permutation task, run with Claude. Question that
drove the whole arc: **the ensemble can recognize rule-obeying pairs almost
perfectly — can that verification ability be turned into generation?**
Answer: no, not in the global architecture — but changing *what nodes see*
(local receptive fields) nearly doubled generation anyway, for a different and
better reason.

All code and per-experiment reports live in this day folder, one folder per
experiment (`<name>/<name>.py`, report in `<name>/results/`); they build on the
March scripts in [`../2026_03_29/`](../2026_03_29/README.md). Config
throughout: PART_DIM=128, sparsity 0.10, 300 train / 100 test pairs, k=10
templates/node, 600 epochs, eta=0.03, seed 42. The `.venv` is broken (uv
python was removed); scripts run on system python via stub-guarded imports in
`permutation_relation_scoring.py`.

## Task

Y = P(X) for a fixed hidden bit-permutation. Ensemble of Zenith nodes (top-1
WTA spherical quantizer, `experiments/2026_03_29/zenith_node/learning6.py`) trained on concatenated
[X, Y] pairs. Recall: clamp X, zero Y, read out reconstruction of the Y half.
Metric: Jaccard vs true Y (chance ≈ 0.05).

## The experiments, in order

| # | experiment | script | headline result |
|---|---|---|---|
| 1 | scoring fix | [`permutation_relation_scoring.py`](permutation_relation_scoring/README.md) | the *norm* of the activation vector ("energy") carries all pair-validity signal: 96.2% ranking vs 15 distractors. March's below-chance result was the normalization deleting this norm. |
| 2 | iterative completion + free-energy search | [`permutation_iterative_completion.py`](permutation_iterative_completion/README.md) | both LOSE to one-shot readout (0.196 / 0.182 vs 0.336). Search finds fake energy peaks above truth on 100% of items — reward hacking. |
| 3 | mismatch (far) negatives | [`permutation_contrastive.py`](permutation_contrastive/README.md) | ranking → 99.4%, one-shot → 0.388; but near-field hacking stays 100%. Negatives generalize only as far as their distribution. |
| 4 | dream/CD unlearning | [`permutation_dreams.py`](permutation_dreams/README.md) | total failure — whack-a-mole, 100% hack rate all 6 rounds. Point negatives can't cover ~10^16 wrong states. |
| 5 | propose-and-verify | [`permutation_propose_verify.py`](permutation_propose_verify/README.md) | proposer fine (pool best 0.576), but truth never ranks #1 in any of 12 cells; N=1000 *worse* than N=100 (over-optimization). |
| 6 | near-miss (hard) negatives | [`permutation_hard_negatives.py`](permutation_hard_negatives/README.md) | truth_wins still 0%, but best global generator: one-shot 0.3995. |
| 7 | consensus decoding | [`permutation_consensus.py`](permutation_consensus/README.md) | verdict on the verifier: truth's energy sits at the **0.7th percentile** (median) of 1000 sampled neighbors — near-field is INVERTED, not noisy. Every vote/soft decode ≤ pool_mean ≈ one-shot. |
| 8 | iterative sampling (population refinement, no verifier) | [`permutation_soft_iteration.py`](permutation_soft_iteration/README.md) | one pass through the regeneration map costs 0.40 → 0.28, then monotone decay to the 0.19 hard-iteration floor; pool collapses unanimously (0.96) on the wrong answer. The feedback map itself is a contraction onto the blend. |
| 9 | **local receptive fields** | [`permutation_local_rf.py`](permutation_local_rf/README.md) | **one-shot 0.336 → 0.66**. Templates crystallize into true wire-facts (60.8% vs 0.4% null at F=64). Far-field verification needs big windows (F=16 → 29%, F=64 → 100%). Near-field improves (0.7% → ~22 pctl) but stays inverted; truth_wins still 0%. |

## Local receptive field detail (the breakthrough)

Each node sees only F dims — F/2 random X bits + F/2 random Y bits — with node
counts scaled for equal wire coverage (~5 nodes per wire). Plain training, no
negatives.

| F | M | wire% (null) | one-shot J | far-field | truth pctl med | argmax J |
|---|---|---|---|---|---|---|
| global | 100 | — | 0.336 (0.3995 w/ negatives) | 97.7% | 0.7% | 0.281 |
| 16 | 1280 | 4.4% (0.8%) | **0.6611** | 29.0% | 29.1% | 0.431 |
| 32 | 320 | 17.0% (0.8%) | 0.5879 | 62.0% | 7.1% | 0.416 |
| 64 | 80 | **60.8% (0.4%)** | 0.6411 | **100.0%** | 21.6% | 0.500 |

`wire%` = fraction of templates whose strongest X-bit and strongest Y-bit form
a true wire pair (null = same stat vs a fake permutation). Direct mechanistic
evidence of factorization: shrink the window and memories stop being blurred
episodes and become the rule's atoms.

## What the arc established

1. **Root cause of every global failure is one thing: blending.** 300 pairs
   into 10 slots/node ⇒ each template averages ~30 pairs. Generation reads the
   blend (0.34–0.40 ceiling), near-field energy *prefers* blend-shifted lures
   over the exact truth (inversion), and the feedback dynamics contract onto
   the blend (iteration always loses to one-shot). Four negative-training
   regimes moved none of it — storage arithmetic isn't trainable.
2. **Two kinds of generalization.** Averaging buys "distill the typical"
   (far-field recognition, solved at ~99–100%). It cannot buy "factor the rule
   into recombinable pieces" (generation). The permutation is 128 independent
   wire-facts; whole-pattern top-1 recall can't mix sub-parts of different
   memories. Local windows make composition happen by construction.
3. **Verification splits by scale.** Far-field wrongness is holistic — a
   mismatched pair looks locally innocent through a 16-bit peephole (29%),
   perfect through a 64-bit one (100%). Near-field exactness is not solved at
   any scale tried. Architecture lesson: **small windows generate, large
   windows verify** — the hypercolumn division of labor.
4. Reproduced from first principles on a toy: reward hacking of a learned
   objective, best-of-N over-optimization, and confident convergence of
   self-feedback onto generic (schema-like) answers.

## Open next steps (ranked)

1. **F=64 + near negatives** — best architecture meets best trainer; does
   0.64 push toward 0.70+?
2. **Scale M at F=64** (80 → 300+) — where does composition saturate?
3. **Two-scale ensemble end-to-end** — F=64 generates, global/large-F verifies
   far-field; measure the full propose-and-verify pipeline.
4. Near-field inversion at F=64 remains the standing open problem — wire-fact
   templates still let lures out-resonate truth; understand why before trying
   to train it away again.

---

# The experiments, one folder each

Each folder has its script, its generated report in `results/`, and a short
README. No separate writeup was kept per experiment; the arc above is the
writeup.

## `permutation_relation_scoring/` — the code's size is the signal

March's lookup on the normalised code scored at chance. Scoring the same
ensemble by the size of its activation instead gives 96.21% against 15
distractors (raw lookup 84.54%, template score 92.54%).

Full writeup: [`permutation_relation_scoring/README.md`](permutation_relation_scoring/README.md).

---

## `permutation_iterative_completion/` — feeding the answer back in

Re-presenting the guess (0.1957) and climbing the energy (0.1817) both lose to
the one-shot readout (0.3362); the search finds states the verifier likes more
than the truth on 100% of items.

Full writeup: [`permutation_iterative_completion/README.md`](permutation_iterative_completion/README.md).

---

## `permutation_contrastive/` — mismatched pairs as negatives

Unlearning on [X, someone else's P(X)] lifts ranking to 99.40% and one-shot to
0.3880, but the search still beats the truth on 100% of items.

Full writeup: [`permutation_contrastive/README.md`](permutation_contrastive/README.md).

---

## `permutation_dreams/` — the model's own fake peaks as negatives

Six rounds of unlearning the states the search finds; the search still beats
the truth on 100% of items every round. One-shot 0.3639.

Full writeup: [`permutation_dreams/README.md`](permutation_dreams/README.md).

---

## `permutation_propose_verify/` — sample candidates, let the energy pick

The candidate pool contains good answers (best 0.5757), but the energy never
ranks the true answer first in any of the 12 settings, and 1000 candidates
never pick better than 100.

Full writeup: [`permutation_propose_verify/README.md`](permutation_propose_verify/README.md).

---

## `permutation_hard_negatives/` — near-miss negatives

Negatives = the true pair with 1-3 bits swapped. Best one-shot of the global
models (0.3995), but the truth still never wins the ranking.

Full writeup: [`permutation_hard_negatives/README.md`](permutation_hard_negatives/README.md).

---

## `permutation_consensus/` — voting over the top candidates

The truth's energy sits at the 0.7th percentile (median) of 1000 nearby
candidates: the near field is inverted, not noisy. No vote beats the plain
pool average (0.3972) or one-shot (0.3995).

Full writeup: [`permutation_consensus/README.md`](permutation_consensus/README.md).

---

## `permutation_soft_iteration/` — refining a population without a verifier

Sample, regenerate, average, repeat: decode quality falls every round (0.2725
→ 0.1840 at T=0.5) while the pool grows unanimous.

Full writeup: [`permutation_soft_iteration/README.md`](permutation_soft_iteration/README.md).

---

## `permutation_local_rf/` — each hypercolumn sees only a window

Hypercolumns that see F bits instead of all 256: one-shot 0.66 at F=16 and
0.64 at F=64 against 0.336 global; at F=64, 60.8% of templates are true wire
pairs (null 0.4%).

Full writeup: [`permutation_local_rf/README.md`](permutation_local_rf/README.md).
