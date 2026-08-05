"""Scoring-variant follow-up to permutation_relation.py.

Hypothesis under test: activation-space NN scored *below chance* because the
candidate activation code is L2-normalized before the cosine lookup. The code's
NORM carries the "how well do my templates explain this pair" signal (coupling
compatibility); dividing it out leaves only direction, and all stored training
codes share a dominant direction (they are all coupling-consistent), so
degraded/false pairs look MORE typical than novel true pairs after
normalization.

Variants compared on the identical ensemble + data (same seeds as
permutation_relation.py):

    raw_nn    : max cosine of candidate pair to training pairs      (control)
    act_cos   : max cosine of NORMALIZED activation to stored codes (control,
                original method — expect at/below chance)
    act_dot   : max dot of UNNORMALIZED activation to stored unit codes
                ( = ||act|| * act_cos — the one-line fix)
    act_norm  : ||act|| alone, no nearest-neighbor lookup at all
    template  : mean per-node top-1 |correlation|                   (control)

Prediction: act_cos <= chance ; act_norm ~ template ; act_dot in between.

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_relation_scoring/permutation_relation_scoring.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The computation needs only numpy. tqdm/matplotlib are stubbed if absent so
# this also runs outside the project venv (which is currently broken).
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    import sys
    import types

    def tqdm(it, **_kw):
        return it

    _tq = types.ModuleType("tqdm")
    _tq.tqdm = tqdm
    sys.modules["tqdm"] = _tq

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    _mpl = types.ModuleType("matplotlib")
    _mpl.use = lambda *a, **k: None
    _plt = types.ModuleType("matplotlib.pyplot")
    _gs = types.ModuleType("matplotlib.gridspec")
    _gs.GridSpec = object
    _mpl.pyplot = _plt
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt
    sys.modules["matplotlib.gridspec"] = _gs

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_03_29/permutation_relation",):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation as pr

OUT_DIR = Path(__file__).resolve().parent / "results"
METHODS = ["raw_nn", "act_cos", "act_dot", "act_norm", "template"]


def build_all_candidate_scores(ensemble, train_pairs, test_sources, test_targets):
    """Precompute all five scores for every (source i, target j) combination."""
    n_test = len(test_sources)
    w_all = np.vstack(ensemble)                      # (M*k, pair_dim)
    m = len(ensemble)
    k = ensemble[0].shape[0]

    # All candidate pairs [X_i, Y_j], centered + normalized like pr._normalize
    pairs = np.array(
        [pr.make_pair(test_sources[i], test_targets[j])
         for i in range(n_test) for j in range(n_test)]
    )
    pairs_hat = pr._normalize_rows(pairs)             # (n_test^2, pair_dim)

    train_pair_norms = pr._normalize_rows(train_pairs)

    # Stored training activation codes, normalized as in the original script
    train_acts = train_pair_norms @ w_all.T           # (N_train, M*k)
    train_act_norms = train_acts / np.maximum(
        np.linalg.norm(train_acts, axis=1, keepdims=True), pr.EPS
    )

    acts = pairs_hat @ w_all.T                        # (n_test^2, M*k)
    act_norms = np.linalg.norm(acts, axis=1)
    acts_hat = acts / np.maximum(act_norms[:, None], pr.EPS)

    scores = {
        "raw_nn":   np.max(pairs_hat @ train_pair_norms.T, axis=1),
        "act_cos":  np.max(acts_hat @ train_act_norms.T, axis=1),
        "act_dot":  np.max(acts @ train_act_norms.T, axis=1),
        "act_norm": act_norms,
        "template": np.mean(np.max(np.abs(acts.reshape(-1, m, k)), axis=2), axis=1),
    }
    return {name: s.reshape(n_test, n_test) for name, s in scores.items()}


def evaluate(score_grids, n_test, distractor_counts, n_trials, rng):
    """Same trial structure and rng usage as pr.evaluate_relation_ranking."""
    results = {
        d: {m: {"correct": 0, "margins": []} for m in METHODS}
        for d in distractor_counts
    }
    test_indices = np.arange(n_test)

    for d in distractor_counts:
        total = 0
        for i in tqdm(range(n_test), desc=f"Distractors={d}", leave=False):
            other_idx = test_indices[test_indices != i]
            for _ in range(n_trials):
                distractor_idx = rng.choice(other_idx, size=d, replace=False)
                cand_j = np.concatenate(([i], distractor_idx))
                for m in METHODS:
                    s = score_grids[m][i, cand_j]
                    results[d][m]["correct"] += int(np.argmax(s) == 0)
                    results[d][m]["margins"].append(float(s[0] - np.max(s[1:])))
                total += 1
        for m in METHODS:
            st = results[d][m]
            st["accuracy"] = st["correct"] / total
            st["avg_margin"] = float(np.mean(st["margins"]))
    return results


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pr.SEED)

    perm = pr.make_hidden_permutation(pr.PART_DIM, rng)
    train_sources = pr.make_sparse_vectors(pr.N_TRAIN, pr.PART_DIM, pr.SPARSITY, rng)
    test_sources = pr.make_sparse_vectors(pr.N_TEST, pr.PART_DIM, pr.SPARSITY, rng)
    train_pairs, _ = pr.make_positive_pairs(train_sources, perm)
    _, test_targets = pr.make_positive_pairs(test_sources, perm)

    ensemble = pr.train(
        train_pairs, pr.NODE_COUNT, pr.TEMPLATE_COUNT, pr.ETA, pr.TRAINING_EPOCHS, rng
    )

    print("Precomputing candidate scores for all (X_i, Y_j) combinations...")
    grids = build_all_candidate_scores(ensemble, train_pairs, test_sources, test_targets)

    results = evaluate(
        grids, len(test_sources), pr.DISTRACTOR_COUNTS, pr.N_CANDIDATE_TRIALS,
        np.random.default_rng(pr.SEED + 100),
    )

    lines = [
        "# Zenith — Permutation Relation, Scoring Variants",
        "",
        "Same data, seeds, and trained ensemble as `permutation_relation.py`.",
        "Only the scoring of the activation code differs between variants.",
        "",
        "| Distractors | Chance | " + " | ".join(METHODS) + " |",
        "|---|---|" + "---|" * len(METHODS),
    ]
    for d in pr.DISTRACTOR_COUNTS:
        chance = 1.0 / (d + 1)
        row = [f"| {d} | {chance:.2%} "]
        for m in METHODS:
            row.append(f"| {results[d][m]['accuracy']:.2%} ")
        lines.append("".join(row) + "|")
        print(f"\ndistractors={d}  chance={chance:.2%}")
        for m in METHODS:
            st = results[d][m]
            print(f"  {m:9s} acc={st['accuracy']:7.2%}  margin={st['avg_margin']:+.4f}")

    lines += [
        "",
        "`act_cos` is the original activation-space NN (normalized cosine).",
        "`act_dot` keeps the activation's magnitude in the NN lookup.",
        "`act_norm` is the magnitude alone — no lookup.",
        "",
    ]
    (OUT_DIR / "scoring_variants.md").write_text("\n".join(lines))
    print(f"\nSaved report to {OUT_DIR / 'scoring_variants.md'}")
