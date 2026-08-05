"""Contrastive (two-phase) training for the hidden-permutation task.

Follow-up to permutation_iterative_completion.py, which showed the ensemble's
energy landscape has fake peaks: free search finds Y states the verifier
scores HIGHER than the true target on 100% of items, because training only
ever raised energy on valid pairs and never lowered it on invalid ones.

Fix under test — a negative phase: for every real pair [X, P(X)], also present
a corrupted pair [X, P(X')] (someone else's target: coupling broken, marginal
statistics intact) and apply the SAME learning rule with reversed rotation
(negative eta — the winner rotates AWAY, proportional to how much it wrongly
resonates).

Measured per neg_scale (0.0 = control, reproduces the one-phase model):
    ranking   : verification accuracy, true target vs 15 distractors (energy score)
    one_shot  : generation Jaccard/precision, single projection readout
    iterative : clamped re-presentation loop
    energy    : greedy swap search on the energy score
    hacked%   : fraction of items where search finds energy above the truth

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_contrastive/permutation_contrastive.py 0.0 0.5
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_iterative_completion"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_iterative_completion import (
    energy_search,
    iterative_completion,
    pair_energy_batch,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
N_DISTRACTORS = 15
N_RANK_TRIALS = 20


def _train_contrastive_node(args: tuple) -> np.ndarray:
    sources, targets, k, eta, neg_scale, epochs, seed = args
    rng = np.random.default_rng(seed)
    n = len(sources)
    positives = pp.make_full_patterns(sources, targets)
    w = pp.init_weights(k, positives.shape[1], rng)
    for _ in range(epochs):
        for idx in rng.permutation(n):
            w = pp.zenith_learn(w, positives[idx], eta)
            if neg_scale > 0.0:
                j = int(rng.integers(n - 1))
                j = j + 1 if j >= idx else j  # any target but idx's own
                neg = np.concatenate([sources[idx], targets[j]])
                w = pp.zenith_learn(w, neg, -eta * neg_scale)
    return w


def train_contrastive(sources, targets, m, k, eta, neg_scale, epochs, rng):
    seeds = [int(rng.integers(0, 2**32)) for _ in range(m)]
    args = [(sources, targets, k, eta, neg_scale, epochs, s) for s in seeds]
    print(f"Training neg_scale={neg_scale}: {m} nodes in parallel...")
    with ProcessPoolExecutor() as pool:
        return list(tqdm(pool.map(_train_contrastive_node, args), total=m, leave=False))


def evaluate(ensemble, train_sources, train_targets, test_sources, test_targets, rng):
    w_all = np.vstack(ensemble)
    n_test = len(test_sources)
    n_active = int(np.sum(test_targets[0]))

    # --- Verification: rank true target among mismatched targets by energy ---
    all_pairs = np.array(
        [np.concatenate([test_sources[i], test_targets[j]])
         for i in range(n_test) for j in range(n_test)]
    )
    e_grid = pair_energy_batch(w_all, all_pairs).reshape(n_test, n_test)
    correct = total = 0
    idx_all = np.arange(n_test)
    for i in range(n_test):
        others = idx_all[idx_all != i]
        for _ in range(N_RANK_TRIALS):
            cand = np.concatenate(([i], rng.choice(others, N_DISTRACTORS, replace=False)))
            correct += int(np.argmax(e_grid[i, cand]) == 0)
            total += 1
    ranking_acc = correct / total

    # --- Generation: one-shot, iterative, energy search ---
    methods = {m_: {"exact": 0, "jaccard": [], "precision": [], "recall": []}
               for m_ in ["one_shot", "iterative", "energy"]}
    hacked, gaps = 0, []
    for i in tqdm(range(n_test), desc="Generation", leave=False):
        x, true_y = test_sources[i], test_targets[i]
        recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
        y_one = pp.binarize_topk(recon[pp.PART_DIM:], n_active)
        pp._score(methods["one_shot"], y_one, true_y)

        y_iter, _ = iterative_completion(ensemble, x, n_active)
        pp._score(methods["iterative"], y_iter, true_y)

        y_en, e_final = energy_search(w_all, x, y_one)
        pp._score(methods["energy"], y_en, true_y)
        e_true = pair_energy_batch(w_all, np.concatenate([x, true_y])[None, :])[0]
        gaps.append(e_final - e_true)
        hacked += int(e_final > e_true + 1e-12)

    return {
        "ranking_acc": ranking_acc,
        "methods": {m_: {k_: (np.mean(v) if isinstance(v, list) else v / n_test)
                         for k_, v in st.items()} for m_, st in methods.items()},
        "hacked_frac": hacked / n_test,
        "mean_gap": float(np.mean(gaps)),
    }


if __name__ == "__main__":
    neg_scales = [float(a) for a in sys.argv[1:]] or [0.0, 0.5]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for neg_scale in neg_scales:
        rng = np.random.default_rng(pp.SEED)
        perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
        train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
        test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
        train_targets = pp.apply_permutation_batch(train_sources, perm)
        test_targets = pp.apply_permutation_batch(test_sources, perm)

        ensemble = train_contrastive(
            train_sources, train_targets, pp.DIRECT_NODE_COUNT,
            pp.DIRECT_TEMPLATE_COUNT, pp.ETA, neg_scale, pp.TRAINING_EPOCHS, rng,
        )
        res = evaluate(ensemble, train_sources, train_targets,
                       test_sources, test_targets, np.random.default_rng(pp.SEED + 100))

        m_ = res["methods"]
        row = (
            f"| {neg_scale} | {res['ranking_acc']:.2%} "
            f"| {m_['one_shot']['jaccard']:.4f} | {m_['iterative']['jaccard']:.4f} "
            f"| {m_['energy']['jaccard']:.4f} | {m_['energy']['precision']:.4f} "
            f"| {res['hacked_frac']:.0%} | {res['mean_gap']:+.4f} |"
        )
        rows.append(row)
        print("\nneg_scale | ranking | one_shot J | iterative J | energy J | energy P | hacked | gap")
        print(row)

    header = [
        "# Zenith — Contrastive Negative Phase (hidden permutation)",
        "",
        "Negatives: [X_i, P(X_j)] mismatched targets, same rule with reversed rotation.",
        "",
        "| neg_scale | ranking acc | one_shot J | iterative J | energy J | energy P | hacked% | search-true gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    report = OUT_DIR / "report.md"
    existing = report.read_text().splitlines() if report.exists() else header
    report.write_text("\n".join(existing + rows) + "\n")
    print(f"\nAppended to {report}")
