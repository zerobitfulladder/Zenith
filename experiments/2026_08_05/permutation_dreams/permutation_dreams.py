"""Dream-phase (contrastive-divergence-style) training for the permutation task.

permutation_contrastive.py showed that mismatched-target negatives improve
verification and one-shot generation but do NOT remove the fake energy peaks:
free search still beats the true target on 100% of items, because on-manifold
lies never visit the off-manifold states the optimizer exploits.

This experiment samples negatives from the model's own behavior instead:

    repeat R rounds:
        1. WAKE  : train on positive pairs (+ current dream pool as negatives)
        2. DREAM : run the energy search on training sources with the CURRENT
                   model; collect the fake peaks it finds (states scored above
                   the true target) as the next round's negative pool

Unlearning uses the same rule with reversed rotation (negative eta), exactly
as in permutation_contrastive.py.

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_dreams/permutation_dreams.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_contrastive", "2026_08_05/permutation_iterative_completion"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_contrastive import evaluate
from permutation_iterative_completion import energy_search, pair_energy_batch

OUT_DIR = Path(__file__).resolve().parent / "results"

ROUNDS = 6
EPOCHS_PER_ROUND = pp.TRAINING_EPOCHS // ROUNDS   # same total budget as before
NEG_SCALE = 0.5
N_DREAM_SOURCES = 100                             # training sources dreamed per round
DREAM_MAX_SWEEPS = 10


def _train_round_node(args: tuple) -> np.ndarray:
    w, positives, negatives, eta, neg_scale, epochs, seed = args
    rng = np.random.default_rng(seed)
    n = len(positives)
    n_neg = len(negatives)
    for _ in range(epochs):
        for idx in rng.permutation(n):
            w = pp.zenith_learn(w, positives[idx], eta)
            if n_neg > 0:
                w = pp.zenith_learn(w, negatives[int(rng.integers(n_neg))], -eta * neg_scale)
    return w


def train_round(ensemble, positives, negatives, eta, neg_scale, epochs, rng):
    seeds = [int(rng.integers(0, 2**32)) for _ in range(len(ensemble))]
    args = [(w, positives, negatives, eta, neg_scale, epochs, s)
            for w, s in zip(ensemble, seeds)]
    with ProcessPoolExecutor() as pool:
        return list(pool.map(_train_round_node, args))


def collect_dreams(ensemble, sources, targets, rng):
    """Run the generation-time search on training sources; keep the fake peaks."""
    w_all = np.vstack(ensemble)
    n_active = int(np.sum(targets[0]))
    idx = rng.choice(len(sources), size=min(N_DREAM_SOURCES, len(sources)), replace=False)
    dreams = []
    hacked = 0
    for i in idx:
        x, true_y = sources[i], targets[i]
        recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
        y_init = pp.binarize_topk(recon[pp.PART_DIM:], n_active)
        y_dream, e_dream = energy_search(w_all, x, y_init, max_sweeps=DREAM_MAX_SWEEPS)
        e_true = pair_energy_batch(w_all, np.concatenate([x, true_y])[None, :])[0]
        if e_dream > e_true + 1e-12 and not np.array_equal(y_dream, true_y):
            dreams.append(np.concatenate([x, y_dream]))
            hacked += 1
    return np.array(dreams), hacked / len(idx)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pp.SEED)

    perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
    train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
    test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
    train_targets = pp.apply_permutation_batch(train_sources, perm)
    test_targets = pp.apply_permutation_batch(test_sources, perm)
    positives = pp.make_full_patterns(train_sources, train_targets)

    init_rng = np.random.default_rng(pp.SEED + 7)
    ensemble = [pp.init_weights(pp.DIRECT_TEMPLATE_COUNT, positives.shape[1], init_rng)
                for _ in range(pp.DIRECT_NODE_COUNT)]

    dreams = np.array([])
    hack_curve = []
    for r in range(ROUNDS):
        ensemble = train_round(ensemble, positives, dreams, pp.ETA, NEG_SCALE,
                               EPOCHS_PER_ROUND, rng)
        dreams, hack_rate = collect_dreams(ensemble, train_sources, train_targets,
                                           np.random.default_rng(pp.SEED + 1000 + r))
        hack_curve.append(hack_rate)
        print(f"round {r + 1}/{ROUNDS}: dreamed {len(dreams)} fake peaks "
              f"(train hack rate {hack_rate:.0%})", flush=True)

    res = evaluate(ensemble, train_sources, train_targets,
                   test_sources, test_targets, np.random.default_rng(pp.SEED + 100))
    m_ = res["methods"]

    lines = [
        "# Zenith — Dream-Phase Training (self-generated negatives)",
        "",
        f"Rounds={ROUNDS}, epochs/round={EPOCHS_PER_ROUND}, neg_scale={NEG_SCALE}, "
        f"dream sources/round={N_DREAM_SOURCES}.",
        "",
        f"Train-time hack rate per round: "
        + " -> ".join(f"{h:.0%}" for h in hack_curve),
        "",
        "| ranking acc | one_shot J | iterative J | energy J | energy P | hacked% | search-true gap |",
        "|---|---|---|---|---|---|---|",
        f"| {res['ranking_acc']:.2%} | {m_['one_shot']['jaccard']:.4f} "
        f"| {m_['iterative']['jaccard']:.4f} | {m_['energy']['jaccard']:.4f} "
        f"| {m_['energy']['precision']:.4f} | {res['hacked_frac']:.0%} "
        f"| {res['mean_gap']:+.4f} |",
        "",
        "Compare: one-phase control 0.336/0.196/0.182, hacked 100%, gap +0.98;",
        "mismatch-negatives 0.388/0.224/0.209, hacked 100%, gap +0.80.",
        "",
    ]
    print("\n".join(lines[6:9]))
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    print(f"Saved report to {OUT_DIR / 'report.md'}")
