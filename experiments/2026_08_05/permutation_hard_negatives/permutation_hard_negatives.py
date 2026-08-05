"""Near-miss (hard) negatives for the permutation verifier.

propose_verify showed the verifier's failure is in the NEAR field: with the
true target planted in the candidate pool it wins the ranking 0% of the time —
lures (truth with a few bits swapped toward template resonance) always score
higher. Far negatives (mismatched targets) fixed far-field discrimination
only; self-generated dreams were too sparse to matter.

This experiment trains with negatives at the blind distance: the item's OWN
true pair with s random Y-bit swaps (s in 1..3), same reversed-rotation rule.

Modes:
    near  : all negatives are near-miss swaps
    mixed : 50% near-miss, 50% mismatched target (far + near curriculum)

Metrics per mode: far ranking acc, one-shot J, free-search hacked%/gap, and
propose-and-verify cells (T=0.5/N=1000 high-coverage, T=1.0/N=100) with
picked J, pool_best J, and truth_wins.

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_hard_negatives/permutation_hard_negatives.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_contrastive", "2026_08_05/permutation_iterative_completion", "2026_08_05/permutation_propose_verify"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_contrastive import evaluate
from permutation_iterative_completion import pair_energy_batch
from permutation_propose_verify import gumbel_topk_samples, jaccard_rows

OUT_DIR = Path(__file__).resolve().parent / "results"

NEG_SCALE = 0.5
MODES = ["near", "mixed"]
PV_CELLS = [(0.5, 1000), (1.0, 100)]
MAX_SWAPS = 3


def _swap_bits(y: np.ndarray, n_swaps: int, rng) -> np.ndarray:
    out = y.copy()
    on = np.nonzero(out)[0]
    off = np.nonzero(out == 0)[0]
    out[rng.choice(on, n_swaps, replace=False)] = 0.0
    out[rng.choice(off, n_swaps, replace=False)] = 1.0
    return out


def _train_hard_node(args: tuple) -> np.ndarray:
    sources, targets, mode, k, eta, neg_scale, epochs, seed = args
    rng = np.random.default_rng(seed)
    n = len(sources)
    positives = pp.make_full_patterns(sources, targets)
    w = pp.init_weights(k, positives.shape[1], rng)
    for _ in range(epochs):
        for idx in rng.permutation(n):
            w = pp.zenith_learn(w, positives[idx], eta)
            if mode == "mixed" and rng.random() < 0.5:
                j = int(rng.integers(n - 1))
                j = j + 1 if j >= idx else j
                y_neg = targets[j]
            else:
                y_neg = _swap_bits(targets[idx], int(rng.integers(1, MAX_SWAPS + 1)), rng)
            w = pp.zenith_learn(w, np.concatenate([sources[idx], y_neg]), -eta * neg_scale)
    return w


def train_hard(sources, targets, mode, m, k, eta, neg_scale, epochs, rng):
    seeds = [int(rng.integers(0, 2**32)) for _ in range(m)]
    args = [(sources, targets, mode, k, eta, neg_scale, epochs, s) for s in seeds]
    print(f"Training mode={mode}: {m} nodes in parallel...", flush=True)
    with ProcessPoolExecutor() as pool:
        return list(tqdm(pool.map(_train_hard_node, args), total=m, leave=False))


def propose_verify_eval(ensemble, test_sources, test_targets, temperature, n_pool):
    w_all = np.vstack(ensemble)
    n_test = len(test_sources)
    n_active = int(np.sum(test_targets[0]))
    srng = np.random.default_rng(pp.SEED + 500)

    picked_j, pool_best_j = [], []
    truth_wins = 0
    for i in range(n_test):
        x, true_y = test_sources[i], test_targets[i]
        recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
        y_scores = recon[pp.PART_DIM:]
        z = (y_scores - y_scores.mean()) / (y_scores.std() + pp.EPS)
        cands = gumbel_topk_samples(z, n_pool, temperature, n_active, srng)
        cands = np.vstack([cands, pp.binarize_topk(y_scores, n_active)[None, :]])
        pairs = np.hstack([np.repeat(x[None, :], len(cands), axis=0), cands])
        energies = pair_energy_batch(w_all, pairs)
        jac = jaccard_rows(cands, true_y)
        picked_j.append(jac[int(np.argmax(energies))])
        pool_best_j.append(float(jac.max()))
        e_true = pair_energy_batch(w_all, np.concatenate([x, true_y])[None, :])[0]
        truth_wins += int(e_true >= energies.max() - 1e-12)
    return float(np.mean(picked_j)), float(np.mean(pool_best_j)), truth_wins / n_test


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Zenith — Near-Miss (Hard) Negative Training",
        "",
        "| mode | ranking | one_shot J | hacked% | gap | PV cell | picked J | pool_best J | truth_wins |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for mode in MODES:
        rng = np.random.default_rng(pp.SEED)
        perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
        train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
        test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
        train_targets = pp.apply_permutation_batch(train_sources, perm)
        test_targets = pp.apply_permutation_batch(test_sources, perm)

        ensemble = train_hard(train_sources, train_targets, mode, pp.DIRECT_NODE_COUNT,
                              pp.DIRECT_TEMPLATE_COUNT, pp.ETA, NEG_SCALE,
                              pp.TRAINING_EPOCHS, rng)

        res = evaluate(ensemble, train_sources, train_targets,
                       test_sources, test_targets, np.random.default_rng(pp.SEED + 100))
        m_ = res["methods"]

        for temperature, n_pool in PV_CELLS:
            pj, pb, tw = propose_verify_eval(ensemble, test_sources, test_targets,
                                             temperature, n_pool)
            row = (
                f"| {mode} | {res['ranking_acc']:.2%} | {m_['one_shot']['jaccard']:.4f} "
                f"| {res['hacked_frac']:.0%} | {res['mean_gap']:+.4f} "
                f"| T={temperature},N={n_pool} | {pj:.4f} | {pb:.4f} | {tw:.0%} |"
            )
            lines.append(row)
            print(row, flush=True)

    lines += [
        "",
        "Reference (mismatch negatives only): ranking 99.4%, one_shot 0.388, "
        "hacked 100%, gap +0.80, truth_wins 0%, best picked J 0.385.",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved report to {OUT_DIR / 'report.md'}")
