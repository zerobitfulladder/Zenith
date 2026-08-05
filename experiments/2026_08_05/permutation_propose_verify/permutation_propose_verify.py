"""Propose-and-verify generation for the hidden-permutation task.

The arc so far: the ensemble's energy is a near-perfect judge of plausible
candidates (99.4% ranking with contrastive training) but an incorrigible
target for free optimization (search finds chimera states above the truth on
100% of items). So: never optimize the energy — SAMPLE proposals from the
model's own blurry readout and let the energy RANK them.

Pipeline per test item:
    1. One-shot readout of [X, 0] -> per-bit scores for the Y half (the blur).
    2. Z-score the Y scores, sample N sparse candidates by Gumbel top-k at
       temperature T (candidates stay on the readout's plausibility manifold).
    3. Add the deterministic one-shot guess to the pool.
    4. Score all candidates with the verifier energy ||W_all @ pair_hat||,
       output the argmax.

Diagnostics separating proposer from verifier:
    pool_best   : best Jaccard present in the pool (coverage ceiling)
    picked=best : how often the verifier selects that pool-best candidate
    truth_wins  : with the true target secretly added, does it win the ranking?

Models: one-phase control and contrastive (mismatch negatives, scale 0.5).

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_propose_verify/permutation_propose_verify.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_contrastive", "2026_08_05/permutation_iterative_completion"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_contrastive import train_contrastive
from permutation_iterative_completion import pair_energy_batch

OUT_DIR = Path(__file__).resolve().parent / "results"

TEMPERATURES = [0.5, 1.0, 2.0]
POOL_SIZES = [100, 1000]
MODELS = [("one_phase", 0.0), ("contrastive", 0.5)]


def gumbel_topk_samples(scores_z, n_samples, temperature, n_active, rng):
    """Sample sparse binary vectors ~ softmax(scores/T) without replacement."""
    gumbel = rng.gumbel(size=(n_samples, scores_z.size))
    keys = scores_z[None, :] / temperature + gumbel
    idx = np.argpartition(keys, -n_active, axis=1)[:, -n_active:]
    out = np.zeros((n_samples, scores_z.size))
    np.put_along_axis(out, idx, 1.0, axis=1)
    return out


def jaccard_rows(cands: np.ndarray, true_y: np.ndarray) -> np.ndarray:
    inter = cands @ true_y
    union = cands.sum(axis=1) + true_y.sum() - inter
    return inter / np.maximum(union, 1.0)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Zenith — Propose-and-Verify Generation",
        "",
        "| model | T | N | picked J | picked P | pool_best J | picked=best | truth_wins | one_shot J |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for model_name, neg_scale in MODELS:
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
        w_all = np.vstack(ensemble)
        n_test = len(test_sources)
        n_active = int(np.sum(test_targets[0]))

        # Per-item readout scores and one-shot guesses (computed once per model)
        readout_z, one_shots, one_shot_stats = [], [], []
        for i in range(n_test):
            recon, _ = pp.reconstruct_pattern(
                ensemble, np.concatenate([test_sources[i], np.zeros(pp.PART_DIM)])
            )
            y_scores = recon[pp.PART_DIM:]
            readout_z.append((y_scores - y_scores.mean()) / (y_scores.std() + pp.EPS))
            y_one = pp.binarize_topk(y_scores, n_active)
            one_shots.append(y_one)
            one_shot_stats.append(jaccard_rows(y_one[None, :], test_targets[i])[0])
        one_shot_j = float(np.mean(one_shot_stats))

        for temperature in TEMPERATURES:
            for n_pool in POOL_SIZES:
                srng = np.random.default_rng(pp.SEED + 500)
                picked_j, picked_p, pool_best_j = [], [], []
                picked_is_best = truth_wins = 0

                for i in tqdm(range(n_test), desc=f"{model_name} T={temperature} N={n_pool}",
                              leave=False):
                    x, true_y = test_sources[i], test_targets[i]
                    cands = gumbel_topk_samples(readout_z[i], n_pool, temperature,
                                                n_active, srng)
                    cands = np.vstack([cands, one_shots[i][None, :]])
                    pairs = np.hstack([np.repeat(x[None, :], len(cands), axis=0), cands])
                    energies = pair_energy_batch(w_all, pairs)

                    pick = int(np.argmax(energies))
                    jac = jaccard_rows(cands, true_y)
                    inter = cands[pick] @ true_y
                    picked_j.append(jac[pick])
                    picked_p.append(inter / n_active)
                    pool_best_j.append(float(jac.max()))
                    picked_is_best += int(jac[pick] >= jac.max() - 1e-12)

                    e_true = pair_energy_batch(
                        w_all, np.concatenate([x, true_y])[None, :]
                    )[0]
                    truth_wins += int(e_true >= energies.max() - 1e-12)

                row = (
                    f"| {model_name} | {temperature} | {n_pool} "
                    f"| {np.mean(picked_j):.4f} | {np.mean(picked_p):.4f} "
                    f"| {np.mean(pool_best_j):.4f} | {picked_is_best / n_test:.0%} "
                    f"| {truth_wins / n_test:.0%} | {one_shot_j:.4f} |"
                )
                lines.append(row)
                print(row, flush=True)

    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved report to {OUT_DIR / 'report.md'}")
