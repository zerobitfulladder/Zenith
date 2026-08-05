"""Iterative completion follow-up to permutation_projection.py.

The scoring-variant experiment showed the ensemble is a strong VERIFIER of the
hidden permutation (activation-norm energy: 96% ranking accuracy) but a weak
one-shot PRODUCER (~50% precision reading Y out of blended templates).

This experiment tests whether iteration closes that gap. Four methods, one
shared ensemble (same config as permutation_projection.py's direct model):

    source_nn : nearest training source, return its stored target  (memory)
    one_shot  : [X, 0] -> projection readout -> top-k binarize     (March method)
    iterative : clamp X, feed [X, Y_t] back in, re-read Y, snap to
                top-k, repeat until Y stops changing               (user's loop)
    energy    : greedy swap search on Y directly maximizing the
                validated verifier signal ||W_all @ pair_hat||     (checker-guided)

Prediction: iterative > one_shot if re-presenting a partly-correct Y sharpens
winner selection; energy >= iterative because it climbs the exact objective the
verifier was shown to compute well.

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_iterative_completion/permutation_iterative_completion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp

OUT_DIR = Path(__file__).resolve().parent / "results"

MAX_ITERS = 20
MAX_SWEEPS = 15


def pair_energy_batch(w_all: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Validated verifier: L2 norm of the full correlation vector per pair."""
    hats = pp._normalize_rows(pairs)
    return np.linalg.norm(hats @ w_all.T, axis=1)


def iterative_completion(ensemble, x_source, n_active, max_iters=MAX_ITERS):
    """Clamp X, evolve Y: readout -> top-k snap -> re-present, until stable."""
    y = np.zeros(pp.PART_DIM)
    seen = set()
    trajectory = []
    for _ in range(max_iters):
        key = tuple(np.nonzero(y)[0])
        if key in seen:  # fixed point or 2-cycle
            break
        seen.add(key)
        pair = np.concatenate([x_source, y])
        recon, _resp = pp.reconstruct_pattern(ensemble, pair)
        y = pp.binarize_topk(recon[pp.PART_DIM:], n_active)
        trajectory.append(y.copy())
    return y, len(trajectory)


def energy_search(w_all, x_source, y_init, max_sweeps=MAX_SWEEPS):
    """Greedy steepest-ascent swap search on Y under the verifier energy.

    A move swaps one active Y bit for one inactive Y bit (sparsity preserved).
    All candidate swaps are scored in one vectorized batch per sweep.
    """
    y = y_init.copy()
    base_pair = np.concatenate([x_source, y])
    energy = pair_energy_batch(w_all, base_pair[None, :])[0]

    for _ in range(max_sweeps):
        on_bits = np.nonzero(y)[0]
        off_bits = np.nonzero(y == 0)[0]
        n_cand = len(on_bits) * len(off_bits)
        cands = np.repeat(base_pair[None, :], n_cand, axis=0)
        rows = np.arange(n_cand)
        off_idx = pp.PART_DIM + np.repeat(on_bits, len(off_bits))
        on_idx = pp.PART_DIM + np.tile(off_bits, len(on_bits))
        cands[rows, off_idx] = 0.0
        cands[rows, on_idx] = 1.0

        energies = pair_energy_batch(w_all, cands)
        best = int(np.argmax(energies))
        if energies[best] <= energy + 1e-12:
            break
        energy = energies[best]
        base_pair = cands[best]
        y = base_pair[pp.PART_DIM:].copy()
    return y, energy


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pp.SEED)

    perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
    train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
    test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
    train_targets = pp.apply_permutation_batch(train_sources, perm)
    test_targets = pp.apply_permutation_batch(test_sources, perm)
    train_full = pp.make_full_patterns(train_sources, train_targets)

    ensemble = pp.train_ensemble(
        train_full, pp.DIRECT_NODE_COUNT, pp.DIRECT_TEMPLATE_COUNT,
        pp.ETA, pp.TRAINING_EPOCHS, rng, "direct",
    )
    w_all = np.vstack(ensemble)

    n_active = int(np.sum(test_targets[0]))
    train_source_norm = pp._normalize_rows(train_sources)

    methods = {
        m: {"exact": 0, "jaccard": [], "precision": [], "recall": []}
        for m in ["source_nn", "one_shot", "iterative", "energy"]
    }
    iters_used, true_energy_gap = [], []

    for i in tqdm(range(len(test_sources)), desc="Evaluating"):
        x, true_y = test_sources[i], test_targets[i]

        # Memory baseline
        nn_idx = int(np.argmax(train_source_norm @ pp._normalize(x)))
        pp._score(methods["source_nn"], train_targets[nn_idx], true_y)

        # One-shot projection (March direct method)
        recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
        y_one = pp.binarize_topk(recon[pp.PART_DIM:], n_active)
        pp._score(methods["one_shot"], y_one, true_y)

        # Iterative clamped completion
        y_iter, n_it = iterative_completion(ensemble, x, n_active)
        pp._score(methods["iterative"], y_iter, true_y)
        iters_used.append(n_it)

        # Energy-guided swap search, initialized from the one-shot guess
        y_en, e_final = energy_search(w_all, x, y_one)
        pp._score(methods["energy"], y_en, true_y)
        e_true = pair_energy_batch(w_all, np.concatenate([x, true_y])[None, :])[0]
        true_energy_gap.append(e_final - e_true)

    lines = [
        "# Zenith — Iterative Completion of the Hidden Permutation",
        "",
        f"Config identical to permutation_projection.py direct model: "
        f"M={pp.DIRECT_NODE_COUNT}, k={pp.DIRECT_TEMPLATE_COUNT}, "
        f"N_train={pp.N_TRAIN}, epochs={pp.TRAINING_EPOCHS}.",
        "",
        "| Method | Jaccard | Precision | Recall | Exact |",
        "|---|---|---|---|---|",
    ]
    print()
    for m, st in methods.items():
        row = (
            f"| {m} | {np.mean(st['jaccard']):.4f} | {np.mean(st['precision']):.4f} "
            f"| {np.mean(st['recall']):.4f} | {st['exact'] / len(test_sources):.2%} |"
        )
        lines.append(row)
        print(row)

    lines += [
        "",
        f"Iterative loop: mean iterations to fixed point = {np.mean(iters_used):.2f} "
        f"(max allowed {MAX_ITERS}).",
        f"Energy search final energy minus true-target energy: "
        f"mean {np.mean(true_energy_gap):+.4f} "
        f"(>0 means search found a state the verifier likes MORE than the truth: "
        f"{np.mean(np.array(true_energy_gap) > 0):.0%} of items).",
        "",
    ]
    print("\n".join(lines[-4:]))
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    print(f"Saved report to {OUT_DIR / 'report.md'}")
