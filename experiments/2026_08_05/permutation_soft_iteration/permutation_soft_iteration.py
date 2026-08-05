"""Iterative sampling / population refinement — verifier-free beam-ish decode.

User-proposed scheme: read out a Y-score distribution from [X, 0], sample N
candidates from it, feed each [X, candidate] back through the ensemble, AVERAGE
the N regenerated Y-score profiles into the next round's distribution, resample,
repeat. No energy ranking anywhere — the network's own regeneration is the only
judge. Private sampling errors decorrelate across particles and wash out in the
average; only shared signal survives to the next round.

Bootstrap hypothesis: with [X, 0] winner selection sees only the X half; with a
mostly-right [X, Y] the templates matching BOTH halves win, and those stored the
true pairs, so their Y-halves are more accurate. Bias hypothesis: the blending
pull toward neighbor-consensus bits is shared by all particles and survives the
average, so rounds drift instead of climbing.

Anchors: one_shot 0.3995 (near-negative model), hard single-trajectory
iteration 0.196, pool_mean 0.3972.

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_soft_iteration/permutation_soft_iteration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_hard_negatives", "2026_08_05/permutation_propose_verify"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_hard_negatives import train_hard
from permutation_propose_verify import gumbel_topk_samples, jaccard_rows

OUT_DIR = Path(__file__).resolve().parent / "results"

N_POOL = 100
N_ROUNDS = 6
CONFIGS = {
    "T0.5": [0.5] * N_ROUNDS,
    "T1.0": [1.0] * N_ROUNDS,
    "anneal": list(np.geomspace(1.0, 0.25, N_ROUNDS)),
}


def batch_reconstruct_y(w_stack: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Vectorized reconstruct_pattern over a batch, returning only Y-half scores."""
    hats = pp._normalize_rows(pairs)                                   # (N, D)
    corr = np.einsum("nd,mkd->nmk", hats, w_stack)                     # (N, M, k)
    winners = np.argmax(np.abs(corr), axis=2)                          # (N, M)
    c_w = np.take_along_axis(corr, winners[:, :, None], axis=2)[:, :, 0]
    w_win = w_stack[np.arange(w_stack.shape[0])[None, :], winners]     # (N, M, D)
    recon = np.einsum("nm,nmd->nd", c_w, w_win) / w_stack.shape[0]
    return recon[:, pp.PART_DIM:]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pp.SEED)

    perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
    train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
    test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
    train_targets = pp.apply_permutation_batch(train_sources, perm)
    test_targets = pp.apply_permutation_batch(test_sources, perm)

    ensemble = train_hard(train_sources, train_targets, "near", pp.DIRECT_NODE_COUNT,
                          pp.DIRECT_TEMPLATE_COUNT, pp.ETA, 0.5, pp.TRAINING_EPOCHS, rng)
    w_stack = np.stack(ensemble)
    n_test = len(test_sources)
    n_active = int(np.sum(test_targets[0]))

    results = {}  # config -> dict of per-round lists
    one_shot_j = []

    for name, temps in CONFIGS.items():
        srng = np.random.default_rng(pp.SEED + 600)
        decode = [[] for _ in range(N_ROUNDS)]
        oracle = [[] for _ in range(N_ROUNDS)]
        collapse = [[] for _ in range(N_ROUNDS)]

        for i in tqdm(range(n_test), desc=f"soft-iter {name}"):
            x, true_y = test_sources[i], test_targets[i]
            recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
            scores = recon[pp.PART_DIM:]
            if name == "T0.5":  # record baseline once
                y0 = pp.binarize_topk(scores, n_active)
                one_shot_j.append(jaccard_rows(y0[None, :], true_y)[0])

            for r, temp in enumerate(temps):
                z = (scores - scores.mean()) / (scores.std() + pp.EPS)
                cands = gumbel_topk_samples(z, N_POOL, temp, n_active, srng)
                pairs = np.hstack([np.repeat(x[None, :], N_POOL, axis=0), cands])
                scores = batch_reconstruct_y(w_stack, pairs).mean(axis=0)

                y_dec = pp.binarize_topk(scores, n_active)
                decode[r].append(jaccard_rows(y_dec[None, :], true_y)[0])
                oracle[r].append(float(np.max(jaccard_rows(cands, true_y))))
                bit_freq = cands.mean(axis=0)
                collapse[r].append(float(np.mean(np.abs(bit_freq - 0.5)) * 2))

        results[name] = {"decode": decode, "oracle": oracle, "collapse": collapse}

    header = "| config | " + " | ".join(f"r{r+1}" for r in range(N_ROUNDS)) + " |"
    sep = "|---" * (N_ROUNDS + 1) + "|"
    lines = [
        "# Zenith — Iterative sampling (population refinement, no verifier)",
        "",
        f"Near-negative model. N={N_POOL} particles/round, {N_ROUNDS} rounds.",
        f"Anchors: one_shot {np.mean(one_shot_j):.4f} | hard iteration 0.196 | pool_mean 0.3972.",
        "",
        "## Decode Jaccard (top-k of round's mean distribution)",
        header, sep,
    ]
    for name in CONFIGS:
        row = " | ".join(f"{np.mean(results[name]['decode'][r]):.4f}" for r in range(N_ROUNDS))
        lines.append(f"| {name} | {row} |")
    lines += ["", "## Oracle best sample in pool (coverage of truth)", header, sep]
    for name in CONFIGS:
        row = " | ".join(f"{np.mean(results[name]['oracle'][r]):.4f}" for r in range(N_ROUNDS))
        lines.append(f"| {name} | {row} |")
    lines += ["", "## Collapse (1 = pool unanimous, 0 = max disagreement)", header, sep]
    for name in CONFIGS:
        row = " | ".join(f"{np.mean(results[name]['collapse'][r]):.4f}" for r in range(N_ROUNDS))
        lines.append(f"| {name} | {row} |")

    print("\n".join(lines[3:]))
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved report to {OUT_DIR / 'report.md'}")
