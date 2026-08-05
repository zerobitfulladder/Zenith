"""Local receptive fields — force factorization by shrinking what each node sees.

Diagnosis so far: global templates blend ~30 whole pairs, so generation,
near-field verification, and iterative dynamics all point at the blend instead
of the truth. The permutation rule is 128 independent wire-facts (x_i <-> y_pi),
but whole-pattern recall can't mix sub-parts of different memories.

Fix under test: each node sees only F dims (F/2 random X bits + F/2 random Y
bits). Within a window, a matched wire pair co-occurs 100% of the time while
unmatched bits co-occur ~SPARSITY of the time, so local templates should
crystallize around wire-facts — and the full answer is assembled from many
nodes' local readouts (composition for free). No weight sharing, no spatial
neighborhoods: "convolution" minus the geometry, because the rule has none.

Configs hold expected wire coverage constant (M * (F/2/128)^2 = 5):
    F=16 / M=1280, F=32 / M=320, F=64 / M=80.
Global anchors (plain training): one_shot 0.336, far-field 97.7%,
truth pctl median 0.7% (INVERTED), truth_wins 0%.

Measured per config:
    wire%     — templates whose strongest X-bit and strongest Y-bit form a TRUE
                wire (null column: same stat vs a random fake permutation)
    one_shot  — generation Jaccard from count-normalized local reconstruction
    far-field — truth vs 15 mismatched targets, rank-1 accuracy by energy
    truth pctl / wins / argmax / pool_best — near-field: truth's energy
                percentile in a T=0.5, N=1000 Gumbel pool (consensus protocol)

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_local_rf/permutation_local_rf.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_propose_verify"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_propose_verify import gumbel_topk_samples, jaccard_rows

OUT_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = [(16, 1280), (32, 320), (64, 80)]  # (window F, nodes M)
N_POOL = 1000
TEMPERATURE = 0.5
N_DISTRACT = 15


def make_windows(m: int, f: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified windows: F/2 dims from the X half, F/2 from the Y half."""
    wins = np.empty((m, f), dtype=np.int64)
    for n in range(m):
        wins[n, : f // 2] = rng.choice(pp.PART_DIM, f // 2, replace=False)
        wins[n, f // 2 :] = rng.choice(pp.PART_DIM, f // 2, replace=False) + pp.PART_DIM
    return wins


def _window_normalize(sub: np.ndarray) -> np.ndarray:
    """Mean-center and unit-normalize along the last axis (window-local)."""
    sub = sub - sub.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(sub, axis=-1, keepdims=True)
    return sub / np.where(n > pp.EPS, n, 1.0)


def train_local(patterns, windows, k, eta, epochs, rng) -> np.ndarray:
    """All-nodes-vectorized Zenith training; per-node pattern orders."""
    m, f = windows.shape
    n_pat = len(patterns)
    w = np.stack([pp.init_weights(k, f, rng) for _ in range(m)])  # (M, k, F)
    rows = np.arange(m)
    for _ in tqdm(range(epochs), desc=f"train F={f} M={m}", leave=False):
        orders = np.argsort(rng.random((m, n_pat)), axis=1)
        for t in range(n_pat):
            x_hat = _window_normalize(patterns[orders[:, t][:, None], windows])
            c = np.einsum("mkf,mf->mk", w, x_hat)
            win = np.argmax(np.abs(c), axis=1)
            c_w = c[rows, win]
            w_win = w[rows, win]
            tau = x_hat - c_w[:, None] * w_win
            tau_n = np.linalg.norm(tau, axis=1, keepdims=True)
            tau_hat = np.where(tau_n > pp.EPS, tau / np.where(tau_n > pp.EPS, tau_n, 1.0), 0.0)
            theta = eta * c_w
            w[rows, win] = w_win * np.cos(theta)[:, None] + tau_hat * np.sin(theta)[:, None]
    return w


def local_reconstruct_scores(w, windows, x_full) -> np.ndarray:
    """Per-node top-1 winner projection, scattered and count-averaged per dim."""
    m = w.shape[0]
    x_hat = _window_normalize(x_full[windows])
    c = np.einsum("mkf,mf->mk", w, x_hat)
    win = np.argmax(np.abs(c), axis=1)
    c_w = c[np.arange(m), win]
    contrib = c_w[:, None] * w[np.arange(m), win]
    recon = np.zeros(x_full.size)
    count = np.zeros(x_full.size)
    np.add.at(recon, windows.ravel(), contrib.ravel())
    np.add.at(count, windows.ravel(), 1.0)
    return recon / np.maximum(count, 1.0)


def local_energy_batch(w, windows, pairs, chunk=256) -> np.ndarray:
    """||all template correlations|| per pair, windows applied per node."""
    out = np.empty(len(pairs))
    for s in range(0, len(pairs), chunk):
        sub = _window_normalize(pairs[s : s + chunk][:, windows])   # (B, M, F)
        c = np.einsum("bmf,mkf->bmk", sub, w)
        out[s : s + chunk] = np.sqrt(np.sum(c * c, axis=(1, 2)))
    return out


def wire_hit_rate(w, windows, perm) -> float:
    """Fraction of templates whose top |weight| X-bit and Y-bit form a true wire.

    apply_permutation_batch gives y[j] = x[perm[j]], so the wire into y-dim j
    comes from x-dim perm[j].
    """
    f = windows.shape[1]
    half = f // 2
    xdims = windows[:, :half]                    # (M, F/2) in 0..127
    ydims = windows[:, half:] - pp.PART_DIM      # (M, F/2) in 0..127
    hits = total = 0
    for n in range(w.shape[0]):
        for t in range(w.shape[1]):
            xi = xdims[n, np.argmax(np.abs(w[n, t, :half]))]
            yj = ydims[n, np.argmax(np.abs(w[n, t, half:]))]
            hits += int(perm[yj] == xi)
            total += 1
    return hits / total


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pp.SEED)

    perm = pp.make_hidden_permutation(pp.PART_DIM, rng)
    train_sources = pp.make_sparse_vectors(pp.N_TRAIN, pp.PART_DIM, pp.SPARSITY, rng)
    test_sources = pp.make_sparse_vectors(pp.N_TEST, pp.PART_DIM, pp.SPARSITY, rng)
    train_targets = pp.apply_permutation_batch(train_sources, perm)
    test_targets = pp.apply_permutation_batch(test_sources, perm)
    train_patterns = pp.make_full_patterns(train_sources, train_targets)

    null_perm = np.random.default_rng(pp.SEED + 900).permutation(pp.PART_DIM)
    n_test = len(test_sources)
    n_active = int(np.sum(test_targets[0]))

    lines = [
        "# Zenith — Local Receptive Fields (factorize by limiting what nodes see)",
        "",
        f"Plain training (no negatives), {pp.TRAINING_EPOCHS} epochs, k={pp.DIRECT_TEMPLATE_COUNT}, "
        f"eta={pp.ETA}. Equal expected wire coverage (~5 nodes/wire) across configs.",
        "Global anchors (plain): one_shot 0.336 | far-field 97.7% | truth pctl med 0.7% | truth_wins 0%.",
        "",
        "| F | M | wire% | wire% null | one_shot J | far-field | truth pctl (med) | truth_wins | argmax J | pool_best J |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for f_dim, m_nodes in CONFIGS:
        crng = np.random.default_rng(pp.SEED + f_dim)
        windows = make_windows(m_nodes, f_dim, crng)
        w = train_local(train_patterns, windows, pp.DIRECT_TEMPLATE_COUNT, pp.ETA,
                        pp.TRAINING_EPOCHS, crng)

        wire_pct = wire_hit_rate(w, windows, perm)
        wire_null = wire_hit_rate(w, windows, null_perm)

        srng = np.random.default_rng(pp.SEED + 500)
        one_shot, far_ok, truth_pctl, argmax_j, pool_best = [], [], [], [], []
        truth_wins = 0
        for i in tqdm(range(n_test), desc=f"eval F={f_dim}"):
            x, true_y = test_sources[i], test_targets[i]
            recon = local_reconstruct_scores(w, windows, np.concatenate([x, np.zeros(pp.PART_DIM)]))
            y_scores = recon[pp.PART_DIM:]
            y_one = pp.binarize_topk(y_scores, n_active)
            one_shot.append(jaccard_rows(y_one[None, :], true_y)[0])

            distract = test_targets[[(i + 1 + d) % n_test for d in range(N_DISTRACT)]]
            far_pairs = np.hstack([np.repeat(x[None, :], N_DISTRACT + 1, axis=0),
                                   np.vstack([true_y[None, :], distract])])
            far_e = local_energy_batch(w, windows, far_pairs)
            far_ok.append(float(far_e[0] > far_e[1:].max()))

            z = (y_scores - y_scores.mean()) / (y_scores.std() + pp.EPS)
            cands = gumbel_topk_samples(z, N_POOL, TEMPERATURE, n_active, srng)
            cands = np.vstack([cands, y_one[None, :]])
            pairs = np.hstack([np.repeat(x[None, :], len(cands), axis=0), cands])
            energies = local_energy_batch(w, windows, pairs)
            e_true = local_energy_batch(w, windows, np.concatenate([x, true_y])[None, :])[0]
            truth_pctl.append(float(np.mean(energies < e_true)))
            truth_wins += int(e_true >= energies.max() - 1e-12)
            jac = jaccard_rows(cands, true_y)
            argmax_j.append(jac[int(np.argmax(energies))])
            pool_best.append(float(jac.max()))

        lines.append(
            f"| {f_dim} | {m_nodes} | {wire_pct:.1%} | {wire_null:.1%} | "
            f"{np.mean(one_shot):.4f} | {np.mean(far_ok):.1%} | "
            f"{np.median(truth_pctl):.1%} | {truth_wins / n_test:.0%} | "
            f"{np.mean(argmax_j):.4f} | {np.mean(pool_best):.4f} |"
        )
        print(lines[-1], flush=True)

    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved report to {OUT_DIR / 'report.md'}")
