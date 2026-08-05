"""Consensus decoding: marginalize over ranked candidates instead of argmax.

Four training regimes all leave truth_wins at 0% — the true target never ranks
#1 of 1001 candidates. Hypothesis: the verifier is not blind but NOISY — truth
ranks high, argmax is just dominated by lures (truth +/- a few resonant bit
swaps). If the high-energy region is a cloud of near-truth variants with
independent private errors, a per-bit MAJORITY VOTE over the top-M candidates
recovers the shared bits (the truth) and cancels the private mistakes.

Measured (near-negative model, T=0.5 / N=1000 pools):
    truth_rank   : mean percentile of the planted truth's energy in the pool
    argmax J     : baseline decode (pick the single highest-energy candidate)
    vote@M J     : top-12 bits of the mean of the top-M candidates by energy
    soft J       : energy-softmax-weighted per-bit average over the whole pool
    pool_mean J  : unweighted average of ALL candidates (control: does the
                   verifier's ranking contribute anything beyond the proposer?)

Usage:
    .venv/bin/python experiments/2026_08_05/permutation_consensus/permutation_consensus.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys  # the modules imported below live in sibling experiment folders
_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _d in ("2026_08_05/permutation_relation_scoring", "2026_03_29/permutation_projection", "2026_08_05/permutation_hard_negatives", "2026_08_05/permutation_iterative_completion", "2026_08_05/permutation_propose_verify"):
    sys.path.insert(0, str(_EXPERIMENTS / _d))

import permutation_relation_scoring  # noqa: F401  (installs tqdm/matplotlib stubs if absent)
from tqdm import tqdm

import permutation_projection as pp
from permutation_hard_negatives import train_hard
from permutation_iterative_completion import pair_energy_batch
from permutation_propose_verify import gumbel_topk_samples, jaccard_rows

OUT_DIR = Path(__file__).resolve().parent / "results"

TEMPERATURE = 0.5
N_POOL = 1000
VOTE_MS = [10, 25, 50, 100]


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
    w_all = np.vstack(ensemble)
    n_test = len(test_sources)
    n_active = int(np.sum(test_targets[0]))
    srng = np.random.default_rng(pp.SEED + 500)

    stats = {f"vote@{m}": [] for m in VOTE_MS}
    stats.update({"argmax": [], "soft": [], "pool_mean": [], "one_shot": []})
    truth_pctl = []

    for i in tqdm(range(n_test), desc="Consensus decoding"):
        x, true_y = test_sources[i], test_targets[i]
        recon, _ = pp.reconstruct_pattern(ensemble, np.concatenate([x, np.zeros(pp.PART_DIM)]))
        y_scores = recon[pp.PART_DIM:]
        z = (y_scores - y_scores.mean()) / (y_scores.std() + pp.EPS)
        y_one = pp.binarize_topk(y_scores, n_active)
        stats["one_shot"].append(jaccard_rows(y_one[None, :], true_y)[0])

        cands = gumbel_topk_samples(z, N_POOL, TEMPERATURE, n_active, srng)
        cands = np.vstack([cands, y_one[None, :]])
        pairs = np.hstack([np.repeat(x[None, :], len(cands), axis=0), cands])
        energies = pair_energy_batch(w_all, pairs)

        e_true = pair_energy_batch(w_all, np.concatenate([x, true_y])[None, :])[0]
        truth_pctl.append(float(np.mean(energies < e_true)))

        jac = jaccard_rows(cands, true_y)
        stats["argmax"].append(jac[int(np.argmax(energies))])

        order = np.argsort(energies)[::-1]
        for m in VOTE_MS:
            bit_freq = cands[order[:m]].mean(axis=0)
            y_vote = pp.binarize_topk(bit_freq, n_active)
            stats[f"vote@{m}"].append(jaccard_rows(y_vote[None, :], true_y)[0])

        beta = 5.0 / (np.std(energies) + pp.EPS)
        wts = np.exp(beta * (energies - energies.max()))
        y_soft = pp.binarize_topk(wts @ cands / wts.sum(), n_active)
        stats["soft"].append(jaccard_rows(y_soft[None, :], true_y)[0])

        y_mean = pp.binarize_topk(cands.mean(axis=0), n_active)
        stats["pool_mean"].append(jaccard_rows(y_mean[None, :], true_y)[0])

    lines = [
        "# Zenith — Consensus Decoding (marginalize over ranked candidates)",
        "",
        f"Near-negative model, pools: T={TEMPERATURE}, N={N_POOL}.",
        "",
        f"Truth energy percentile in pool: mean {np.mean(truth_pctl):.1%} "
        f"(median {np.median(truth_pctl):.1%}) — rank-1 rate was 0%.",
        "",
        "| decode | Jaccard |",
        "|---|---|",
    ]
    for name in ["one_shot", "argmax"] + [f"vote@{m}" for m in VOTE_MS] + ["soft", "pool_mean"]:
        lines.append(f"| {name} | {np.mean(stats[name]):.4f} |")
    print("\n".join(lines[4:]))
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved report to {OUT_DIR / 'report.md'}")
