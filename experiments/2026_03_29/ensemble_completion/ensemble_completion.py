"""Zenith ensemble pattern completion experiment.

Multiple independent Zenith nodes see the same input. Each produces a
winner index + correlation score. The ensemble code (tuple of winners)
forms a distributed representation that can handle load > 1.

Usage:
    .venv/bin/python experiments/2026_03_29/ensemble_completion/ensemble_completion.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration (single values, no sweeps)
# ---------------------------------------------------------------------------

DIM = 200                   # vector dimensionality
SPARSITY = 0.05             # fraction of active bits
TEMPLATE_COUNT = 10         # k per node
PATTERN_COUNT = 200          # N  (load = N/k = 4.0 here)
NODE_COUNT = 50              # M  (number of independent Zenith nodes)
MASK_FRACTIONS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

TRAINING_EPOCHS = 1000
ETA = 0.03
N_TRIALS = 50               # random mask trials per pattern per mask fraction

SEED = 42
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Pattern generation
# ---------------------------------------------------------------------------

def make_sparse_patterns(n: int, dim: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    n_active = max(1, int(round(dim * sparsity)))
    patterns = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        idx = rng.choice(dim, size=n_active, replace=False)
        patterns[i, idx] = 1.0
    return patterns


def mask_pattern(pattern: np.ndarray, mask_frac: float, rng: np.random.Generator) -> np.ndarray:
    masked = pattern.copy()
    active_idx = np.nonzero(pattern)[0]
    n_mask = int(round(len(active_idx) * mask_frac))
    if n_mask > 0:
        to_mask = rng.choice(active_idx, size=n_mask, replace=False)
        masked[to_mask] = 0.0
    return masked


# ---------------------------------------------------------------------------
# Zenith core
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def init_weights(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal((k, dim))
    w -= np.mean(w, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True) + EPS
    return w


def zenith_forward(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return full activation vector (k,) — all correlations."""
    x_hat = _normalize(x)
    return w @ x_hat


def zenith_learn(w: np.ndarray, x: np.ndarray, eta: float) -> np.ndarray:
    x_hat = _normalize(x)
    c = w @ x_hat
    winner = np.argmax(np.abs(c))
    c_w = c[winner]

    w_win = w[winner]
    tau = x_hat - c_w * w_win
    tau_norm = np.linalg.norm(tau)
    tau_hat = tau / tau_norm if tau_norm > EPS else np.zeros_like(tau)

    theta = eta * c_w
    w[winner] = w_win * np.cos(theta) + tau_hat * np.sin(theta)
    return w


# ---------------------------------------------------------------------------
# Ensemble: array of M independent Zenith nodes
# ---------------------------------------------------------------------------

def init_ensemble(m: int, k: int, dim: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [init_weights(k, dim, rng) for _ in range(m)]


def ensemble_forward(ensemble: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Return concatenated activation vector (M*k,) from all nodes."""
    activations = [zenith_forward(w, x) for w in ensemble]
    return np.concatenate(activations)


def ensemble_learn(ensemble: list[np.ndarray], x: np.ndarray, eta: float) -> list[np.ndarray]:
    return [zenith_learn(w, x, eta) for w in ensemble]


# ---------------------------------------------------------------------------
# Training with tracking
# ---------------------------------------------------------------------------

def train(
    patterns: np.ndarray,
    m: int,
    k: int,
    eta: float,
    epochs: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Train ensemble.

    Returns:
        ensemble      — list of M weight matrices
        sim_curves    — (epochs, n_patterns, M) top-1 cosine sim per node
        code_accuracy — (epochs,) fraction of patterns with a unique ensemble code
    """
    n, dim = patterns.shape
    ensemble = init_ensemble(m, k, dim, rng)

    pats_normed = np.array([_normalize(p) for p in patterns])
    sim_curves = np.zeros((epochs, n, m), dtype=np.float64)
    code_accuracy = np.zeros(epochs, dtype=np.float64)

    for ep in tqdm(range(epochs), desc="Training", leave=False):
        order = rng.permutation(n)
        for idx in order:
            ensemble = ensemble_learn(ensemble, patterns[idx], eta)

        # track per-node top-1 similarities and ensemble code uniqueness
        codes = []
        for i in range(n):
            act = ensemble_forward(ensemble, patterns[i])
            # extract per-node winner indices for uniqueness check
            per_node = act.reshape(m, k)
            winners = tuple(int(np.argmax(np.abs(per_node[j]))) for j in range(m))
            codes.append(winners)
            for j in range(m):
                sim_curves[ep, i, j] = ensemble[j][winners[j]] @ pats_normed[i]

        unique_codes = len(set(codes))
        code_accuracy[ep] = unique_codes / n

    return ensemble, sim_curves, code_accuracy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(a @ b / (na * nb))


def evaluate(
    ensemble: list[np.ndarray],
    patterns: np.ndarray,
    mask_frac: float,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """Evaluate ensemble pattern completion via activation-space cosine similarity.

    For each pattern, compares the ensemble activation vector from the masked
    input against all full-input activation vectors. The nearest one (by cosine
    sim) is the retrieval result.

    Returns:
        retrieval_acc  — fraction of trials where nearest full-activation was correct
        avg_cos_sim    — avg cosine sim between masked and its own full activation
        avg_cos_rank   — avg rank of the correct pattern (0 = best)
        full_acts      — (n_patterns, M*k) full-input activations
    """
    n = patterns.shape[0]

    # baseline: full-input activation vectors
    full_acts = np.zeros((n, len(ensemble) * ensemble[0].shape[0]), dtype=np.float64)
    for i in range(n):
        full_acts[i] = ensemble_forward(ensemble, patterns[i])

    if mask_frac == 0.0:
        return {
            "retrieval_acc": 1.0,
            "avg_cos_sim": 1.0,
            "avg_cos_rank": 0.0,
            "full_acts": full_acts,
        }

    correct = 0
    cos_sims = []
    ranks = []
    total = n * n_trials

    for i in range(n):
        for _ in range(n_trials):
            masked = mask_pattern(patterns[i], mask_frac, rng)
            masked_act = ensemble_forward(ensemble, masked)

            # cosine sim to all full-input activations
            sims = np.array([_cosine_sim(masked_act, full_acts[j]) for j in range(n)])
            best = int(np.argmax(sims))

            if best == i:
                correct += 1

            cos_sims.append(sims[i])  # sim to own full activation
            rank = int(np.sum(sims > sims[i]))  # how many others are more similar
            ranks.append(rank)

    return {
        "retrieval_acc": correct / total,
        "avg_cos_sim": float(np.mean(cos_sims)),
        "avg_cos_rank": float(np.mean(ranks)),
        "full_acts": full_acts,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    sim_curves: np.ndarray,
    code_accuracy_curve: np.ndarray,
    eval_results: dict[float, dict],
):
    m = sim_curves.shape[2]
    mfs = sorted(eval_results.keys())
    load = PATTERN_COUNT / TEMPLATE_COUNT

    fig = plt.figure(figsize=(20, 16), dpi=150)
    fig.suptitle(
        f"Zenith Ensemble  —  dim={DIM}, sparsity={SPARSITY}, k={TEMPLATE_COUNT}, "
        f"N={PATTERN_COUNT} (load={load:.1f}), M={NODE_COUNT} nodes, "
        f"η={ETA}, epochs={TRAINING_EPOCHS}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30,
                  top=0.95, bottom=0.06, left=0.07, right=0.96)

    # --- (0,0) Training: avg cosine sim per node over epochs ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Training: avg cosine sim per node", fontweight="bold", fontsize=10)
    for j in range(m):
        avg = np.mean(sim_curves[:, :, j], axis=1)
        ax.plot(avg, linewidth=1.0, label=f"Node {j}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg top-1 cosine sim")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Training: ensemble avg cosine sim (mean +/- std) ---
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Training: ensemble avg cosine sim", fontweight="bold", fontsize=10)
    all_mean = np.mean(sim_curves, axis=(1, 2))
    all_std = np.std(np.mean(sim_curves, axis=2), axis=1)
    ax.plot(all_mean, color="C0", linewidth=1.2)
    ax.fill_between(range(len(all_mean)), all_mean - all_std, all_mean + all_std,
                     alpha=0.2, color="C0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine sim (mean +/- std)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # --- (0,2) Training: code uniqueness over epochs ---
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Training: ensemble code uniqueness", fontweight="bold", fontsize=10)
    ax.plot(code_accuracy_curve, color="C2", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of unique codes")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Retrieval accuracy vs mask fraction ---
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Retrieval accuracy vs masking", fontweight="bold", fontsize=10)
    accs = [eval_results[mf]["retrieval_acc"] for mf in mfs]
    ax.plot(mfs, accs, "o-", color="C0", linewidth=1.5, markersize=5)
    for x, y in zip(mfs, accs):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- (1,1) Avg cosine sim (masked act vs own full act) ---
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Activation cosine sim vs masking", fontweight="bold", fontsize=10)
    cos_sims = [eval_results[mf]["avg_cos_sim"] for mf in mfs]
    ax.plot(mfs, cos_sims, "o-", color="C1", linewidth=1.5, markersize=5)
    for x, y in zip(mfs, cos_sims):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Cosine sim (masked vs own full)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # --- (1,2) Avg rank of correct pattern ---
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Avg rank of correct pattern vs masking", fontweight="bold", fontsize=10)
    ranks = [eval_results[mf]["avg_cos_rank"] for mf in mfs]
    ax.plot(mfs, ranks, "o-", color="C3", linewidth=1.5, markersize=5)
    for x, y in zip(mfs, ranks):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Avg rank (0 = best)")
    ax.set_ylim(-0.5, max(ranks) + 2 if ranks else 5)
    ax.grid(True, alpha=0.3)

    out_path = OUT_DIR / "ensemble_completion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def write_report(
    eval_results: dict[float, dict],
    sim_curves: np.ndarray,
    code_accuracy_curve: np.ndarray,
):
    load = PATTERN_COUNT / TEMPLATE_COUNT
    mfs = sorted(eval_results.keys())
    final_sim = np.mean(sim_curves[-1])
    final_uniqueness = code_accuracy_curve[-1]

    lines = [
        f"# Zenith Ensemble — Pattern Completion Report",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Dimensionality | {DIM} |",
        f"| Sparsity | {SPARSITY} |",
        f"| Templates per node (k) | {TEMPLATE_COUNT} |",
        f"| Stored patterns (N) | {PATTERN_COUNT} |",
        f"| Load (N/k) | {load:.1f} |",
        f"| Ensemble nodes (M) | {NODE_COUNT} |",
        f"| Learning rate | {ETA} |",
        f"| Training epochs | {TRAINING_EPOCHS} |",
        f"| Mask trials per pattern | {N_TRIALS} |",
        f"",
        f"## Training Summary",
        f"",
        f"- Final avg top-1 cosine similarity: **{final_sim:.4f}**",
        f"- Final code uniqueness: **{final_uniqueness:.2%}** ({int(final_uniqueness * PATTERN_COUNT)}/{PATTERN_COUNT} unique codes)",
        f"",
        f"## Retrieval Results",
        f"",
        f"| Mask % | Retrieval Acc | Cos Sim (masked vs full) | Avg Rank | Retention |",
        f"|--------|--------------|--------------------------|----------|-----------|",
    ]

    baseline_sim = eval_results[0.0]["avg_cos_sim"] if 0.0 in eval_results else 1.0
    for mf in mfs:
        r = eval_results[mf]
        retention = r["avg_cos_sim"] / baseline_sim if baseline_sim > EPS else 0.0
        lines.append(
            f"| {mf:.0%} | {r['retrieval_acc']:.2%} | {r['avg_cos_sim']:.4f} "
            f"| {r['avg_cos_rank']:.1f} | {retention:.2%} |"
        )

    lines += [
        f"",
        f"## Interpretation",
        f"",
        f"- **Retrieval Acc**: fraction of masked trials where the correct pattern's "
        f"full-input activation was the nearest neighbor (by cosine similarity).",
        f"- **Cos Sim**: average cosine similarity between masked-input activation "
        f"and the same pattern's full-input activation. Higher = more stable representation.",
        f"- **Avg Rank**: where the correct pattern falls in the similarity ranking "
        f"(0 = best). Lower is better.",
        f"- **Retention**: ratio of masked cos sim to full cos sim (mask=0%). "
        f"Shows how much of the activation structure is preserved under masking.",
        f"",
    ]

    out_path = OUT_DIR / "report.md"
    out_path.write_text("\n".join(lines))
    print(f"Saved report to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Config: dim={DIM}, sparsity={SPARSITY}, k={TEMPLATE_COUNT}, "
          f"N={PATTERN_COUNT}, M={NODE_COUNT}, load={PATTERN_COUNT/TEMPLATE_COUNT:.1f}")

    # generate patterns
    patterns = make_sparse_patterns(PATTERN_COUNT, DIM, SPARSITY, rng)

    # train
    ensemble, sim_curves, code_accuracy_curve = train(
        patterns, NODE_COUNT, TEMPLATE_COUNT, ETA, TRAINING_EPOCHS, rng
    )

    # evaluate at each mask fraction
    eval_results = {}
    for mf in MASK_FRACTIONS:
        eval_rng = np.random.default_rng(SEED + 100)
        eval_results[mf] = evaluate(ensemble, patterns, mf, N_TRIALS, eval_rng)
        r = eval_results[mf]
        print(f"  mask={mf:.1f}  retrieval_acc={r['retrieval_acc']:.3f}  "
              f"cos_sim={r['avg_cos_sim']:.3f}  avg_rank={r['avg_cos_rank']:.1f}")

    # plot
    plot_results(sim_curves, code_accuracy_curve, eval_results)

    # write human-readable report
    write_report(eval_results, sim_curves, code_accuracy_curve)
    print("Done.")
