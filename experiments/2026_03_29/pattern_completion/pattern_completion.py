"""Zenith pattern completion experiment.

Trains Zenith (top-1 winner-take-all on the unit hypersphere) on sparse
binary patterns, then tests retrieval under partial masking.

Sweeps over: template count, pattern count, sparsity, mask fraction.
Produces a single composite figure with all metrics.

Usage:
    .venv/bin/python experiments/2026_03_29/pattern_completion/pattern_completion.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global sweep parameters
# ---------------------------------------------------------------------------

DIM = 200                                     # vector dimensionality
SPARSITIES = [0.05, 0.10, 0.20]              # fraction of active bits
TEMPLATE_COUNTS = [10, 25, 50]                # k  (number of templates)
PATTERN_COUNTS = [5, 10, 20, 40]             # N  (number of stored patterns)
MASK_FRACTIONS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]  # fraction of active bits zeroed

TRAINING_EPOCHS = 3000                         # passes over the pattern set
ETA = 0.01                                    # geodesic step fraction

SEED = 42
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Pattern generation
# ---------------------------------------------------------------------------

def make_sparse_patterns(n: int, dim: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """Create n sparse binary vectors of given dimensionality and sparsity."""
    n_active = max(1, int(round(dim * sparsity)))
    patterns = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        idx = rng.choice(dim, size=n_active, replace=False)
        patterns[i, idx] = 1.0
    return patterns


def mask_pattern(pattern: np.ndarray, mask_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out mask_frac of the active bits in a pattern."""
    masked = pattern.copy()
    active_idx = np.nonzero(pattern)[0]
    n_mask = int(round(len(active_idx) * mask_frac))
    if n_mask > 0:
        to_mask = rng.choice(active_idx, size=n_mask, replace=False)
        masked[to_mask] = 0.0
    return masked


# ---------------------------------------------------------------------------
# Zenith core (standalone, no Node framework)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    """Mean-center and L2-normalize."""
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def init_weights(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize k templates on the unit hypersphere."""
    w = rng.standard_normal((k, dim))
    w -= np.mean(w, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True) + EPS
    return w


def zenith_forward(w: np.ndarray, x: np.ndarray):
    """Return (winner_index, correlations, winner_correlation)."""
    x_hat = _normalize(x)
    c = w @ x_hat
    winner = np.argmax(np.abs(c))
    return winner, c, c[winner]


def zenith_learn(w: np.ndarray, x: np.ndarray, eta: float) -> np.ndarray:
    """One learning step: geodesic rotation of the winner toward x."""
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
# Training with tracking
# ---------------------------------------------------------------------------

def train(
    patterns: np.ndarray,
    k: int,
    eta: float,
    epochs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Train and return (W, similarity_curve).

    similarity_curve: shape (epochs, n_patterns) — after each epoch, for each
    pattern the cosine similarity between that pattern (normalized) and its
    best-matching template.
    """
    n, dim = patterns.shape
    w = init_weights(k, dim, rng)

    # pre-compute normalized patterns for tracking
    pats_normed = np.array([_normalize(p) for p in patterns])

    sim_curve = np.zeros((epochs, n), dtype=np.float64)

    for ep in range(epochs):
        # present patterns in random order
        order = rng.permutation(n)
        for idx in order:
            w = zenith_learn(w, patterns[idx], eta)

        # track: best cosine similarity per pattern
        for i in range(n):
            sims = w @ pats_normed[i]  # (k,)
            sim_curve[ep, i] = np.max(sims)  # top-1 cosine sim

    return w, sim_curve


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    w: np.ndarray,
    patterns: np.ndarray,
    mask_frac: float,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """Evaluate pattern completion under masking.

    Returns dict with:
        accuracy       — fraction of trials where correct template won
        full_sim       — avg cosine sim when full (unmasked) pattern is given
        masked_sim     — avg cosine sim when masked pattern is given
        retention      — masked_sim / full_sim  (how much similarity is retained)
    """
    n = patterns.shape[0]
    pats_normed = np.array([_normalize(p) for p in patterns])

    # full-input similarities (baseline)
    full_sims = np.zeros(n)
    full_winners = np.zeros(n, dtype=int)
    for i in range(n):
        winner, c, c_w = zenith_forward(w, patterns[i])
        full_sims[i] = w[winner] @ pats_normed[i]
        full_winners[i] = winner

    if mask_frac == 0.0:
        return {
            "accuracy": 1.0,
            "full_sim": float(np.mean(full_sims)),
            "masked_sim": float(np.mean(full_sims)),
            "retention": 1.0,
        }

    correct = 0
    masked_sims = np.zeros(n * n_trials)

    for i in range(n):
        for t in range(n_trials):
            masked = mask_pattern(patterns[i], mask_frac, rng)
            winner, c, c_w = zenith_forward(w, masked)

            # cosine sim between winning template and the FULL (unmasked) normalized pattern
            sim = w[winner] @ pats_normed[i]
            masked_sims[i * n_trials + t] = sim

            if winner == full_winners[i]:
                correct += 1

    total = n * n_trials
    avg_masked_sim = float(np.mean(masked_sims))
    avg_full_sim = float(np.mean(full_sims))

    return {
        "accuracy": correct / total,
        "full_sim": avg_full_sim,
        "masked_sim": avg_masked_sim,
        "retention": avg_masked_sim / avg_full_sim if avg_full_sim > EPS else 0.0,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(SPARSITIES, TEMPLATE_COUNTS, PATTERN_COUNTS))
    # filter out combos where patterns > templates (keep them for capacity test)
    # but skip where it's absurdly unbalanced
    combos = [(s, k, n) for s, k, n in combos if n <= k * 2]

    results = []          # flat list of result dicts
    training_curves = {}  # key: (sparsity, k, n) -> sim_curve

    total = len(combos)
    print(f"Running {total} training configs x {len(MASK_FRACTIONS)} mask levels")

    for sparsity, k, n_patterns in tqdm(combos, desc="Configs"):
        # generate patterns with a deterministic sub-seed
        sub_seed = hash((SEED, sparsity, k, n_patterns)) % (2**31)
        sub_rng = np.random.default_rng(sub_seed)

        patterns = make_sparse_patterns(n_patterns, DIM, sparsity, sub_rng)

        # train
        train_rng = np.random.default_rng(sub_seed + 1)
        w, sim_curve = train(patterns, k, ETA, TRAINING_EPOCHS, train_rng)
        training_curves[(sparsity, k, n_patterns)] = sim_curve

        # evaluate at each mask fraction
        for mf in MASK_FRACTIONS:
            eval_rng = np.random.default_rng(sub_seed + 2)
            metrics = evaluate(w, patterns, mf, n_trials=20, rng=eval_rng)
            results.append({
                "sparsity": sparsity,
                "k": k,
                "n_patterns": n_patterns,
                "load": n_patterns / k,
                "mask_frac": mf,
                **metrics,
            })

    return results, training_curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], training_curves: dict):
    """Create a single composite figure with all metrics."""

    res = results  # shorthand
    sparsities = sorted(set(r["sparsity"] for r in res))
    ks = sorted(set(r["k"] for r in res))
    ns = sorted(set(r["n_patterns"] for r in res))
    mfs = sorted(set(r["mask_frac"] for r in res))

    # -----------------------------------------------------------------------
    # Layout: 4 rows x 3 columns (one column per sparsity)
    #   Row 0: Training curves (avg top-1 cosine sim over epochs)
    #   Row 1: Retrieval accuracy vs mask fraction
    #   Row 2: Cosine similarity (full vs masked) vs mask fraction
    #   Row 3: Retention (%) vs mask fraction
    # -----------------------------------------------------------------------

    n_sp = len(sparsities)
    fig = plt.figure(figsize=(7 * n_sp, 24), dpi=150)
    fig.suptitle(
        f"Zenith Pattern Completion  —  dim={DIM}, η={ETA}, epochs={TRAINING_EPOCHS}",
        fontsize=16, fontweight="bold", y=0.995,
    )

    gs = GridSpec(4, n_sp, figure=fig, hspace=0.30, wspace=0.28,
                  top=0.97, bottom=0.04, left=0.06, right=0.96)

    cmap_k = plt.cm.viridis
    cmap_load = plt.cm.plasma

    # helper to pick results
    def pick(sp, k, n, mf=None):
        return [r for r in res
                if r["sparsity"] == sp and r["k"] == k and r["n_patterns"] == n
                and (mf is None or r["mask_frac"] == mf)]

    for col, sp in enumerate(sparsities):

        # --- Row 0: Training curves ---
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_title(f"Training curves  (sparsity={sp})", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Avg top-1 cosine sim")

        line_idx = 0
        configs_for_sp = [(k, n) for s, k, n in training_curves if s == sp]
        n_lines = len(configs_for_sp)
        for k_val in ks:
            for n_val in ns:
                key = (sp, k_val, n_val)
                if key not in training_curves:
                    continue
                curve = training_curves[key]  # (epochs, n_patterns)
                avg_curve = np.mean(curve, axis=1)
                color = cmap_load(line_idx / max(n_lines - 1, 1))
                ax0.plot(avg_curve, color=color, linewidth=1.0,
                         label=f"k={k_val} N={n_val}")
                line_idx += 1
        ax0.legend(fontsize=6, ncol=2, loc="lower right")
        ax0.set_ylim(0, 1.05)
        ax0.grid(True, alpha=0.3)

        # --- Row 1: Retrieval accuracy vs mask fraction ---
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_title(f"Retrieval accuracy  (sparsity={sp})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Mask fraction")
        ax1.set_ylabel("Accuracy")

        line_idx = 0
        for k_val in ks:
            for n_val in ns:
                accs = []
                for mf in mfs:
                    rows = pick(sp, k_val, n_val, mf)
                    if rows:
                        accs.append(rows[0]["accuracy"])
                    else:
                        accs.append(np.nan)
                if all(np.isnan(a) for a in accs):
                    continue
                color = cmap_load(line_idx / max(n_lines - 1, 1))
                ax1.plot(mfs, accs, "o-", color=color, linewidth=1.2, markersize=3,
                         label=f"k={k_val} N={n_val}")
                line_idx += 1
        ax1.legend(fontsize=6, ncol=2, loc="lower left")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # --- Row 2: Cosine similarity (full vs masked) ---
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_title(f"Cosine similarity  (sparsity={sp})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mask fraction")
        ax2.set_ylabel("Cosine similarity")

        line_idx = 0
        for k_val in ks:
            for n_val in ns:
                full_sims = []
                masked_sims = []
                for mf in mfs:
                    rows = pick(sp, k_val, n_val, mf)
                    if rows:
                        full_sims.append(rows[0]["full_sim"])
                        masked_sims.append(rows[0]["masked_sim"])
                    else:
                        full_sims.append(np.nan)
                        masked_sims.append(np.nan)
                if all(np.isnan(a) for a in masked_sims):
                    continue
                color = cmap_load(line_idx / max(n_lines - 1, 1))
                ax2.plot(mfs, masked_sims, "o-", color=color, linewidth=1.2, markersize=3,
                         label=f"k={k_val} N={n_val}")
                # dashed line for full (baseline) — just first value repeated
                if not np.isnan(full_sims[0]):
                    ax2.axhline(full_sims[0], color=color, linestyle="--",
                                linewidth=0.7, alpha=0.5)
                line_idx += 1
        ax2.legend(fontsize=6, ncol=2, loc="lower left")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        # --- Row 3: Retention (%) vs mask fraction ---
        ax3 = fig.add_subplot(gs[3, col])
        ax3.set_title(f"Retention %  (sparsity={sp})", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Mask fraction")
        ax3.set_ylabel("Retention  (masked_sim / full_sim)")

        line_idx = 0
        for k_val in ks:
            for n_val in ns:
                rets = []
                for mf in mfs:
                    rows = pick(sp, k_val, n_val, mf)
                    if rows:
                        rets.append(rows[0]["retention"])
                    else:
                        rets.append(np.nan)
                if all(np.isnan(a) for a in rets):
                    continue
                color = cmap_load(line_idx / max(n_lines - 1, 1))
                ax3.plot(mfs, rets, "o-", color=color, linewidth=1.2, markersize=3,
                         label=f"k={k_val} N={n_val}")
                line_idx += 1
        ax3.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax3.legend(fontsize=6, ncol=2, loc="lower left")
        ax3.set_ylim(0, 1.15)
        ax3.grid(True, alpha=0.3)

    out_path = OUT_DIR / "pattern_completion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")

    # Also save a capacity-focused plot: accuracy heatmaps (load vs mask)
    plot_capacity_heatmaps(results)


def plot_capacity_heatmaps(results: list[dict]):
    """One heatmap per sparsity: x=mask_fraction, y=load (N/k), value=accuracy."""

    sparsities = sorted(set(r["sparsity"] for r in results))
    mfs = sorted(set(r["mask_frac"] for r in results))

    n_sp = len(sparsities)
    fig, axes = plt.subplots(2, n_sp, figsize=(7 * n_sp, 10), dpi=150)
    if n_sp == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Capacity analysis  —  Load (N/k) vs Mask fraction",
                 fontsize=14, fontweight="bold")

    for col, sp in enumerate(sparsities):
        sub = [r for r in results if r["sparsity"] == sp]
        loads = sorted(set(r["load"] for r in sub))

        # accuracy heatmap
        acc_grid = np.full((len(loads), len(mfs)), np.nan)
        ret_grid = np.full((len(loads), len(mfs)), np.nan)
        for r in sub:
            li = loads.index(r["load"])
            mi = mfs.index(r["mask_frac"])
            # average if multiple configs share the same load
            if np.isnan(acc_grid[li, mi]):
                acc_grid[li, mi] = r["accuracy"]
                ret_grid[li, mi] = r["retention"]
            else:
                acc_grid[li, mi] = (acc_grid[li, mi] + r["accuracy"]) / 2
                ret_grid[li, mi] = (ret_grid[li, mi] + r["retention"]) / 2

        load_labels = [f"{l:.2f}" for l in loads]
        mf_labels = [f"{m:.1f}" for m in mfs]

        ax_acc = axes[0, col]
        im = ax_acc.imshow(acc_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                           origin="lower")
        ax_acc.set_xticks(range(len(mfs)))
        ax_acc.set_xticklabels(mf_labels, fontsize=8)
        ax_acc.set_yticks(range(len(loads)))
        ax_acc.set_yticklabels(load_labels, fontsize=8)
        ax_acc.set_xlabel("Mask fraction")
        ax_acc.set_ylabel("Load (N / k)")
        ax_acc.set_title(f"Accuracy  (sparsity={sp})", fontweight="bold")
        # annotate cells
        for i in range(len(loads)):
            for j in range(len(mfs)):
                v = acc_grid[i, j]
                if not np.isnan(v):
                    ax_acc.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                color="black" if v > 0.4 else "white")
        fig.colorbar(im, ax=ax_acc, shrink=0.8)

        ax_ret = axes[1, col]
        im2 = ax_ret.imshow(ret_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.1,
                            origin="lower")
        ax_ret.set_xticks(range(len(mfs)))
        ax_ret.set_xticklabels(mf_labels, fontsize=8)
        ax_ret.set_yticks(range(len(loads)))
        ax_ret.set_yticklabels(load_labels, fontsize=8)
        ax_ret.set_xlabel("Mask fraction")
        ax_ret.set_ylabel("Load (N / k)")
        ax_ret.set_title(f"Retention  (sparsity={sp})", fontweight="bold")
        for i in range(len(loads)):
            for j in range(len(mfs)):
                v = ret_grid[i, j]
                if not np.isnan(v):
                    ax_ret.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                                color="black" if v > 0.4 else "white")
        fig.colorbar(im2, ax=ax_ret, shrink=0.8)

    out_path = OUT_DIR / "capacity_heatmaps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity heatmaps to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, training_curves = run_sweep()
    plot_results(results, training_curves)
    print("Done.")
