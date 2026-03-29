"""Zenith fixed-permutation completion via projection.

Stage 1 learns a sparse code S(X) from source-only patterns [X, 0].
Stage 2 learns full patterns [S(X), P(X)] where P is one fixed hidden
permutation. At test time:

    Y -> [Y, 0] -> Stage 1 -> S(Y)
    [S(Y), 0] -> Stage 2 -> projection reconstruction -> P_hat(Y)

For comparison, the script also trains a direct model on [X, P(X)] and tests it
with [Y, 0] -> projection reconstruction, plus a nearest-neighbor baseline in
the raw source space.

Usage:
    .venv/bin/python experiments/2026_03_29/permutation_projection/permutation_projection.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PART_DIM = 128
SPARSITY = 0.10

N_TRAIN = 300
N_TEST = 100

STAGE1_TEMPLATE_COUNT = 10
STAGE1_NODE_COUNT = 100

STAGE2_TEMPLATE_COUNT = 10
STAGE2_NODE_COUNT = 100

DIRECT_TEMPLATE_COUNT = 10
DIRECT_NODE_COUNT = 100

TRAINING_EPOCHS = 600
ETA = 0.03

SEED = 42
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_sparse_vectors(
    n: int,
    dim: int,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create n sparse binary vectors of fixed sparsity."""
    n_active = max(1, int(round(dim * sparsity)))
    patterns = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        idx = rng.choice(dim, size=n_active, replace=False)
        patterns[i, idx] = 1.0
    return patterns


def make_hidden_permutation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Create one fixed hidden permutation over indices [0, dim)."""
    return rng.permutation(dim)


def apply_permutation_batch(vectors: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply the fixed permutation to a batch of sparse vectors."""
    return vectors[:, perm]


def make_source_only_patterns(sources: np.ndarray) -> np.ndarray:
    """Create [X, 0] patterns for stage 1."""
    zeros = np.zeros_like(sources)
    return np.concatenate([sources, zeros], axis=1)


def make_full_patterns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Create [left, right] concatenated patterns."""
    return np.concatenate([left, right], axis=1)


# ---------------------------------------------------------------------------
# Zenith core
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, EPS)


def init_weights(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal((k, dim))
    w -= np.mean(w, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True) + EPS
    return w


def top1_activation_from_normalized(w: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Return the sparse top-1 activation vector for one node."""
    c = w @ x_hat
    winner = int(np.argmax(np.abs(c)))
    a = np.zeros_like(c)
    a[winner] = c[winner]
    return a


def zenith_learn(w: np.ndarray, x: np.ndarray, eta: float) -> np.ndarray:
    x_hat = _normalize(x)
    c = w @ x_hat
    winner = int(np.argmax(np.abs(c)))
    c_w = c[winner]

    w_win = w[winner]
    tau = x_hat - c_w * w_win
    tau_norm = np.linalg.norm(tau)
    tau_hat = tau / tau_norm if tau_norm > EPS else np.zeros_like(tau)

    theta = eta * c_w
    w[winner] = w_win * np.cos(theta) + tau_hat * np.sin(theta)
    return w


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def _train_single_node(args: tuple) -> np.ndarray:
    patterns, k, eta, epochs, seed = args
    rng = np.random.default_rng(seed)
    n, dim = patterns.shape
    w = init_weights(k, dim, rng)
    for _ in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            w = zenith_learn(w, patterns[idx], eta)
    return w


def train_ensemble(
    patterns: np.ndarray,
    m: int,
    k: int,
    eta: float,
    epochs: int,
    rng: np.random.Generator,
    label: str,
) -> list[np.ndarray]:
    """Train M independent Zenith nodes in parallel."""
    seeds = [int(rng.integers(0, 2**32)) for _ in range(m)]
    args = [(patterns, k, eta, epochs, s) for s in seeds]

    print(f"Training {label}: {m} nodes in parallel...")
    with ProcessPoolExecutor() as pool:
        ensemble = list(tqdm(pool.map(_train_single_node, args), total=m, desc=label))

    return ensemble


def encode_pattern(ensemble: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Encode one input as concatenated sparse top-1 activations from all nodes."""
    x_hat = _normalize(x)
    codes = [top1_activation_from_normalized(w, x_hat) for w in ensemble]
    return np.concatenate(codes)


def encode_batch(ensemble: list[np.ndarray], patterns: np.ndarray) -> np.ndarray:
    """Encode a batch of inputs."""
    return np.array([encode_pattern(ensemble, x) for x in patterns])


def reconstruct_pattern(ensemble: list[np.ndarray], x: np.ndarray) -> tuple[np.ndarray, float]:
    """Projection reconstruction from an ensemble.

    Each node computes its sparse top-1 activation and reconstructs via W^T a.
    The final reconstruction is the average over nodes.

    Returns:
        reconstruction  — average projected reconstruction
        avg_response    — mean top-1 |activation| across nodes
    """
    x_hat = _normalize(x)
    recons = []
    top1_mags = []
    for w in ensemble:
        a = top1_activation_from_normalized(w, x_hat)
        recons.append(w.T @ a)
        top1_mags.append(float(np.max(np.abs(a))))
    return np.mean(recons, axis=0), float(np.mean(top1_mags))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def binarize_topk(v: np.ndarray, k: int) -> np.ndarray:
    """Convert a dense score vector to a binary sparse vector with k active bits."""
    out = np.zeros_like(v)
    if k <= 0:
        return out
    idx = np.argpartition(v, -k)[-k:]
    out[idx] = 1.0
    return out


def _score(metrics_dict: dict, retrieved: np.ndarray, true: np.ndarray):
    """Append set-based reconstruction metrics."""
    r_set = set(np.nonzero(retrieved)[0])
    t_set = set(np.nonzero(true)[0])

    if np.array_equal(retrieved, true):
        metrics_dict["exact"] += 1

    intersection = len(r_set & t_set)
    union = len(r_set | t_set)
    jaccard = intersection / union if union > 0 else 1.0
    metrics_dict["jaccard"].append(jaccard)

    precision = intersection / len(r_set) if len(r_set) > 0 else (1.0 if len(t_set) == 0 else 0.0)
    metrics_dict["precision"].append(precision)

    recall = intersection / len(t_set) if len(t_set) > 0 else (1.0 if len(r_set) == 0 else 0.0)
    metrics_dict["recall"].append(recall)


def evaluate_methods(
    stage1_ensemble: list[np.ndarray],
    stage2_ensemble: list[np.ndarray],
    direct_ensemble: list[np.ndarray],
    train_sources: np.ndarray,
    train_targets: np.ndarray,
    test_sources: np.ndarray,
    test_targets: np.ndarray,
) -> dict:
    """Evaluate source NN, direct completion, and two-stage completion."""
    methods = {
        "source_nn": {"exact": 0, "jaccard": [], "precision": [], "recall": [], "response": []},
        "direct": {"exact": 0, "jaccard": [], "precision": [], "recall": [], "response": []},
        "two_stage": {"exact": 0, "jaccard": [], "precision": [], "recall": [], "response": []},
    }

    train_source_norm = _normalize_rows(train_sources)
    n_active = int(np.sum(test_targets[0]))

    for i in tqdm(range(len(test_sources)), desc="Evaluating", leave=False):
        true_target = test_targets[i]

        # Baseline: nearest training source in raw source space
        src_hat = _normalize(test_sources[i])
        nn_idx = int(np.argmax(train_source_norm @ src_hat))
        _score(methods["source_nn"], train_targets[nn_idx], true_target)
        methods["source_nn"]["response"].append(float(np.max(train_source_norm @ src_hat)))

        # Direct completion: train on [X, P(X)], test with [Y, 0]
        direct_input = make_full_patterns(test_sources[i:i+1], np.zeros((1, PART_DIM), dtype=np.float64))[0]
        direct_recon, direct_resp = reconstruct_pattern(direct_ensemble, direct_input)
        direct_target = binarize_topk(direct_recon[PART_DIM:], n_active)
        _score(methods["direct"], direct_target, true_target)
        methods["direct"]["response"].append(direct_resp)

        # Two-stage completion: [Y, 0] -> S(Y), then [S(Y), 0] -> P_hat(Y)
        stage1_input = make_full_patterns(test_sources[i:i+1], np.zeros((1, PART_DIM), dtype=np.float64))[0]
        code = encode_pattern(stage1_ensemble, stage1_input)
        stage2_input = make_full_patterns(code[None, :], np.zeros((1, PART_DIM), dtype=np.float64))[0]
        stage2_recon, stage2_resp = reconstruct_pattern(stage2_ensemble, stage2_input)
        stage2_target = binarize_topk(stage2_recon[len(code):], n_active)
        _score(methods["two_stage"], stage2_target, true_target)
        methods["two_stage"]["response"].append(stage2_resp)

    result = {}
    n_test = len(test_sources)
    for method, metrics in methods.items():
        result[method] = {
            "exact_match": metrics["exact"] / n_test,
            "jaccard": float(np.mean(metrics["jaccard"])),
            "precision": float(np.mean(metrics["precision"])),
            "recall": float(np.mean(metrics["recall"])),
            "avg_response": float(np.mean(metrics["response"])),
            "per_sample_jaccard": metrics["jaccard"],
        }

    result["gaps"] = {
        "two_stage_minus_direct": result["two_stage"]["jaccard"] - result["direct"]["jaccard"],
        "two_stage_minus_nn": result["two_stage"]["jaccard"] - result["source_nn"]["jaccard"],
        "direct_minus_nn": result["direct"]["jaccard"] - result["source_nn"]["jaccard"],
    }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(result: dict, code_dim: int):
    method_order = [
        ("source_nn", "Source NN", "C1"),
        ("direct", "Direct", "C0"),
        ("two_stage", "Two-stage", "C2"),
    ]

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.suptitle(
        f"Zenith — Fixed Permutation Completion via Projection  |  "
        f"part_dim={PART_DIM}, sparsity={SPARSITY}, "
        f"N_train={N_TRAIN}, N_test={N_TEST}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  top=0.93, bottom=0.08, left=0.07, right=0.96)

    # --- (0,0) Jaccard comparison ---
    ax = fig.add_subplot(gs[0, 0])
    labels = [label for _, label, _ in method_order]
    vals = [result[key]["jaccard"] for key, _, _ in method_order]
    colors = [color for _, _, color in method_order]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Target reconstruction\nTwo-stage gap vs direct: {result['gaps']['two_stage_minus_direct']:+.3f}",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,1) Precision / recall comparison ---
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(method_order))
    w = 0.35
    prec = [result[key]["precision"] for key, _, _ in method_order]
    rec = [result[key]["recall"] for key, _, _ in method_order]
    ax.bar(x - w / 2, prec, w, label="Precision", color="C4", alpha=0.85)
    ax.bar(x + w / 2, rec, w, label="Recall", color="C5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision and Recall", fontweight="bold", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,2) Exact match comparison ---
    ax = fig.add_subplot(gs[0, 2])
    exacts = [result[key]["exact_match"] for key, _, _ in method_order]
    bars = ax.bar(labels, exacts, color=colors, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Exact match rate")
    ax.set_ylim(0, min(1.05, max(exacts) * 1.35 + 0.02))
    ax.set_title("Exact target reconstruction", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,0) Per-sample jaccard distributions ---
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0.0, 1.0, 21)
    for key, label, color in method_order:
        ax.hist(result[key]["per_sample_jaccard"], bins=bins, alpha=0.45, label=label, color=color)
    ax.set_xlabel("Per-sample Jaccard")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction quality distribution", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Summary text ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    summary_lines = [
        f"Source dim: {PART_DIM}",
        f"Target dim: {PART_DIM}",
        f"Stage-1 code dim: {code_dim}",
        f"Stage-1 sparsity: ~{STAGE1_NODE_COUNT} active",
        f"Sparsity: {SPARSITY:.0%} ({int(PART_DIM * SPARSITY)} active bits)",
        f"Train: {N_TRAIN}",
        f"Test:  {N_TEST}",
        f"",
        f"Two-stage Jaccard: {result['two_stage']['jaccard']:.3f}",
        f"Direct Jaccard:    {result['direct']['jaccard']:.3f}",
        f"Source NN:         {result['source_nn']['jaccard']:.3f}",
        f"",
        f"Two-stage - direct: {result['gaps']['two_stage_minus_direct']:+.3f}",
        f"Two-stage - NN:     {result['gaps']['two_stage_minus_nn']:+.3f}",
    ]
    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    # --- (1,2) Pipeline text ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    pipeline = [
        "Stage 1 train:",
        "  [X, 0] -> S(X)",
        "",
        "Stage 2 train:",
        "  [S(X), P(X)]",
        "",
        "Test:",
        "  [Y, 0] -> S(Y)",
        "  [S(Y), 0] -> projection",
        "  take output half -> P_hat(Y)",
        "",
        "Baseline direct:",
        "  [Y, 0] -> projection",
        "  from model trained on [X, P(X)]",
    ]
    ax.text(0.1, 0.95, "\n".join(pipeline), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

    out_path = OUT_DIR / "permutation_projection.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(result: dict, code_dim: int):
    lines = [
        "# Zenith — Fixed Permutation Completion via Projection Report",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Source dimensionality | {PART_DIM} |",
        f"| Target dimensionality | {PART_DIM} |",
        f"| Source/target sparsity | {SPARSITY} |",
        f"| Active bits per vector | {int(PART_DIM * SPARSITY)} |",
        f"| Training examples | {N_TRAIN} |",
        f"| Test examples | {N_TEST} |",
        f"| Stage-1 nodes | {STAGE1_NODE_COUNT} |",
        f"| Stage-1 templates per node | {STAGE1_TEMPLATE_COUNT} |",
        f"| Stage-1 code dimensionality | {code_dim} |",
        f"| Stage-2 nodes | {STAGE2_NODE_COUNT} |",
        f"| Stage-2 templates per node | {STAGE2_TEMPLATE_COUNT} |",
        f"| Direct-model nodes | {DIRECT_NODE_COUNT} |",
        f"| Direct-model templates per node | {DIRECT_TEMPLATE_COUNT} |",
        f"| Learning rate | {ETA} |",
        f"| Training epochs | {TRAINING_EPOCHS} |",
        "",
        "## Task",
        "",
        "One fixed hidden permutation P is used for all examples.",
        "Training sources X are random sparse vectors. Targets are P(X).",
        "The test sources Y are unseen sparse vectors; the goal is to reconstruct P(Y).",
        "",
        "## Methods",
        "",
        "- **Source NN**: nearest neighbor in source space, return that training example's P(X).",
        "- **Direct**: train Zenith on [X, P(X)], then test with [Y, 0] and reconstruct by projection.",
        "- **Two-stage**: train Stage 1 on [X, 0] to get S(X); train Stage 2 on [S(X), P(X)]; test with [S(Y), 0] and reconstruct by projection.",
        "",
        "## Results",
        "",
        "| Method | Jaccard | Precision | Recall | Exact Match | Avg Response |",
        "|--------|---------|-----------|--------|-------------|--------------|",
    ]

    for key, label in [("source_nn", "Source NN"), ("direct", "Direct"), ("two_stage", "Two-stage")]:
        stats = result[key]
        lines.append(
            f"| {label} | {stats['jaccard']:.4f} | {stats['precision']:.4f} | "
            f"{stats['recall']:.4f} | {stats['exact_match']:.2%} | {stats['avg_response']:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- Two-stage minus direct (Jaccard): **{result['gaps']['two_stage_minus_direct']:+.4f}**",
        f"- Two-stage minus source NN (Jaccard): **{result['gaps']['two_stage_minus_nn']:+.4f}**",
        f"- Direct minus source NN (Jaccard): **{result['gaps']['direct_minus_nn']:+.4f}**",
        "",
        "- **Jaccard** measures overlap between predicted and true active target bits.",
        "- **Precision** asks: of the predicted active target bits, how many are correct?",
        "- **Recall** asks: of the true target bits, how many were recovered?",
        "- **Exact match** is the fraction of targets reconstructed perfectly.",
        "",
    ]

    if result["two_stage"]["jaccard"] > result["direct"]["jaccard"] + 0.01:
        lines.append(
            "The stage-1 code helps. Zenith's learned source representation supports better downstream target reconstruction than direct [X, P(X)] completion."
        )
    elif result["direct"]["jaccard"] > result["two_stage"]["jaccard"] + 0.01:
        lines.append(
            "The stage-1 code hurts or discards useful information. Direct completion from [Y, 0] works better than reconstructing through S(Y)."
        )
    else:
        lines.append(
            "Two-stage and direct completion perform similarly. The stage-1 code neither clearly helps nor hurts under this configuration."
        )

    lines += [
        "",
        "### Caveat",
        "",
        "This is still prototype completion, not symbolic rule execution.",
        "A good result means the learned code and templates support completion of a fixed transformation on unseen sparse vectors.",
        "",
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

    print(f"Config: part_dim={PART_DIM}, sparsity={SPARSITY}, "
          f"train={N_TRAIN}, test={N_TEST}, epochs={TRAINING_EPOCHS}")

    perm = make_hidden_permutation(PART_DIM, rng)
    print("Generated hidden permutation P.")

    train_sources = make_sparse_vectors(N_TRAIN, PART_DIM, SPARSITY, rng)
    test_sources = make_sparse_vectors(N_TEST, PART_DIM, SPARSITY, rng)
    train_targets = apply_permutation_batch(train_sources, perm)
    test_targets = apply_permutation_batch(test_sources, perm)

    # Stage 1: train on source-only patterns [X, 0]
    stage1_train = make_source_only_patterns(train_sources)
    stage1_ensemble = train_ensemble(
        stage1_train,
        STAGE1_NODE_COUNT,
        STAGE1_TEMPLATE_COUNT,
        ETA,
        TRAINING_EPOCHS,
        rng,
        label="Stage1",
    )

    # Build stage-1 codes for stage-2 training
    print("Encoding training set through Stage 1...")
    stage1_codes = encode_batch(stage1_ensemble, stage1_train)
    code_dim = stage1_codes.shape[1]
    print(f"Stage-1 code dimensionality: {code_dim}")

    # Stage 2: train on [S(X), P(X)]
    stage2_train = make_full_patterns(stage1_codes, train_targets)
    stage2_ensemble = train_ensemble(
        stage2_train,
        STAGE2_NODE_COUNT,
        STAGE2_TEMPLATE_COUNT,
        ETA,
        TRAINING_EPOCHS,
        rng,
        label="Stage2",
    )

    # Direct baseline: train on [X, P(X)]
    direct_train = make_full_patterns(train_sources, train_targets)
    direct_ensemble = train_ensemble(
        direct_train,
        DIRECT_NODE_COUNT,
        DIRECT_TEMPLATE_COUNT,
        ETA,
        TRAINING_EPOCHS,
        rng,
        label="Direct",
    )

    print("\nEvaluating reconstruction...")
    result = evaluate_methods(
        stage1_ensemble,
        stage2_ensemble,
        direct_ensemble,
        train_sources,
        train_targets,
        test_sources,
        test_targets,
    )

    for key, label in [("source_nn", "Source NN"), ("direct", "Direct"), ("two_stage", "Two-stage")]:
        stats = result[key]
        print(f"  {label:<9} jaccard={stats['jaccard']:.4f}  "
              f"prec={stats['precision']:.4f}  "
              f"rec={stats['recall']:.4f}  "
              f"exact={stats['exact_match']:.2%}")

    print(f"  Two-stage - direct jaccard: {result['gaps']['two_stage_minus_direct']:+.4f}")
    print(f"  Two-stage - NN jaccard:     {result['gaps']['two_stage_minus_nn']:+.4f}")

    plot_results(result, code_dim)
    write_report(result, code_dim)
    print("\nDone.")
