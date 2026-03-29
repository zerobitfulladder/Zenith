"""Zenith ensemble intersection (AND) learning test.

Tests whether the ensemble can learn the AND operation from examples.

Data model:
    Each training pattern = [A, B, A∩B] where A and B are sparse binary vectors
    and A∩B is their bitwise intersection (overlapping active bits).
    Train on many such triplets.
    Test: present [X, Y, 0] with never-seen X,Y and check if the nearest stored
    pattern's third part matches X∩Y.

Comparison:
    - Ensemble NN: find nearest training pattern in activation space, take its C part
    - Raw NN: find nearest training pattern in raw input space, take its C part
    - Measure how close the retrieved C is to the true X∩Y

Usage:
    .venv/bin/python experiments/2026_03_29/intersection_test/intersection_test.py
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

PART_DIM = 100               # dimensionality of each part (A, B, C)
SPARSITY = 0.20              # fraction of active bits in A and B

N_TRAIN = 500                # number of training triplets
N_TEST = 200                 # number of test triplets

TEMPLATE_COUNT = 10          # k per Zenith node
NODE_COUNT = 30              # M ensemble nodes

TRAINING_EPOCHS = 1000
ETA = 0.03

SEED = 42
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_sparse_vector(dim: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    n_active = max(1, int(round(dim * sparsity)))
    v = np.zeros(dim, dtype=np.float64)
    idx = rng.choice(dim, size=n_active, replace=False)
    v[idx] = 1.0
    return v


def make_triplets(
    n: int,
    part_dim: int,
    sparsity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate n triplets [A, B, C] where C = A AND B.

    Returns:
        patterns   — (n, 3*part_dim) concatenated [A, B, C]
        parts_a    — (n, part_dim)
        parts_b    — (n, part_dim)
        parts_c    — (n, part_dim) the true intersections
    """
    patterns = np.zeros((n, 3 * part_dim), dtype=np.float64)
    parts_a = np.zeros((n, part_dim), dtype=np.float64)
    parts_b = np.zeros((n, part_dim), dtype=np.float64)
    parts_c = np.zeros((n, part_dim), dtype=np.float64)

    for i in range(n):
        a = make_sparse_vector(part_dim, sparsity, rng)
        b = make_sparse_vector(part_dim, sparsity, rng)
        c = a * b  # bitwise AND for binary vectors

        parts_a[i] = a
        parts_b[i] = b
        parts_c[i] = c

        patterns[i, :part_dim] = a
        patterns[i, part_dim:2*part_dim] = b
        patterns[i, 2*part_dim:] = c

    return patterns, parts_a, parts_b, parts_c


def make_masked_input(a: np.ndarray, b: np.ndarray, part_dim: int) -> np.ndarray:
    """Create [A, B, 0] — C part is zeroed out."""
    pattern = np.zeros(3 * part_dim, dtype=np.float64)
    pattern[:part_dim] = a
    pattern[part_dim:2*part_dim] = b
    return pattern


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
# Ensemble
# ---------------------------------------------------------------------------

def init_ensemble(m: int, k: int, dim: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [init_weights(k, dim, rng) for _ in range(m)]


def ensemble_forward(ensemble: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    activations = [zenith_forward(w, x) for w in ensemble]
    return np.concatenate(activations)


def ensemble_learn(ensemble: list[np.ndarray], x: np.ndarray, eta: float) -> list[np.ndarray]:
    return [zenith_learn(w, x, eta) for w in ensemble]


# ---------------------------------------------------------------------------
# Training (parallel)
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


def train(
    patterns: np.ndarray,
    m: int,
    k: int,
    eta: float,
    epochs: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    seeds = [int(rng.integers(0, 2**32)) for _ in range(m)]
    args = [(patterns, k, eta, epochs, s) for s in seeds]

    print(f"Training {m} nodes in parallel...")
    with ProcessPoolExecutor() as pool:
        ensemble = list(tqdm(pool.map(_train_single_node, args), total=m, desc="Training nodes"))

    return ensemble


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(a @ b / (na * nb))


def evaluate_intersection(
    ensemble: list[np.ndarray],
    train_patterns: np.ndarray,
    train_c: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    test_c: np.ndarray,
) -> dict:
    """Evaluate AND reconstruction quality.

    For each test pair (X, Y):
        - Present [X, Y, 0] to the ensemble
        - Find nearest training pattern in activation space → retrieve its C part
        - Find nearest training pattern in raw input space → retrieve its C part
        - Compare retrieved C against true X∩Y

    Metrics:
        - Exact match rate: retrieved C == true C (all bits correct)
        - Jaccard similarity: |retrieved ∩ true| / |retrieved ∪ true|
        - Precision: fraction of retrieved active bits that are correct
        - Recall: fraction of true active bits that are retrieved
    """
    n_train = train_patterns.shape[0]
    n_test = test_a.shape[0]

    # precompute training activations
    train_acts = np.array([ensemble_forward(ensemble, p) for p in train_patterns])
    train_raw_norm = np.array([_normalize(p) for p in train_patterns])

    metrics = {
        "ensemble": {"exact": 0, "jaccard": [], "precision": [], "recall": []},
        "raw": {"exact": 0, "jaccard": [], "precision": [], "recall": []},
    }

    for i in tqdm(range(n_test), desc="Evaluating", leave=False):
        masked_input = make_masked_input(test_a[i], test_b[i], PART_DIM)
        true_c = test_c[i]

        # ensemble NN
        test_act = ensemble_forward(ensemble, masked_input)
        act_sims = np.array([_cosine_sim(test_act, train_acts[j]) for j in range(n_train)])
        ens_retrieved_c = train_c[np.argmax(act_sims)]

        # raw NN
        test_raw = _normalize(masked_input)
        raw_sims = train_raw_norm @ test_raw
        raw_retrieved_c = train_c[np.argmax(raw_sims)]

        # compute metrics for both
        for method, retrieved_c in [("ensemble", ens_retrieved_c), ("raw", raw_retrieved_c)]:
            _score(metrics[method], retrieved_c, true_c)

    # aggregate
    n = n_test
    result = {}
    for method in ["ensemble", "raw"]:
        m = metrics[method]
        result[method] = {
            "exact_match": m["exact"] / n,
            "jaccard": float(np.mean(m["jaccard"])),
            "precision": float(np.mean(m["precision"])),
            "recall": float(np.mean(m["recall"])),
        }
    result["gap_jaccard"] = result["ensemble"]["jaccard"] - result["raw"]["jaccard"]
    result["gap_exact"] = result["ensemble"]["exact_match"] - result["raw"]["exact_match"]

    return result


def _score(metrics_dict: dict, retrieved: np.ndarray, true: np.ndarray):
    """Compute and append binary set metrics."""
    r_set = set(np.nonzero(retrieved)[0])
    t_set = set(np.nonzero(true)[0])

    # exact match (as binary vectors)
    if np.array_equal(retrieved, true):
        metrics_dict["exact"] += 1

    # jaccard
    intersection = len(r_set & t_set)
    union = len(r_set | t_set)
    jaccard = intersection / union if union > 0 else 1.0
    metrics_dict["jaccard"].append(jaccard)

    # precision: of the bits I retrieved, how many are correct?
    precision = intersection / len(r_set) if len(r_set) > 0 else (1.0 if len(t_set) == 0 else 0.0)
    metrics_dict["precision"].append(precision)

    # recall: of the true bits, how many did I retrieve?
    recall = intersection / len(t_set) if len(t_set) > 0 else (1.0 if len(r_set) == 0 else 0.0)
    metrics_dict["recall"].append(recall)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(result: dict, train_stats: dict):
    total_dim = 3 * PART_DIM
    load = N_TRAIN / TEMPLATE_COUNT

    fig = plt.figure(figsize=(16, 10), dpi=150)
    fig.suptitle(
        f"Zenith Ensemble — AND Reconstruction  |  "
        f"part_dim={PART_DIM}, sparsity={SPARSITY}, "
        f"N_train={N_TRAIN}, N_test={N_TEST}, "
        f"M={NODE_COUNT}, k={TEMPLATE_COUNT}, load={load:.0f}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  top=0.93, bottom=0.08, left=0.07, right=0.96)

    # --- (0,0) Jaccard comparison ---
    ax = fig.add_subplot(gs[0, 0])
    methods = ["Raw NN", "Ensemble NN"]
    jaccards = [result["raw"]["jaccard"], result["ensemble"]["jaccard"]]
    bars = ax.bar(methods, jaccards, color=["C1", "C0"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"AND reconstruction (Jaccard)\nGap: {result['gap_jaccard']:+.3f}",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,1) Precision / Recall comparison ---
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    w = 0.3
    prec = [result["raw"]["precision"], result["ensemble"]["precision"]]
    rec = [result["raw"]["recall"], result["ensemble"]["recall"]]
    ax.bar(x - w/2, prec, w, label="Precision", color="C2", alpha=0.8)
    ax.bar(x + w/2, rec, w, label="Recall", color="C4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision & Recall", fontweight="bold", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,2) Exact match comparison ---
    ax = fig.add_subplot(gs[0, 2])
    exacts = [result["raw"]["exact_match"], result["ensemble"]["exact_match"]]
    bars = ax.bar(methods, exacts, color=["C1", "C0"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exact match rate")
    ax.set_ylim(0, max(max(exacts) * 1.3, 0.1))
    ax.set_title("Exact C reconstruction", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,0) Distribution of intersection sizes ---
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Intersection size distribution (train)", fontweight="bold", fontsize=10)
    ax.hist(train_stats["intersection_sizes"], bins=range(0, max(train_stats["intersection_sizes"]) + 2),
            color="C3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("|A ∩ B| (number of overlapping bits)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Summary text ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_active = int(PART_DIM * SPARSITY)
    expected_overlap = n_active * SPARSITY  # E[|A∩B|] = n_active * (n_active/dim)
    summary_lines = [
        f"Part dim: {PART_DIM}",
        f"Sparsity: {SPARSITY:.0%} ({n_active} active bits)",
        f"Total dim: {total_dim}",
        f"Expected |A∩B|: ~{expected_overlap:.1f} bits",
        f"Actual mean |A∩B|: {np.mean(train_stats['intersection_sizes']):.1f}",
        f"",
        f"Train: {N_TRAIN} triplets",
        f"Test:  {N_TEST} triplets",
        f"Ensemble: {NODE_COUNT} nodes, k={TEMPLATE_COUNT}",
        f"",
        f"Jaccard gap: {result['gap_jaccard']:+.3f}",
        f"Exact gap:   {result['gap_exact']:+.1%}",
    ]
    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    # --- (1,2) How the task works ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    task_desc = [
        "Task: Learn A∩B from examples",
        "",
        "Train patterns:  [A, B, A∩B]",
        "Test input:      [X, Y,  0 ]",
        "Expected output: X∩Y",
        "",
        "Method: find nearest stored",
        "pattern, take its C part.",
        "",
        "Ensemble NN: search in",
        "  activation space",
        "Raw NN: search in",
        "  raw input space",
    ]
    ax.text(0.1, 0.95, "\n".join(task_desc), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

    out_path = OUT_DIR / "intersection.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(result: dict, train_stats: dict):
    load = N_TRAIN / TEMPLATE_COUNT
    n_active = int(PART_DIM * SPARSITY)
    expected_overlap = n_active * SPARSITY

    lines = [
        "# Zenith Ensemble — Intersection (AND) Learning Report",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Part dimensionality | {PART_DIM} |",
        f"| Total dimensionality | {3 * PART_DIM} |",
        f"| Sparsity | {SPARSITY} |",
        f"| Active bits per part | {n_active} |",
        f"| Expected \\|A∩B\\| | ~{expected_overlap:.1f} |",
        f"| Actual mean \\|A∩B\\| (train) | {np.mean(train_stats['intersection_sizes']):.1f} |",
        f"| Training triplets | {N_TRAIN} |",
        f"| Test triplets | {N_TEST} |",
        f"| Templates per node (k) | {TEMPLATE_COUNT} |",
        f"| Ensemble nodes (M) | {NODE_COUNT} |",
        f"| Load (N_train/k) | {load:.0f} |",
        f"| Learning rate | {ETA} |",
        f"| Training epochs | {TRAINING_EPOCHS} |",
        "",
        "## Task",
        "",
        "Train on [A, B, A∩B] triplets. Test: present [X, Y, 0] with never-seen X, Y.",
        "Retrieve nearest stored pattern, take its C part. Compare to true X∩Y.",
        "",
        "## Results",
        "",
        "| Method | Jaccard | Precision | Recall | Exact Match |",
        "|--------|---------|-----------|--------|-------------|",
        f"| Raw NN | {result['raw']['jaccard']:.4f} | {result['raw']['precision']:.4f} "
        f"| {result['raw']['recall']:.4f} | {result['raw']['exact_match']:.2%} |",
        f"| Ensemble NN | {result['ensemble']['jaccard']:.4f} | {result['ensemble']['precision']:.4f} "
        f"| {result['ensemble']['recall']:.4f} | {result['ensemble']['exact_match']:.2%} |",
        f"| **Gap** | **{result['gap_jaccard']:+.4f}** | "
        f"**{result['ensemble']['precision'] - result['raw']['precision']:+.4f}** | "
        f"**{result['ensemble']['recall'] - result['raw']['recall']:+.4f}** | "
        f"**{result['gap_exact']:+.2%}** |",
        "",
        "## Interpretation",
        "",
        "- **Jaccard**: |retrieved ∩ true| / |retrieved ∪ true|. 1.0 = perfect reconstruction.",
        "- **Precision**: of the bits the retrieved C has active, how many are correct?",
        "- **Recall**: of the true intersection bits, how many were retrieved?",
        "- **Exact match**: fraction where retrieved C is bit-for-bit identical to true X∩Y.",
        "",
    ]

    if result["gap_jaccard"] > 0.005:
        lines += [
            "The ensemble outperforms raw NN at reconstructing the intersection. "
            "This suggests the learned representation captures something about the "
            "AND structure beyond raw input similarity.",
        ]
    elif result["gap_jaccard"] < -0.005:
        lines += [
            "Raw NN outperforms the ensemble. The learned representation does not "
            "help with AND reconstruction — the ensemble's partition structure doesn't "
            "align with intersection computation.",
        ]
    else:
        lines += [
            "Both methods perform similarly. The ensemble doesn't gain or lose much "
            "over raw similarity for AND reconstruction.",
        ]

    lines += [
        "",
        "### Important caveat",
        "",
        "Even if the ensemble outperforms raw NN, this doesn't necessarily mean it ",
        "\"learned AND\". Both methods retrieve the C part of the nearest stored ",
        "training pattern. The question is whether the ensemble's activation space ",
        "groups training patterns in a way that makes the nearest neighbor's C part ",
        "a better approximation of the true X∩Y than raw similarity provides.",
        "",
        "True AND computation would require outputting X∩Y for inputs far from any ",
        "training pair — which nearest-neighbor retrieval fundamentally cannot do.",
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

    total_dim = 3 * PART_DIM
    load = N_TRAIN / TEMPLATE_COUNT
    n_active = int(PART_DIM * SPARSITY)
    expected_overlap = n_active * SPARSITY

    print(f"Config: part_dim={PART_DIM}, sparsity={SPARSITY} ({n_active} active), "
          f"total_dim={total_dim}")
    print(f"Expected |A∩B| ~ {expected_overlap:.1f} bits")
    print(f"Train: {N_TRAIN}, Test: {N_TEST}, M={NODE_COUNT}, k={TEMPLATE_COUNT}, load={load:.0f}")

    # generate data
    train_patterns, train_a, train_b, train_c = make_triplets(N_TRAIN, PART_DIM, SPARSITY, rng)
    test_patterns, test_a, test_b, test_c = make_triplets(N_TEST, PART_DIM, SPARSITY, rng)

    train_intersection_sizes = [int(np.sum(train_c[i])) for i in range(N_TRAIN)]
    test_intersection_sizes = [int(np.sum(test_c[i])) for i in range(N_TEST)]
    print(f"Train |A∩B|: mean={np.mean(train_intersection_sizes):.1f}, "
          f"range=[{min(train_intersection_sizes)}, {max(train_intersection_sizes)}]")
    print(f"Test  |A∩B|: mean={np.mean(test_intersection_sizes):.1f}, "
          f"range=[{min(test_intersection_sizes)}, {max(test_intersection_sizes)}]")

    train_stats = {"intersection_sizes": train_intersection_sizes}

    # train ensemble on full triplets [A, B, A∩B]
    ensemble = train(train_patterns, NODE_COUNT, TEMPLATE_COUNT, ETA, TRAINING_EPOCHS, rng)

    # evaluate
    print("\nEvaluating AND reconstruction...")
    result = evaluate_intersection(ensemble, train_patterns, train_c,
                                   test_a, test_b, test_c)

    print(f"\n  Ensemble: jaccard={result['ensemble']['jaccard']:.4f}  "
          f"prec={result['ensemble']['precision']:.4f}  "
          f"rec={result['ensemble']['recall']:.4f}  "
          f"exact={result['ensemble']['exact_match']:.2%}")
    print(f"  Raw NN:   jaccard={result['raw']['jaccard']:.4f}  "
          f"prec={result['raw']['precision']:.4f}  "
          f"rec={result['raw']['recall']:.4f}  "
          f"exact={result['raw']['exact_match']:.2%}")
    print(f"  Gap:      jaccard={result['gap_jaccard']:+.4f}  "
          f"exact={result['gap_exact']:+.2%}")

    # plot and report
    plot_results(result, train_stats)
    write_report(result, train_stats)
    print("\nDone.")
