"""Zenith ensemble fixed-permutation relation test.

Train on positive pairs [X, P(X)] where P is a single hidden permutation shared
by all examples. Test on unseen X values. For each test X, rank the true output
P(X) against distractor outputs.

This separates three levels of behavior:
    1. Raw pair-space nearest-neighbor similarity
    2. Activation-space nearest-neighbor similarity (memory-like manifold lookup)
    3. Direct template response score (mean top-1 |correlation| across nodes)

If the direct template score succeeds on unseen X, that is stronger evidence that
Zenith templates encode the fixed transformation itself, not just stored pair
instances.

Usage:
    .venv/bin/python experiments/2026_03_29/permutation_relation/permutation_relation.py
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

PART_DIM = 128              # dimensionality of X and Y separately
SPARSITY = 0.10             # fraction of active bits in X

N_TRAIN = 400               # number of positive training pairs [X, P(X)]
N_TEST = 120                # number of unseen test X values

TEMPLATE_COUNT = 10         # k per Zenith node
NODE_COUNT = 30             # M ensemble nodes

TRAINING_EPOCHS = 800
ETA = 0.03

DISTRACTOR_COUNTS = [1, 3, 7, 15]
N_CANDIDATE_TRIALS = 20

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
    """Create one fixed hidden permutation P over indices [0, dim)."""
    return rng.permutation(dim)


def apply_permutation_batch(vectors: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply the fixed permutation to a batch of sparse vectors."""
    return vectors[:, perm]


def make_positive_pairs(sources: np.ndarray, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build positive relation pairs [X, P(X)]."""
    targets = apply_permutation_batch(sources, perm)
    pairs = np.concatenate([sources, targets], axis=1)
    return pairs, targets


def make_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build one candidate pair [X, Y]."""
    return np.concatenate([x, y])


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


def node_forward_from_normalized(w: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    return w @ x_hat


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_forward_from_normalized(
    ensemble: list[np.ndarray],
    x_hat: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (concatenated activations, mean top-1 magnitude across nodes)."""
    activations = []
    top1_scores = []
    for w in ensemble:
        c = node_forward_from_normalized(w, x_hat)
        activations.append(c)
        top1_scores.append(float(np.max(np.abs(c))))
    return np.concatenate(activations), float(np.mean(top1_scores))


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
    """Train M independent Zenith nodes in parallel."""
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


def _score_candidate(
    ensemble: list[np.ndarray],
    train_pair_norms: np.ndarray,
    train_act_norms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    pair = make_pair(x, y)
    pair_hat = _normalize(pair)

    raw_score = float(np.max(train_pair_norms @ pair_hat))

    act, template_score = ensemble_forward_from_normalized(ensemble, pair_hat)
    act_score = float(np.max(train_act_norms @ (act / max(np.linalg.norm(act), EPS))))

    return {
        "raw_nn": raw_score,
        "activation_nn": act_score,
        "template": template_score,
    }


def evaluate_relation_ranking(
    ensemble: list[np.ndarray],
    train_pairs: np.ndarray,
    test_sources: np.ndarray,
    test_targets: np.ndarray,
    distractor_counts: list[int],
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """Rank the true P(X) among distractors for unseen X.

    For each test source X:
        - Candidate 0 is the true Y = P(X)
        - Remaining candidates are distractor Y values taken from other test examples

    A method succeeds if it gives the true candidate the highest score.
    """
    methods = ["raw_nn", "activation_nn", "template"]

    train_pair_norms = _normalize_rows(train_pairs)
    train_acts = np.array(
        [ensemble_forward_from_normalized(ensemble, _normalize(p))[0] for p in train_pairs]
    )
    train_act_norms = train_acts / np.maximum(np.linalg.norm(train_acts, axis=1, keepdims=True), EPS)

    results = {
        d: {
            method: {"correct": 0, "ranks": [], "margins": []}
            for method in methods
        }
        for d in distractor_counts
    }

    test_indices = np.arange(len(test_sources))

    for distractor_count in distractor_counts:
        total = 0
        desc = f"Distractors={distractor_count}"
        for i in tqdm(range(len(test_sources)), desc=desc, leave=False):
            other_idx = test_indices[test_indices != i]
            for _ in range(n_trials):
                distractor_idx = rng.choice(other_idx, size=distractor_count, replace=False)
                candidate_targets = [test_targets[i]]
                candidate_targets.extend(test_targets[j] for j in distractor_idx)

                candidate_scores = {method: [] for method in methods}
                for y in candidate_targets:
                    scores = _score_candidate(
                        ensemble,
                        train_pair_norms,
                        train_act_norms,
                        test_sources[i],
                        y,
                    )
                    for method in methods:
                        candidate_scores[method].append(scores[method])

                for method in methods:
                    scores = np.array(candidate_scores[method], dtype=np.float64)
                    true_score = float(scores[0])
                    pred = int(np.argmax(scores))
                    rank = int(np.sum(scores > true_score))
                    false_best = float(np.max(scores[1:])) if len(scores) > 1 else true_score

                    results[distractor_count][method]["correct"] += int(pred == 0)
                    results[distractor_count][method]["ranks"].append(rank)
                    results[distractor_count][method]["margins"].append(true_score - false_best)

                total += 1

        for method in methods:
            stats = results[distractor_count][method]
            stats["accuracy"] = stats["correct"] / total
            stats["avg_rank"] = float(np.mean(stats["ranks"]))
            stats["avg_margin"] = float(np.mean(stats["margins"]))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, perm: np.ndarray):
    methods = [
        ("raw_nn", "Raw NN", "C1"),
        ("activation_nn", "Activation NN", "C0"),
        ("template", "Template score", "C2"),
    ]
    distractors = sorted(results.keys())

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.suptitle(
        f"Zenith Ensemble — Fixed Permutation Relation  |  "
        f"part_dim={PART_DIM}, sparsity={SPARSITY}, "
        f"N_train={N_TRAIN}, N_test={N_TEST}, "
        f"M={NODE_COUNT}, k={TEMPLATE_COUNT}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  top=0.93, bottom=0.08, left=0.07, right=0.96)

    # --- (0,0) Accuracy vs distractor count ---
    ax = fig.add_subplot(gs[0, 0])
    for method, label, color in methods:
        vals = [results[d][method]["accuracy"] for d in distractors]
        ax.plot(distractors, vals, "o-", color=color, linewidth=1.6, markersize=5, label=label)
    chance = [1.0 / (d + 1) for d in distractors]
    ax.plot(distractors, chance, "k--", linewidth=1.0, label="Chance")
    ax.set_xlabel("Number of distractors")
    ax.set_ylabel("True-candidate accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Relation completion accuracy", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Avg rank vs distractor count ---
    ax = fig.add_subplot(gs[0, 1])
    for method, label, color in methods:
        vals = [results[d][method]["avg_rank"] for d in distractors]
        ax.plot(distractors, vals, "o-", color=color, linewidth=1.6, markersize=5, label=label)
    ax.set_xlabel("Number of distractors")
    ax.set_ylabel("Avg rank of true candidate (0 = best)")
    ax.set_title("Ranking quality", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- (0,2) Margin vs distractor count ---
    ax = fig.add_subplot(gs[0, 2])
    for method, label, color in methods:
        vals = [results[d][method]["avg_margin"] for d in distractors]
        ax.plot(distractors, vals, "o-", color=color, linewidth=1.6, markersize=5, label=label)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Number of distractors")
    ax.set_ylabel("Avg margin (true - best false)")
    ax.set_title("Score margin", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Hardest setting bar chart ---
    hardest = distractors[-1]
    ax = fig.add_subplot(gs[1, 0])
    labels = [label for _, label, _ in methods]
    vals = [results[hardest][method]["accuracy"] for method, _, _ in methods]
    colors = [color for _, _, color in methods]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0 / (hardest + 1), color="black", linestyle="--", linewidth=0.9, label="Chance")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hardest setting ({hardest} distractors)", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Example of the hidden permutation ---
    ax = fig.add_subplot(gs[1, 1])
    n_show = min(32, PART_DIM)
    ax.scatter(np.arange(n_show), perm[:n_show], color="C4", s=26)
    ax.set_xlabel("Input index i")
    ax.set_ylabel("Output index P(i)")
    ax.set_title("Hidden permutation sample", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- (1,2) Summary text ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    hardest = distractors[-1]
    summary_lines = [
        f"Part dim: {PART_DIM}",
        f"Sparsity: {SPARSITY:.0%} ({int(PART_DIM * SPARSITY)} active bits)",
        f"Train pairs: {N_TRAIN}",
        f"Test X values: {N_TEST}",
        f"Ensemble: {NODE_COUNT} nodes, k={TEMPLATE_COUNT}",
        f"Epochs: {TRAINING_EPOCHS}, eta={ETA}",
        f"",
        f"{hardest} distractors:",
        f"  Raw NN:       {results[hardest]['raw_nn']['accuracy']:.1%}",
        f"  Activation:   {results[hardest]['activation_nn']['accuracy']:.1%}",
        f"  Template:     {results[hardest]['template']['accuracy']:.1%}",
        f"  Chance:       {1.0 / (hardest + 1):.1%}",
    ]
    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    out_path = OUT_DIR / "permutation_relation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results: dict):
    methods = [
        ("raw_nn", "Raw pair NN"),
        ("activation_nn", "Activation-space NN"),
        ("template", "Direct template score"),
    ]
    distractors = sorted(results.keys())

    lines = [
        "# Zenith Ensemble — Fixed Permutation Relation Report",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Part dimensionality | {PART_DIM} |",
        f"| Total pair dimensionality | {2 * PART_DIM} |",
        f"| Sparsity | {SPARSITY} |",
        f"| Active bits per part | {int(PART_DIM * SPARSITY)} |",
        f"| Training positive pairs | {N_TRAIN} |",
        f"| Test sources | {N_TEST} |",
        f"| Templates per node (k) | {TEMPLATE_COUNT} |",
        f"| Ensemble nodes (M) | {NODE_COUNT} |",
        f"| Learning rate | {ETA} |",
        f"| Training epochs | {TRAINING_EPOCHS} |",
        f"| Candidate trials per test source | {N_CANDIDATE_TRIALS} |",
        "",
        "## Task",
        "",
        "Training data contains only positive pairs [X, P(X)], where P is one fixed hidden permutation.",
        "At test time, X is unseen. The model must rank the true Y = P(X) against distractor Y values.",
        "",
        "## Methods",
        "",
        "- **Raw pair NN**: score candidate [X, Y] by its maximum cosine similarity to any training pair.",
        "- **Activation-space NN**: score candidate [X, Y] by the cosine similarity of its ensemble activation to stored training activations.",
        "- **Direct template score**: score candidate [X, Y] by the mean top-1 |correlation| across Zenith nodes, without nearest-neighbor lookup.",
        "",
        "## Results",
        "",
        "| Distractors | Method | Accuracy | Avg Rank | Avg Margin | Chance |",
        "|-------------|--------|----------|----------|------------|--------|",
    ]

    for d in distractors:
        chance = 1.0 / (d + 1)
        for method, label in methods:
            stats = results[d][method]
            lines.append(
                f"| {d} | {label} | {stats['accuracy']:.2%} | "
                f"{stats['avg_rank']:.2f} | {stats['avg_margin']:+.4f} | {chance:.2%} |"
            )

    hardest = distractors[-1]
    raw_acc = results[hardest]["raw_nn"]["accuracy"]
    act_acc = results[hardest]["activation_nn"]["accuracy"]
    template_acc = results[hardest]["template"]["accuracy"]

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Activation-space NN > Raw NN** means the learned activation manifold organizes valid permutation pairs better than raw input similarity does.",
        "- **Direct template score > Raw NN** is stronger: it suggests the templates themselves capture some of the fixed transformation, not only nearest-neighbor memory.",
        "- **Avg rank** asks where the true candidate falls among all candidates. 0 means best.",
        "- **Avg margin** is true score minus best false score. Positive is good.",
        "",
    ]

    if template_acc > raw_acc + 0.02:
        lines.append(
            "The direct template score beats raw NN in the hardest setting. "
            "That is evidence that Zenith templates learned something about the fixed permutation itself."
        )
    elif act_acc > raw_acc + 0.02:
        lines.append(
            "Activation-space NN beats raw NN, but direct template scoring does not clearly pull ahead. "
            "That points more to a learned memory manifold than to explicit relation encoding."
        )
    else:
        lines.append(
            "Neither activation-space NN nor direct template scoring clearly beats raw NN. "
            "For this configuration, Zenith does not appear to encode the fixed permutation better than raw similarity."
        )

    lines += [
        "",
        "### Caveat",
        "",
        "A fixed permutation is a very specific linear relation. Success here would not imply general relational reasoning.",
        "It would only show that Zenith can internalize one reusable transformation on sparse vectors.",
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
          f"train={N_TRAIN}, test={N_TEST}, M={NODE_COUNT}, k={TEMPLATE_COUNT}")

    perm = make_hidden_permutation(PART_DIM, rng)
    print("Generated hidden permutation P.")

    train_sources = make_sparse_vectors(N_TRAIN, PART_DIM, SPARSITY, rng)
    test_sources = make_sparse_vectors(N_TEST, PART_DIM, SPARSITY, rng)
    train_pairs, train_targets = make_positive_pairs(train_sources, perm)
    test_pairs, test_targets = make_positive_pairs(test_sources, perm)
    _ = train_targets, test_pairs  # silence unused-variable intent

    print(f"Train pairs: {len(train_pairs)}  Test sources: {len(test_sources)}")

    ensemble = train(train_pairs, NODE_COUNT, TEMPLATE_COUNT, ETA, TRAINING_EPOCHS, rng)

    print("\nEvaluating permutation relation ranking...")
    results = evaluate_relation_ranking(
        ensemble,
        train_pairs,
        test_sources,
        test_targets,
        DISTRACTOR_COUNTS,
        N_CANDIDATE_TRIALS,
        np.random.default_rng(SEED + 100),
    )

    for d in DISTRACTOR_COUNTS:
        chance = 1.0 / (d + 1)
        print(f"\n  distractors={d}  chance={chance:.2%}")
        print(f"    raw_nn         acc={results[d]['raw_nn']['accuracy']:.2%}  "
              f"rank={results[d]['raw_nn']['avg_rank']:.2f}  "
              f"margin={results[d]['raw_nn']['avg_margin']:+.4f}")
        print(f"    activation_nn  acc={results[d]['activation_nn']['accuracy']:.2%}  "
              f"rank={results[d]['activation_nn']['avg_rank']:.2f}  "
              f"margin={results[d]['activation_nn']['avg_margin']:+.4f}")
        print(f"    template       acc={results[d]['template']['accuracy']:.2%}  "
              f"rank={results[d]['template']['avg_rank']:.2f}  "
              f"margin={results[d]['template']['avg_margin']:+.4f}")

    plot_results(results, perm)
    write_report(results)
    print("\nDone.")
