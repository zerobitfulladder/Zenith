"""Zenith ensemble generalization test.

Tests whether the ensemble learns representations that capture category
structure beyond what raw input similarity provides.

Data model:
    Each pattern = signal bits (shared within category) + nuisance bits (random per exemplar).
    Train on some exemplars, test on held-out exemplars with NEW nuisance bits.
    Compare category retrieval accuracy:
        - Ensemble activation-space nearest neighbor
        - Raw input-space nearest neighbor (baseline)
    If ensemble > baseline, the difference is genuine generalization.

Usage:
    .venv/bin/python experiments/2026_03_29/generalization_test/generalization_test.py
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

DIM = 300                   # total vector dimensionality
SIGNAL_DIM = 100            # how many dims are "signal" (first SIGNAL_DIM dims)
NUISANCE_DIM = 200          # remaining dims are "nuisance" (DIM - SIGNAL_DIM)

SIGNAL_SPARSITY = 0.10      # fraction of signal dims active per category
NUISANCE_SPARSITY = 0.20    # fraction of nuisance dims active per exemplar

NUM_CATEGORIES = 20          # number of distinct categories
TRAIN_PER_CAT = 15           # training exemplars per category
TEST_PER_CAT = 10            # test exemplars per category

TEMPLATE_COUNT = 10          # k per Zenith node
NODE_COUNT = 30              # M ensemble nodes

TRAINING_EPOCHS = 1000
ETA = 0.03
N_TRIALS = 20               # repeated trials for test evaluation

SEED = 42
EPS = 1e-8

OUT_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_category_prototypes(
    num_categories: int,
    signal_dim: int,
    signal_sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create sparse signal prototypes for each category. Shape (num_categories, signal_dim)."""
    n_active = max(1, int(round(signal_dim * signal_sparsity)))
    prototypes = np.zeros((num_categories, signal_dim), dtype=np.float64)
    for i in range(num_categories):
        idx = rng.choice(signal_dim, size=n_active, replace=False)
        prototypes[i, idx] = 1.0
    return prototypes


def make_exemplars(
    prototypes: np.ndarray,
    num_per_cat: int,
    nuisance_dim: int,
    nuisance_sparsity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exemplars: signal from prototype + random nuisance bits.

    Returns:
        patterns — (num_categories * num_per_cat, signal_dim + nuisance_dim)
        labels   — (num_categories * num_per_cat,) category index
    """
    num_categories, signal_dim = prototypes.shape
    total_dim = signal_dim + nuisance_dim
    n_nuisance_active = max(1, int(round(nuisance_dim * nuisance_sparsity)))

    n_total = num_categories * num_per_cat
    patterns = np.zeros((n_total, total_dim), dtype=np.float64)
    labels = np.zeros(n_total, dtype=np.int64)

    for cat in range(num_categories):
        for j in range(num_per_cat):
            idx = cat * num_per_cat + j
            # signal part: copy from prototype
            patterns[idx, :signal_dim] = prototypes[cat]
            # nuisance part: random sparse
            nuisance_idx = rng.choice(nuisance_dim, size=n_nuisance_active, replace=False)
            patterns[idx, signal_dim + nuisance_idx] = 1.0
            labels[idx] = cat

    return patterns, labels


# ---------------------------------------------------------------------------
# Zenith core (same as ensemble_completion.py)
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
# Training
# ---------------------------------------------------------------------------

def _train_single_node(args: tuple) -> np.ndarray:
    """Train one Zenith node. Runs in a separate process."""
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
    """Train ensemble on patterns in parallel. Returns trained ensemble."""
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


def evaluate_generalization(
    ensemble: list[np.ndarray],
    train_patterns: np.ndarray,
    train_labels: np.ndarray,
    test_patterns: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Compare category retrieval: ensemble activation NN vs raw input NN.

    For each test exemplar:
        - Ensemble: find nearest training exemplar in activation space, check same category
        - Raw: find nearest training exemplar in raw input space, check same category

    Returns dict with ensemble_acc, raw_acc, gap, and per-category breakdowns.
    """
    n_train = train_patterns.shape[0]
    n_test = test_patterns.shape[0]

    # precompute training activations
    train_acts = np.array([ensemble_forward(ensemble, p) for p in train_patterns])

    # precompute training raw normalized vectors for cosine sim
    train_raw_norm = np.array([_normalize(p) for p in train_patterns])

    ensemble_correct = 0
    raw_correct = 0
    per_cat_ensemble = {}
    per_cat_raw = {}

    for i in tqdm(range(n_test), desc="Evaluating", leave=False):
        test_act = ensemble_forward(ensemble, test_patterns[i])
        test_raw_norm = _normalize(test_patterns[i])
        true_cat = int(test_labels[i])

        # ensemble NN
        act_sims = np.array([_cosine_sim(test_act, train_acts[j]) for j in range(n_train)])
        ens_pred_cat = int(train_labels[np.argmax(act_sims)])
        ens_hit = ens_pred_cat == true_cat
        ensemble_correct += ens_hit

        # raw NN
        raw_sims = train_raw_norm @ test_raw_norm
        raw_pred_cat = int(train_labels[np.argmax(raw_sims)])
        raw_hit = raw_pred_cat == true_cat
        raw_correct += raw_hit

        # per-category tracking
        per_cat_ensemble.setdefault(true_cat, []).append(ens_hit)
        per_cat_raw.setdefault(true_cat, []).append(raw_hit)

    ensemble_acc = ensemble_correct / n_test
    raw_acc = raw_correct / n_test

    return {
        "ensemble_acc": ensemble_acc,
        "raw_acc": raw_acc,
        "gap": ensemble_acc - raw_acc,
        "per_cat_ensemble": {c: np.mean(v) for c, v in per_cat_ensemble.items()},
        "per_cat_raw": {c: np.mean(v) for c, v in per_cat_raw.items()},
    }


def evaluate_with_masking(
    ensemble: list[np.ndarray],
    train_patterns: np.ndarray,
    train_labels: np.ndarray,
    test_patterns: np.ndarray,
    test_labels: np.ndarray,
    mask_fractions: list[float],
    n_trials: int,
    rng: np.random.Generator,
) -> dict[float, dict]:
    """Evaluate at multiple mask fractions. Returns {mask_frac: result_dict}."""
    n_train = train_patterns.shape[0]
    n_test = test_patterns.shape[0]

    # precompute training activations and raw norms
    train_acts = np.array([ensemble_forward(ensemble, p) for p in train_patterns])
    train_raw_norm = np.array([_normalize(p) for p in train_patterns])

    results = {}

    for mf in mask_fractions:
        ensemble_correct = 0
        raw_correct = 0
        total = 0

        desc = f"Mask {mf:.0%}"
        for i in tqdm(range(n_test), desc=desc, leave=False):
            true_cat = int(test_labels[i])
            trials = 1 if mf == 0.0 else n_trials

            for _ in range(trials):
                if mf == 0.0:
                    test_input = test_patterns[i]
                else:
                    test_input = mask_pattern(test_patterns[i], mf, rng)

                # ensemble NN
                test_act = ensemble_forward(ensemble, test_input)
                act_sims = np.array([_cosine_sim(test_act, train_acts[j]) for j in range(n_train)])
                ens_pred = int(train_labels[np.argmax(act_sims)])
                ensemble_correct += (ens_pred == true_cat)

                # raw NN
                test_raw = _normalize(test_input)
                raw_sims = train_raw_norm @ test_raw
                raw_pred = int(train_labels[np.argmax(raw_sims)])
                raw_correct += (raw_pred == true_cat)

                total += 1

        results[mf] = {
            "ensemble_acc": ensemble_correct / total,
            "raw_acc": raw_correct / total,
            "gap": (ensemble_correct - raw_correct) / total,
        }

    return results


def mask_pattern(pattern: np.ndarray, mask_frac: float, rng: np.random.Generator) -> np.ndarray:
    masked = pattern.copy()
    active_idx = np.nonzero(pattern)[0]
    n_mask = int(round(len(active_idx) * mask_frac))
    if n_mask > 0:
        to_mask = rng.choice(active_idx, size=n_mask, replace=False)
        masked[to_mask] = 0.0
    return masked


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    base_result: dict,
    masking_results: dict[float, dict],
):
    load = (NUM_CATEGORIES * TRAIN_PER_CAT) / TEMPLATE_COUNT
    mfs = sorted(masking_results.keys())

    fig = plt.figure(figsize=(18, 10), dpi=150)
    fig.suptitle(
        f"Zenith Ensemble Generalization  —  dim={DIM} ({SIGNAL_DIM}sig+{NUISANCE_DIM}nui), "
        f"cats={NUM_CATEGORIES}, train={TRAIN_PER_CAT}/cat, test={TEST_PER_CAT}/cat, "
        f"M={NODE_COUNT}, k={TEMPLATE_COUNT}, load={load:.1f}",
        fontsize=11, fontweight="bold", y=0.995,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                  top=0.93, bottom=0.08, left=0.07, right=0.96)

    # --- (0,0) Base comparison: ensemble vs raw ---
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Category retrieval (no masking)", fontweight="bold", fontsize=10)
    bars = ax.bar(["Raw NN", "Ensemble NN"], [base_result["raw_acc"], base_result["ensemble_acc"]],
                  color=["C1", "C0"], width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", fontsize=11, fontweight="bold")
    gap = base_result["gap"]
    ax.set_ylabel("Category accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Category retrieval (no masking)\nGap: {gap:+.1%}", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,1) Per-category accuracy comparison ---
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Per-category accuracy", fontweight="bold", fontsize=10)
    cats = sorted(base_result["per_cat_ensemble"].keys())
    ens_accs = [base_result["per_cat_ensemble"][c] for c in cats]
    raw_accs = [base_result["per_cat_raw"][c] for c in cats]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, raw_accs, w, label="Raw NN", color="C1", alpha=0.8)
    ax.bar(x + w/2, ens_accs, w, label="Ensemble NN", color="C0", alpha=0.8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (0,2) Gap per category ---
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Ensemble advantage per category", fontweight="bold", fontsize=10)
    gaps = [base_result["per_cat_ensemble"][c] - base_result["per_cat_raw"][c] for c in cats]
    colors = ["C2" if g >= 0 else "C3" for g in gaps]
    ax.bar(x, gaps, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy gap (ens - raw)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,0) Ensemble acc vs mask fraction ---
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Accuracy vs masking", fontweight="bold", fontsize=10)
    ens_acc = [masking_results[mf]["ensemble_acc"] for mf in mfs]
    raw_acc = [masking_results[mf]["raw_acc"] for mf in mfs]
    ax.plot(mfs, ens_acc, "o-", color="C0", linewidth=1.5, markersize=5, label="Ensemble NN")
    ax.plot(mfs, raw_acc, "s--", color="C1", linewidth=1.5, markersize=5, label="Raw NN")
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Category accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,1) Gap vs mask fraction ---
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Generalization gap vs masking", fontweight="bold", fontsize=10)
    gap_vals = [masking_results[mf]["gap"] for mf in mfs]
    ax.plot(mfs, gap_vals, "o-", color="C2", linewidth=1.5, markersize=5)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    for xv, yv in zip(mfs, gap_vals):
        ax.annotate(f"{yv:+.1%}", (xv, yv), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Gap (ensemble - raw)")
    ax.grid(True, alpha=0.3)

    # --- (1,2) Summary text ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary_lines = [
        f"Signal: {SIGNAL_DIM} dims, {SIGNAL_SPARSITY:.0%} sparse",
        f"Nuisance: {NUISANCE_DIM} dims, {NUISANCE_SPARSITY:.0%} sparse",
        f"Categories: {NUM_CATEGORIES}",
        f"Train: {TRAIN_PER_CAT}/cat ({NUM_CATEGORIES * TRAIN_PER_CAT} total)",
        f"Test: {TEST_PER_CAT}/cat ({NUM_CATEGORIES * TEST_PER_CAT} total)",
        f"Ensemble: {NODE_COUNT} nodes, k={TEMPLATE_COUNT}",
        f"Load: {load:.1f}",
        f"",
        f"No-mask gap: {base_result['gap']:+.1%}",
        f"  Ensemble: {base_result['ensemble_acc']:.1%}",
        f"  Raw NN:   {base_result['raw_acc']:.1%}",
    ]
    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    out_path = OUT_DIR / "generalization.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(base_result: dict, masking_results: dict[float, dict]):
    load = (NUM_CATEGORIES * TRAIN_PER_CAT) / TEMPLATE_COUNT
    mfs = sorted(masking_results.keys())

    lines = [
        "# Zenith Ensemble — Generalization Test Report",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Total dimensionality | {DIM} |",
        f"| Signal dimensions | {SIGNAL_DIM} |",
        f"| Nuisance dimensions | {NUISANCE_DIM} |",
        f"| Signal sparsity | {SIGNAL_SPARSITY} |",
        f"| Nuisance sparsity | {NUISANCE_SPARSITY} |",
        f"| Categories | {NUM_CATEGORIES} |",
        f"| Train per category | {TRAIN_PER_CAT} |",
        f"| Test per category | {TEST_PER_CAT} |",
        f"| Templates per node (k) | {TEMPLATE_COUNT} |",
        f"| Ensemble nodes (M) | {NODE_COUNT} |",
        f"| Load (N_train/k) | {load:.1f} |",
        f"| Learning rate | {ETA} |",
        f"| Training epochs | {TRAINING_EPOCHS} |",
        "",
        "## Data Design",
        "",
        "Each pattern consists of two parts:",
        f"- **Signal** (first {SIGNAL_DIM} dims): sparse binary pattern shared by all "
        "exemplars in the same category. Defines category identity.",
        f"- **Nuisance** (last {NUISANCE_DIM} dims): sparse binary pattern randomized "
        "independently per exemplar. Creates misleading raw similarity.",
        "",
        "Training uses one set of exemplars. Testing uses held-out exemplars with the "
        "same signal bits but completely new nuisance bits.",
        "",
        "## Generalization Test (no masking)",
        "",
        f"| Method | Category Accuracy |",
        f"|--------|-------------------|",
        f"| Raw input NN | {base_result['raw_acc']:.2%} |",
        f"| Ensemble activation NN | {base_result['ensemble_acc']:.2%} |",
        f"| **Gap (ensemble - raw)** | **{base_result['gap']:+.2%}** |",
        "",
    ]

    if base_result["gap"] > 0.005:
        lines.append("The ensemble outperforms raw similarity — it has learned to "
                      "suppress nuisance dimensions and focus on category-defining signal.")
    elif base_result["gap"] < -0.005:
        lines.append("Raw similarity outperforms the ensemble — the learned representation "
                      "does not improve over raw input distance for this configuration.")
    else:
        lines.append("Ensemble and raw similarity perform similarly — the representation "
                      "neither helps nor hurts category retrieval.")

    lines += [
        "",
        "## Generalization Under Masking",
        "",
        "| Mask % | Ensemble Acc | Raw NN Acc | Gap |",
        "|--------|-------------|------------|-----|",
    ]
    for mf in mfs:
        r = masking_results[mf]
        lines.append(f"| {mf:.0%} | {r['ensemble_acc']:.2%} | {r['raw_acc']:.2%} | {r['gap']:+.2%} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Positive gap** = generalization. The ensemble learned structure that "
        "raw similarity misses.",
        "- **Zero gap** = the ensemble merely reflects input similarity (no generalization).",
        "- **Negative gap** = the ensemble representation is worse than raw input for "
        "this task (possibly overtrained or too few nodes).",
        "",
        "The key question: does the gap grow as nuisance increases relative to signal? "
        "If so, the ensemble is genuinely filtering noise.",
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

    load = (NUM_CATEGORIES * TRAIN_PER_CAT) / TEMPLATE_COUNT
    print(f"Config: dim={DIM} ({SIGNAL_DIM}sig+{NUISANCE_DIM}nui), "
          f"cats={NUM_CATEGORIES}, train={TRAIN_PER_CAT}/cat, test={TEST_PER_CAT}/cat, "
          f"M={NODE_COUNT}, k={TEMPLATE_COUNT}, load={load:.1f}")

    # generate category prototypes
    prototypes = make_category_prototypes(NUM_CATEGORIES, SIGNAL_DIM, SIGNAL_SPARSITY, rng)
    print(f"Generated {NUM_CATEGORIES} category prototypes "
          f"(signal: {int(SIGNAL_DIM * SIGNAL_SPARSITY)} active bits each)")

    # generate train and test exemplars
    train_patterns, train_labels = make_exemplars(
        prototypes, TRAIN_PER_CAT, NUISANCE_DIM, NUISANCE_SPARSITY, rng)
    test_patterns, test_labels = make_exemplars(
        prototypes, TEST_PER_CAT, NUISANCE_DIM, NUISANCE_SPARSITY, rng)
    print(f"Train: {len(train_patterns)} exemplars, Test: {len(test_patterns)} exemplars")

    # train ensemble on training exemplars only
    ensemble = train(train_patterns, NODE_COUNT, TEMPLATE_COUNT, ETA, TRAINING_EPOCHS, rng)

    # evaluate: no masking (base generalization test)
    print("\nEvaluating generalization (no masking)...")
    base_result = evaluate_generalization(
        ensemble, train_patterns, train_labels, test_patterns, test_labels)
    print(f"  Ensemble: {base_result['ensemble_acc']:.2%}")
    print(f"  Raw NN:   {base_result['raw_acc']:.2%}")
    print(f"  Gap:      {base_result['gap']:+.2%}")

    # evaluate: with masking
    mask_fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    print("\nEvaluating with masking...")
    masking_results = evaluate_with_masking(
        ensemble, train_patterns, train_labels, test_patterns, test_labels,
        mask_fractions, N_TRIALS, np.random.default_rng(SEED + 200))
    for mf in mask_fractions:
        r = masking_results[mf]
        print(f"  mask={mf:.0%}  ensemble={r['ensemble_acc']:.2%}  "
              f"raw={r['raw_acc']:.2%}  gap={r['gap']:+.2%}")

    # plot and report
    plot_results(base_result, masking_results)
    write_report(base_result, masking_results)
    print("\nDone.")
