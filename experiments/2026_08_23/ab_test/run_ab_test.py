"""A/B test: is the Variant B contrast-feedback effect real, and does graded
learning give it a bigger lever?

Harder task (Fashion-MNIST), 2x2 factorial:
    learning  in {top-1, top-5}   (k_active)
    feedback  in {off, on}        (gamma = 0 vs 2, Variant B, contrast source)
across N seeds, with paired per-seed deltas and a paired t-test.

gamma=2 and the contrast source were selected on the MNIST sweeps;
Fashion-MNIST is fresh data for this hypothesis.

Run:  .venv/bin/python experiments/2026_08_23/ab_test/run_ab_test.py
"""

from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import (
    EPS,
    ClassTemplates,
    ZenithLayer,
    center_norm,
    encode_batch,
    plasticity_gain,
)

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000
K1 = 100
ETA1 = 0.04
ETA2 = 0.10
EPOCHS = 2
GAMMA_ON = 2.0
SEEDS = list(range(8))
K_ACTIVES = [1, 5]
# Fashion-MNIST classes: 0 tshirt, 1 trouser, 2 pullover, 3 dress, 4 coat,
# 5 sandal, 6 shirt, 7 sneaker, 8 bag, 9 ankle boot. Known-confusable pairs:
PAIRS = [(0, 6), (2, 6), (4, 6), (2, 4), (7, 9)]
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "ab_test" / "results"
# ---------------------------------------------------------------------------


def load_data(seed):
    X = np.load(ROOT / "data/mnist/fashion/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/fashion/train_labels.npy").astype(np.int64)
    X = X.reshape(len(X), -1)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def run_once(seed, k_active, gamma):
    Xtr, ytr, Xte, yte = load_data(seed)
    rng = np.random.default_rng(seed)
    l1 = ZenithLayer(K1, Xtr.shape[1], ETA1, rng)
    l2 = ClassTemplates(10, K1, ETA2, rng)

    for _ in range(EPOCHS):
        for x, y in zip(Xtr, ytr):
            x_c = x - x.mean()
            n = np.linalg.norm(x_c)
            if n < EPS:
                continue
            x_hat = x_c / n

            plast = None
            if gamma > 0.0:
                src = l2.W[y] - l2.W.mean(axis=0)
                plast = plasticity_gain(src, gamma)

            c1 = l1.forward(x_hat)
            l1.learn(x_hat, c1, step_gain=plast, k_active=k_active)

            h_hat, h_norm = center_norm(np.maximum(c1, 0.0))
            if h_norm > EPS:
                l2.learn(h_hat, y)

    # ---- Evaluation (always neutral: no label, no gain) -------------------
    A1_te, C2_te = encode_batch(Xte, l1.W, l2.W)
    pred = C2_te.argmax(axis=1)
    acc = float((pred == yte).mean())

    pair_ms = []
    for a, b in PAIRS:
        ma = C2_te[yte == a, a] - C2_te[yte == a, b]
        mb = C2_te[yte == b, b] - C2_te[yte == b, a]
        pair_ms.append(float(np.concatenate([ma, mb]).mean()))
    pair_margin = float(np.mean(pair_ms))

    A1_tr, C2_tr = encode_batch(Xtr[:PROBE_N], l1.W, l2.W)
    probe1 = LogisticRegression(max_iter=1000)
    probe1.fit(A1_tr, ytr[:PROBE_N])
    probe_l1 = float(probe1.score(A1_te, yte))
    probe2 = LogisticRegression(max_iter=1000)
    probe2.fit(C2_tr, ytr[:PROBE_N])
    probe_l2 = float(probe2.score(C2_te, yte))

    Xc = Xte - Xte.mean(axis=1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + EPS
    top1_corr = float((Xc @ l1.W.T).max(axis=1).mean())

    p = l1.win_counts / l1.win_counts.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(l1.k))

    return {
        "acc": acc, "pair_margin": pair_margin, "probe_l1": probe_l1,
        "probe_l2": probe_l2, "top1_corr": top1_corr, "entropy": entropy,
    }


METRICS = ["acc", "probe_l1", "probe_l2", "pair_margin", "top1_corr", "entropy"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}  # (k_active, gamma) -> list over seeds of metric dicts
    combos = [(k, g) for k in K_ACTIVES for g in (0.0, GAMMA_ON)]
    for k_active, gamma in combos:
        rows = []
        for seed in tqdm(SEEDS, desc=f"k={k_active} gamma={gamma}", ncols=80):
            rows.append(run_once(seed, k_active, gamma))
        results[(k_active, gamma)] = rows
        means = {m: np.mean([r[m] for r in rows]) for m in METRICS}
        print(f"k={k_active} gamma={gamma}: " +
              "  ".join(f"{m}={means[m]:.4f}" for m in METRICS))

    lines = [
        "# A/B test — Variant B contrast feedback on Fashion-MNIST",
        "",
        f"Config: TRAIN_N={TRAIN_N}, TEST_N={TEST_N}, K1={K1}, ETA1={ETA1}, "
        f"ETA2={ETA2}, EPOCHS={EPOCHS}, gamma_on={GAMMA_ON}, seeds={SEEDS}.",
        "Evaluation is feedforward with neutral gain (no label at test).",
        "",
        "## Condition means (± sd over seeds)",
        "",
        "| k_active | feedback | " + " | ".join(METRICS) + " |",
        "|---" * (len(METRICS) + 2) + "|",
    ]
    for (k_active, gamma), rows in results.items():
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in rows])
            cells.append(f"{v.mean():.4f} ± {v.std():.4f}")
        lines.append(f"| {k_active} | {'on' if gamma > 0 else 'off'} | " + " | ".join(cells) + " |")

    lines += ["", "## Paired per-seed deltas (feedback on − off)", ""]
    for k_active in K_ACTIVES:
        off = results[(k_active, 0.0)]
        on = results[(k_active, GAMMA_ON)]
        lines += [f"### k_active = {k_active}", "",
                  "| metric | mean delta | seeds improved | paired t p-value |",
                  "|---|---|---|---|"]
        for m in METRICS:
            d = np.array([a[m] - b[m] for a, b in zip(on, off)])
            t = stats.ttest_rel([a[m] for a in on], [b[m] for b in off])
            lines.append(f"| {m} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | {t.pvalue:.4f} |")
        lines.append("")

    lines += ["## Graded-learning main effect (k=5 − k=1, feedback off)", "",
              "| metric | mean delta | seeds improved | paired t p-value |",
              "|---|---|---|---|"]
    off1, off5 = results[(1, 0.0)], results[(5, 0.0)]
    for m in METRICS:
        d = np.array([a[m] - b[m] for a, b in zip(off5, off1)])
        t = stats.ttest_rel([a[m] for a in off5], [b[m] for b in off1])
        lines.append(f"| {m} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | {t.pvalue:.4f} |")

    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
