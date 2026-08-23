"""Warm-up test: unsupervised settling first, feedback second.

Fashion-MNIST, top-1 learning, Variant B contrast feedback (gamma=2).
Three paired conditions over the same seeds/data orders:
    off    — gamma 0 in both epochs (baseline)
    always — gamma 2 in both epochs (replicates the A/B's k=1 condition)
    warmup — gamma 0 in epoch 1, gamma 2 in epoch 2

Run:  .venv/bin/python experiments/2026_08_23/warmup_test/run_warmup_test.py
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
GAMMA = 2.0
SEEDS = list(range(8))
CONDITIONS = {"off": (0.0, 0.0), "always": (GAMMA, GAMMA), "warmup": (0.0, GAMMA)}
PAIRS = [(0, 6), (2, 6), (4, 6), (2, 4), (7, 9)]
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "warmup_test" / "results"
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


def run_once(seed, gammas_per_epoch):
    Xtr, ytr, Xte, yte = load_data(seed)
    rng = np.random.default_rng(seed)
    l1 = ZenithLayer(K1, Xtr.shape[1], ETA1, rng)
    l2 = ClassTemplates(10, K1, ETA2, rng)

    for gamma in gammas_per_epoch:
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
            l1.learn(x_hat, c1, step_gain=plast)

            h_hat, h_norm = center_norm(np.maximum(c1, 0.0))
            if h_norm > EPS:
                l2.learn(h_hat, y)

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

    return {
        "acc": acc, "probe_l1": probe_l1, "probe_l2": probe_l2,
        "pair_margin": pair_margin, "top1_corr": top1_corr,
    }


METRICS = ["acc", "probe_l1", "probe_l2", "pair_margin", "top1_corr"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, gammas in CONDITIONS.items():
        rows = []
        for seed in tqdm(SEEDS, desc=name, ncols=80):
            rows.append(run_once(seed, gammas))
        results[name] = rows
        means = {m: np.mean([r[m] for r in rows]) for m in METRICS}
        print(f"{name}: " + "  ".join(f"{m}={means[m]:.4f}" for m in METRICS))

    lines = [
        "# Warm-up test — unsupervised settling, then Variant B contrast feedback",
        "",
        f"Config: Fashion-MNIST, TRAIN_N={TRAIN_N}, TEST_N={TEST_N}, K1={K1}, "
        f"ETA1={ETA1}, ETA2={ETA2}, gamma={GAMMA}, seeds={SEEDS}, top-1 learning.",
        "Conditions (gamma per epoch): off=(0,0), always=(2,2), warmup=(0,2).",
        "",
        "## Condition means (± sd over seeds)",
        "",
        "| condition | " + " | ".join(METRICS) + " |",
        "|---" * (len(METRICS) + 1) + "|",
    ]
    for name, rows in results.items():
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in rows])
            cells.append(f"{v.mean():.4f} ± {v.std():.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    for a, b in [("warmup", "off"), ("always", "off"), ("warmup", "always")]:
        lines += ["", f"## Paired per-seed deltas ({a} − {b})", "",
                  "| metric | mean delta | seeds improved | paired t p-value |",
                  "|---|---|---|---|"]
        for m in METRICS:
            d = np.array([x[m] - y[m] for x, y in zip(results[a], results[b])])
            t = stats.ttest_rel([x[m] for x in results[a]], [y[m] for y in results[b]])
            lines.append(f"| {m} | {d.mean():+.4f} | {(d > 0).sum()}/{len(d)} | {t.pvalue:.4f} |")

    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
