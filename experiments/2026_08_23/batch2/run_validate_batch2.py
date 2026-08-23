"""8-seed validation of the batch-2 rig, plus a per-class style grid.

Each seed gets its own train/test split, data order, and initialization.
Metrics: probe, hard and top-10 readouts, label-only retrieval consistency.
From the first seed, a style grid renders up to 5 stored constellations per
class — the sampled-variety preview (distinct complete digits, one memory).

Run:  .venv/bin/python experiments/2026_08_23/batch2/run_validate_batch2.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import EPS, ZenithLayer, center_norm
from run_batch2 import (
    CODE_DIM,
    EPOCHS,
    ETA1,
    ETA2,
    K1,
    K2,
    LAM,
    PATCH,
    TEST_N,
    TRAIN_N,
    encode_batch,
    encode_image,
    render,
)

ROOT = Path(__file__).resolve().parents[3]

SEEDS = list(range(8))
PROBE_N = 5000
STYLES_PER_CLASS = 5
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "batch2" / "results" / "validation"


def load_data(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


def run_once(seed, make_style_grid):
    Xtr, ytr, Xte, yte = load_data(seed)
    rng = np.random.default_rng(seed)
    dic = ZenithLayer(K1, PATCH * PATCH, ETA1, rng)
    l2 = ZenithLayer(K2, CODE_DIM + 10, ETA2, rng)

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc=f"seed {seed}", ncols=80):
            code = encode_image(x, dic, learning=True)
            h_hat, h_norm = center_norm(code)
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = LAM
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                l2.learn(z_hat, l2.forward(z_hat))

    label_half = l2.W[:, CODE_DIM:]
    owner = label_half.argmax(axis=1)

    consistent = 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE_DIM), label]))
        consistent += int(owner[int(np.argmax(l2.forward(z_hat)))] == j)

    Cte = encode_batch(Xte, dic)
    Ctr = encode_batch(Xtr[:PROBE_N], dic)

    H = Cte - Cte.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    Z = np.concatenate([H, np.zeros((len(H), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ l2.W.T
    acc_hard = float((owner[C2.argmax(axis=1)] == yte).mean())
    kv = 10
    part = np.argpartition(-C2, kv, axis=1)[:, :kv]
    mask = np.zeros_like(C2)
    mask[np.arange(len(C2))[:, None], part] = 1.0
    acc_topk = float((((np.maximum(C2, 0.0) * mask) @ label_half).argmax(axis=1) == yte).mean())

    probe = LogisticRegression(max_iter=1000)
    probe.fit(Ctr, ytr[:PROBE_N])
    probe_acc = float(probe.score(Cte, yte))

    if make_style_grid:
        fig, axes = plt.subplots(10, STYLES_PER_CLASS, figsize=(STYLES_PER_CLASS * 1.6, 16.5))
        for j in range(10):
            members = np.argsort(-label_half[:, j])
            members = [t for t in members if owner[t] == j][:STYLES_PER_CLASS]
            for s in range(STYLES_PER_CLASS):
                ax = axes[j, s]
                if s < len(members):
                    ax.imshow(render(l2.W[members[s], :CODE_DIM], dic), cmap="gray")
                if s == 0:
                    ax.set_ylabel(str(j), rotation=0, fontsize=11, labelpad=12)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Style grid — up to 5 stored constellations per class (one seed)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "style_grid.png", dpi=110)
        plt.close(fig)

    return {"acc_hard": acc_hard, "acc_topk": acc_topk, "probe": probe_acc,
            "consistent": consistent}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, seed in enumerate(SEEDS):
        r = run_once(seed, make_style_grid=(i == 0))
        rows.append(r)
        print(f"SEED {seed}: hard={r['acc_hard']:.4f} top10={r['acc_topk']:.4f} "
              f"probe={r['probe']:.4f} consistent={r['consistent']}/10")

    lines = [
        "# Batch 2 — 8-seed validation",
        "",
        f"Config identical to run_batch2.py; split/order/init vary per seed. Seeds: {SEEDS}.",
        "",
        "| seed | hard | top10 | probe | label-only consistent |",
        "|---|---|---|---|---|",
    ]
    for seed, r in zip(SEEDS, rows):
        lines.append(f"| {seed} | {r['acc_hard']:.4f} | {r['acc_topk']:.4f} | "
                     f"{r['probe']:.4f} | {r['consistent']}/10 |")
    lines.append("")
    for m in ["acc_hard", "acc_topk", "probe"]:
        v = np.array([r[m] for r in rows])
        lines.append(f"- **{m}**: {v.mean():.4f} ± {v.std():.4f}  (min {v.min():.4f}, max {v.max():.4f})")
    cons = np.array([r["consistent"] for r in rows])
    lines.append(f"- **label-only retrieval**: {cons.sum()}/{10 * len(rows)} across all seeds")
    lines += ["", "Figure: style_grid.png (first seed)"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
