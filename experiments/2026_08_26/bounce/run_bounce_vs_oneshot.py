"""Head-to-head: anchored bounce generation vs one-shot k=1 generation.

Neutral judge (the network's own matcher can't recognize renders): a
logistic classifier on raw pixels trained on real MNIST, applied to
ink-normalized renders (relu + peak-scale, identical for both arms).
Reports per-arm: judge accuracy /10 and mean probability on the
intended class. Also a side-by-side figure.

Run:  .venv/bin/python experiments/2026_08_26/bounce/run_bounce_vs_oneshot.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_24" / "rich_palette_8x8"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "bounce"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from run_4layer_topk import LAM, expand, load_data  # noqa: E402
from run_rich_palette_8x8 import G1, G2, POS1, POS2, S2, W1, W2, render8  # noqa: E402
from gain_feedback import center_norm  # noqa: E402
from run_bounce import (  # noqa: E402
    _encode_windows,
    _topk_map,
    BLEND,
    CODE_DIM,
    K1,
    K2,
    K_SCHEDULE,
)

RESULTS = ROOT / "experiments" / "2026_08_26" / "bounce" / "results"


def ink(img):
    v = np.maximum(img, 0.0)
    return v / (v.max() + 1e-9)


def main():
    wz = np.load(ROOT / "experiments" / "2026_08_26"
                 / "base" / "results" / "weights.npz")
    W1n, W2n, Wtn = wz["W1"], wz["W2"], wz["Wtop"]

    class Bank:
        def __init__(s, Wb):
            s.W, s.k = Wb, Wb.shape[0]

    b2 = Bank(W2n)

    def top_match(c2, label=None):
        H, _ = center_norm(c2.reshape(-1))
        lab = np.zeros(10)
        if label is not None:
            lab[label] = LAM
        z, _ = center_norm(np.concatenate([H, lab]))
        return int(np.argmax(Wtn @ z))

    def memory_code(u):
        return np.maximum(Wtn[u, :CODE_DIM], 0.0).reshape(G2, G2, K2)

    def render_from_code(c2, k):
        m1 = _topk_map(expand(_topk_map(c2, k), b2, W2, S2, (G1, G1, K1)), k)
        return render8(m1, W1n)

    def norm_flat(c2):
        f = c2.reshape(-1)
        return (f / (np.linalg.norm(f) + 1e-9)).reshape(c2.shape)

    oneshot, bounced = [], []
    for j in range(10):
        c2 = memory_code(top_match(np.zeros((G2, G2, K2)), label=j))
        oneshot.append(render_from_code(c2, 1))
        for bi, k in enumerate(K_SCHEDULE):
            if bi > 0:
                m1 = _encode_windows(img[..., None], POS1, W1, W1n)
                c2e = _encode_windows(m1, POS2, W2, W2n)
                c_mem = memory_code(top_match(c2e, label=j))
                c2 = (1 - BLEND) * norm_flat(c2e) + BLEND * norm_flat(c_mem)
            img = render_from_code(c2, k)
        bounced.append(img)

    Xtr, ytr, _, _ = load_data()
    judge = LogisticRegression(max_iter=1000)
    judge.fit(Xtr[:10000].reshape(10000, -1), ytr[:10000])

    rows = []
    for name, imgs in [("one-shot k=1", oneshot), ("bounce k=1", bounced)]:
        P = judge.predict_proba(np.stack([ink(i).reshape(-1) for i in imgs]))
        acc = int((P.argmax(axis=1) == np.arange(10)).sum())
        conf = float(np.mean([P[j, j] for j in range(10)]))
        rows.append((name, acc, conf))
        print(f"{name}: judge {acc}/10, mean p(intended)={conf:.3f}", flush=True)

    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.9))
    for j in range(10):
        for r, imgs in enumerate([oneshot, bounced]):
            ax = axes[r, j]
            ax.imshow(imgs[j], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel("one-shot", fontsize=8, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("bounce", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("One-shot k=1 vs anchored bounce (final k=1)")
    fig.tight_layout()
    fig.savefig(RESULTS / "oneshot_vs_bounce.png", dpi=110)
    plt.close(fig)

    with open(RESULTS / "report.md", "a") as f:
        f.write("\n## Head-to-head vs one-shot (pixel-LR judge, ink-normalized)\n\n")
        for name, acc, conf in rows:
            f.write(f"- {name}: {acc}/10, mean p(intended) {conf:.3f}\n")
        f.write("\nFigure: oneshot_vs_bounce.png\n")
    print("Done.")


if __name__ == "__main__":
    main()
