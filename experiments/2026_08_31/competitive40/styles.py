"""When a digit has several experts, what distinguishes them?"""
import sys, numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "competitive"))
from compete40 import H, K, OUT, predict
from compete import load, join, EPS

Xtr, ytr, Xte, yte = load(); n_img = Xtr.shape[1]
z = np.load(OUT / "weights.npz")
for mode in ("competitive", "conscience"):
    W, wins = z[mode].astype(np.float64), z[mode + "_wins"]
    # recompute who would win each TRAINING sample (the training rule)
    win = np.zeros(len(Xtr), dtype=int)
    for s in range(0, len(Xtr), 2000):
        b = slice(s, s + 2000)
        P, _ = predict(W, join(Xtr[b]), n_img)
        c = P[:, np.arange(P.shape[1]), ytr[b]] / np.maximum(np.linalg.norm(P, axis=2), EPS)
        win[b] = c.argmax(0)
    tot = wins.sum(1); live = np.nonzero(tot > tot.sum() * 0.002)[0]
    dom = wins.argmax(1)
    print(f"\n=== {mode}: {len(live)} live experts ===")
    rows = []
    for c in range(10):
        who = [int(h) for h in live if dom[h] == c]
        sh = [wins[h, c] / max(wins[:, c].sum(), 1) for h in who]
        rows.append((c, who, sh))
        print(f"  digit {c}: {len(who)} expert(s)  " +
              "  ".join(f"h{h} takes {s*100:4.1f}% of the class" for h, s in zip(who, sh)))
    # do same-class experts span more similar subspaces than cross-class ones?
    def ov(i, j):
        return float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
                     np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                             (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
    same = [ov(a, b) for c, who, _ in rows for i, a in enumerate(who) for b in who[i+1:]]
    cross = [ov(a, b) for i, a in enumerate(live) for b in live[i+1:] if dom[a] != dom[b]]
    print(f"  subspace overlap: same-digit pairs {np.mean(same) if same else float('nan'):.4f}   "
          f"different-digit pairs {np.mean(cross):.4f}")

    if mode == "conscience":
        mx = max(len(w) for _, w, _ in rows)
        fig, axes = plt.subplots(10, mx + 1, figsize=(1.05 * (mx + 1), 10.5))
        for c, who, sh in rows:
            m = Xtr[ytr == c].mean(0).reshape(28, 28)
            axes[c, 0].imshow(m, cmap="gray_r"); axes[c, 0].set_ylabel(str(c), fontsize=8)
            axes[c, 0].set_title("all" if c == 0 else "", fontsize=7)
            for k, (h, s) in enumerate(zip(who, sh)):
                sel = (win == h) & (ytr == c)
                axes[c, k + 1].imshow(Xtr[sel].mean(0).reshape(28, 28) if sel.sum()
                                      else np.zeros((28, 28)), cmap="gray_r")
                axes[c, k + 1].set_title(f"h{h} {s*100:.0f}%  n={sel.sum()}", fontsize=5.5)
            for k in range(len(who), mx):
                axes[c, k + 1].axis("off")
        for a in axes.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle("mean of the digits each expert actually won\n"
                     "(left column: the class average, for comparison)", fontsize=9)
        fig.subplots_adjust(left=.05, right=.99, top=.93, bottom=.01, wspace=.08, hspace=.25)
        fig.savefig(OUT / "03_expert_styles.png", dpi=140); plt.close(fig)
print(f"\n-> {OUT/'03_expert_styles.png'}")
