"""Why the fit gate collapses on Fashion-MNIST, and the board as one picture.

Two questions:
  1. is the WITHIN-class spread larger, or the BETWEEN-class distance smaller?
  2. does the failure happen at training time (one expert claims two classes)
     or at read time (the right expert exists but loses the gate)?
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, center_norm, EPS, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MNIST_CLASSES = [str(i) for i in range(10)]


def spread(X, y, tag, names):
    """Cosine geometry of the raw data, after the same centring the model sees."""
    Z = center_norm(X)
    C = np.stack([Z[y == c].mean(0) for c in range(10)])
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), EPS)
    within = np.array([float((Z[y == c] @ Cn[c]).mean()) for c in range(10)])
    G = Cn @ Cn.T; np.fill_diagonal(G, -1.0)
    nearest = G.max(1); who = G.argmax(1)
    between = float(G[np.triu_indices(10, 1)].mean())
    print(f"\n[{tag}]  mean sim to own class mean {within.mean():.4f}   "
          f"mean sim between class means {between:.4f}   "
          f"ratio {within.mean()/max(between,1e-9):.2f}")
    for c in np.argsort(nearest)[::-1][:4]:
        print(f"    {names[c]:<12} own {within[c]:.3f}   nearest other "
              f"{names[who[c]]:<12} {nearest[c]:.3f}")
    return {"within_mean": float(within.mean()), "between_mean": between,
            "within_per_class": within.tolist(),
            "nearest_other": {names[c]: [names[who[c]], float(nearest[c])]
                              for c in range(10)}}


def gate_autopsy(W, wins, Xte, yte, n_img):
    """Split the fit gate's mistakes into 'no expert for this class' and
    'the right expert was there and lost'."""
    live = wins.sum(1) > 0
    claim = np.where(live, wins.argmax(1), -1)                 # expert -> its class
    Q = join(Xte)
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k).transpose(1, 0, 2)
    R = np.matmul(S, W)
    err = np.linalg.norm(Q[None, :, :n_img] - R[:, :, :n_img], axis=2) / np.maximum(
        np.linalg.norm(Q[None, :, :n_img], axis=2), EPS)
    err = np.where(live[:, None], err, np.inf)
    pick = err.argmin(0)
    got = claim[pick]
    acc = float((got == yte).mean())
    has = np.array([(claim == c).any() for c in range(10)])
    # rank of the best same-class expert in the error ordering
    order = err.argsort(0)
    rank_of_right = np.full(len(yte), -1)
    for i in range(len(yte)):
        same = np.where(claim[order[:, i]] == yte[i])[0]
        if len(same):
            rank_of_right[i] = same[0]
    wrong = got != yte
    print(f"\n  fit gate {acc:.4f}   experts alive {int(live.sum())}   "
          f"classes with an expert {int(has.sum())}/10")
    print(f"  of the mistakes: right-class expert existed but ranked "
          f"{np.median(rank_of_right[wrong & (rank_of_right>=0)]):.0f} on average "
          f"(1st = would have been correct)")
    print(f"  right-class expert was 1st for {float((rank_of_right==0).mean()):.4f} "
          f"of all test samples, in top-3 for "
          f"{float(((rank_of_right>=0)&(rank_of_right<3)).mean()):.4f}")
    per = {CLASSES[c]: float((got[yte == c] == c).mean()) for c in range(10)}
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (yte, np.where(got >= 0, got, 0)), 1)
    return acc, per, cm, claim, {
        "top1_right": float((rank_of_right == 0).mean()),
        "top3_right": float(((rank_of_right >= 0) & (rank_of_right < 3)).mean()),
        "classes_with_expert": int(has.sum()),
        "experts_alive": int(live.sum()),
        "per_class": per}


def main():
    out = {}
    Xf, yf, Xfe, yfe = load("fashion_mnist")
    Xm, ym, _, _ = load("mnist")
    out["spread_mnist"] = spread(Xm, ym, "MNIST", MNIST_CLASSES)
    out["spread_fashion"] = spread(Xf, yf, "Fashion-MNIST", CLASSES)

    z = np.load(OUT / "l1_weights.npz")
    n_img = 784
    print("\n--- fit-gate autopsy (Fashion, competitive arm) ---")
    acc, per, cm, claim, au = gate_autopsy(z["competitive"].astype(np.float64),
                                           z["competitive_wins"], Xfe, yfe, n_img)
    out["autopsy_competitive"] = au
    print("  per class:", {k: round(v, 3) for k, v in per.items()})
    print("  expert -> claimed class:",
          np.bincount(claim[claim >= 0], minlength=10).tolist(), "(experts per class)")

    # ---- the board, one picture -------------------------------------------
    bl = json.loads((OUT / "baselines_fashion_mnist.json").read_text())["results"]
    l1 = json.loads((OUT / "l1_metrics.json").read_text())["results"]
    rows = [(f"baseline: {k}", v["acc"], "#1b6ca8") for k, v in
            sorted(bl.items(), key=lambda kv: kv[1]["acc"])]
    ours = [("ours: competitive + fit gate", l1["competitive"]["gate_fit"], "#2f8f4e"),
            ("ours: conscience + fit gate", l1["conscience"]["gate_fit"], "#2f8f4e"),
            ("ours: control + fit gate", l1["control"]["gate_fit"], "#7a9f5a"),
            ("ours: control + conf gate", l1["control"]["gate_confidence"], "#7a9f5a"),
            ("ours: competitive + conf gate", l1["competitive"]["gate_confidence"], "#c1462d"),
            ("ours: competitive + soft vote", l1["competitive"]["soft_vote"], "#c1462d")]
    rows = sorted(rows + ours, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] + .004, i, f"{r[1]:.4f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0.1, color="k", lw=.8, ls=":"); ax.text(0.105, -0.6, "chance", fontsize=7)
    ax.set_xlim(0, 1.02); ax.grid(alpha=.25, axis="x")
    ax.set_xlabel("test accuracy — Fashion-MNIST, 20000 train / 5000 test")
    ax.set_title("off-the-shelf classifiers vs the hypercolumn architecture", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "20_board.png", dpi=140); plt.close(fig)

    # ---- where the difficulty is ------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    w = 0.38
    for i, (tag, s) in enumerate((("MNIST", out["spread_mnist"]),
                                  ("Fashion", out["spread_fashion"]))):
        ax[0].bar([i - w/2, i + w/2], [s["within_mean"], s["between_mean"]],
                  width=w, color=["#2f8f4e", "#c1462d"])
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["MNIST", "Fashion"])
    ax[0].set_ylabel("cosine similarity")
    ax[0].set_title("green: sample to its own class mean\nred: class mean to class mean",
                    fontsize=9)
    ax[0].grid(alpha=.25, axis="y")

    lp = json.loads((OUT / "baselines_fashion_mnist.json").read_text())["logistic_per_class"]
    o = np.argsort([lp[c] for c in CLASSES])
    ax[1].barh(range(10), [lp[CLASSES[c]] for c in o], color="#1b6ca8", height=.38,
               label="logistic")
    ax[1].barh(np.arange(10) + .4, [per[CLASSES[c]] for c in o], color="#2f8f4e",
               height=.38, label="ours (competitive + fit)")
    ax[1].set_yticks(np.arange(10) + .2)
    ax[1].set_yticklabels([CLASSES[c] for c in o], fontsize=7.5)
    ax[1].set_xlim(0, 1); ax[1].grid(alpha=.25, axis="x"); ax[1].legend(fontsize=7)
    ax[1].set_title("per class", fontsize=9)

    ax[2].imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="magma")
    ax[2].set_xticks(range(10)); ax[2].set_xticklabels(CLASSES, rotation=90, fontsize=6.5)
    ax[2].set_yticks(range(10)); ax[2].set_yticklabels(CLASSES, fontsize=6.5)
    ax[2].set_xlabel("said"); ax[2].set_ylabel("was")
    ax[2].set_title(f"competitive + fit gate — {acc:.4f}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "21_difficulty.png", dpi=140); plt.close(fig)

    (OUT / "diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}/20_board.png, 21_difficulty.png, diagnostics.json")


if __name__ == "__main__":
    main()
