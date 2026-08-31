"""One layer, K templates, whole images joined with the label. Nothing else.

Each training vector is [image ; label] with the label block scaled so it holds
the same energy as the image, then mean-centred and L2-normalised -- so a
template's inner product with a vector is their Pearson correlation.

Online spherical k-means, k-means++ seeding, no conscience, no teacher signal
beyond the label being part of the input.

    classify   write the image, leave the label blank, take the nearest
               template, read the label half it completes
    generate   write the label, leave the image blank, take the nearest
               template, look at the image half
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
K, RHO, EPOCHS, ETA_MIN, SEED = 100, 1.0, 12, 0.01, 0
N_TRAIN, N_TEST, EPS = 20000, 5000, 1e-12
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"


def load(name):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_images.npy").astype(np.float64).reshape(-1, 784)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def cn(V):
    V = V - V.mean(1, keepdims=True)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


def join(X, y=None):
    V = np.zeros((len(X), 794))
    V[:, :784] = X
    if y is not None:
        L = np.zeros((len(X), 10)); L[np.arange(len(X)), y] = 1.0
        g = RHO * np.linalg.norm(X, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, 784:] = L * g[:, None]
    return cn(V)


def kmeanspp(Q, k, rng):
    W = np.empty((k, Q.shape[1]))
    W[0] = Q[rng.integers(len(Q))]
    d2 = 2.0 - 2.0 * (Q @ W[0])
    for i in range(1, k):
        p = np.maximum(d2, 0); s = p.sum()
        j = rng.integers(len(Q)) if s <= 0 else rng.choice(len(Q), p=p / s)
        W[i] = Q[j]
        d2 = np.minimum(d2, 2.0 - 2.0 * (Q @ W[i]))
    return W


def train(Q, rng, batch=512):
    W = kmeanspp(Q, K, rng)
    n = np.zeros(K)
    for ep in range(EPOCHS):
        for s in range(0, len(Q), batch):
            B = Q[rng.permutation(len(Q))[:0] if False else slice(s, s + batch)]
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], ETA_MIN) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
        print(f"  epoch {ep+1}/{EPOCHS}", flush=True)
    return W, n


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load(DS)
    Q = join(Xtr, ytr)
    print(f"{DS}: {len(Xtr)} train / {len(Xte)} test, K={K}, "
          f"label energy = image energy", flush=True)
    W, n = train(Q, np.random.default_rng(SEED + 1))

    lab = W[:, 784:].argmax(1)
    counts = np.bincount(lab, minlength=10)
    train_win = (Q @ W.T).argmax(1)
    pur = np.array([np.bincount(ytr[train_win == j], minlength=10).max() /
                    max((train_win == j).sum(), 1) for j in range(K)])

    # classify: image in, label blank
    Qq = join(Xte)
    S = Qq @ W.T
    win = S.argmax(1)
    acc = float((lab[win] == yte).mean())
    # same, scoring on the image half only (a template with a big image half
    # would otherwise be favoured)
    Wi = W[:, :784] / np.maximum(np.linalg.norm(W[:, :784], axis=1, keepdims=True), EPS)
    win2 = (cn(Xte) @ Wi.T).argmax(1)
    acc2 = float((lab[win2] == yte).mean())
    print(f"\n  classification (nearest template, label blank) {acc:.4f}")
    print(f"  same, scored on the image half only            {acc2:.4f}")
    print(f"  dead templates {int((n == 0).sum())}/{K}   "
          f"templates per class {counts.tolist()}")
    print(f"  mean class purity of what each template won {pur.mean():.3f}")

    # generate: label in, image blank
    fig, axes = plt.subplots(10, 10, figsize=(10, 10.4))
    V = np.zeros((10, 794))
    V[np.arange(10), 784 + np.arange(10)] = 1.0
    Sg = cn(V) @ W.T
    for d in range(10):
        order = np.argsort(Sg[d])[::-1][:10]
        for j, t in enumerate(order):
            a = axes[d][j]
            a.imshow(W[t, :784].reshape(28, 28), cmap="gray")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{lab[t]}·{int(n[t])}", fontsize=5, pad=1.2)
        axes[d][0].set_ylabel(f"asked {d}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.suptitle(f"{DS}: label in, image out — the 10 templates closest to each "
                 f"label (title: the label the template itself says · how many it won)",
                 fontsize=9)
    fig.subplots_adjust(left=.075, right=.995, top=.945, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / f"joint_generate_{DS}.png", dpi=140); plt.close(fig)

    o = np.argsort(n)[::-1]
    fig, axes = plt.subplots(10, 10, figsize=(10, 10.4))
    for i, a in enumerate(np.ravel(axes)):
        a.imshow(W[o[i], :784].reshape(28, 28), cmap="gray")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{lab[o[i]]}·{int(n[o[i]])}", fontsize=5, pad=1.2)
    fig.suptitle(f"{DS}: all {K} templates, most used first "
                 f"(title: label it says · how many it won)", fontsize=9)
    fig.subplots_adjust(left=.005, right=.995, top=.955, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / f"joint_templates_{DS}.png", dpi=140); plt.close(fig)

    np.savez_compressed(OUT / f"joint_{DS}.npz", W=W.astype(np.float32), n=n)
    (OUT / f"joint_{DS}.json").write_text(json.dumps(
        {"K": K, "rho": RHO, "acc": acc, "acc_image_half": acc2,
         "dead": int((n == 0).sum()), "per_class": counts.tolist(),
         "purity": float(pur.mean()), "seconds": round(time.time() - t0, 1)},
        indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> joint_generate_{DS}.png, "
          f"joint_templates_{DS}.png")


if __name__ == "__main__":
    main()
