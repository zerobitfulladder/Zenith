"""The two-rung stack on CelebA, with all 40 attributes as the second stream.

    L1  5x5 patches over 48x48 faces, stride 1, K1=128     -> 44x44 identities
        pooled to a GxG grid
    L2  one k-means layer over the pooled map joined with the 40 attributes
        at matched energy, K2=256

    image in  -> nearest template -> read its attribute half  (40 predictions)
    attrs in  -> nearest template -> its image half

Attributes are written as +/-1, so present and absent are symmetric and an
unspecified attribute is simply 0.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
PS, K1, K2, GRID = 5, 128, 256, 4
SIDE = 48 - PS + 1
N_TRAIN, N_TEST, FLOOR, EPS = 20000, 5000, 0.05, 1e-12
EPOCHS, ETA_MIN, SEED = 10, 0.01, 0


def load():
    d = ROOT / "data"
    X = np.load(d / "celeba/images_48.npy").astype(np.float32)
    A = np.load(d / "celeba/attrs_48.npy").astype(np.float64) * 2 - 1     # +/-1
    names = json.loads((d / "celeba/attr_names.json").read_text())
    p = np.random.default_rng(SEED).permutation(len(X))
    X, A = X[p], A[p]
    return (X[:N_TRAIN], A[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST],
            A[N_TRAIN:N_TRAIN + N_TEST], names)


def patches(X):
    V = sliding_window_view(X, (PS, PS), axis=(1, 2))
    return np.ascontiguousarray(V.reshape(len(X), -1, PS * PS), dtype=np.float64)


def prep(P):
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    return C / np.maximum(n, EPS)[..., None], n > FLOOR, n


def kmeans(Q, K, rng, epochs=EPOCHS, batch=4096):
    W = joint.kmeanspp(Q[rng.choice(len(Q), min(8000, len(Q)), False)], K, rng)
    n = np.zeros(K)
    for ep in range(epochs):
        for s in range(0, len(Q), batch):
            B = Q[s:s + batch]
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], ETA_MIN) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
    return W, n


def l1_map(W1, X, chunk=256):
    idx = np.full((len(X), SIDE * SIDE), -1, np.int16)
    mag = np.zeros((len(X), SIDE * SIDE), np.float32)
    for a in range(0, len(X), chunk):
        Q, keep, norm = prep(patches(X[a:a + chunk]))
        w = np.einsum("ipd,kd->ipk", Q, W1).argmax(-1)
        idx[a:a + len(Q)] = np.where(keep, w, -1)
        mag[a:a + len(Q)] = np.where(keep, norm, 0.0)
    return idx, mag


def pooled(idx, mag, g=GRID, A=None):
    n = len(idx)
    M = np.zeros((n, SIDE * SIDE, K1), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = mag[r, c]
    s = SIDE // g
    M = M.reshape(n, SIDE, SIDE, K1)[:, :g * s, :g * s]
    M = M.reshape(n, g, s, g, s, K1).max(axis=(2, 4)).reshape(n, -1)
    V = np.zeros((n, M.shape[1] + 40))
    V[:, :M.shape[1]] = M
    if A is not None:
        e = np.linalg.norm(M, axis=1)
        V[:, M.shape[1]:] = A * (e / np.sqrt(40))[:, None]
    return joint.cn(V)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    Xtr, Atr, Xte, Ate, names = load()
    print(f"celeba 48x48: {len(Xtr)} train / {len(Xte)} test, 40 attributes",
          flush=True)

    # ---- L1 -----------------------------------------------------------------
    samp = []
    for a in range(0, 6000, 256):
        Q, keep, _ = prep(patches(Xtr[a:a + 256]))
        for i in range(len(Q)):
            k = np.nonzero(keep[i])[0]
            if len(k):
                samp.append(Q[i, rng.choice(k, min(40, len(k)), False)])
    Qs = np.concatenate(samp)
    W1, n1 = kmeans(Qs, K1, rng)
    print(f"  L1 K={K1} on {len(Qs)} patches, {int((n1==0).sum())} dead", flush=True)

    itr, mtr = l1_map(W1, Xtr)
    ite, mte = l1_map(W1, Xte)
    print(f"  map {SIDE}x{SIDE} -> pooled {GRID}x{GRID}, "
          f"L2 input {GRID*GRID*K1 + 40}", flush=True)

    # ---- L2 -----------------------------------------------------------------
    A2 = np.concatenate([pooled(itr[s:s+2000], mtr[s:s+2000], GRID, Atr[s:s+2000])
                         for s in range(0, len(itr), 2000)])
    W2, n2 = kmeans(A2, K2, rng)
    P = GRID * GRID * K1
    print(f"  L2 K={K2}, {int((n2==0).sum())} dead", flush=True)

    # ---- predict ------------------------------------------------------------
    B = np.concatenate([pooled(ite[s:s+2000], mte[s:s+2000], GRID)
                        for s in range(0, len(ite), 2000)])
    win = (B @ W2.T).argmax(1)
    pred = np.sign(W2[win, P:])
    acc = (pred == np.sign(Ate)).mean(0)
    base = np.maximum((Ate > 0).mean(0), 1 - (Ate > 0).mean(0))
    print(f"\n  attribute accuracy  mean {acc.mean():.4f}   "
          f"majority-class baseline {base.mean():.4f}")
    order = np.argsort(acc - base)[::-1]
    for i in list(order[:5]) + list(order[-5:]):
        print(f"    {names[i]:<22} {acc[i]:.3f}  (baseline {base[i]:.3f})")

    np.savez_compressed(OUT / "celeba.npz", W1=W1.astype(np.float32),
                        W2=W2.astype(np.float32), n1=n1, n2=n2)
    (OUT / "celeba.json").write_text(json.dumps(
        {"K1": K1, "K2": K2, "grid": GRID, "attr_acc_mean": float(acc.mean()),
         "baseline_mean": float(base.mean()),
         "per_attr": {names[i]: [float(acc[i]), float(base[i])] for i in range(40)},
         "params": int(K1 * PS * PS + K2 * (P + 40)),
         "seconds": round(time.time() - t0, 1)}, indent=2))

    # ---- ask for things ------------------------------------------------------
    def ask(spec):
        q = np.zeros(P + 40)
        for k, v in spec.items():
            q[P + names.index(k)] = v
        return (joint.cn(q[None])[0] @ W2.T)

    asks = [("a woman", {"Male": -1}),
            ("a man", {"Male": +1}),
            ("a mustache", {"Mustache": +1}),
            ("a man with a mustache", {"Male": +1, "Mustache": +1}),
            ("A WOMAN WITH A MUSTACHE", {"Male": -1, "Mustache": +1}),
            ("a smiling blonde woman", {"Male": -1, "Smiling": +1, "Blond_Hair": +1}),
            ("an old bald man", {"Male": +1, "Bald": +1, "Young": -1}),
            ("a woman with eyeglasses", {"Male": -1, "Eyeglasses": +1})]
    fig, axes = plt.subplots(len(asks), 8, figsize=(8.6, 1.15 * len(asks)))
    rows = {}
    for r, (title, spec) in enumerate(asks):
        s = ask(spec)
        top = np.argsort(s)[::-1][:8]
        rows[title] = []
        for j, t in enumerate(top):
            im = W2[t, :P]
            a = axes[r][j]
            a.imshow(W2[t, :P].reshape(GRID, GRID, K1).max(-1), cmap="magma")
            a.set_xticks([]); a.set_yticks([])
            lab = W2[t, P:]
            rows[title].append({"template": int(t),
                                "male": float(lab[names.index("Male")]),
                                "mustache": float(lab[names.index("Mustache")])})
            a.set_title(f"M{lab[names.index('Male')]:+.2f} "
                        f"m{lab[names.index('Mustache')]:+.2f}", fontsize=4.5, pad=1)
        axes[r][0].set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("asked for an attribute combination — what the nearest templates say",
                 fontsize=9)
    fig.subplots_adjust(left=.20, right=.995, top=.93, bottom=.01,
                        wspace=.05, hspace=.30)
    fig.savefig(OUT / "celeba_ask.png", dpi=150); plt.close(fig)
    (OUT / "celeba_ask.json").write_text(json.dumps(rows, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
