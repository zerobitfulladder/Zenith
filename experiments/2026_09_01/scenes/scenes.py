"""Three-digit scenes, and the splits that make the question about arrangement.

One object centred in a frame cannot pose the question this project is stuck on.
A scene with three of them can:

    Q1  presence          "is there a 3?"                    a set property
    Q2  global order      "which is leftmost / rightmost?"   a FIXED region
    Q3  anchored relation "what is left of the 7?"           a region the IMAGE
                                                             decides

Q3 is the one that matters. The cells holding the answer are different in every
scene, because they depend on where the query digit happens to be. A readout
wired to fixed positions cannot find them; it has to locate the anchor first.

The splits are the measurement, not the accuracy. Ordered triples of distinct
classes are partitioned so that a test scene can be:

    seen     an arrangement trained on, with digit images never seen
    order    the same three classes in an order never trained
    triple   three classes that never appeared together at all

Anything that answered by memorising configurations is fine on `seen` and lost
on `order` -- same three digits, same ink, only the arrangement is new.

Canvas 48 x 100 so that after 5x5 patching (44 x 96 positions) the pooling grid
divides exactly: 4 x 8 blocks of 11 x 12.
"""

import json
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
W1_PATH = ROOT / "experiments/2026_08_31/kmeans/results/km_mnist.npz"

H, W = 48, 100                  # canvas
STAMP = 28                      # an MNIST digit
PS, FLOOR, EPS = 5, 0.05, 1e-12
SY, SX = H - PS + 1, W - PS + 1                 # 44 x 96 patch positions
GY, GX = 4, 8                                   # pooling grid, 11 x 12 blocks
MIN_GAP = 20                    # between x-centres, so left/right is never close
NDIG = 3
POOL_SPLIT = 40000              # digit images 0..40k build train scenes, rest test


# ---------------------------------------------------------------- the splits

def splits(seed=0):
    """Partition the 720 ordered triples of distinct classes.

    20 unordered triples are held out whole. Of the remaining 100, four of the
    six orderings train and two are reserved -- so `order` always tests an
    arrangement whose classes the model has seen together, repeatedly, just
    never like this.
    """
    rng = np.random.default_rng(seed)
    unordered = [(a, b, c) for a in range(10) for b in range(a + 1, 10)
                 for c in range(b + 1, 10)]
    rng.shuffle(unordered)
    held, rest = unordered[:20], unordered[20:]

    def perms(t):
        return [(t[i], t[j], t[k]) for i in range(3) for j in range(3)
                for k in range(3) if len({i, j, k}) == 3]

    train, order = [], []
    for t in rest:
        p = perms(t)
        rng.shuffle(p)
        train += p[:4]
        order += p[4:]
    triple = [q for t in held for q in perms(t)]
    return {"train": train, "order": order, "triple": triple}


# ---------------------------------------------------------------- the scenes

def load_digits():
    d = ROOT / "data"
    X = np.load(d / "mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 28, 28)
    y = np.load(d / "mnist/digits/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    by = [np.nonzero(y[:POOL_SPLIT] == c)[0] for c in range(10)]        # train pool
    bt = [np.nonzero(y[POOL_SPLIT:] == c)[0] + POOL_SPLIT for c in range(10)]
    return X, by, bt


def make(n, triples, X, bucket, seed):
    """n scenes. Returns images, the left-to-right classes, and their x-centres."""
    rng = np.random.default_rng(seed)
    T = np.asarray(triples)
    pick = T[rng.integers(len(T), size=n)]                       # (n, 3) left->right
    imgs = np.zeros((n, H, W), np.float32)

    # three x-centres, sorted, each at least MIN_GAP apart, uniform over the slack
    lo, hi = STAMP // 2, W - STAMP // 2                          # 14 .. 86
    slack = (hi - lo) - (NDIG - 1) * MIN_GAP                     # 32
    cuts = np.sort(rng.integers(0, slack + 1, size=(n, NDIG)), axis=1)
    xs = lo + cuts + MIN_GAP * np.arange(NDIG)
    ys = rng.integers(STAMP // 2, H - STAMP // 2 + 1, size=(n, NDIG))

    for i in range(n):
        for j in range(NDIG):
            src = X[rng.choice(bucket[pick[i, j]])]
            y0, x0 = ys[i, j] - STAMP // 2, xs[i, j] - STAMP // 2
            tile = imgs[i, y0:y0 + STAMP, x0:x0 + STAMP]
            np.maximum(tile, src, out=tile)
    return imgs, pick, xs


def make_single(n, X, bucket, seed):
    """One digit, anywhere on the same canvas. The recognition ceiling."""
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 10, n)
    imgs = np.zeros((n, H, W), np.float32)
    xs = rng.integers(STAMP // 2, W - STAMP // 2 + 1, n)
    ys = rng.integers(STAMP // 2, H - STAMP // 2 + 1, n)
    for i in range(n):
        src = X[rng.choice(bucket[cls[i]])]
        y0, x0 = ys[i] - STAMP // 2, xs[i] - STAMP // 2
        tile = imgs[i, y0:y0 + STAMP, x0:x0 + STAMP]
        np.maximum(tile, src, out=tile)
    return imgs, cls.astype(np.int64)


# ---------------------------------------------------------------- the questions

def q_presence(cls):
    P = np.zeros((len(cls), 10), np.float32)
    P[np.repeat(np.arange(len(cls)), NDIG), cls.ravel()] = 1.0
    return P


def q_order(cls):
    return cls[:, 0], cls[:, -1]            # leftmost, rightmost


def q_anchored(cls, seed):
    """(scene, query class, class immediately left of it or 10 for 'none')."""
    rows = np.repeat(np.arange(len(cls)), NDIG)
    slot = np.tile(np.arange(NDIG), len(cls))
    query = cls[rows, slot]
    answer = np.where(slot == 0, 10, cls[rows, np.maximum(slot - 1, 0)])
    return rows, query.astype(np.int64), answer.astype(np.int64)


# ---------------------------------------------------------------- layer 1

def patches(X):
    V = sliding_window_view(X, (PS, PS), axis=(1, 2))
    return np.ascontiguousarray(V.reshape(len(X), -1, PS * PS), dtype=np.float32)


def prep(P):
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    return C / np.maximum(n, EPS)[..., None], n > FLOOR, n


def l1_pooled(W1, X, chunk=32, gy=GY, gx=GX):
    """Winner-take-all over 5x5 patches, then max-pool per channel onto GY x GX.

    Exactly yesterday's message format -- an identity per position carrying the
    patch's contrast -- and exactly yesterday's pooling, which is what makes an
    identity map survive a shift. The grid is KEPT, not collapsed: the baseline
    gets full positional information, so the comparison is honest.
    """
    K = len(W1)
    by, bx = SY // gy, SX // gx
    out = np.zeros((len(X), gy, gx, K), np.float32)
    for a in range(0, len(X), chunk):
        B = X[a:a + chunk]
        Q, keep, norm = prep(patches(B))
        win = np.einsum("ipd,kd->ipk", Q, W1).argmax(-1)
        M = np.zeros((len(B), SY * SX, K), np.float32)
        r, c = np.nonzero(keep)
        M[r, c, win[r, c]] = norm[r, c]
        M = M.reshape(len(B), gy, by, gx, bx, K).max(axis=(2, 4))
        out[a:a + len(B)] = M
    return out.reshape(len(X), -1)


def load_w1():
    return np.load(W1_PATH)["64"].astype(np.float32)


# ---------------------------------------------------------------- eyeball it

def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = splits()
    X, btr, bte = load_digits()
    imgs, cls, xs = make(8, sp["train"], X, btr, 1)
    rows, query, answer = q_anchored(cls, 0)
    fig, ax = plt.subplots(8, 1, figsize=(7, 9))
    for i in range(8):
        ax[i].imshow(imgs[i], cmap="gray_r"); ax[i].axis("off")
        qs = [f"left of {query[j]} -> " + ("none" if answer[j] == 10 else str(answer[j]))
              for j in np.nonzero(rows == i)[0]]
        ax[i].set_title(f"{list(cls[i])}   " + " | ".join(qs), fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "scenes.png", dpi=110)
    print({k: len(v) for k, v in sp.items()})
    print("overlap train/order:", len(set(sp["train"]) & set(sp["order"])),
          " train/triple:", len(set(sp["train"]) & set(sp["triple"])))
    W1 = load_w1()
    M = l1_pooled(W1, imgs)
    print("pooled map", M.shape, "nonzero per scene", (M > 0).sum(1).mean())


if __name__ == "__main__":
    main()
