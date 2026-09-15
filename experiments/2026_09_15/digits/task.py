"""Real digits for the hypothesis machine.

28x28 -> 14x14 by 2x2 mean, then four ink levels. Each image is shifted so its ink sits
at the centre, which is what makes "region 0" mean the same thing on every image -- the
machine addresses relatively, so without centring a coarse region would be meaningless.

The machine only ever sees one cell at a time.
"""

import numpy as np

OOB = 9
DIGITS = (0, 1, 7)


class Digits:
    name = "DIGITS"
    W = H = 14
    palette = 4
    regions = 9            # opt in to the where-did-this-come-from facts
    slots = 9              # one per region, so a whole coarse view fits in memory
    relations = False      # the region is already in each fact; pairs would only bloat
    count_fact = False     # a count lies after one slip: unrecoverable
    next_fact = True       # 'next empty slot is k': sayable AND self-correcting
    answers = list(DIGITS)

    def __init__(self, root, n_train=4000, n_test=1000, seed=0):
        X = np.load(root / "mnist/digits/train_images.npy").astype(np.float32)
        Y = np.load(root / "mnist/digits/train_labels.npy").astype(int)
        keep = np.isin(Y, DIGITS)
        X, Y = X[keep], Y[keep]
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(X))[: n_train + n_test]
        X, Y = X[p], Y[p]
        X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
        X = np.stack([centre(a) for a in X])
        self.G = np.digitize(X, [0.15, 0.40, 0.70]).astype(np.int8)
        self.Y = Y
        self.train = list(range(n_train))
        self.test = list(range(n_train, n_train + n_test))
        self.answer = None

    def take(self, i):
        self.answer = int(self.Y[i])
        return self.G[i]

    def sample(self, rng):
        return self.take(rng.choice(self.train))

    def sym(self, g, x, y):
        if not (0 <= x < self.W and 0 <= y < self.H):
            return OOB
        return int(g[y, x])

    def nearest(self, g, s, x, y):
        ys, xs = np.nonzero(g == s)
        if len(xs) == 0:
            return None
        d = np.abs(xs - x) + np.abs(ys - y)
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])

    def majority(self, idx):
        c = np.bincount(self.Y[idx])
        return c.max() / len(idx)


def centre(a):
    ys, xs = np.nonzero(a > 0.15)
    if len(xs) == 0:
        return a
    dy = a.shape[0] // 2 - int(round(ys.mean()))
    dx = a.shape[1] // 2 - int(round(xs.mean()))
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)
