"""Digits seen through a learned vocabulary instead of ink levels.

4x4 patches of the centred 28x28 image at stride 2 -> a 13x13 map. k-means with k=10 on
the training patches; each cell of the map is the index of the nearest template. The
cursor now reports "which of ten local shapes is here" rather than "how dark is here".
Everything else -- regions, slots, the next-empty-slot fact, the teacher -- is unchanged.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from task import DIGITS, Digits, centre

P, S = 4, 2


def patches(img):
    return np.stack([img[y:y + P, x:x + P].ravel()
                     for y in range(0, 28 - P + 1, S) for x in range(0, 28 - P + 1, S)])


class Patches(Digits):
    name = "PATCHES"
    W = H = 13
    palette = 10

    def __init__(self, root, n_train=4000, n_test=1000, seed=0, k=10):
        X = np.load(root / "mnist/digits/train_images.npy").astype(np.float32)
        Y = np.load(root / "mnist/digits/train_labels.npy").astype(int)
        keep = np.isin(Y, DIGITS)
        X, Y = X[keep], Y[keep]
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(X))[: n_train + n_test]
        X, Y = X[p], Y[p]
        X = np.stack([centre(a) for a in X])
        allp = np.stack([patches(a) for a in X])                    # n x 169 x 16
        km = MiniBatchKMeans(k, random_state=seed, n_init=10).fit(
            allp[:n_train].reshape(-1, P * P))
        self.G = km.predict(allp.reshape(-1, P * P)).reshape(len(X), 13, 13).astype(np.int8)
        self.templates = km.cluster_centers_.reshape(k, P, P)
        self.Y = Y
        self.train = list(range(n_train))
        self.test = list(range(n_train, n_train + n_test))
        self.answer = None
        self.palette = k
