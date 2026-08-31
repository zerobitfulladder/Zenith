"""Shared loader for the Fashion-MNIST arm.

Identical protocol to every MNIST run of 08-31 so the two boards can be put
side by side: one shuffle with SEED=0, first 20000 train, next 5000 test,
pixels in [0,1], flattened.
"""

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "competitive"))
from compete import center_norm, geo_step, EPS          # noqa: E402,F401

N_TRAIN, N_TEST, SEED = 20000, 5000, 0
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load(name="fashion_mnist"):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_images.npy").astype(np.float64).reshape(-1, 784)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in name else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X = X / 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def join(X, y=None, rho=1.0):
    """[image ; label] with the label block scaled to rho * ||image||."""
    V = np.zeros((len(X), X.shape[1] + 10))
    V[:, :X.shape[1]] = X
    if y is not None:
        L = np.zeros((len(X), 10)); L[np.arange(len(X)), y] = 1.0
        g = rho * np.linalg.norm(X, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, X.shape[1]:] = L * g[:, None]
    return center_norm(V)
