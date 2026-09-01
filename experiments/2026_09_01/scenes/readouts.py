"""Three ways to answer a question from one feedforward pass of the pooled map.

The point of having three is that "the baseline" should be the strongest fair
opponent, not a strawman. All three read the SAME 4 x 8 x 64 pooled map, which
still carries position -- nothing is hidden from them.

    quantiser   yesterday's second rung. Nearest template over
                [map ; query ; answer] at matched energy, read the answer half.
                This is the architecture's own answer, and it stores template
                means, so it can only return arrangements it has seen.
    logistic    a linear discriminative readout, which could in principle
                learn "leftness" as a direction.
    mlp         one hidden layer. The reference for what plain capacity buys.

Rows are gathered per minibatch rather than materialised: Q3 asks three
questions of every scene, and a dense design matrix for it would be 500 MB.
"""

import numpy as np

EPS = 1e-12


def cn(V):
    V = V - V.mean(1, keepdims=True)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


def rownorm(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


# ---------------------------------------------------------------- gradient nets

class Net:
    def __init__(self, d, out, hidden, seed):
        r = np.random.default_rng(seed)
        if hidden:
            self.W = [r.normal(0, np.sqrt(2.0 / d), (d, hidden)).astype(np.float32),
                      r.normal(0, np.sqrt(1.0 / hidden), (hidden, out)).astype(np.float32)]
        else:
            self.W = [r.normal(0, np.sqrt(1.0 / d), (d, out)).astype(np.float32)]
        self.b = [np.zeros(w.shape[1], np.float32) for w in self.W]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        if len(self.W) == 1:
            return X @ self.W[0] + self.b[0], None
        h = np.maximum(X @ self.W[0] + self.b[0], 0)
        return h @ self.W[1] + self.b[1], h

    def step(self, X, dZ, h, lr):
        if len(self.W) == 1:
            g = [X.T @ dZ, dZ.sum(0)]
        else:
            dh = (dZ @ self.W[1].T) * (h > 0)
            g = [X.T @ dh, h.T @ dZ, dh.sum(0), dZ.sum(0)]
        self.t += 1
        p = self.W + self.b
        for i in range(len(p)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] * g[i]
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p[i] -= lr * mh / (np.sqrt(vh) + 1e-8)


def _design(base, rows, extra, b):
    X = base[rows[b]]
    return X if extra is None else np.hstack([X, extra[b]])


def fit_net(base, rows, extra, Y, out, mode, hidden=0,
            epochs=25, batch=256, lr=1e-3, seed=0):
    d = base.shape[1] + (0 if extra is None else extra.shape[1])
    net = Net(d, out, hidden, seed)
    rng = np.random.default_rng(seed + 7)
    idx = np.arange(len(rows))
    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            X = _design(base, rows, extra, b)
            Z, h = net.forward(X)
            if mode == "softmax":
                Z -= Z.max(1, keepdims=True)
                P = np.exp(Z); P /= P.sum(1, keepdims=True)
                P[np.arange(len(b)), Y[b]] -= 1.0
            else:
                P = 1.0 / (1.0 + np.exp(-Z)) - Y[b]
            net.step(X, (P / len(b)).astype(np.float32), h, lr)
    return net


def predict_net(net, base, rows, extra, batch=2048):
    out = []
    for s in range(0, len(rows), batch):
        b = np.arange(s, min(s + batch, len(rows)))
        out.append(net.forward(_design(base, rows, extra, b))[0])
    return np.concatenate(out)


# ---------------------------------------------------------------- the quantiser

def _joint(base, rows, extra, A, rho, na):
    """[map ; query ; answer], each extra block scaled to rho of the map's energy.

    At read time A is None and the answer block is blank -- the match is decided
    by the map and the query, and the winner's answer half is what it returns.
    """
    blocks = [rownorm(base[rows])]
    if extra is not None:
        blocks.append(rownorm(extra) * rho)
    blocks.append(np.zeros((len(rows), na), np.float32) if A is None
                  else rownorm(A) * rho)
    return cn(np.hstack(blocks).astype(np.float32))


def kmeanspp(Q, K, rng):
    W = np.empty((K, Q.shape[1]), np.float32)
    W[0] = Q[rng.integers(len(Q))]
    d2 = 2.0 - 2.0 * (Q @ W[0])
    for k in range(1, K):
        p = np.maximum(d2, 0); s = p.sum()
        i = rng.integers(len(Q)) if s <= 0 else rng.choice(len(Q), p=p / s)
        W[k] = Q[i]
        d2 = np.minimum(d2, 2.0 - 2.0 * (Q @ W[k]))
    return W


def fit_quant(base, rows, extra, A, K=400, epochs=12, rho=0.7,
              batch=512, eta_min=0.01, seed=0):
    rng = np.random.default_rng(seed)
    na = A.shape[1]
    seed_rows = rng.choice(len(rows), min(4000, len(rows)), replace=False)
    W = kmeanspp(_joint(base, rows[seed_rows], None if extra is None else extra[seed_rows],
                        A[seed_rows], rho, na), K, rng)
    n = np.zeros(K)
    idx = np.arange(len(rows))
    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            B = _joint(base, rows[b], None if extra is None else extra[b], A[b], rho, na)
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], eta_min) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
    return W, int((n == 0).sum())


def predict_quant(W, base, rows, extra, na, rho=0.7, batch=2048):
    out = []
    for s in range(0, len(rows), batch):
        b = slice(s, min(s + batch, len(rows)))
        B = _joint(base, rows[b], None if extra is None else extra[b], None, rho, na)
        out.append(W[(B @ W.T).argmax(1)][:, -na:])
    return np.concatenate(out)
