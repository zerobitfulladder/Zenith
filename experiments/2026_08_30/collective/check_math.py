"""Proves the fast rank-1 update equals the literal leave-one-out rule.

The fast path never builds the (k, dim) leave-one-out array; it relies on
tau_i = e - g_i w_i, which drops the s_i w_i term. This checks that claim
against a direct transcription of the definition.
"""
import numpy as np
from collective import Hypercolumn, EPS


def naive_step(W, x_hat, eta, weighted):
    """The rule exactly as written: build every leave-one-out residual."""
    s = W @ x_hat
    recon = s @ W
    others = recon[None, :] - s[:, None] * W        # what the others built
    r = x_hat[None, :] - others                     # the residue left for i
    if weighted:
        r = s[:, None] * r
    tau = r - np.sum(r * W, axis=1, keepdims=True) * W
    tn = np.linalg.norm(tau, axis=1)
    out = W.copy()
    live = tn > EPS
    th = eta * tn[live]
    out[live] = (W[live] * np.cos(th)[:, None]
                 + (tau[live] / tn[live, None]) * np.sin(th)[:, None])
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + EPS)


rng = np.random.default_rng(0)
for rule, weighted in (("raw", False), ("weighted", True)):
    worst = 0.0
    for trial in range(200):
        k, d = rng.integers(3, 40), rng.integers(5, 120)
        hc = Hypercolumn(k, d, eta=0.3, rng=np.random.default_rng(trial), rule=rule)
        W0 = hc.W.copy()
        x = rng.standard_normal(d)
        x -= x.mean()
        x /= np.linalg.norm(x)
        hc.learn_one(x)
        worst = max(worst, np.abs(hc.W - naive_step(W0, x, 0.3, weighted)).max())
    print(f"{rule:9s} max |fast - naive| over 200 random layers: {worst:.3e}")

# The identity the speedup rests on, stated on its own.
W = np.linalg.qr(rng.standard_normal((60, 60)))[0][:17]
W /= np.linalg.norm(W, axis=1, keepdims=True)
x = rng.standard_normal(60); x -= x.mean(); x /= np.linalg.norm(x)
s = W @ x
e = x - s @ W
g = W @ e
r = x[None, :] - ((s @ W)[None, :] - s[:, None] * W)
tau_naive = r - np.sum(r * W, axis=1, keepdims=True) * W
print("tau_i == e - g_i w_i          :", np.allclose(tau_naive, e[None, :] - g[:, None] * W))
print("||tau_i||^2 == ||e||^2 - g_i^2:", np.allclose(np.linalg.norm(tau_naive, axis=1) ** 2,
                                                    (e @ e) - g ** 2))
