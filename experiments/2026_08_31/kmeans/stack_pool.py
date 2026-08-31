"""Same two rungs, but the L1 identity map is max-pooled before L2 sees it.

At full resolution a one-pixel shift changes which template wins where, and
one-hots of different templates are orthogonal -- so two images of the same
digit can have almost no similarity. Pooling over a block restores the
tolerance: the same template winning anywhere in the block counts the same.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km, joint, stack

OUT = Path(__file__).resolve().parent / "results"
EPS, K2, EPOCHS, ETA_MIN = 1e-12, 100, 12, 0.01
GRIDS = [4, 3, 2, 1]


def dens(idx, mag, k1, g, y=None):
    n = len(idx)
    M = np.zeros((n, stack.SIDE * stack.SIDE, k1), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = mag[r, c]
    if g != stack.SIDE:
        M = M.reshape(n, stack.SIDE, stack.SIDE, k1)
        s = stack.SIDE // g
        M = M[:, :g * s, :g * s].reshape(n, g, s, g, s, k1).max(axis=(2, 4))
    M = M.reshape(n, -1)
    V = np.zeros((n, M.shape[1] + 10))
    V[:, :M.shape[1]] = M
    if y is not None:
        e = np.linalg.norm(M, axis=1)
        L = np.zeros((n, 10)); L[np.arange(n), y] = 1.0
        V[:, M.shape[1]:] = L * e[:, None]
    return joint.cn(V)


def main():
    W1 = np.load(OUT / "km_mnist.npz")["64"].astype(np.float64)
    k1 = len(W1)
    Xtr, ytr, Xte, yte = joint.load("mnist")
    itr, mtr = stack.l1_map(W1, Xtr)
    ite, mte = stack.l1_map(W1, Xte)
    res = {}
    for g in GRIDS:
        rng = np.random.default_rng(1)
        seed = rng.choice(len(Xtr), 3000, replace=False)
        W = joint.kmeanspp(dens(itr[seed], mtr[seed], k1, g, ytr[seed]), K2, rng)
        n = np.zeros(K2)
        order = np.arange(len(Xtr))
        for ep in range(EPOCHS):
            rng.shuffle(order)
            for s in range(0, len(order), 512):
                b = order[s:s + 512]
                B = dens(itr[b], mtr[b], k1, g, ytr[b])
                win = (B @ W.T).argmax(1)
                for j in np.unique(win):
                    m = B[win == j]
                    n[j] += len(m)
                    W[j] += min(max(1.0 / n[j], ETA_MIN) * len(m), 1.0) * (m.mean(0) - W[j])
                    W[j] /= np.linalg.norm(W[j]) + EPS
        P = g * g * k1
        lab = W[:, P:].argmax(1)
        acc = []
        for s in range(0, len(Xte), 1000):
            B = dens(ite[s:s + 1000], mte[s:s + 1000], k1, g)
            acc.append(lab[(B @ W.T).argmax(1)] == yte[s:s + 1000])
        a = float(np.concatenate(acc).mean())
        res[str(g)] = {"acc": a, "dims": int(P + 10),
                       "params": int(K2 * (P + 10)), "dead": int((n == 0).sum())}
        print(f"  pool {g:>2}x{g:<2}  dims {P+10:>6}  params {K2*(P+10):>9,}  "
              f"accuracy {a:.4f}  dead {int((n==0).sum())}", flush=True)
    res["reference"] = {"one layer on raw pixels": 0.9210, "logistic": 0.9074}
    (OUT / "stack_pool.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
