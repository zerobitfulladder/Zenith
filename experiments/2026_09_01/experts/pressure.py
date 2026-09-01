"""Let the table decide which patches each template gets, and see what they become.

Until now the templates followed the ink alone: the winner was whoever rebuilt
the patch best. This puts the counted table into the competition, so a template
wins patches partly because it is EVIDENCE FOR THE CLASS the system currently
believes. Over rounds, template 12 stops being "a diagonal stroke that occurs
here" and becomes "the diagonal stroke that occurs here in 6s".

    winner = argmin ( ink error  -  beta * sum_c q_c T[t, cell, c] )

The learning rule is untouched -- the winner still rotates toward the patch it
won. Only the assignment changes.

Why this is not the thing that collapsed earlier (0.4143 -> 0.1543): that used
the TRUE label to pick winners during training, and reading has no label, so the
partition could not be reproduced. Here the bias comes from the INFERRED class,
computed by a first feedforward pass through the same table -- available at
training and at reading, identically. Both are two-pass:

    pass 1   winners on ink alone -> table -> belief q
    pass 2   winners with the bias -> table -> class, and the code

Annealed, because at round 0 the beliefs are weak and biasing hard on a bad
table would shape templates by noise. Accuracy is tracked every round: if it
turns down, the loop is confirming itself and we stop.
"""

import json, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import experts as E, blank as B, settle as S, readouts as R, single as SG

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = 4, 10, 1e-12
H, ROUNDS, BETAS = 180, 3, [0.0, 0.75, 1.5]
EPOCHS, BATCH, ETA = 2, 2048, 0.5
CELL = B.cellmap(GRID)


def tagged(X, y, rng, per_img):
    """Patches with the image they came from and the cell they sat in."""
    U, I, C = [], [], []
    for a in range(0, len(X), 256):
        Q, keep = E.patches(X[a:a + 256])
        for i in range(len(Q)):
            k = np.nonzero(keep[i])[0]
            if len(k):
                p = rng.choice(k, min(per_img, len(k)), False)
                U.append(Q[i, p]); I.append(np.full(len(p), a + i)); C.append(CELL[p])
    U = np.concatenate(U).astype(np.float32)
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), EPS), \
        np.concatenate(I), np.concatenate(C)


def belief(idx, T):
    Tc = T[:, CELL, :]
    P = np.arange(S.SIDE * S.SIDE)
    out = np.zeros((len(idx), NL))
    for a in range(0, len(idx), 500):
        b = idx[a:a + 500]; v = b >= 0
        out[a:a + len(b)] = (Tc[np.where(v, b, 0), P[None, :]] * v[..., None]).sum(1)
    z = (out - out.mean(1, keepdims=True)) / (out.std(1, keepdims=True) + 1e-9)
    q = np.exp(z - z.max(1, keepdims=True))
    return out, q / q.sum(1, keepdims=True)


def biased_winners(W, X, T, q, chunk=200):
    """Pass 2: winners with the table's bias, per image."""
    Tc = np.ascontiguousarray(T.transpose(1, 0, 2))          # (ncell, H, NL)
    Wf, wn = W[:, 0, :], (W[:, 0, :25] ** 2).sum(1)
    idx = np.full((len(X), S.SIDE * S.SIDE), -1, np.int16)
    for a in range(0, len(X), chunk):
        Q, keep = E.patches(X[a:a + chunk])
        P = Q.reshape(-1, 25)
        Sc = P @ Wf.T
        err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
        m = len(Q)
        bias = np.einsum('ik,ihk->ih', np.repeat(q[a:a + m], S.SIDE * S.SIDE, 0),
                         Tc[np.tile(CELL, m)], optimize=True)
        w = (err - bias).argmin(1).reshape(m, -1)
        idx[a:a + m] = np.where(keep.reshape(m, -1), w, -1)
    return idx


def train_biased(W, U, I, C, q, T, beta, rng):
    Tc = np.ascontiguousarray(T.transpose(1, 0, 2))
    Wf = W[:, 0, :].copy()
    wn = (Wf[:, :25] ** 2).sum(1)
    for ep in range(EPOCHS):
        order = rng.permutation(len(U))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            Bp = U[b]
            Sc = Bp @ Wf.T
            err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
            if beta > 0:
                err = err - beta * np.einsum('ik,ihk->ih', q[I[b]], Tc[C[b]],
                                             optimize=True)
            win = err.argmin(1)
            for j in np.unique(win):
                m = Bp[win == j]
                if len(m) >= 4:
                    Wf[j] = E.geo_step(Wf[j:j + 1].astype(np.float64),
                                       m.astype(np.float64), ETA)[0].astype(np.float32)
            wn = (Wf[:, :25] ** 2).sum(1)
    return Wf[:, None, :].astype(np.float64)


def pooled(idx, h):
    b = S.SIDE // GRID
    M = np.zeros((len(idx), S.SIDE * S.SIDE, h), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = 1.0
    return M.reshape(len(idx), GRID, b, GRID, b, h).max(axis=(2, 4)).reshape(len(idx), -1)


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)


def evaluate(W, Xtr, ytr, Xte, yte, beta):
    itr0, ite0 = SG.winners(W, Xtr), SG.winners(W, Xte)
    T0 = B.table(itr0, ytr, len(W), GRID)
    _, qtr = belief(itr0, T0); _, qte = belief(ite0, T0)
    if beta > 0:
        itr, ite = (biased_winners(W, Xtr, T0, qtr),
                    biased_winners(W, Xte, T0, qte))
    else:
        itr, ite = itr0, ite0
    T = B.table(itr, ytr, len(W), GRID)
    acc = float((B.score(ite, T, GRID).argmax(1) == yte).mean())
    Atr, Ate = pooled(itr, len(W)), pooled(ite, len(W))
    M = unit(np.stack([Atr[ytr == c].mean(0) for c in range(NL)]))
    nm = float(((unit(Ate) @ M.T).argmax(1) == yte).mean())
    net = R.fit_net(Atr, np.arange(len(Atr)), None, ytr, NL, "softmax", hidden=0, epochs=30)
    lin = float((R.predict_net(net, Ate, np.arange(len(Ate)), None).argmax(1) == yte).mean())
    return acc, nm, lin, itr, T, qtr


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    rng = np.random.default_rng(E.SEED + 1)
    U, I, C = tagged(Xtr, ytr, rng, E.PER_IMG)
    W = np.load(OUT / "single_unlabelled.npz")["W"].astype(np.float64)
    print(f"{len(U):,} tagged patches, {H} templates, starting from the "
          f"unsupervised vocabulary\n", flush=True)

    res = []
    for r, beta in enumerate(BETAS):
        if r > 0:
            _, _, _, itr, T, qtr = evaluate(W, Xtr, ytr, Xte, yte, BETAS[r - 1])
            W = train_biased(W, U, I, C, qtr, T, beta, np.random.default_rng(100 + r))
        acc, nm, lin, *_ = evaluate(W, Xtr, ytr, Xte, yte, beta)
        row = {"round": r, "beta": beta, "table_acc": acc, "nearest_mean": nm,
               "linear": lin, "ladder_gap": lin - nm}
        res.append(row)
        print(f"  round {r}  beta {beta:<5} table {acc:.4f}   nearest-mean {nm:.4f}   "
              f"linear {lin:.4f}   gap {lin-nm:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(OUT / f"pressure_r{r}.npz", W=W.astype(np.float32))

    print("\n  baseline (no pressure): table 0.9613  nearest-mean 0.9100  "
          "linear 0.9767  gap 0.0667")
    print("  weighting only:                       nearest-mean 0.9283  "
          "linear 0.9783  gap 0.0500")
    (OUT / "pressure.json").write_text(json.dumps(
        {"rounds": res, "reference": {"table": 0.9613, "nearest_mean": 0.9100,
                                      "linear": 0.9767}}, indent=2))


if __name__ == "__main__":
    main()
