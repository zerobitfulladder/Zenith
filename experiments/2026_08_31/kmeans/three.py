"""Three rungs, all plain k-means, identity only, with downward memories.

    L1  5x5 patches, stride 1, K1=64            -> 24x24 identities
        pool 2                                  -> 12x12 x 64
    L2  3x3 window over that, shared, K2=128    -> 10x10 identities  (~13px field)
        pool 2                                  ->  5x5 x 128
    L3  global over that, joined with the label, K3=100

Recognition goes up through pooled maps. Generation comes back down through
separate memories: each L3 template remembers the mean UNPOOLED L2 map it won,
and each L2 template remembers the mean fine L1 map over its own receptive
field. Feedback is not the forward pass transposed.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km, joint, stack

OUT = Path(__file__).resolve().parent / "results"
K1KEY, K2, K3 = "64", 128, 100
S1, P1, W2, S2, P2 = 24, 12, 3, 10, 5          # sizes of each stage
EPOCHS, ETA_MIN, EPS, SEED = 10, 0.01, 1e-12, 0
RF = 6                                          # fine L1 positions an L2 unit sees


def pool_map(idx, mag, k1, g=P1):
    """(n, g, g, k1) max-pooled one-hots from the fine L1 map."""
    n = len(idx)
    M = np.zeros((n, S1 * S1, k1), np.float32)
    r, c = np.nonzero(idx >= 0)
    M[r, c, idx[r, c]] = mag[r, c]
    s = S1 // g
    return M.reshape(n, g, s, g, s, k1).max(axis=(2, 4))


def windows(M):
    """(n, 10*10, 3*3*k1) -- every 3x3 window of the pooled L1 map."""
    n, g, _, k = M.shape
    o = S2
    out = np.empty((n, o * o, W2 * W2 * k), np.float32)
    for i in range(o):
        for j in range(o):
            out[:, i * o + j] = M[:, i:i + W2, j:j + W2].reshape(n, -1)
    return out


def unit(V):
    nv = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(nv, EPS), nv[..., 0]


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


def l2_code(W2t, M):
    """Winner index and magnitude at each of the 10x10 L2 positions."""
    Q, nv = unit(windows(M))
    S = np.einsum("npd,kd->npk", Q, W2t)
    return S.argmax(-1).astype(np.int16), nv


def l3_input(i2, m2, y=None):
    """pool the L2 map to 5x5, flatten, append the label at matched energy."""
    n = len(i2)
    M = np.zeros((n, S2 * S2, K2), np.float32)
    r, c = np.nonzero(m2 > 0)
    M[r, c, i2[r, c]] = m2[r, c]
    M = M.reshape(n, S2, S2, K2)[:, :P2 * 2, :P2 * 2]
    M = M.reshape(n, P2, 2, P2, 2, K2).max(axis=(2, 4)).reshape(n, -1)
    V = np.zeros((n, M.shape[1] + 10))
    V[:, :M.shape[1]] = M
    if y is not None:
        e = np.linalg.norm(M, axis=1)
        L = np.zeros((n, 10)); L[np.arange(n), y] = 1.0
        V[:, M.shape[1]:] = L * e[:, None]
    return joint.cn(V)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED + 1)
    W1 = np.load(OUT / "km_mnist.npz")[K1KEY].astype(np.float64)
    k1 = len(W1)
    Xtr, ytr, Xte, yte = joint.load("mnist")
    itr, mtr = stack.l1_map(W1, Xtr)
    ite, mte = stack.l1_map(W1, Xte)
    print(f"L1 K={k1}: map {S1}x{S1}, pooled to {P1}x{P1}", flush=True)

    # ---- L2 on sampled windows ---------------------------------------------
    sub = rng.choice(len(Xtr), 6000, replace=False)
    Mp = pool_map(itr[sub], mtr[sub], k1)
    Qw, nv = unit(windows(Mp))
    Qw = Qw[nv > 0]
    Qw = Qw[rng.choice(len(Qw), min(400000, len(Qw)), False)]
    W2t, n2 = kmeans(Qw.astype(np.float64), K2, rng)
    print(f"L2 K={K2} trained on {len(Qw)} windows of {W2*W2*k1} "
          f"({int((n2==0).sum())} dead)", flush=True)

    def encode(idx, mag, chunk=2000):
        I, M = [], []
        for a in range(0, len(idx), chunk):
            Mp = pool_map(idx[a:a + chunk], mag[a:a + chunk], k1)
            i, m = l2_code(W2t, Mp)
            I.append(i); M.append(m)
        return np.concatenate(I), np.concatenate(M)

    i2tr, m2tr = encode(itr, mtr)
    i2te, m2te = encode(ite, mte)
    print(f"L2 map {S2}x{S2}, pooled to {P2}x{P2} -> L3 input "
          f"{P2*P2*K2 + 10}", flush=True)

    # ---- L3 ------------------------------------------------------------------
    A = np.concatenate([l3_input(i2tr[s:s + 2000], m2tr[s:s + 2000], ytr[s:s + 2000])
                        for s in range(0, len(i2tr), 2000)])
    W3, n3 = kmeans(A, K3, rng)
    P = P2 * P2 * K2
    lab = W3[:, P:].argmax(1)
    B = np.concatenate([l3_input(i2te[s:s + 2000], m2te[s:s + 2000])
                        for s in range(0, len(i2te), 2000)])
    win3 = (B @ W3.T).argmax(1)
    acc = float((lab[win3] == yte).mean())
    tw = (A @ W3.T).argmax(1)
    pur = np.mean([np.bincount(ytr[tw == j], minlength=10).max() / max((tw == j).sum(), 1)
                   for j in range(K3) if (tw == j).any()])
    print(f"\n  classification (image in, label out) {acc:.4f}")
    print(f"  L3 dead {int((n3==0).sum())}/{K3}  purity {pur:.3f}  "
          f"params L2 {K2*W2*W2*k1:,} + L3 {K3*(P+10):,}", flush=True)

    # ---- downward memories ---------------------------------------------------
    down3 = np.zeros((K3, S2 * S2, K2), np.float32); c3 = np.zeros(K3)
    for j in np.unique(tw):
        sel = np.nonzero(tw == j)[0]
        r, c = np.nonzero(m2tr[sel] > 0)
        np.add.at(down3[j], (c, i2tr[sel][r, c]), m2tr[sel][r, c])
        c3[j] += len(sel)
    down3 /= np.maximum(c3, 1)[:, None, None]

    down2 = np.zeros((K2, RF * RF, k1), np.float32); c2 = np.zeros(K2)
    for a in range(0, 8000, 1000):                      # a sample is enough
        sl = slice(a, a + 1000)
        for p in range(S2 * S2):
            pr, pc = divmod(p, S2)
            fr, fc = pr * 2, pc * 2
            wj = i2tr[sl][:, p]
            good = m2tr[sl][:, p] > 0
            block_i = itr[sl][:, [(fr + dy) * S1 + (fc + dx)
                                  for dy in range(RF) for dx in range(RF)]]
            block_m = mtr[sl][:, [(fr + dy) * S1 + (fc + dx)
                                  for dy in range(RF) for dx in range(RF)]]
            for j in np.unique(wj[good]):
                s2 = good & (wj == j)
                bi, bm = block_i[s2], block_m[s2]
                r, c = np.nonzero(bi >= 0)
                np.add.at(down2[j], (c, bi[r, c]), bm[r, c])
                c2[j] += s2.sum()
    down2 /= np.maximum(c2, 1)[:, None, None]

    def paint(t3):
        M2 = down3[t3]                                   # (100, K2)
        canvas = np.zeros((S1 * S1, k1))
        for p in range(S2 * S2):
            j, w = M2[p].argmax(), M2[p].max()
            if w <= 0:
                continue
            pr, pc = divmod(p, S2); fr, fc = pr * 2, pc * 2
            blk = down2[j].reshape(RF, RF, k1)
            for dy in range(RF):
                for dx in range(RF):
                    if fr + dy < S1 and fc + dx < S1:
                        canvas[(fr + dy) * S1 + fc + dx] += w * blk[dy, dx]
        sel, mag = canvas.argmax(1), canvas.max(1)
        img, ct = np.zeros((28, 28)), np.zeros((28, 28))
        for p in range(S1 * S1):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, S1)
            img[r:r + km.PS, c:c + km.PS] += W1[sel[p]].reshape(km.PS, km.PS) * mag[p]
            ct[r:r + km.PS, c:c + km.PS] += 1
        return img / np.maximum(ct, 1)

    V = np.zeros((10, P + 10)); V[np.arange(10), P + np.arange(10)] = 1.0
    Sg = joint.cn(V) @ W3.T
    fig, axes = plt.subplots(10, 8, figsize=(8.4, 10.6))
    for d in range(10):
        for jj, t in enumerate(np.argsort(Sg[d])[::-1][:8]):
            axes[d][jj].imshow(paint(t), cmap="gray")
            axes[d][jj].set_xticks([]); axes[d][jj].set_yticks([])
            axes[d][jj].set_title(f"{lab[t]}·{int(n3[t])}", fontsize=5, pad=1.2)
        axes[d][0].set_ylabel(f"asked {d}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.suptitle("three rungs: label -> L3 -> L2 map -> L1 map -> pixels",
                 fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.95, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / "three_generate.png", dpi=140); plt.close(fig)

    (OUT / "three.json").write_text(json.dumps(
        {"K1": k1, "K2": K2, "K3": K3, "acc": acc, "purity": float(pur),
         "dead_l2": int((n2 == 0).sum()), "dead_l3": int((n3 == 0).sum()),
         "params": int(K2 * W2 * W2 * k1 + K3 * (P + 10)),
         "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / "three.npz", W2=W2t.astype(np.float32),
                        W3=W3.astype(np.float32), down2=down2, down3=down3,
                        n2=n2, n3=n3, lab=lab)
    print(f"\ndone in {time.time()-t0:.0f}s -> three_generate.png")


if __name__ == "__main__":
    main()
