"""Generation from a pooled stack: a separate, finer downward memory.

Pooling makes recognition shift-tolerant and makes the downward projection
ambiguous -- "template 12 fires somewhere in this block" does not say where.
So the downward path is not the transpose of the upward one. While training,
each L2 template also accumulates the mean UNPOOLED L1 map of the images it
wins. Recognition compares pooled maps; generation reads the fine memory.

This is what cortical feedback is: its own connections, not the feedforward
weights run backwards.
"""

import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import km, joint, stack, stack_pool

OUT = Path(__file__).resolve().parent / "results"
G, K2, EPOCHS, ETA_MIN, EPS = 4, 100, 12, 0.01, 1e-12


def main():
    W1 = np.load(OUT / "km_mnist.npz")["64"].astype(np.float64)
    k1 = len(W1)
    Xtr, ytr, Xte, yte = joint.load("mnist")
    itr, mtr = stack.l1_map(W1, Xtr)
    ite, mte = stack.l1_map(W1, Xte)

    rng = np.random.default_rng(1)
    seed = rng.choice(len(Xtr), 3000, replace=False)
    W = joint.kmeanspp(stack_pool.dens(itr[seed], mtr[seed], k1, G, ytr[seed]), K2, rng)
    n = np.zeros(K2)
    order = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        rng.shuffle(order)
        for s in range(0, len(order), 512):
            b = order[s:s + 512]
            B = stack_pool.dens(itr[b], mtr[b], k1, G, ytr[b])
            win = (B @ W.T).argmax(1)
            for j in np.unique(win):
                m = B[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j], ETA_MIN) * len(m), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
        print(f"  epoch {ep+1}/{EPOCHS}", flush=True)

    P = G * G * k1
    lab = W[:, P:].argmax(1)
    acc = []
    for s in range(0, len(Xte), 1000):
        B = stack_pool.dens(ite[s:s + 1000], mte[s:s + 1000], k1, G)
        acc.append(lab[(B @ W.T).argmax(1)] == yte[s:s + 1000])
    print(f"\n  classification {float(np.concatenate(acc).mean()):.4f}")

    # ---- the downward memory: mean UNPOOLED map per L2 template -------------
    S = stack.SIDE * stack.SIDE
    down = np.zeros((K2, S, k1), np.float32)
    cnt = np.zeros(K2)
    for s in range(0, len(Xtr), 1000):
        b = slice(s, s + 1000)
        B = stack_pool.dens(itr[b], mtr[b], k1, G, ytr[b])
        win = (B @ W.T).argmax(1)
        ii, mm = itr[b], mtr[b]
        for j in np.unique(win):
            sel = np.nonzero(win == j)[0]
            r, c = np.nonzero(ii[sel] >= 0)
            np.add.at(down[j], (c, ii[sel][r, c]), mm[sel][r, c])
            cnt[j] += len(sel)
    down /= np.maximum(cnt, 1)[:, None, None]

    def paint(M):
        sel, mag = M.argmax(1), M.max(1)
        canvas, ct = np.zeros((28, 28)), np.zeros((28, 28))
        for p in range(S):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, stack.SIDE)
            canvas[r:r + km.PS, c:c + km.PS] += W1[sel[p]].reshape(km.PS, km.PS) * mag[p]
            ct[r:r + km.PS, c:c + km.PS] += 1
        return canvas / np.maximum(ct, 1)

    V = np.zeros((10, P + 10)); V[np.arange(10), P + np.arange(10)] = 1.0
    Sg = joint.cn(V) @ W.T
    fig, axes = plt.subplots(10, 8, figsize=(8.4, 10.6))
    for d in range(10):
        for j, t in enumerate(np.argsort(Sg[d])[::-1][:8]):
            axes[d][j].imshow(paint(down[t]), cmap="gray")
            axes[d][j].set_xticks([]); axes[d][j].set_yticks([])
            axes[d][j].set_title(f"{lab[t]}·{int(n[t])}", fontsize=5, pad=1.2)
        axes[d][0].set_ylabel(f"asked {d}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.suptitle("pooled stack, label in -> image out, via a separate finer "
                 "downward memory", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.95, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / "stack_pool_generate.png", dpi=140); plt.close(fig)

    # the alternative: paint the pooled template directly, smeared over blocks
    fig, axes = plt.subplots(10, 8, figsize=(8.4, 10.6))
    step = stack.SIDE // G
    for d in range(10):
        for j, t in enumerate(np.argsort(Sg[d])[::-1][:8]):
            M = W[t, :P].reshape(G, G, k1)
            full = np.zeros((S, k1))
            for by in range(G):
                for bx in range(G):
                    for py in range(by * step, (by + 1) * step):
                        for px in range(bx * step, (bx + 1) * step):
                            full[py * stack.SIDE + px] = M[by, bx] / step ** 2
            axes[d][j].imshow(paint(full), cmap="gray")
            axes[d][j].set_xticks([]); axes[d][j].set_yticks([])
        axes[d][0].set_ylabel(f"asked {d}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.suptitle("the same, painted from the POOLED template alone — "
                 "the placement is genuinely unknown", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.95, bottom=.005,
                        wspace=.05, hspace=.22)
    fig.savefig(OUT / "stack_pool_generate_naive.png", dpi=140); plt.close(fig)
    np.savez_compressed(OUT / "gen_pool.npz", W=W.astype(np.float32),
                        down=down.astype(np.float32), n=n, lab=lab)
    print("-> stack_pool_generate.png, stack_pool_generate_naive.png")


if __name__ == "__main__":
    main()
