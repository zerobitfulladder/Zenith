"""Up and back down with ZERO learned decoder.

    up    patch -> L1 chord -> [quantise to a vocabulary part] -> transpose by offset -> SUM
    down  polychord -> for each KNOWN offset: un-transpose, score the vocabulary, take the
          winner -> that part's stored mean patch -> place it back

Nothing in the downward direction is trained.  It is un-transposition, consensus across the
64 voices, and a table lookup of the part's stored appearance.

Three numbers:
    vocabulary ceiling - quantise each patch to its nearest part and put that part back.
                         No pooling at all.  This is the best any of this can do.
    round trip         - the same, but the parts go through pool-and-interrogate first.
    gap                - what the pooling round trip costs.
"""
from pathlib import Path
import numpy as np
import layer1 as L1M, recur as RC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
P, B, L = 7, 64, 41
NG = 11                                   # 11x11 cells of 7x7 patches at stride 2
POS = list(range(0, 28 - P + 1, 2))
OFF = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
rng = np.random.default_rng(0)

W = np.load(ROOT / "results" / "W_coherent.npy"); Wc, Ws = W[0], W[1]
rho = np.load(ROOT / "results" / "rho_coherent.npy")
RHO = np.stack([dy*rho[3] + dx*rho[2] for dy, dx in OFF])      # composed cell intervals


def bump(th, w=0.7):
    slot = np.arange(L) * (2*np.pi/L)
    d = th[..., None] - slot
    d = np.arctan2(np.sin(d), np.cos(d)) * (L/(2*np.pi))
    e = np.exp(-d*d/(2*w*w))
    return e / e.sum(-1, keepdims=True)


def build_vocabulary(K=128, n_img=1000):
    """K parts.  Each stores a chord AND the mean patch it stands for -- the lookup table."""
    Xp, _, _ = RC.labelled_patches(n_img=n_img)
    Z = RC.chord_vecs(Xp, Wc, Ws)
    a, C = RC.kmeans(Z, K)
    chords = np.arctan2(C[:, B:], C[:, :B])
    patches = np.stack([Xp[a == k].mean(0) if (a == k).sum() else np.zeros(P*P)
                        for k in range(K)])
    return chords, patches


def grids(img):
    g = np.stack([img[i:i+P, j:j+P].ravel() for i in POS for j in POS])
    return g - g.mean(1, keepdims=True)


def quantise(patches_flat, VOC):
    """patch -> L1 chord -> nearest vocabulary part (a cleanup, not a learned map)."""
    th = np.arctan2(patches_flat @ Ws.T, patches_flat @ Wc.T)
    return np.argmax(np.cos(th[:, None, :] - VOC[None]).sum(-1), 1)


def assemble(cell_patches, cells):
    img = np.zeros((28, 28)); cnt = np.zeros((28, 28))
    for (ci, cj), pa in zip(cells, cell_patches):
        i, j = POS[ci], POS[cj]
        img[i:i+P, j:j+P] += pa.reshape(P, P); cnt[i:i+P, j:j+P] += 1
    return img / np.maximum(cnt, 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--test", type=int, default=300); a = ap.parse_args()

    VOC, VPATCH = build_vocabulary(a.K)
    X, _ = L1M.mnist_labelled("t10k", a.test)
    print(f"{a.K}-part vocabulary, B={B} voices, {len(X)} test digits")
    print("no learned weights anywhere in the downward direction\n")

    # windows tile the 9x9 sub-grid of cells with stride 3 (cells 9,10 are the blank margin)
    WINS = [(i, j) for i in (1, 4, 7) for j in (1, 4, 7)]
    num_c = num_r = num_q = den = 0.0
    hit = tot = 0
    shots = []
    for n, im in enumerate(X):
        g = grids(im)
        ids = quantise(g, VOC)                                   # true part per cell
        cells, ceil_pa, rt_pa = [], [], []
        for (wi, wj) in WINS:
            idx = [( wi+dy)*NG + (wj+dx) for dy, dx in OFF]
            live = [k for k, c in enumerate(idx) if np.abs(g[c]).sum() > 1.0]
            if not live:
                continue
            Z = bump(VOC[ids[[idx[k] for k in live]]] + RHO[live]).sum(0)   # POOL
            for k in live:                                       # INTERROGATE each offset
                sc = np.einsum("bl,vbl->v", Z, bump(VOC + RHO[k]))
                got = int(sc.argmax())
                hit += int(got == ids[idx[k]]); tot += 1
                dy, dx = OFF[k]
                cells.append((wi+dy, wj+dx))
                ceil_pa.append(VPATCH[ids[idx[k]]])              # ceiling: true part back
                rt_pa.append(VPATCH[got])                        # round trip: what came back
        tgt = assemble([g[ci*NG+cj] for ci, cj in cells], cells)
        rec_c, rec_r = assemble(ceil_pa, cells), assemble(rt_pa, cells)
        num_c += ((rec_c - tgt)**2).sum(); num_r += ((rec_r - tgt)**2).sum()
        den += (tgt**2).sum()
        if n < 10: shots.append((im, tgt, rec_c, rec_r))

    print(f"  parts recovered by interrogation        {hit/tot:.3f}  ({tot} queries)")
    print(f"  vocabulary ceiling   R2                 {1 - num_c/den:.3f}")
    print(f"  full round trip      R2                 {1 - num_r/den:.3f}")
    print(f"  cost of the pool + interrogate step     {(num_r - num_c)/den:.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 10, figsize=(10*.85, 4*.95))
    rows = ["digit", "patches\n(target)", "vocab\nceiling", "round trip\nno decoder"]
    for j, (im, tgt, rc, rr) in enumerate(shots):
        for r, pic in enumerate((im, tgt, rc, rr)):
            ax[r, j].imshow(pic, cmap="gray"); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if j == 0: ax[r, j].set_ylabel(rows[r], fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Up and back down with no learned decoder: pool, interrogate, look up", fontsize=10)
    fig.tight_layout(rect=[.06, 0, 1, .92]); fig.savefig(OUT / "nodecoder.png", dpi=150)
    print(f"\n  figure -> {OUT}/nodecoder.png")
