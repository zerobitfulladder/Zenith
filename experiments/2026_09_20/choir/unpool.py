"""Can the parts be read back out of a pooled polychord?

Pooling is  Z = sum_c cluster(theta_c + rho_c).  It expands rather than compresses, but it
is not invertible in closed form: per voice you get a multiset of peaks and nothing says
which peak came from which cell.  Recovery has to go through cleanup -- un-transpose by a
known offset and match against the vocabulary.

Measured here: with n cells contributing, how often does cell c's true part come back?
Swept against the number of contributors, and against how the parts are rendered
(clusters keep peaks apart; arrows average them into one).
"""
from pathlib import Path
import numpy as np
import layer1 as L1M, recur as RC

ROOT = Path(__file__).resolve().parent
P, B, L = 7, 64, 41
rng = np.random.default_rng(0)
W = np.load(ROOT / "results" / "W_coherent.npy"); Wc, Ws = W[0], W[1]
rho_px = np.load(ROOT / "results" / "rho_coherent.npy")


def bump(th, w=0.7):
    slot = np.arange(L) * (2*np.pi/L)
    d = th[..., None] - slot
    d = np.arctan2(np.sin(d), np.cos(d)) * (L/(2*np.pi))
    e = np.exp(-d*d/(2*w*w))
    return e / e.sum(-1, keepdims=True)


def vocabulary(K=128):
    Xp, cls, _ = RC.labelled_patches(n_img=800)
    Z = RC.chord_vecs(Xp, Wc, Ws)
    a, C = RC.kmeans(Z, K)
    th = np.arctan2(C[:, B:], C[:, :B])            # K vocabulary chords
    return th, Xp, np.arctan2(Xp @ Ws.T, Xp @ Wc.T)


if __name__ == "__main__":
    VOC, Xp, TH = vocabulary()
    K = len(VOC)
    # composed cell intervals: rho(dy,dx) = dy*rho(2,0) + dx*rho(0,2)
    OFF = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    RHO = np.stack([dy*rho_px[3] + dx*rho_px[2] for dy, dx in OFF])

    print(f"vocabulary of {K} chords, B={B} voices, gamut {L}\n")
    print(f"  {'contributors':>13}{'clusters':>11}{'arrows':>9}{'chance':>9}")

    TRIALS = 400
    for n in (1, 2, 3, 4, 6, 9):
        hit_c = hit_a = 0
        for _ in range(TRIALS):
            ids = rng.integers(0, K, n)                       # n parts at n distinct offsets
            slots = rng.permutation(9)[:n]
            th = VOC[ids] + RHO[slots]                        # transposed chords (n, B)
            Zc = bump(th).sum(0)                              # cluster pooling  (B, L)
            Za = np.exp(1j*th).sum(0)                         # arrow pooling    (B,)
            q = rng.integers(0, n)                            # query one cell
            back = VOC + RHO[slots[q]]                        # every candidate, transposed
            sc_c = np.einsum("bl,kbl->k", Zc, bump(back))
            sc_a = (np.cos(np.angle(Za)[None] - back)).sum(1)
            hit_c += int(sc_c.argmax() == ids[q]); hit_a += int(sc_a.argmax() == ids[q])
        print(f"  {n:>13}{hit_c/TRIALS:>11.3f}{hit_a/TRIALS:>9.3f}{1/K:>9.3f}")

    print("\n  Same thing on REAL neighbouring cells rather than random vocabulary draws:")
    NG = 11
    pos = list(range(0, 28 - P + 1, 2))
    X, _ = L1M.mnist_labelled("t10k", 400)
    hit = tot = 0
    for im in X:
        g = np.array([[im[i:i+P, j:j+P].ravel() for j in pos] for i in pos])
        g = g - g.mean(-1, keepdims=True)
        for i in range(1, NG-1, 3):
            for j in range(1, NG-1, 3):
                cells = [(dy, dx) for dy, dx in OFF if g[i+dy, j+dx].std() > 0.02]
                if len(cells) < 2:
                    continue
                th = np.arctan2(g[[i+d[0] for d in cells], [j+d[1] for d in cells]] @ Ws.T,
                                g[[i+d[0] for d in cells], [j+d[1] for d in cells]] @ Wc.T)
                true = np.argmax(np.cos(th[:, None, :] - VOC[None]).sum(-1), 1)   # nearest vocab
                k = [OFF.index(c) for c in cells]
                Z = bump(VOC[true] + RHO[k]).sum(0)
                for q in range(len(cells)):
                    sc = np.einsum("bl,kbl->k", Z, bump(VOC + RHO[k[q]]))
                    hit += int(sc.argmax() == true[q]); tot += 1
    print(f"    {tot} queries over real 3x3 windows: {hit/tot:.3f} recovered")
