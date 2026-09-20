"""Add the shift-coherence term to layer 1, and see whether it moves the filters off pixels.

    E  =  ||xhat - x||^2                       separation: stops collapse
       +  mu * sum_b (1 - cos(dtheta_b - rho_b[t]))    coherence: makes rho(t) exist

rho is LEARNED alongside the encoder -- it is the transposition vector the pooling step
needs, so the same term both shapes the basis and estimates the intervals.

A coherence-only encoder is degenerate (a constant encoder has coherence 1.0 and holds no
information), so the reconstruction term is load-bearing, not decoration.
"""
import time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer1 as L1
import shift as SH

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
P, B = L1.P, L1.B
rng = np.random.default_rng(0)
SHIFTS = [(0, 1), (1, 0), (0, 2), (2, 0)]


def pair_data(n_img=2500, stride=3):
    """X0 and, for each shift, the same content moved through the window."""
    IM = L1.mnist("train", n_img)
    base, moved = [], [[] for _ in SHIFTS]
    for im in IM:
        for i in range(2, 28 - P - 3, stride):
            for j in range(2, 28 - P - 3, stride):
                p = im[i:i+P, j:j+P]
                if p.sum() <= 1.0:
                    continue
                ok, ms = True, []
                for (dy, dx) in SHIFTS:
                    q = im[i+dy:i+dy+P, j+dx:j+dx+P]
                    if q.sum() <= 1.0: ok = False; break
                    ms.append(q.ravel())
                if ok:
                    base.append(p.ravel())
                    for k, m in enumerate(ms): moved[k].append(m)
    c = lambda a: (lambda z: z - z.mean(1, keepdims=True))(np.asarray(a))
    return c(base), np.stack([c(m) for m in moved])


def enc(X, Wc, Ws):
    A, S = X @ Wc.T, X @ Ws.T
    return A, S, np.arctan2(S, A), A*A + S*S + 1e-3


def train(X0, XS, Wc, Ws, mu, epochs=10, lr=3e-3, bs=256):
    Wc, Ws = Wc.copy(), Ws.copy()
    U, V = rng.normal(size=(B, P*P)) * .05, rng.normal(size=(B, P*P)) * .05
    rho = np.zeros((len(SHIFTS), B))
    opt = L1.Adam([U, V, Wc, Ws, rho], lr)
    for _ in range(epochs):
        for i in range(0, len(X0) - bs, bs):
            X = X0[i:i+bs]; n = len(X)
            A, S, TH, R2 = enc(X, Wc, Ws)
            C, Sn = np.cos(TH), np.sin(TH)
            E = (C @ U + Sn @ V - X) / n
            gU, gV = C.T @ E, Sn.T @ E
            dTH = -Sn * (E @ U.T) + C * (E @ V.T)               # from reconstruction
            gWc = np.zeros_like(Wc); gWs = np.zeros_like(Ws); grho = np.zeros_like(rho)
            if mu > 0:
                for k in range(len(SHIFTS)):
                    Y = XS[k][i:i+bs]
                    Ay, Sy, THy, R2y = enc(Y, Wc, Ws)
                    d = THy - TH - rho[k]
                    g = mu * np.sin(d) / n
                    grho[k] = -g.sum(0)
                    dTH -= g                                     # theta0 pushed the other way
                    gWc += ((g * (-Sy / R2y)).T @ Y); gWs += ((g * (Ay / R2y)).T @ Y)
            gWc += (dTH * (-S / R2)).T @ X
            gWs += (dTH * (A / R2)).T @ X
            opt.step([gU, gV, gWc, gWs, grho])
    return Wc, Ws, U, V, rho


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.3, 1.0, 3.0])
    ap.add_argument("--epochs", type=int, default=10); a = ap.parse_args()

    t0 = time.time()
    X0, XS = pair_data()
    TE = L1.patchify(L1.mnist("t10k", 400))
    A_, B_ = SH.pairs(1)                                        # held-out coherence probe
    print(f"{len(X0)} patch tuples, {len(SHIFTS)} shifts, B={B} voices\n")
    print(f"  {'mu':>5} {'R2':>8} {'top-1 px':>10} {'coh mean':>10} {'worst':>8} {'all-pairs sim':>15}")

    def allpairs(Wc, Ws, n=3000):
        _, _, T, _ = enc(TE[:n], Wc, Ws)
        i, j = rng.integers(0, n, 60000), rng.integers(0, n, 60000)
        m = i != j
        return float(np.cos(T[i[m]] - T[j[m]]).mean())      # 0 = unrelated, 1 = collapsed

    Rb = L1.random_bank(B)
    banks = {}
    for mu in a.mu:
        Wc, Ws, U, V, rho = train(X0, XS, *Rb, mu, a.epochs)
        _, _, TH, _ = enc(TE, Wc, Ws)
        coh = SH.coherence(A_, B_, Wc, Ws)
        print(f"  {mu:>5} {L1.r2(TE, np.cos(TH) @ U + np.sin(TH) @ V):8.3f} "
              f"{L1.energy_top1(Wc, Ws):10.1%} {coh.mean():10.3f} {coh.min():8.3f} "
              f"{allpairs(Wc, Ws):15.3f}")
        banks[f"mu = {mu}"] = (Wc, Ws)
        np.save(ROOT / "results" / "W_coherent.npy", np.stack([Wc, Ws])); np.save(ROOT / "results" / "rho_coherent.npy", rho)

    gc, gs = SH.gabor_bank(); cg = SH.coherence(A_, B_, gc, gs)
    cr = SH.coherence(A_, B_, *Rb)
    print(f"\n  {'Gabor':>5} {'--':>8} {L1.energy_top1(gc, gs):10.1%} {cg.mean():10.3f} "
          f"{cg.min():8.3f} {allpairs(gc, gs):15.3f}")
    print(f"  {'random':>5} {'--':>8} {L1.energy_top1(*Rb):10.1%} {cr.mean():10.3f} "
          f"{cr.min():8.3f} {allpairs(*Rb):15.3f}")

    banks["Gabor (reference)"] = (gc, gs)
    L1.fig_filters(banks, OUT / "coherent_filters.png")
    print(f"\n  ({time.time()-t0:.0f}s)  figure -> {OUT}/coherent_filters.png")
