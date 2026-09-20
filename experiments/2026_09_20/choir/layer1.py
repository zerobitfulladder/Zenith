"""Layer 1 of the choir: a patch becomes a chord.

    patch (49 numbers)  --linear-->  2B numbers  --atan2-->  B angles  --linear-->  patch

atan2 is the only nonlinearity.  Each voice is a quadrature pair of projections; the angle
is its note, the magnitude is how hard it fired.  Question: trained only to reconstruct,
what does a voice learn to measure?

Arms:
    random / Gabor, frozen     - what an untrained encoder scores
    random / Gabor, learned    - the experiment
    sparse                     - magnitude decoder with a group-L1 penalty on r
                                 (reconstruction alone cannot prefer one basis over
                                  another; sparsity is the classical tie-breaker)
"""
import gzip, struct, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "mnist"
OUT  = ROOT / "results"
P, B, L = 7, 64, 41                       # patch side, voices, notes per gamut
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def mnist(split, n):
    with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz") as f:
        _, N, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(N, r, c).astype(np.float32) / 255.
    return X[rng.permutation(N)[:n]]


def mnist_labelled(split, n):
    with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz") as f:
        _, N, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(N, r, c).astype(np.float32) / 255.
    with gzip.open(DATA / f"{split}-labels-idx1-ubyte.gz") as f:
        _, _n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), np.uint8).astype(int)
    sel = rng.permutation(N)[:n]
    return X[sel], y[sel]


def patchify(imgs):
    pos = range(0, 28 - P + 1, 2)
    pa = np.stack([im[i:i+P, j:j+P].ravel() for im in imgs for i in pos for j in pos])
    pa = pa[pa.sum(1) > 1.0]                        # inked patches only
    return pa - pa.mean(1, keepdims=True)           # mean-centre: no DC voice


# ------------------------------------------------------------------ encoders
def gabor_bank(B):
    yy, xx = np.mgrid[0:P, 0:P] - P // 2
    env = np.exp(-(xx**2 + yy**2) / (2 * 2.0**2))
    Wc, Ws = [], []
    for k in range(B):
        fr, ori = 0.5 + 3.0 * ((k // 8) / 7.), np.pi * (k % 8) / 8.
        rot = xx * np.cos(ori) + yy * np.sin(ori)
        Wc.append((env * np.cos(2 * np.pi * fr * rot / P)).ravel())
        Ws.append((env * np.sin(2 * np.pi * fr * rot / P)).ravel())
    return np.array(Wc), np.array(Ws)


def random_bank(B):
    return rng.normal(size=(B, P * P)) / P, rng.normal(size=(B, P * P)) / P


# ------------------------------------------------------------------ model
def chord(X, Wc, Ws, quant=0):
    """patch -> B angles.  quant=1 rounds each note onto the 41-slot gamut."""
    A, S = X @ Wc.T, X @ Ws.T
    TH = np.arctan2(S, A)
    if quant:
        TH = np.round(TH * L / (2 * np.pi)) * 2 * np.pi / L
    return A, S, TH


def decode_phase(TH, U, V):
    return np.cos(TH) @ U + np.sin(TH) @ V


def r2(X, Xh):
    return 1 - ((X - Xh) ** 2).sum() / (X ** 2).sum()


class Adam:
    def __init__(s, ps, lr): s.ps, s.lr, s.m, s.v, s.t = ps, lr, [np.zeros_like(p) for p in ps], [np.zeros_like(p) for p in ps], 0
    def step(s, gs):
        s.t += 1
        for k, (p, g) in enumerate(zip(s.ps, gs)):
            s.m[k] = .9 * s.m[k] + .1 * g
            s.v[k] = .999 * s.v[k] + .001 * g * g
            p -= s.lr * (s.m[k] / (1 - .9**s.t)) / (np.sqrt(s.v[k] / (1 - .999**s.t)) + 1e-8)


def train(TR, TE, Wc, Ws, learn_enc, epochs, lr=3e-3, bs=256):
    """Phase-only decoder: the chord is the whole code, magnitudes discarded."""
    Wc, Ws = Wc.copy(), Ws.copy()
    U, V = rng.normal(size=(B, P*P)) * .05, rng.normal(size=(B, P*P)) * .05
    ps = [U, V] + ([Wc, Ws] if learn_enc else [])
    opt = Adam(ps, lr)
    for _ in range(epochs):
        for i in range(0, len(TR) - bs, bs):
            X = TR[i:i+bs]; n = len(X)
            A, S, TH = chord(X, Wc, Ws)
            C, Sn = np.cos(TH), np.sin(TH)
            E = (C @ U + Sn @ V - X) / n
            gs = [C.T @ E, Sn.T @ E]
            if learn_enc:
                G = -Sn * (E @ U.T) + C * (E @ V.T)          # d/d(note)
                R2 = A*A + S*S + 1e-3                        # the 1/r^2 blow-up, damped
                gs += [(G * (-S / R2)).T @ X, (G * (A / R2)).T @ X]
            opt.step(gs)
    _, _, TH = chord(TE, Wc, Ws)
    _, _, THq = chord(TE, Wc, Ws, quant=1)
    return Wc, Ws, U, V, r2(TE, decode_phase(TH, U, V)), r2(TE, decode_phase(THq, U, V))


def train_sparse(TR, TE, Wc, Ws, epochs, lam=0.04, lr=3e-3, bs=256):
    """Magnitude decoder + group-L1 on r.  Note r*cos(th)=a exactly, so the decoder is
    linear and this is complex sparse coding: penalise sum_b |r_b|, unit-norm the atoms."""
    Wc, Ws = Wc.copy(), Ws.copy()
    U, V = rng.normal(size=(B, P*P)) * .05, rng.normal(size=(B, P*P)) * .05
    opt = Adam([U, V, Wc, Ws], lr)
    for _ in range(epochs):
        for i in range(0, len(TR) - bs, bs):
            X = TR[i:i+bs]; n = len(X)
            A, S = X @ Wc.T, X @ Ws.T
            R = np.sqrt(A*A + S*S + 1e-6)
            E = (A @ U + S @ V - X) / n
            dA = E @ U.T + lam * A / R / n
            dS = E @ V.T + lam * S / R / n
            opt.step([A.T @ E, S.T @ E, dA.T @ X, dS.T @ X])
            nrm = np.sqrt((U**2).sum(1) + (V**2).sum(1))[:, None] + 1e-9   # unit-norm atoms
            U /= nrm; V /= nrm
    A, S = TE @ Wc.T, TE @ Ws.T
    R = np.sqrt(A*A + S*S)
    return Wc, Ws, r2(TE, A @ U + S @ V), (R > 0.1 * R.max(1, keepdims=True)).mean()


# ------------------------------------------------------------------ figures
def energy_top1(Wc, Ws):
    W = np.vstack([Wc, Ws]) ** 2
    return float(np.mean(W.max(1) / W.sum(1)))


def fig_filters(banks, path):
    ncol = 8
    fig, axes = plt.subplots(len(banks) * 2, ncol, figsize=(ncol * 1.05, len(banks) * 2.3))
    for bi, (name, (Wc, Ws)) in enumerate(banks.items()):
        order = np.argsort(-((Wc**2 + Ws**2).sum(1)))[:ncol]
        for row, (Wx, tag) in enumerate(((Wc, "cos"), (Ws, "sin"))):
            for j, k in enumerate(order):
                ax = axes[bi * 2 + row, j]
                v = Wx[k].reshape(P, P); m = np.abs(v).max() + 1e-9
                ax.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f"{name}\nw_{tag}" if row == 0 else f"w_{tag}",
                                  fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("What each voice learned to measure  (one quadrature pair per column)", fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96]); fig.savefig(path, dpi=140); plt.close(fig)


def fig_concentration(banks, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ks = np.arange(1, 13)
    for name, (Wc, Ws) in banks.items():
        W = np.sort(np.vstack([Wc, Ws]) ** 2, 1)[:, ::-1]
        frac = (W[:, :12].cumsum(1) / W.sum(1, keepdims=True)).mean(0)
        ax.plot(ks, frac, marker="o", ms=3.5, label=name)
    ax.axhline(1.0, color="0.8", lw=.8)
    ax.set_xlabel("pixels, strongest first"); ax.set_ylabel("fraction of filter energy")
    ax.set_title("A filter that is one pixel is not a part", fontsize=10)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, frameon=False); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_recon(TE, Wc, Ws, U, V, path, n=10):
    _, _, TH = chord(TE[:n], Wc, Ws)
    Xh = decode_phase(TH, U, V)
    fig, axes = plt.subplots(2, n, figsize=(n * .95, 2.3))
    for j in range(n):
        for row, img in ((0, TE[j]), (1, Xh[j])):
            m = np.abs(TE[j]).max() + 1e-9
            axes[row, j].imshow(img.reshape(P, P), cmap="gray", vmin=-m, vmax=m)
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
    axes[0, 0].set_ylabel("patch", fontsize=8, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("from\nchord", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Reconstruction from B angles alone", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 0.93]); fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lam", type=float, default=0.04)
    a = ap.parse_args()

    TR, TE = patchify(mnist("train", a.digits)), patchify(mnist("t10k", 400))
    print(f"{len(TR)} train / {len(TE)} test mean-centred {P}x{P} patches, B={B} voices, "
          f"gamut {L}\n")

    Xc = TR - TR.mean(0); ev, Q = np.linalg.eigh(Xc.T @ Xc / len(Xc)); Q = Q[:, ::-1]
    for k in (16, 32, 48):
        Z = (TE - TR.mean(0)) @ Q[:, :k]
        print(f"  {'PCA, ' + str(k) + ' components':<32} R2 = {r2(TE, Z @ Q[:, :k].T + TR.mean(0)):6.3f}")
    print(f"  {'(the patch has 49 dimensions)':<32}\n")

    Rb, Gb = random_bank(B), gabor_bank(B)
    banks, t0 = {}, time.time()
    for name, bank, learn in (("random, frozen", Rb, False), ("Gabor, frozen", Gb, False),
                              ("random, learned", Rb, True), ("Gabor, learned", Gb, True)):
        Wc, Ws, U, V, s, sq = train(TR, TE, *bank, learn, a.epochs)
        if name == "random, learned": UV = (Wc, Ws, U, V)
        print(f"  {name:<32} R2 = {s:6.3f}   quantised {sq:6.3f}   "
              f"top-1 pixel {energy_top1(Wc, Ws):5.1%}")
        banks[name] = (Wc, Ws)

    Wc, Ws, s, act = train_sparse(TR, TE, *Rb, a.epochs, a.lam)
    print(f"  {'sparse (L1 on r), learned':<32} R2 = {s:6.3f}   active voices {act:5.1%}   "
          f"top-1 pixel {energy_top1(Wc, Ws):5.1%}")
    banks["sparse, learned"] = (Wc, Ws)
    print(f"\n  ({time.time() - t0:.0f}s)")

    show = {k: banks[k] for k in ("random, frozen", "Gabor, frozen", "random, learned", "sparse, learned")}
    fig_filters(show, OUT / "layer1_filters.png")
    fig_concentration(show, OUT / "layer1_concentration.png")
    fig_recon(TE, *UV, OUT / "layer1_reconstruction.png")
    print(f"  figures -> {OUT}/")
