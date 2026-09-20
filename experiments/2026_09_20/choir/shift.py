"""Are the learned filters usable as a substrate for transposition?

The architecture applies a cell offset to a chord as a transposition: every voice advances
by its own fixed interval.  That is only sound if a SHIFTED patch yields a COHERENTLY
ROTATED chord -- i.e. voice b's phase change under a 1px shift is the same number for every
patch.  If instead each patch gets an unrelated new phase, the same stroke at two positions
has two unrelated names, and the vocabulary shatters.

Metric: for a fixed shift, collect d(theta_b) over many patches and take the circular
resultant length  |mean exp(i*d)|.   1.0 = a fixed interval (transposition works).
                                     0.0 = noise (transposition is meaningless).
"""
import gzip, struct
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "mnist"
OUT = ROOT / "results"
P, B = 7, 64
rng = np.random.default_rng(0)

with gzip.open(DATA / "train-images-idx3-ubyte.gz") as f:
    _, N, r, c = struct.unpack(">IIII", f.read(16))
    IM = np.frombuffer(f.read(), np.uint8).reshape(N, r, c).astype(np.float32) / 255.
IM = IM[rng.permutation(N)[:4000]]


def pairs(dx):
    """patch, and the same content shifted by dx pixels through the same window."""
    a, b = [], []
    for im in IM:
        for i in range(4, 28 - P - 4, 3):
            for j in range(4, 28 - P - 4 - dx, 3):
                p0, p1 = im[i:i+P, j:j+P], im[i:i+P, j+dx:j+dx+P]
                if p0.sum() > 1.0 and p1.sum() > 1.0:
                    a.append(p0.ravel()); b.append(p1.ravel())
    A, Bp = np.array(a), np.array(b)
    return A - A.mean(1, keepdims=True), Bp - Bp.mean(1, keepdims=True)


def gabor_bank():
    """Sane parameters for a 7x7 window: low frequencies, envelope inside the patch."""
    yy, xx = np.mgrid[0:P, 0:P] - P // 2
    Wc, Ws = [], []
    for k in range(B):
        fr = [0.6, 1.0, 1.5, 2.0][(k // 16) % 4]
        ori = np.pi * (k % 8) / 8.
        sig = [1.4, 2.0][(k // 8) % 2]
        env = np.exp(-(xx**2 + yy**2) / (2 * sig**2))
        rot = xx * np.cos(ori) + yy * np.sin(ori)
        Wc.append((env * np.cos(2*np.pi*fr*rot/P)).ravel())
        Ws.append((env * np.sin(2*np.pi*fr*rot/P)).ravel())
    return np.array(Wc), np.array(Ws)


def theta(X, Wc, Ws):
    return np.arctan2(X @ Ws.T, X @ Wc.T)


def coherence(A, Bp, Wc, Ws):
    d = theta(Bp, Wc, Ws) - theta(A, Wc, Ws)
    return np.abs(np.exp(1j * d).mean(0))                    # per voice, 0..1


if __name__ == "__main__":
    W = np.load(ROOT / "results" / "W_learned.npy") if (ROOT / "results" / "W_learned.npy").exists() else None
    banks = {"random": (rng.normal(size=(B, P*P))/P, rng.normal(size=(B, P*P))/P),
             "Gabor (fixed params)": gabor_bank()}
    if W is not None:
        banks["learned (pixel pairs)"] = (W[0], W[1])

    shifts = [1, 2, 3]
    res = {k: [] for k in banks}
    sim = {k: [] for k in banks}
    for dx in shifts:
        A, Bp = pairs(dx)
        for name, (Wc, Ws) in banks.items():
            res[name].append(coherence(A, Bp, Wc, Ws).mean())
            sim[name].append(np.cos(theta(Bp, Wc, Ws) - theta(A, Wc, Ws)).mean())
        print(f"shift {dx}px  ({len(A)} patch pairs)")
        for name in banks:
            print(f"    {name:<24} coherence {res[name][-1]:.3f}   chord similarity {sim[name][-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name in banks:
        ax[0].plot(shifts, res[name], marker="o", label=name)
        ax[1].plot(shifts, sim[name], marker="o", label=name)
    for a_, t, yl in ((ax[0], "Is the phase change a FIXED interval?", "coherence  |mean e^{i·Δθ}|"),
                      (ax[1], "Does the chord stay recognisable?", "chord similarity")):
        a_.set_title(t, fontsize=10); a_.set_xlabel("shift (pixels)"); a_.set_ylabel(yl, fontsize=9)
        a_.set_ylim(0, 1.02); a_.set_xticks(shifts); a_.spines[["top","right"]].set_visible(False)
        a_.axhline(0, color="0.85", lw=.8)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Transposition needs a coherent phase shift, not just similarity", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT / "shift_coherence.png", dpi=140)
    print(f"\nfigure -> {OUT}/shift_coherence.png")
