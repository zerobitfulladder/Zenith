"""Could we skip learning layer 1 and just take a fixed Fourier transform?

The coherence-trained filters look like full-patch oriented gratings, so the obvious question
is whether a fixed DFT basis does the same job for free.  In theory it should be BETTER:
the shift theorem is exact for plane waves, and rho(t) is known in closed form, no learning.

The catch this tests: our shift is not a CIRCULAR shift.  Content moves through the window,
new pixels enter one edge and old ones leave.  The DFT shift theorem does not cover that.
"""
from pathlib import Path
import numpy as np
import layer1 as L1
import shift as SH

ROOT = Path(__file__).resolve().parent
P = L1.P
rng = np.random.default_rng(0)


def dft_bank(independent=True):
    """w_cos/w_sin for spatial frequencies (u,v).  Real patches make (u,v) and (-u,-v)
    conjugate, so only about half the grid carries independent information."""
    yy, xx = np.mgrid[0:P, 0:P]
    Wc, Ws, freqs = [], [], []
    for u in range(P):
        for v in range(P):
            if u == 0 and v == 0:
                continue                                    # DC: killed by mean-centring
            if independent and (u > P//2 or (u == 0 and v > P//2)):
                continue                                    # keep one of each conjugate pair
            ph = 2*np.pi*(u*yy + v*xx)/P
            Wc.append(np.cos(ph).ravel()); Ws.append(np.sin(ph).ravel()); freqs.append((u, v))
    return np.array(Wc), np.array(Ws), freqs


def phase_recon_r2(TE, Wc, Ws, epochs=12):
    """Best linear decoder from the phases alone -- the fair readout for a chord."""
    B = len(Wc)
    TH = np.arctan2(TE @ Ws.T, TE @ Wc.T)
    C, S = np.cos(TH), np.sin(TH)
    U, V = rng.normal(size=(B, P*P))*.05, rng.normal(size=(B, P*P))*.05
    opt = L1.Adam([U, V], 3e-3)
    for _ in range(epochs):
        for i in range(0, len(TE)-256, 256):
            c, s, X = C[i:i+256], S[i:i+256], TE[i:i+256]
            E = (c @ U + s @ V - X)/len(X)
            opt.step([c.T @ E, s.T @ E])
    return L1.r2(TE, C @ U + S @ V)


if __name__ == "__main__":
    TE = L1.patchify(L1.mnist("t10k", 500))
    A, Bp = SH.pairs(1); A2, B2 = SH.pairs(2)
    n = 3000
    def allpairs(Wc, Ws):
        T = np.arctan2(TE[:n] @ Ws.T, TE[:n] @ Wc.T)
        i, j = rng.integers(0, n, 60000), rng.integers(0, n, 60000); m = i != j
        return float(np.cos(T[i[m]] - T[j[m]]).mean())

    banks = {}
    Wc, Ws, fq = dft_bank(True);  banks[f"DFT, {len(Wc)} independent"] = (Wc, Ws)
    Wc, Ws, _  = dft_bank(False); banks[f"DFT, {len(Wc)} all non-DC"] = (Wc, Ws)
    gc, gs = SH.gabor_bank();     banks["Gabor, 64"] = (gc, gs)
    banks["random, 64"] = L1.random_bank(64)

    print(f"{len(TE)} test patches, {len(A)} shift pairs\n")
    print(f"  {'bank':<24}{'voices':>7}{'coh 1px':>9}{'coh 2px':>9}"
          f"{'phase-only R2':>15}{'all-pairs sim':>15}")
    for name, (wc, ws) in banks.items():
        print(f"  {name:<24}{len(wc):>7}{SH.coherence(A,Bp,wc,ws).mean():9.3f}"
              f"{SH.coherence(A2,B2,wc,ws).mean():9.3f}"
              f"{phase_recon_r2(TE, wc, ws):15.3f}{allpairs(wc,ws):15.3f}")

    print("\n  Is the DFT shift theorem exact here?  predicted vs measured interval,")
    print("  for a 1px horizontal shift (predicted = 2*pi*v/P):\n")
    Wc, Ws, fq = dft_bank(True)
    t0 = np.arctan2(A @ Ws.T, A @ Wc.T); t1 = np.arctan2(Bp @ Ws.T, Bp @ Wc.T)
    d = np.angle(np.exp(1j*(t1-t0)).mean(0)); coh = np.abs(np.exp(1j*(t1-t0)).mean(0))
    for k in np.argsort([u*u+v*v for u, v in fq])[:8]:
        u, v = fq[k]
        print(f"    freq (u={u},v={v})  predicted {np.degrees(-2*np.pi*v/P):+8.1f}  "
              f"measured {np.degrees(d[k]):+8.1f}   coherence {coh[k]:.3f}")
