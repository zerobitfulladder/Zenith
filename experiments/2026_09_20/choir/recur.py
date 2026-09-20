"""Does a vocabulary emerge?

Reconstruction and coherence are both satisfied by a smooth embedding in which no two
patches ever land in the same place.  A vocabulary needs the opposite: a modest number of
regions on the torus, each visited many times, by patches from different digits.

Measured, for each encoder:
  concentration - how tightly patches pile onto a fixed number of centres
  reuse         - how many digit classes visit each region
  vs pixels     - the same clustering done in raw pixel space (what k-means used to do)
"""
from pathlib import Path
import numpy as np
import layer1 as L1
import coherent as CO
import shift as SH

ROOT = Path(__file__).resolve().parent
P, B = L1.P, L1.B
rng = np.random.default_rng(0)


def labelled_patches(n_img=1200, stride=3):
    """patches with the digit class they came from, and the cell they sat in."""
    X, y = L1.mnist_labelled("train", n_img)
    pa, cls, pos = [], [], []
    for im, lab in zip(X, y):
        for ci, i in enumerate(range(0, 28 - P + 1, stride)):
            for cj, j in enumerate(range(0, 28 - P + 1, stride)):
                p = im[i:i+P, j:j+P]
                if p.sum() > 1.0:
                    pa.append(p.ravel()); cls.append(lab); pos.append(ci * 10 + cj)
    pa = np.asarray(pa)
    return pa - pa.mean(1, keepdims=True), np.asarray(cls), np.asarray(pos)


def kmeans(Z, K, iters=25):
    """spherical-ish k-means on whatever vectors it is given (rows already scaled)."""
    C = Z[rng.permutation(len(Z))[:K]].copy()
    for _ in range(iters):
        a = np.argmax(Z @ C.T, 1)
        for k in range(K):
            m = a == k
            if m.sum():
                v = Z[m].sum(0); C[k] = v / (np.linalg.norm(v) + 1e-9)
    return np.argmax(Z @ C.T, 1), C


def chord_vecs(Xp, Wc, Ws):
    TH = np.arctan2(Xp @ Ws.T, Xp @ Wc.T)
    Z = np.concatenate([np.cos(TH), np.sin(TH)], 1)
    return Z / np.sqrt(B)                      # unit norm; Z.Zt = mean cos(dtheta)


def report(name, Z, cls, pos, K):
    a, C = kmeans(Z, K)
    tight = float((Z * C[a]).sum(1).mean())                    # mean sim to own centre
    ent, sizes, covered = [], [], []
    for k in range(K):
        m = a == k
        if m.sum() < 20:
            continue
        p = np.bincount(cls[m], minlength=10) / m.sum()
        ent.append(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(10))
        covered.append((p > 0.02).sum())
        sizes.append(m.sum())
    used = len(ent)
    top = np.sort(sizes)[::-1]
    print(f"  {name:<26}{used:>6}{tight:>10.3f}{np.mean(ent):>10.3f}{np.mean(covered):>9.1f}"
          f"{top[:int(.1*used)].sum()/len(Z):>10.1%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--mu", type=float, default=0.05); a = ap.parse_args()

    Xp, cls, pos = labelled_patches()
    print(f"{len(Xp)} inked {P}x{P} patches, {len(set(cls))} classes, K={a.K} regions\n")

    X0, XS = CO.pair_data(n_img=2000)
    Rb = L1.random_bank(B)
    Wc, Ws, *_ = CO.train(X0, XS, *Rb, a.mu, epochs=10)
    np.save(ROOT / "results" / "W_coherent.npy", np.stack([Wc, Ws]))

    banks = {"learned (coherent)": (Wc, Ws), "random": Rb, "Gabor": SH.gabor_bank()}
    Wr = np.load(ROOT / "results" / "W_learned.npy"); banks["learned (recon only)"] = (Wr[0], Wr[1])

    print(f"  {'encoder':<26}{'used':>6}{'tightness':>10}{'class ent':>10}{'classes':>9}"
          f"{'top 10%':>10}")
    for name, (wc, ws) in banks.items():
        report(name, chord_vecs(Xp, wc, ws), cls, pos, a.K)
    Zp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)
    report("raw pixels", Zp, cls, pos, a.K)

    print("\n  used      - regions with >=20 patches (of K)")
    print("  tightness - mean chord similarity to own centre (1 = identical, 0 = unrelated)")
    print("  class ent - digit-class entropy inside a region, normalised (1 = all 10 equally)")
    print("  classes   - mean number of classes with >2% share in a region")
    print("  top 10%   - share of all patches held by the largest 10% of regions")


def fig_regions(banks, Xp, cls, K, path, ncol=12):
    """The mean patch of each region: what the vocabulary actually looks like."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(banks.items()) + [("raw pixels", None)]
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * .82, len(rows) * 1.0))
    for ri, (name, bank) in enumerate(rows):
        Z = (Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-9)) if bank is None \
            else chord_vecs(Xp, *bank)
        a, _ = kmeans(Z, K)
        big = np.argsort(-np.bincount(a, minlength=K))[:ncol]
        for j, k in enumerate(big):
            m = Xp[a == k].mean(0).reshape(P, P)
            v = np.abs(m).max() + 1e-9
            axes[ri, j].imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            axes[ri, j].set_xticks([]); axes[ri, j].set_yticks([])
            ent = np.bincount(cls[a == k], minlength=10) / max((a == k).sum(), 1)
            axes[ri, j].set_title(f"{(a==k).sum()}", fontsize=5.5, pad=1.5)
            if j == 0:
                axes[ri, j].set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Mean patch of the 12 largest regions  (count above each)", fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.94]); fig.savefig(path, dpi=150); plt.close(fig)
