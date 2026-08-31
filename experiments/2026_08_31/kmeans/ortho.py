"""Disentangle the attribute direction, then compose.

The mustache direction learned from data carries maleness with it, because the
two co-occur. Adding it to a female base either drags the face male (large gain)
or cancels against the female differential (equal gain). So strip it:

    d_mustache_orth = d_mustache - (d_mustache . d_male_hat) d_male_hat

Now "mustache" means the upper-lip change and holds no opinion about sex, and it
cannot cancel against a female base because the two directions are orthogonal.

Reported per panel, so this is falsifiable rather than a vibe:
    dM = how far the composed map moved ALONG the male direction
    dm = how far it moved along the mustache direction
Orthogonalisation should keep dM near zero while dm rises.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import celeba as C

OUT = Path(__file__).resolve().parent / "results"
DSf, EPS = 2, 1e-12
DN = C.SIDE // DSf
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    t0 = time.time()
    W1 = np.load(OUT / "celeba.npz")["W1"].astype(np.float64)
    k1 = len(W1)
    Xtr, Atr, Xte, Ate, names = C.load()
    itr, mtr = C.l1_map(W1, Xtr)
    pos = np.arange(C.SIDE * C.SIDE)
    dpos = (pos // C.SIDE // DSf) * DN + (pos % C.SIDE // DSf)

    def mean_map(sel):
        acc = np.zeros((DN * DN, k1)); n = 0
        for a in range(0, len(itr), 2000):
            sl = slice(a, a + 2000); s = sel[sl]
            if not s.any():
                continue
            ii, mm = itr[sl][s], mtr[sl][s]
            r, c = np.nonzero(ii >= 0)
            np.add.at(acc, (dpos[c], ii[r, c]), mm[r, c])
            n += int(s.sum())
        return acc / max(n, 1)

    def delta(attr):
        i = names.index(attr)
        return mean_map(Atr[:, i] > 0) - mean_map(Atr[:, i] < 0)

    male = names.index("Male")
    base_f = mean_map(Atr[:, male] < 0)
    d_male = delta("Male")
    d_must = delta("Mustache")
    d_glas = delta("Eyeglasses")
    unit = lambda v: v / (np.linalg.norm(v) + EPS)
    hm = unit(d_male)
    orth = lambda d: d - (d * hm).sum() * hm
    print("cosine(mustache, male) =",
          f"{float((unit(d_must) * hm).sum()):+.3f}   "
          f"cosine(glasses, male) = {float((unit(d_glas) * hm).sum()):+.3f}")

    def paint(M):
        M = np.maximum(M, 0.0)
        sel, mag = M.argmax(1), M.max(1)
        img, ct = np.zeros((48, 48)), np.zeros((48, 48))
        for p in range(DN * DN):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, DN); r, c = r * DSf, c * DSf
            img[r:r + C.PS, c:c + C.PS] += W1[sel[p]].reshape(C.PS, C.PS) * mag[p]
            ct[r:r + C.PS, c:c + C.PS] += 1
        return img / np.maximum(ct, 1)

    rows = [("woman + mustache (raw direction)", d_must),
            ("woman + mustache (male component removed)", orth(d_must)),
            ("woman + glasses (raw direction)", d_glas),
            ("woman + glasses (male component removed)", orth(d_glas))]
    hmu, hgl = unit(d_must), unit(d_glas)
    bn = np.linalg.norm(base_f)
    log = {}
    fig, axes = plt.subplots(len(rows), len(BETAS), figsize=(2.05 * len(BETAS), 2.35 * len(rows)))
    for r, (title, d) in enumerate(rows):
        dh = unit(d)
        for ci, b in enumerate(BETAS):
            M = base_f + b * bn * dh
            mv = float(((M - base_f) * hm).sum())
            uv = float(((M - base_f) * (hgl if "glasses" in title else hmu)).sum())
            axes[r][ci].imshow(paint(M), cmap="gray")
            axes[r][ci].set_xticks([]); axes[r][ci].set_yticks([])
            axes[r][ci].set_xlabel(f"dM {mv:+.2f}   d{'g' if 'glasses' in title else 'm'} {uv:+.2f}",
                                   fontsize=5.5)
            if r == 0:
                axes[r][ci].set_title(f"beta = {b}", fontsize=8)
            log[f"{title}|beta={b}"] = {"along_male": mv, "along_attr": uv}
        axes[r][0].set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("removing the male component from an attribute direction "
                 "(base = the female mean)", fontsize=10)
    fig.subplots_adjust(left=.235, right=.995, top=.93, bottom=.02,
                        wspace=.06, hspace=.30)
    fig.savefig(OUT / "celeba_ortho.png", dpi=150); plt.close(fig)
    (OUT / "ortho.json").write_text(json.dumps(
        {"cos_mustache_male": float((unit(d_must) * hm).sum()),
         "cos_glasses_male": float((unit(d_glas) * hm).sum()), "panels": log},
        indent=2))
    print(f"done in {time.time()-t0:.0f}s -> celeba_ortho.png")


if __name__ == "__main__":
    main()
