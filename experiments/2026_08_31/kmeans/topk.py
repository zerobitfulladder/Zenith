"""The small change: let the top rung activate a POPULATION, not a winner.

No retraining -- the same L1 and L2 weights from celeba.py. Only the read
changes. Each L2 template keeps a downward memory (mean half-resolution L1 map
of the faces it won), and generation is the weighted SUM of the memories of
however many templates are active:

    k = 1     the argmax. pure retrieval, what we did before.
    k > 1     relu(correlation) over the top k, normalised, summed.

Two ways to drive it, because they are not the same thing:

    joint     rank templates by correlation with the whole attribute query
    per-attr  activate a population for EACH requested attribute separately and
              superimpose -- so a rare attribute gets an equal voice instead of
              being outvoted by a common one
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint as J, celeba as C

OUT = Path(__file__).resolve().parent / "results"
DS, EPS = 2, 1e-12
DN = C.SIDE // DS
KS = [1, 4, 16, 64]


def main():
    t0 = time.time()
    z = np.load(OUT / "celeba.npz")
    W1, W2 = z["W1"].astype(np.float64), z["W2"].astype(np.float64)
    Xtr, Atr, Xte, Ate, names = C.load()
    P = C.GRID * C.GRID * C.K1
    itr, mtr = C.l1_map(W1, Xtr)
    pos = np.arange(C.SIDE * C.SIDE)
    dpos = (pos // C.SIDE // DS) * DN + (pos % C.SIDE // DS)

    down = np.zeros((C.K2, DN * DN, C.K1), np.float32)
    cnt = np.zeros(C.K2)
    for s in range(0, len(itr), 2000):
        sl = slice(s, s + 2000)
        win = (C.pooled(itr[sl], mtr[sl], C.GRID, Atr[sl]) @ W2.T).argmax(1)
        ii, mm = itr[sl], mtr[sl]
        for j in np.unique(win):
            sel = np.nonzero(win == j)[0]
            r, c = np.nonzero(ii[sel] >= 0)
            np.add.at(down[j], (dpos[c], ii[sel][r, c]), mm[sel][r, c])
            cnt[j] += len(sel)
    down /= np.maximum(cnt, 1)[:, None, None]
    print(f"downward memories built ({int((cnt>0).sum())} live templates)", flush=True)

    def paint(M):
        sel, mag = M.argmax(1), M.max(1)
        img, ct = np.zeros((48, 48)), np.zeros((48, 48))
        for p in range(DN * DN):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, DN)
            r, c = r * DS, c * DS
            img[r:r + C.PS, c:c + C.PS] += W1[sel[p]].reshape(C.PS, C.PS) * mag[p]
            ct[r:r + C.PS, c:c + C.PS] += 1
        return img / np.maximum(ct, 1)

    def corr(spec):
        q = np.zeros(P + 40)
        for k, v in spec.items():
            q[P + names.index(k)] = v
        return J.cn(q[None])[0] @ W2.T

    def population(s, k):
        top = np.argsort(s)[::-1][:k]
        w = np.maximum(s[top], 0.0)
        if w.sum() <= 0:
            w = np.ones(len(top))
        w = w / w.sum()
        return (down[top] * w[:, None, None]).sum(0), top, w

    mus, mal = names.index("Mustache"), names.index("Male")
    queries = [("a woman with a mustache", {"Male": -1, "Mustache": +1}),
               ("a man with a mustache", {"Male": +1, "Mustache": +1}),
               ("a woman with eyeglasses", {"Male": -1, "Eyeglasses": +1})]
    log = {}
    fig, axes = plt.subplots(len(queries) * 2, len(KS) + 1,
                             figsize=(2.0 * (len(KS) + 1), 2.15 * len(queries) * 2))
    for qi, (title, spec) in enumerate(queries):
        s_joint = corr(spec)
        for mode in (0, 1):
            r = qi * 2 + mode
            for ci, k in enumerate(KS):
                if mode == 0:
                    M, top, w = population(s_joint, k)
                else:                              # per-attribute, equal voice
                    M = np.zeros((DN * DN, C.K1))
                    for a, v in spec.items():
                        Ma, top, w = population(corr({a: v}), k)
                        M += Ma / len(spec)
                axes[r][ci].imshow(paint(M), cmap="gray")
                axes[r][ci].set_xticks([]); axes[r][ci].set_yticks([])
                if r == 0:
                    axes[r][ci].set_title(f"k = {k}", fontsize=8)
                mv = float((W2[top, P + mal] * w).sum())
                uv = float((W2[top, P + mus] * w).sum())
                axes[r][ci].set_xlabel(f"M{mv:+.2f} m{uv:+.2f}", fontsize=5)
                log[f"{title}|{'joint' if mode==0 else 'per-attr'}|k={k}"] = \
                    {"male": mv, "mustache": uv}
            # for reference: the k=64 population driven by the rare attribute alone
            Ma, _, _ = population(corr({list(spec)[1]: spec[list(spec)[1]]}), 64)
            axes[r][-1].imshow(paint(Ma), cmap="gray")
            axes[r][-1].set_xticks([]); axes[r][-1].set_yticks([])
            axes[r][-1].set_title(f"{list(spec)[1]} alone, k=64", fontsize=6)
            axes[r][0].set_ylabel(f"{title}\n({'joint' if mode==0 else 'per-attribute'})",
                                  fontsize=5.5, rotation=0, ha="right", va="center")
    fig.suptitle("a population instead of a winner — same weights, only the read "
                 "changed", fontsize=10)
    fig.subplots_adjust(left=.19, right=.995, top=.945, bottom=.01,
                        wspace=.06, hspace=.30)
    fig.savefig(OUT / "celeba_topk.png", dpi=150); plt.close(fig)
    (OUT / "topk.json").write_text(json.dumps(log, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> celeba_topk.png")


if __name__ == "__main__":
    main()
