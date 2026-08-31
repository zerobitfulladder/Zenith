"""Send the DIFFERENTIAL, not the mean.

Each L2 template stored the mean L1 map of the faces it won. Superimposing two
such means re-adds the shared face structure twice and averages away what makes
each population distinctive -- which is why summing a "woman" population and a
"mustache" population produced neither.

The change: every template sends what it ADDS relative to the global mean.

    base   = mean L1 map over all faces
    d[j]   = mean map of template j's members  -  base
    render = base + gain * sum_a  (population differential for attribute a)

Now the common structure is contributed once by the base, and each active
population only adds its own contribution. This is the predictive-coding form:
send the difference from the expectation.

Same weights as celeba.py. Only the downward message changed.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint as J, celeba as C

OUT = Path(__file__).resolve().parent / "results"
DSf, EPS, KPOP = 2, 1e-12, 16
DN = C.SIDE // DSf
GAINS = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    t0 = time.time()
    z = np.load(OUT / "celeba.npz")
    W1, W2 = z["W1"].astype(np.float64), z["W2"].astype(np.float64)
    Xtr, Atr, Xte, Ate, names = C.load()
    P = C.GRID * C.GRID * C.K1
    itr, mtr = C.l1_map(W1, Xtr)
    pos = np.arange(C.SIDE * C.SIDE)
    dpos = (pos // C.SIDE // DSf) * DN + (pos % C.SIDE // DSf)

    down = np.zeros((C.K2, DN * DN, C.K1), np.float64)
    cnt = np.zeros(C.K2)
    base = np.zeros((DN * DN, C.K1), np.float64)
    for s in range(0, len(itr), 2000):
        sl = slice(s, s + 2000)
        win = (C.pooled(itr[sl], mtr[sl], C.GRID, Atr[sl]) @ W2.T).argmax(1)
        ii, mm = itr[sl], mtr[sl]
        r, c = np.nonzero(ii >= 0)
        np.add.at(base, (dpos[c], ii[r, c]), mm[r, c])
        for j in np.unique(win):
            sel = np.nonzero(win == j)[0]
            r, c = np.nonzero(ii[sel] >= 0)
            np.add.at(down[j], (dpos[c], ii[sel][r, c]), mm[sel][r, c])
            cnt[j] += len(sel)
    base /= len(itr)
    down /= np.maximum(cnt, 1)[:, None, None]
    dmem = down - base[None]                      # what each template ADDS
    print(f"base + {int((cnt>0).sum())} differentials built", flush=True)

    def paint(M):
        M = np.maximum(M, 0.0)
        sel, mag = M.argmax(1), M.max(1)
        img, ct = np.zeros((48, 48)), np.zeros((48, 48))
        for p in range(DN * DN):
            if mag[p] <= 0:
                continue
            r, c = divmod(p, DN)
            r, c = r * DSf, c * DSf
            img[r:r + C.PS, c:c + C.PS] += W1[sel[p]].reshape(C.PS, C.PS) * mag[p]
            ct[r:r + C.PS, c:c + C.PS] += 1
        return img / np.maximum(ct, 1)

    def pop(spec, k=KPOP):
        q = np.zeros(P + 40)
        for a, v in spec.items():
            q[P + names.index(a)] = v
        s = J.cn(q[None])[0] @ W2.T
        top = np.argsort(s)[::-1][:k]
        w = np.maximum(s[top], 0.0)
        w = w / max(w.sum(), EPS)
        return (dmem[top] * w[:, None, None]).sum(0), (W2[top, P:] * w[:, None]).sum(0)

    mus, mal, gla = (names.index("Mustache"), names.index("Male"),
                     names.index("Eyeglasses"))
    queries = [("a woman with a mustache", [("Male", -1), ("Mustache", +1)]),
               ("a man with a mustache", [("Male", +1), ("Mustache", +1)]),
               ("a woman with eyeglasses", [("Male", -1), ("Eyeglasses", +1)]),
               ("a bald woman", [("Male", -1), ("Bald", +1)])]
    log = {}
    fig, axes = plt.subplots(len(queries), len(GAINS) + 1,
                             figsize=(2.0 * (len(GAINS) + 1), 2.3 * len(queries)))
    for r, (title, spec) in enumerate(queries):
        parts = [pop({a: v}) for a, v in spec]
        for ci, g in enumerate(GAINS):
            M = base + g * sum(p[0] for p in parts)
            av = sum(p[1] for p in parts) / len(parts)
            axes[r][ci].imshow(paint(M), cmap="gray")
            axes[r][ci].set_xticks([]); axes[r][ci].set_yticks([])
            axes[r][ci].set_xlabel(f"M{av[mal]:+.2f} m{av[mus]:+.2f} g{av[gla]:+.2f}",
                                   fontsize=5)
            if r == 0:
                axes[r][ci].set_title(f"gain {g}", fontsize=8)
            log[f"{title}|gain={g}"] = {"male": float(av[mal]),
                                        "mustache": float(av[mus]),
                                        "glasses": float(av[gla])}
        axes[r][-1].imshow(paint(base), cmap="gray")
        axes[r][-1].set_xticks([]); axes[r][-1].set_yticks([])
        axes[r][-1].set_title("the base (all faces)", fontsize=6)
        axes[r][0].set_ylabel(title, fontsize=6.5, rotation=0, ha="right", va="center")
    fig.suptitle("each template sends what it ADDS to the average face; "
                 "populations then superimpose", fontsize=10)
    fig.subplots_adjust(left=.20, right=.995, top=.93, bottom=.02,
                        wspace=.06, hspace=.28)
    fig.savefig(OUT / "celeba_diff.png", dpi=150); plt.close(fig)
    (OUT / "diff.json").write_text(json.dumps(log, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> celeba_diff.png")


if __name__ == "__main__":
    main()
