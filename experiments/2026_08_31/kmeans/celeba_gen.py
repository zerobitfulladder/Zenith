"""Faces out. Each L2 template remembers the mean (half-resolution) L1 map of
the faces it won, and that is painted back through the L1 codebook.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint, celeba as C

OUT = Path(__file__).resolve().parent / "results"
DS = 2                                     # downward memory at half resolution
DN = C.SIDE // DS                          # 22 x 22 positions


def main():
    t0 = time.time()
    z = np.load(OUT / "celeba.npz")
    W1, W2 = z["W1"].astype(np.float64), z["W2"].astype(np.float64)
    Xtr, Atr, Xte, Ate, names = C.load()
    itr, mtr = C.l1_map(W1, Xtr)
    P = C.GRID * C.GRID * C.K1

    down = np.zeros((C.K2, DN * DN, C.K1), np.float32)
    cnt = np.zeros(C.K2)
    for s in range(0, len(itr), 2000):
        sl = slice(s, s + 2000)
        B = C.pooled(itr[sl], mtr[sl], C.GRID, Atr[sl])
        win = (B @ W2.T).argmax(1)
        ii, mm = itr[sl], mtr[sl]
        pos = np.arange(C.SIDE * C.SIDE)
        dpos = (pos // C.SIDE // DS) * DN + (pos % C.SIDE // DS)
        for j in np.unique(win):
            sel = np.nonzero(win == j)[0]
            r, c = np.nonzero(ii[sel] >= 0)
            np.add.at(down[j], (dpos[c], ii[sel][r, c]), mm[sel][r, c])
            cnt[j] += len(sel)
    down /= np.maximum(cnt, 1)[:, None, None]

    def paint(t):
        M = down[t]
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

    def ask(spec):
        q = np.zeros(P + 40)
        for k, v in spec.items():
            q[P + names.index(k)] = v
        return joint.cn(q[None])[0] @ W2.T

    asks = [("a woman", {"Male": -1}),
            ("a man", {"Male": +1}),
            ("a man with a mustache", {"Male": +1, "Mustache": +1}),
            ("A WOMAN WITH A MUSTACHE", {"Male": -1, "Mustache": +1}),
            ("a smiling blonde woman", {"Male": -1, "Smiling": +1, "Blond_Hair": +1}),
            ("an old bald man", {"Male": +1, "Bald": +1, "Young": -1}),
            ("a woman with eyeglasses", {"Male": -1, "Eyeglasses": +1}),
            ("a child", {"Young": +1, "No_Beard": +1, "Heavy_Makeup": -1})]
    im_, mu_ = names.index("Male"), names.index("Mustache")
    fig, axes = plt.subplots(len(asks), 8, figsize=(9, 1.35 * len(asks)))
    log = {}
    for r, (title, spec) in enumerate(asks):
        top = np.argsort(ask(spec))[::-1][:8]
        log[title] = []
        for j, t in enumerate(top):
            axes[r][j].imshow(paint(t), cmap="gray")
            axes[r][j].set_xticks([]); axes[r][j].set_yticks([])
            lab = W2[t, P:]
            axes[r][j].set_title(f"M{lab[im_]:+.2f} m{lab[mu_]:+.2f} n{int(cnt[t])}",
                                 fontsize=4.5, pad=1)
            log[title].append({"male": float(lab[im_]), "mustache": float(lab[mu_]),
                               "n": int(cnt[t])})
        axes[r][0].set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("attributes in, face out — M = the template's own Male value, "
                 "m = its Mustache value, n = faces it won", fontsize=9)
    fig.subplots_adjust(left=.21, right=.995, top=.93, bottom=.01,
                        wspace=.05, hspace=.32)
    fig.savefig(OUT / "celeba_generate.png", dpi=150); plt.close(fig)
    (OUT / "celeba_ask_log.json").write_text(json.dumps(log, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> celeba_generate.png")


if __name__ == "__main__":
    main()
