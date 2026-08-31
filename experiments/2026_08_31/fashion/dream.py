"""Two generation probes on the best model (λ=4).

1. DREAM -- hand an expert coefficients it never saw and let it draw.
     pure random    c ~ N(0, 1), no knowledge of what real coefficients look like
     matched random c ~ N(mean, sd) of that expert's own-class coefficients
   If the expert really owns one class, matched noise should come out as a
   plausible member of that class.

2. ROUND TRIP -- give it a real image of its class, let it answer, then feed
   its own answer back: image -> (rebuilt image + label) -> draw from that
   label alone -> repeat. How well does it actually draw once nothing but its
   own output is left?
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG, LAM = 784, "4.0"
RNG = np.random.default_rng(0)


def gray(a, v, title=None):
    v = v.reshape(28, 28)
    a.imshow(v, cmap="gray", vmin=v.min(), vmax=v.max())
    a.set_xticks([]); a.set_yticks([])
    if title:
        a.set_title(title, fontsize=5.5, pad=1.5)


def main():
    z = np.load(OUT / f"both_w_lam{LAM}.npz")
    W, wins = z["W"].astype(np.float64), z["wins"]
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    proto = center_norm(np.stack([Xtr[ytr == c].mean(0) for c in range(10)]))
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), EPS)

    # the busiest expert of each class
    picks = []
    for c in range(10):
        cand = np.where(claim == c)[0]
        if len(cand):
            picks.append(int(cand[wins[cand].sum(1).argmax()]))
    stats, rows = {}, []
    for h in picks:
        c = claim[h]
        Q = join(Xtr[ytr == c][:2000])
        S = Q @ W[h].T                                   # real coefficients
        rows.append((h, c, S.mean(0), S.std(0), np.abs(S).mean()))

    # ---------- 1. dream ---------------------------------------------------
    n_draw = 6
    fig, axes = plt.subplots(len(rows), n_draw + 3,
                             figsize=(1.05 * (n_draw + 3), 1.15 * len(rows)))
    cos_m, cos_p = [], []
    for r, (h, c, mu, sd, scale) in enumerate(rows):
        gray(axes[r, 0], proto[c], "class mean")
        axes[r, 0].set_ylabel(f"e{h}\n{CLASSES[c][:9]}", fontsize=5.5,
                              rotation=0, ha="right", va="center")
        for k in range(2):                               # pure random
            v = (RNG.standard_normal(W.shape[1]) * scale) @ W[h]
            g = v[:N_IMG]
            cos_p.append(float(g @ proto[c] / (np.linalg.norm(g) + EPS)))
            gray(axes[r, 1 + k], g, f"pure {cos_p[-1]:+.2f}")
        for k in range(n_draw):                          # matched random
            v = (mu + sd * RNG.standard_normal(W.shape[1])) @ W[h]
            g = v[:N_IMG]
            cos_m.append(float(g @ proto[c] / (np.linalg.norm(g) + EPS)))
            gray(axes[r, 3 + k], g, f"matched {cos_m[-1]:+.2f}")
    fig.suptitle("dreams: coefficients drawn from noise, image drawn by the expert  "
                 "(number = cosine with the class mean)", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.90, bottom=.005,
                        wspace=.06, hspace=.30)
    fig.savefig(OUT / "70_dream.png", dpi=140); plt.close(fig)

    # ---------- 2. round trip ----------------------------------------------
    n_it = 4
    fig, axes = plt.subplots(len(rows), 2 + n_it,
                             figsize=(1.05 * (2 + n_it), 1.15 * len(rows)))
    keep = []
    for r, (h, c, *_ ) in enumerate(rows):
        i = np.where(yte == c)[0][0]
        gray(axes[r, 0], Xte[i], "real")
        axes[r, 0].set_ylabel(f"e{h}\n{CLASSES[c][:9]}", fontsize=5.5,
                              rotation=0, ha="right", va="center")
        q = join(Xte[i:i + 1])
        img = None
        for t in range(n_it + 1):
            R = (q @ W[h].T) @ W[h]
            img, lab = R[0, :N_IMG], R[0, N_IMG:]
            cs = float(img @ proto[c] / (np.linalg.norm(img) + EPS))
            said = CLASSES[int(lab.argmax())][:8]
            if t == 0:
                gray(axes[r, 1], img, f"rebuilt {cs:+.2f}\nsays {said}")
            else:
                gray(axes[r, 1 + t], img, f"loop {t} {cs:+.2f}\nsays {said}")
            # feed its OWN label back, image blank: draw what you just named
            v = np.zeros((1, N_IMG + 10)); v[0, N_IMG:] = lab
            q = center_norm(v)
            keep.append(cs)
    fig.suptitle("round trip: real image → rebuild → draw from the label it just "
                 "produced → repeat", fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.90, bottom=.005,
                        wspace=.06, hspace=.30)
    fig.savefig(OUT / "71_roundtrip.png", dpi=140); plt.close(fig)

    print(f"experts probed: {[ (h, CLASSES[c]) for h, c, *_ in rows ]}")
    print(f"dream cosine with class mean:  matched {np.mean(cos_m):+.3f}   "
          f"pure random {np.mean(cos_p):+.3f}")
    (OUT / "dream_metrics.json").write_text(json.dumps(
        {"lam": LAM, "experts": [[int(h), int(c)] for h, c, *_ in rows],
         "dream_cos_matched": float(np.mean(cos_m)),
         "dream_cos_pure": float(np.mean(cos_p))}, indent=2))
    print("-> 70_dream.png, 71_roundtrip.png")


if __name__ == "__main__":
    main()
