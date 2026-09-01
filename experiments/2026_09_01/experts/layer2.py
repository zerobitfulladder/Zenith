"""Layer 2: competing templates over the pooled layer-1 code, with its own table.

Layer 1 is the co-adaptive run -- 180 templates trained from scratch alongside
their table, match-against-average 0.9487. Frozen here.

Layer 2 is the same machinery one level up:

    input       the pooled L1 code (4x4 cells x 180 templates)
    templates   K2 of them, competing, winner takes all, annealed step
    table       T2[L2 template, class]   -- no cell index needed, L2 sees
                                            the whole image already
    read        which L2 template won -> its class

The prediction from yesterday's stack, which did this with plain k-means: the
templates should come out CLASS-PURE (purity 1.000, 8-13 per class), because at
this level a cluster IS a class. That is the claim this tests directly.

Also measured: whether two images of the same digit land on the SAME template.
They mostly will not -- there are ~20 templates per class and they split by
style. That is the class-and-style code, and it means the stable thing is the
GROUP of templates sharing a label, not any single index.
"""

import json, time
from pathlib import Path
import numpy as np
import experts as E, blank as B, settle as S, readouts as R, single as SG, pressure as PR

OUT = Path(__file__).resolve().parent / "results"
GRID, NL, EPS = 4, 10, 1e-12
K2S, EPOCHS, ETA_MIN = [100, 200, 400], 12, 0.02


def cn(V):
    V = V - V.mean(1, keepdims=True)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


def kmeanspp(Q, k, rng):
    W = np.empty((k, Q.shape[1]), np.float32)
    W[0] = Q[rng.integers(len(Q))]
    d2 = 2.0 - 2.0 * (Q @ W[0])
    for i in range(1, k):
        p = np.maximum(d2, 0); s = p.sum()
        j = rng.integers(len(Q)) if s <= 0 else rng.choice(len(Q), p=p / s)
        W[i] = Q[j]
        d2 = np.minimum(d2, 2.0 - 2.0 * (Q @ W[i]))
    return W


def train(Q, k, rng):
    W = kmeanspp(Q[rng.choice(len(Q), min(4000, len(Q)), False)], k, rng)
    n = np.zeros(k)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Q))
        for s in range(0, len(order), 512):
            Bt = Q[order[s:s + 512]]
            win = (Bt @ W.T).argmax(1)
            for j in np.unique(win):
                m = Bt[win == j]
                n[j] += len(m)
                W[j] += min(max(1.0 / n[j] * len(m), ETA_MIN), 1.0) * (m.mean(0) - W[j])
                W[j] /= np.linalg.norm(W[j]) + EPS
    return W, n


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load()
    W1 = np.load(OUT / "coadapt.npz")["W"].astype(np.float64)
    itr, ite = SG.winners(W1, Xtr), SG.winners(W1, Xte)
    Qtr = cn(PR.pooled(itr, len(W1))).astype(np.float32)
    Qte = cn(PR.pooled(ite, len(W1))).astype(np.float32)
    print(f"L1 frozen: {len(W1)} templates -> pooled code {Qtr.shape[1]} numbers\n",
          flush=True)

    res = {}
    for k in K2S:
        t1 = time.time()
        rng = np.random.default_rng(3)
        W2, n = train(Qtr, k, rng)
        wtr = (Qtr @ W2.T).argmax(1)
        wte = (Qte @ W2.T).argmax(1)
        C = np.zeros((k, NL))
        np.add.at(C, (wtr, ytr), 1.0)
        lab = C.argmax(1)
        acc = float((lab[wte] == yte).mean())
        live = C.sum(1) > 0
        purity = float((C[live].max(1) / C[live].sum(1)).mean())
        per_class = np.bincount(lab[live], minlength=NL)
        # do two images of the same class land on the same template?
        same = []
        for c in range(NL):
            w = wte[yte == c]
            same.append(float((w[:, None] == w[None, :]).mean()))
        res[str(k)] = {"acc": acc, "purity": purity, "live": int(live.sum()),
                       "templates_per_class": per_class.tolist(),
                       "same_template_within_class": float(np.mean(same)),
                       "params": int(k * Qtr.shape[1]), "seconds": round(time.time() - t1, 1)}
        print(f"  K2={k:<4} acc {acc:.4f}   purity {purity:.3f}   live {int(live.sum())}/{k}   "
              f"per class {per_class.min()}-{per_class.max()}   "
              f"two same-class images share a template {np.mean(same)*100:.1f}% "
              f"({time.time()-t1:.0f}s)", flush=True)

    res["reference"] = {"L1 alone, counted table": 0.9540,
                        "L1 alone, match-against-average": 0.9487,
                        "L1 alone, trained linear probe": 0.9787,
                        "yesterday's kmeans L2 purity": 1.000}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "layer2.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
