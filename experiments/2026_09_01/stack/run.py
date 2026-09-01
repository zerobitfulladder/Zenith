"""The whole stack, end to end, on any dataset. Nothing tuned per dataset.

    L1   180 templates over 5x5 patches, trained CO-ADAPTIVELY from scratch --
         random templates, empty table, and the table biasing which patches each
         template wins as it fills (beta 1.5)
    pool 8 x 8 regions of 3 x 3 patch positions
    L2   200 templates over 3x3-region windows (13 x 13 pixels, about half an
         object), SLID over 36 window positions, shared weights
    read T2[L2 template, window cell, class], counted; sum the 36 lookups

MNIST reference for this exact pipeline: 0.9593.
Fashion references from earlier today and yesterday, same 12k/3k split unless
noted: single layer + counted table 0.7793, dense probe 0.8570, logistic on raw
pixels 0.8512 (20k/5k), one layer with a teacher 0.8278 (20k/5k), conv stack
0.7950 (20k/5k).
"""

import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
sys.path.insert(0, str(HERE.parent / "scenes"))
import experts as E, settle as S, single as SG, layer2 as L2, conv2layer as C2, coadapt as CA

OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "fashion_mnist"
GRADED = "graded" in sys.argv[2:]
H, K2, BETA, EPOCHS, IMG_BATCH, ETA, NL, EPS = 180, 200, 1.5, 3, 64, 0.5, 10, 1e-12
NC = C2.PG


def train_l1(Xtr, ytr, rng):
    Wf = rng.standard_normal((H, 25)).astype(np.float32)
    Wf -= Wf.mean(1, keepdims=True)
    Wf /= np.linalg.norm(Wf, axis=1, keepdims=True) + EPS
    N = np.zeros((H, CA.NC, NL)); T = CA.table_from(N)
    order = np.arange(len(Xtr))
    for ep in range(EPOCHS):
        rng.shuffle(order)
        for s in range(0, len(order), IMG_BATCH):
            ids = order[s:s + IMG_BATCH]
            Q, keep = E.patches(Xtr[ids]); m = len(Q)
            P = Q.reshape(-1, 25).astype(np.float32); k = keep.reshape(-1)
            wn = (Wf ** 2).sum(1)
            Sc = P @ Wf.T
            err = 1.0 - 2 * Sc * Sc + Sc ** 2 * wn[None, :]
            cells = np.tile(CA.CELL, m)
            w0 = err.argmin(1)
            sc = (T[w0, cells] * k[:, None]).reshape(m, -1, NL).sum(1)
            z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)
            q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            bias = np.einsum('ik,ihk->ih', np.repeat(q, S.SIDE * S.SIDE, 0),
                             np.ascontiguousarray(T.transpose(1, 0, 2))[cells],
                             optimize=True)
            win = (err - BETA * bias).argmin(1)
            lab = np.repeat(ytr[ids], S.SIDE * S.SIDE)
            np.add.at(N, (win[k], cells[k], lab[k]), 1.0)
            T = CA.table_from(N)
            for j in np.unique(win[k]):
                sel = P[k][win[k] == j]
                if len(sel) >= 4:
                    Wf[j] = E.geo_step(Wf[j:j + 1].astype(np.float64),
                                       sel.astype(np.float64), ETA)[0].astype(np.float32)
        print(f"    L1 epoch {ep+1}/{EPOCHS}", flush=True)
    return Wf[:, None, :].astype(np.float64)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    print(f"{DS}: {len(Xtr)} train / {len(Xte)} test, graded pooling {GRADED}\n", flush=True)
    W1 = train_l1(Xtr, ytr, np.random.default_rng(7))

    P8tr, P8te = C2.code(W1, Xtr, GRADED), C2.code(W1, Xte, GRADED)
    rng = np.random.default_rng(5)
    samp = []
    for a in range(0, len(P8tr), 256):
        V = C2.windows(P8tr[a:a + 256])
        for i in range(len(V)):
            samp.append(V[i, rng.choice(C2.WG ** 2, C2.PER_IMG, False)])
    Q = L2.cn(np.concatenate(samp)).astype(np.float32)
    W2, _ = L2.train(Q, K2, np.random.default_rng(3))
    mtr, mte = C2.win_map(W2, P8tr), C2.win_map(W2, P8te)
    T2, N2 = C2.table(mtr, ytr, K2)

    P = np.arange(C2.WG ** 2)
    sc = np.stack([T2[mte[a], P].sum(0) for a in range(len(mte))])
    pred = sc.argmax(1)
    acc = float((pred == yte).mean())
    cl = N2.sum(1); live = cl.sum(1) > 0
    pur = float((cl[live].max(1) / cl[live].sum(1)).mean())

    cm = np.zeros((NL, NL), int); np.add.at(cm, (yte, pred), 1)
    tp = np.diag(cm); rec = tp / np.maximum(cm.sum(1), 1)
    names = (["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
              "Shirt", "Sneaker", "Bag", "Ankle boot"] if "fashion" in DS
             else [str(i) for i in range(NL)])
    print(f"\n  {DS}  ACCURACY {acc:.4f}   L2 purity {pur:.3f}   live {int(live.sum())}/{K2}")
    print("  per-class recall: " + "  ".join(f"{n}:{r:.2f}" for n, r in zip(names, rec)))
    off = cm - np.diag(tp)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:5], off.shape))[0]
    print("  worst confusions: " + "   ".join(
        f"{names[a]}->{names[b]} {off[a,b]}" for a, b in top))
    (OUT / f"full_stack_{DS}{'_graded' if GRADED else ''}.json").write_text(json.dumps(
        {"dataset": DS, "acc": acc, "purity": pur, "confusion": cm.tolist(),
         "recall": rec.tolist(), "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / f"full_stack_{DS}{'_graded' if GRADED else ''}.npz", W1=W1.astype(np.float32), W2=W2)
    print(f"  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
