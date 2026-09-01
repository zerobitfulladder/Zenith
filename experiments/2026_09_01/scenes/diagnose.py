"""Is the baseline recognition-limited, and is the pooling grid why?

Q1 came back at 0.43 exact-set-of-three, about 76% per digit, against 0.9492 for
this same stack on one centred digit. Before any Q3 conclusion is allowed, find
out whether the encoder can even see a digit on this canvas.

The suspicion is the grid. Yesterday's best had a 28x28 digit spanning 4x4
pooling blocks. The 4x8 grid used for the baseline gives a digit about 2x2 --
the degenerate end of yesterday's own sweep, which scored 0.518. The grid was
sized to the canvas instead of to the digit.

    single    one digit anywhere on the canvas, 10-way. The ceiling.
    presence  three digits, PER-DIGIT accuracy (not exact-set), same encoder.
"""

import json, time
from pathlib import Path
import numpy as np
import scenes as S
import readouts as R

OUT = Path(__file__).resolve().parent / "results"
N_TR, N_TE, EP = 6000, 2000, 25
GRIDS = [(4, 8), (11, 12), (11, 16)]        # blocks of 11x12, 4x8, 4x6


def main():
    t0 = time.time()
    sp = S.splits(); X, ptr, pte = S.load_digits(); W1 = S.load_w1()
    s_tr = S.make_single(N_TR, X, ptr, 11)
    s_te = S.make_single(N_TE, X, pte, 12)
    m_tr = S.make(N_TR, sp["train"], X, ptr, 13)
    m_te = S.make(N_TE, sp["train"], X, pte, 14)
    res = {}
    for gy, gx in GRIDS:
        te = time.time()
        A = S.l1_pooled(W1, s_tr[0], gy=gy, gx=gx)
        B = S.l1_pooled(W1, s_te[0], gy=gy, gx=gx)
        C = S.l1_pooled(W1, m_tr[0], gy=gy, gx=gx)
        D = S.l1_pooled(W1, m_te[0], gy=gy, gx=gx)
        r = {"dims": int(A.shape[1]), "block": [S.SY // gy, S.SX // gx]}
        ra, rb = np.arange(len(A)), np.arange(len(B))
        for lab, hid in (("logistic", 0), ("mlp", 256)):
            n1 = R.fit_net(A, ra, None, s_tr[1], 10, "softmax", hidden=hid, epochs=EP)
            r[f"single_{lab}"] = float(
                (R.predict_net(n1, B, rb, None).argmax(1) == s_te[1]).mean())
            Ytr, Yte = S.q_presence(m_tr[1]), S.q_presence(m_te[1])
            n2 = R.fit_net(C, np.arange(len(C)), None, Ytr, 10, "bce", hidden=hid, epochs=EP)
            P = R.predict_net(n2, D, np.arange(len(D)), None)
            top3 = np.argsort(-P, 1)[:, :3]
            hit = np.take_along_axis(Yte, top3, 1)
            r[f"presence_perdigit_{lab}"] = float(hit.mean())
            r[f"presence_set_{lab}"] = float(hit.all(1).mean())
        res[f"{gy}x{gx}"] = r
        print(f"  grid {gy}x{gx:<3} blocks {r['block']}  dims {r['dims']:>6}  "
              f"single log {r['single_logistic']:.4f} mlp {r['single_mlp']:.4f}   "
              f"per-digit log {r['presence_perdigit_logistic']:.4f} "
              f"mlp {r['presence_perdigit_mlp']:.4f}  ({time.time()-te:.0f}s)", flush=True)
    res["reference"] = {"one centred digit, yesterday's stack": 0.9492}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "diagnose.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
