"""Dedicated gallery: original vs the full from-L3 residual reconstruction.

The complete chain on Exp 4's seq weights: residual voices at L3 (R=4) ->
side-channel expansion to L2 code -> side-channel expansion to L1 code ->
mean/norm pixel decode. The rightmost column of ladder_exp4seq.png, shown
properly: 16 test images, pairs.

Run: .venv/bin/python experiments/2026_08_28/residual_full_gpu/show_l3_roundtrip.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
import run_recon3 as R                     # noqa: E402
import run_residual_full_gpu as G          # noqa: E402

xp = G.xp
N_SHOW = 16


def main():
    w = np.load(HERE.parent / "residual_learning" / "results" / "weights_seq.npz")
    W1 = xp.asarray(w["W1"], dtype=G.DTYPE)
    W2 = xp.asarray(w["W2"], dtype=G.DTYPE)
    W3 = xp.asarray(w["W3"], dtype=G.DTYPE)
    _, _, Xte, _ = R.load()
    xb = xp.asarray(Xte[:N_SHOW], dtype=G.DTYPE)
    n = len(xb)

    P1 = G.window_stack(xb[..., None], G.POS1, R.W1_WIN).reshape(n * len(G.POS1), -1)
    Xh, means1, norms1, ok1 = G.center_norm_rows(P1)
    C1 = xp.maximum(Xh @ W1.T, 0.0) * ok1[:, None]
    means1 = means1.reshape(n, -1)
    norms1 = norms1.reshape(n, -1)
    M1map = C1.reshape(n, R.G1, R.G1, R.K1)

    B2 = G.window_stack(M1map, G.POS2, R.W2_WIN).reshape(n * len(G.POS2), -1)
    Bh2, bm2, bn2, ok2 = G.center_norm_rows(B2)
    seg2 = (xp.maximum(Bh2 @ W2.T, 0.0) * ok2[:, None]).reshape(n, len(G.POS2), -1)
    bm2 = bm2.reshape(n, -1)
    bn2 = bn2.reshape(n, -1)
    M2map = seg2.reshape(n, R.G2, R.G2, R.K2)

    B3 = G.window_stack(M2map, G.POS3, R.W3_WIN).reshape(n * len(G.POS3), -1)
    Bh3, bm3, bn3, _ = G.center_norm_rows(B3)
    bm3 = bm3.reshape(n, -1)
    bn3 = bn3.reshape(n, -1)

    w3r, cf3, _ = G.residual_code(Bh3, W3, G.RMAX_DEEP)
    est3 = xp.zeros((n * len(G.POS3), W3.shape[1]), dtype=G.DTYPE)
    for t in range(G.RMAX_DEEP):
        est3 = est3 + cf3[:, t, None] * W3[w3r[:, t]]
    conf3 = xp.maximum(cf3[:, 0], 1e-6).reshape(n, len(G.POS3))
    c2 = G.expand_level((est3.reshape(n, len(G.POS3), -1), conf3),
                        G.POS3, R.W3_WIN, R.G2, R.K2, W3, "sc", bm3, bn3)
    c1 = G.expand_level(c2.reshape(n, len(G.POS2), -1),
                        G.POS2, R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)
    recon = G.decode_pixels_graded(c1.reshape(n, len(G.POS1), -1),
                                   means1, norms1, W1)

    to_np = (lambda a: a.get()) if G.XP_NAME == "cupy" else np.asarray
    recon_n = to_np(recon)
    imgs, titles = [], []
    for i in range(n):
        imgs.append(Xte[i]); titles.append("orig")
        imgs.append(recon_n[i]); titles.append("from L3")
    R.gallery(imgs, titles,
              HERE / "results" / "l3_roundtrip_pairs.png",
              "Original vs full from-L3 residual reconstruction "
              "(4 voices/position, all side channels)", 8)
    print("written l3_roundtrip_pairs.png")


if __name__ == "__main__":
    main()
