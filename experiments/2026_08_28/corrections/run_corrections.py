"""Exp 6 — across-level correction scheme, pure read on Exp 4's weights.

The residual principle rotated sideways: instead of subtracting the
previous VOICE at the same position (Exp 3), each level subtracts the
STORY FROM THE LEVEL ABOVE and residual-codes only the gap, in its own
vocabulary.

Encode-time chain (all on frozen exp4seq weights, GPU):
  1. L3 blocks residual-coded (R3=4 voices)          -> the coarse story
  2. expand down to a predicted L1-code map
  3. at each L2 position: actual block minus its projection on the
     predicted block = the gap; residual-code the gap vs W2 (R2c voices)
     -> corrected block estimates -> corrected L1-code map
  4. at each L1 position: actual window minus its projection on the
     predicted window direction; residual-code vs W1 (R1c voices)
     -> final windows -> mean/norm decode

Stored description = 9x2xR3 + 25x2xR2c + 121x2xR1c numbers (+310 side).
Arms sweep (R2c, R1c); (0,0) ~ the plain from-L3 read, (4,4) should
approach the from-L1 ceiling.

Run:  .venv/bin/python experiments/2026_08_28/corrections/run_corrections.py
Env:  CR_TEST (default 5000), CR_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_full_gpu"))
import run_recon3 as R                    # noqa: E402
import run_residual_full_gpu as G         # noqa: E402

xp = G.xp
OUTPUT_DIR = HERE / "results" / os.environ.get('CR_TAG', '').lstrip('_')
TEST_N = int(os.environ.get("CR_TEST", "5000"))
R3 = 4
ARMS = [(0, 0), (2, 0), (4, 0), (0, 4), (2, 2), (4, 4)]
EB = 500


def corrected_recon(xb, W1, W2, W3, r2c, r1c):
    n = len(xb)
    P1 = G.window_stack(xb[..., None], G.POS1, R.W1_WIN).reshape(
        n * len(G.POS1), -1)
    Xh, means1, norms1, ok1 = G.center_norm_rows(P1)
    C1 = xp.maximum(Xh @ W1.T, 0.0) * ok1[:, None]
    M1map = C1.reshape(n, R.G1, R.G1, R.K1)
    means1 = means1.reshape(n, -1)
    norms1 = norms1.reshape(n, -1)
    ok1 = ok1.reshape(n, -1)

    B2 = G.window_stack(M1map, G.POS2, R.W2_WIN).reshape(n * len(G.POS2), -1)
    Bh2, bm2, bn2, _ = G.center_norm_rows(B2)
    seg2 = xp.maximum(Bh2 @ W2.T, 0.0).reshape(n, len(G.POS2), -1)
    bm2 = bm2.reshape(n, -1)
    bn2 = bn2.reshape(n, -1)
    M2map = seg2.reshape(n, R.G2, R.G2, R.K2)

    B3 = G.window_stack(M2map, G.POS3, R.W3_WIN).reshape(n * len(G.POS3), -1)
    Bh3, bm3, bn3, _ = G.center_norm_rows(B3)
    bm3 = bm3.reshape(n, -1)
    bn3 = bn3.reshape(n, -1)

    # 1. coarse story: L3 voices
    w3r, cf3, _ = G.residual_code(Bh3, W3, R3)
    est3 = xp.zeros((n * len(G.POS3), W3.shape[1]), dtype=G.DTYPE)
    for t in range(R3):
        est3 = est3 + cf3[:, t, None] * W3[w3r[:, t]]
    conf3 = xp.maximum(cf3[:, 0], 1e-6).reshape(n, len(G.POS3))
    M2_hat = G.expand_level((est3.reshape(n, len(G.POS3), -1), conf3),
                            G.POS3, R.W3_WIN, R.G2, R.K2, W3, "sc", bm3, bn3)
    M1_hat = G.expand_level(M2_hat.reshape(n, len(G.POS2), -1),
                            G.POS2, R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)

    # 2. L2 corrections: gap between actual blocks and the predicted map
    Bp = G.window_stack(M1_hat, G.POS2, R.W2_WIN).reshape(n * len(G.POS2), -1)
    Bp = Bp - Bp.mean(axis=1, keepdims=True)
    bpn = xp.linalg.norm(Bp, axis=1, keepdims=True)
    Bp_hat = Bp / xp.maximum(bpn, G.EPS)
    proj2 = xp.sum(Bh2 * Bp_hat, axis=1)
    est2 = proj2[:, None] * Bp_hat
    if r2c > 0:
        gap2 = Bh2 - est2
        w2c, cf2c, _ = G.residual_code(gap2, W2, r2c)
        for t in range(r2c):
            est2 = est2 + cf2c[:, t, None] * W2[w2c[:, t]]
    conf2 = xp.maximum(proj2, 1e-6).reshape(n, len(G.POS2))
    M1_corr = G.expand_level((est2.reshape(n, len(G.POS2), -1), conf2),
                             G.POS2, R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)

    # 3. L1 corrections: gap between actual windows and predicted direction
    seg1 = M1_corr.reshape(n * len(G.POS1), -1)
    v = seg1 @ W1
    v = v - v.mean(axis=1, keepdims=True)
    vn = xp.linalg.norm(v, axis=1, keepdims=True)
    v_hat = v / xp.maximum(vn, G.EPS)
    proj1 = xp.sum(Xh * v_hat, axis=1)
    estw = proj1[:, None] * v_hat
    if r1c > 0:
        gap1 = Xh - estw
        w1c, cf1c, _ = G.residual_code(gap1, W1, r1c)
        for t in range(r1c):
            estw = estw + cf1c[:, t, None] * W1[w1c[:, t]]
    conf1 = xp.maximum(proj1, 1e-6).reshape(n, len(G.POS1))
    conf1 = xp.where(ok1, conf1, 1e-6)
    estw = estw.reshape(n, len(G.POS1), -1)

    num = xp.zeros((n, R.SIDE, R.SIDE), dtype=G.DTYPE)
    den = xp.zeros((n, R.SIDE, R.SIDE), dtype=G.DTYPE)
    for p, (r, c) in enumerate(G.POS1):
        win = xp.where(ok1[:, p, None],
                       means1[:, p, None] + norms1[:, p, None] * estw[:, p],
                       means1[:, p, None] * xp.ones_like(estw[:, p]))
        w2d = win.reshape(n, R.W1_WIN, R.W1_WIN)
        num[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += conf1[:, p, None, None] * w2d
        den[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += conf1[:, p, None, None]
    return num / xp.maximum(den, 1e-6)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    w = np.load(HERE.parent / "residual_learning" / "results" / "weights_seq.npz")
    W1 = xp.asarray(w["W1"], dtype=G.DTYPE)
    W2 = xp.asarray(w["W2"], dtype=G.DTYPE)
    W3 = xp.asarray(w["W3"], dtype=G.DTYPE)
    _, _, Xte, _ = R.load()
    Xte = Xte[:TEST_N]

    res = {"test_n": len(Xte), "r3": R3}
    gal = {}
    for (r2c, r1c) in ARMS:
        ms, cs = [], []
        for s in range(0, len(Xte), EB):
            xb = xp.asarray(Xte[s:s + EB], dtype=G.DTYPE)
            rec = corrected_recon(xb, W1, W2, W3, r2c, r1c)
            m = ((rec - xb) ** 2).mean(axis=(1, 2))
            ms.append(xp.asnumpy(m) if G.XP_NAME == "cupy" else m)
            cc = G._corr_rows(rec, xb)
            cs.append(xp.asnumpy(cc) if G.XP_NAME == "cupy" else cc)
            if s == 0:
                gal[(r2c, r1c)] = (xp.asnumpy(rec[:6])
                                   if G.XP_NAME == "cupy" else rec[:6].copy())
        nums = 9 * 2 * R3 + 25 * 2 * r2c + 121 * 2 * r1c
        res[f"c{r2c}_{r1c}_mse"] = round(float(np.concatenate(ms).mean()), 5)
        res[f"c{r2c}_{r1c}_corr"] = round(float(np.concatenate(cs).mean()), 3)
        res[f"c{r2c}_{r1c}_stored_numbers"] = nums
        print(f"({r2c},{r1c}) done", flush=True)

    imgs, titles = [], []
    show = [(0, 0), (4, 0), (0, 4), (4, 4)]
    for i in range(6):
        imgs.append(Xte[i]); titles.append("orig" if i == 0 else None)
        for a in show:
            imgs.append(gal[a][i])
            titles.append(f"L2c={a[0]},L1c={a[1]}" if i == 0 else None)
    R.gallery(imgs, titles, OUTPUT_DIR / "corrections_ladder.png",
              "From L3 + across-level corrections (Exp 4 weights)",
              1 + len(show))
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
