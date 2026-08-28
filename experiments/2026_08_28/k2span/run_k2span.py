"""Exp 7 — is L2's residual-read ceiling its vocabulary SPAN?

Hypothesis from the transfer-failure analysis: L2's bank is 9x
undercomplete (64 templates for 576-dim blocks; L1 is 64-in-64), so
residual voices saturate at 2 — extra rounds can only recombine a span
that covers a fraction of block space. Test: K2 in {64, 256, 512}
(L3's input dim follows K2), GPU mini-batch, L1 residual learning R=4,
L2/L3 plain skeleton learning (full-stack rounds are refuted-pending-
Exp-8). If the span story is right: bigger K2 raises round-1 match
quality and the L2 residual read keeps climbing past 2 voices.
All comparisons are within the GPU-trained regime (which is dull for
residual reads vs CPU-online — known; internal comparison still valid).

Run:  .venv/bin/python experiments/2026_08_28/k2span/run_k2span.py
Env:  KS_TEST (default 5000), KS_K2S (default "64,256,512"), KS_TAG
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
OUTPUT_DIR = HERE / "results" / os.environ.get('KS_TAG', '').lstrip('_')
TEST_N = int(os.environ.get("KS_TEST", "5000"))
K2S = [int(s) for s in os.environ.get("KS_K2S", "64,256,512").split(",")]
RMAX2 = 6
EB = 500


def train(k2, Xtr_x):
    d1 = G.Dict(R.K1, R.W1_WIN * R.W1_WIN, R.ETA1)
    d2 = G.Dict(k2, R.W2_WIN * R.W2_WIN * R.K1, R.ETA2)
    d3 = G.Dict(R.K3, R.W3_WIN * R.W3_WIN * k2, R.ETA3)
    for s in range(0, len(Xtr_x), G.B):
        xb = Xtr_x[s:s + G.B]
        m1 = G.level_pass(xb[..., None], d1, G.POS1, R.W1_WIN, 1, R.G1,
                          "dense", True, G.R_TRAIN)
        m2 = G.level_pass(m1, d2, G.POS2, R.W2_WIN, R.K1, R.G2,
                          "skeleton", True, 1)
        G.level_pass(m2, d3, G.POS3, R.W3_WIN, k2, R.G3, "skeleton", True, 1)
    if G.XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    return d1, d2, d3


def eval_k2(tag, k2, W1, W2, W3, Xte, res):
    mets = {}
    mq2_all = []
    for s in range(0, len(Xte), EB):
        xb = xp.asarray(Xte[s:s + EB], dtype=G.DTYPE)
        n = len(xb)
        P1 = G.window_stack(xb[..., None], G.POS1, R.W1_WIN).reshape(
            n * len(G.POS1), -1)
        Xh, means1, norms1, ok1 = G.center_norm_rows(P1)
        C1 = xp.maximum(Xh @ W1.T, 0.0) * ok1[:, None]
        means1 = means1.reshape(n, -1)
        norms1 = norms1.reshape(n, -1)
        M1map = C1.reshape(n, R.G1, R.G1, R.K1)

        B2 = G.window_stack(M1map, G.POS2, R.W2_WIN).reshape(
            n * len(G.POS2), -1)
        Bh2, bm2, bn2, _ = G.center_norm_rows(B2)
        seg2 = xp.maximum(Bh2 @ W2.T, 0.0).reshape(n, len(G.POS2), -1)
        bm2 = bm2.reshape(n, -1)
        bn2 = bn2.reshape(n, -1)
        out = {}

        c1p = G.expand_level(seg2, G.POS2, R.W2_WIN, R.G1, R.K1, W2, "parity")
        out["L2_parity"] = G.decode_pixels_graded(
            c1p.reshape(n, len(G.POS1), -1), means1, norms1, W1)

        w2r, cf2, mq2 = G.residual_code(Bh2, W2, RMAX2)
        mq2_all.append(mq2)
        est2 = xp.zeros((n * len(G.POS2), W2.shape[1]), dtype=G.DTYPE)
        conf2 = xp.maximum(cf2[:, 0], 1e-6).reshape(n, len(G.POS2))
        for t in range(RMAX2):
            est2 = est2 + cf2[:, t, None] * W2[w2r[:, t]]
            c1r = G.expand_level((est2.reshape(n, len(G.POS2), -1), conf2),
                                 G.POS2, R.W2_WIN, R.G1, R.K1, W2, "sc",
                                 bm2, bn2)
            out[f"L2_res_R{t + 1}"] = G.decode_pixels_graded(
                c1r.reshape(n, len(G.POS1), -1), means1, norms1, W1)

        # from-L3 (parity + residual R4), dims follow k2
        M2map = seg2.reshape(n, R.G2, R.G2, k2)
        B3 = G.window_stack(M2map, G.POS3, R.W3_WIN).reshape(
            n * len(G.POS3), -1)
        Bh3, bm3, bn3, _ = G.center_norm_rows(B3)
        seg3 = xp.maximum(Bh3 @ W3.T, 0.0).reshape(n, len(G.POS3), -1)
        bm3 = bm3.reshape(n, -1)
        bn3 = bn3.reshape(n, -1)
        c2p = G.expand_level(seg3, G.POS3, R.W3_WIN, R.G2, k2, W3, "parity")
        c1p3 = G.expand_level(c2p.reshape(n, len(G.POS2), -1), G.POS2,
                              R.W2_WIN, R.G1, R.K1, W2, "parity")
        out["L3_parity"] = G.decode_pixels_graded(
            c1p3.reshape(n, len(G.POS1), -1), means1, norms1, W1)
        w3r, cf3, _ = G.residual_code(Bh3, W3, 4)
        est3 = xp.zeros((n * len(G.POS3), W3.shape[1]), dtype=G.DTYPE)
        for t in range(4):
            est3 = est3 + cf3[:, t, None] * W3[w3r[:, t]]
        conf3 = xp.maximum(cf3[:, 0], 1e-6).reshape(n, len(G.POS3))
        c2r = G.expand_level((est3.reshape(n, len(G.POS3), -1), conf3),
                             G.POS3, R.W3_WIN, R.G2, k2, W3, "sc", bm3, bn3)
        c1r3 = G.expand_level(c2r.reshape(n, len(G.POS2), -1), G.POS2,
                              R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)
        out["L3_res"] = G.decode_pixels_graded(
            c1r3.reshape(n, len(G.POS1), -1), means1, norms1, W1)

        for key, img in out.items():
            mets.setdefault(key, {"mse": [], "corr": []})
            m = ((img - xb) ** 2).mean(axis=(1, 2))
            cc = G._corr_rows(img, xb)
            mets[key]["mse"].append(xp.asnumpy(m) if G.XP_NAME == "cupy" else m)
            mets[key]["corr"].append(xp.asnumpy(cc) if G.XP_NAME == "cupy" else cc)

    for key, d in mets.items():
        res[f"{tag}_{key}_mse"] = round(float(np.concatenate(d["mse"]).mean()), 5)
        res[f"{tag}_{key}_corr"] = round(float(np.concatenate(d["corr"]).mean()), 3)
    m = xp.concatenate(mq2_all)
    m = xp.asnumpy(m) if G.XP_NAME == "cupy" else m
    res[f"{tag}_matchq_L2"] = [round(float(np.nanmean(m[:, t])), 3)
                               for t in range(m.shape[1])]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, _, Xte, _ = R.load()
    Xtr_x = xp.asarray(Xtr, dtype=G.DTYPE)
    Xte = Xte[:TEST_N]
    res = {"test_n": len(Xte), "k2s": K2S}
    for k2 in K2S:
        d1, d2, d3 = train(k2, Xtr_x)
        tag = f"k{k2}"
        res[f"{tag}_units_used"] = [int((d.win_counts > 0).sum())
                                    for d in (d1, d2, d3)]
        eval_k2(tag, k2, d1.W, d2.W, d3.W, Xte, res)
        print(f"{tag} done", flush=True)
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
