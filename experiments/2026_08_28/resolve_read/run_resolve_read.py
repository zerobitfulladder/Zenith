"""Exp 11 — greedy selection + closed-form coefficient re-solve (pure read).

On the standing best weights (fullseq_cpu). Selection unchanged (the
sequential rounds pick the committee); then the committee's strengths
are re-solved JOINTLY via the Gram inverse (orthogonal projection onto
the committee's span — MSE-optimal for that selection; the partial-
correlation operation). Same ids, same storage, better numbers.

Measured against the greedy coefficients at every rung:
  from-L1  R = 1..6
  from-L2  residual read (R=4 committee) -> expansion -> decode
  from-L3  residual read (R=4)           -> expansion -> decode
Diagnostics: mean |coef shift| and fraction of negative resolved
coefficients (the committee-overlap signature).

Run:  .venv/bin/python experiments/2026_08_28/resolve_read/run_resolve_read.py
Env:  RR_TEST (default 5000), RR_TAG
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
OUTPUT_DIR = HERE / "results" / os.environ.get('RR_TAG', '').lstrip('_')
TEST_N = int(os.environ.get("RR_TEST", "5000"))
EB = 500
REG = 1e-5


def resolve_prefix(V, W, winners, rmax):
    """Joint least-squares coefficients for each prefix committee.
    V (M, D) targets; winners (M, rmax). Returns list of est (M, D) per
    prefix length, plus (coef shift, neg fraction) at full length."""
    M = len(V)
    ests, diags = [], None
    for rr in range(1, rmax + 1):
        Ws = W[winners[:, :rr]]                       # (M, rr, D)
        Gm = xp.einsum("mrd,msd->mrs", Ws, Ws)
        Gm = Gm + REG * xp.eye(rr, dtype=G.DTYPE)[None]
        b = xp.einsum("mrd,md->mr", Ws, V)
        coef = xp.linalg.solve(Gm, b[..., None])[..., 0]
        ests.append(xp.einsum("mr,mrd->md", coef, Ws))
        if rr == rmax:
            diags = coef
    return ests, diags


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    w = np.load(HERE.parent / "fullseq_cpu" / "results" / "weights.npz")
    W1 = xp.asarray(w["W1"], dtype=G.DTYPE)
    W2 = xp.asarray(w["W2"], dtype=G.DTYPE)
    W3 = xp.asarray(w["W3"], dtype=G.DTYPE)
    _, _, Xte, _ = R.load()
    Xte = Xte[:TEST_N]

    mets = {}
    coef_stats = {"L1": [], "L2": [], "L3": []}

    def score(key, img, xb):
        mets.setdefault(key, {"mse": [], "corr": []})
        m = ((img - xb) ** 2).mean(axis=(1, 2))
        cc = G._corr_rows(img, xb)
        mets[key]["mse"].append(xp.asnumpy(m) if G.XP_NAME == "cupy" else m)
        mets[key]["corr"].append(xp.asnumpy(cc) if G.XP_NAME == "cupy" else cc)

    for s in range(0, len(Xte), EB):
        xb = xp.asarray(Xte[s:s + EB], dtype=G.DTYPE)
        n = len(xb)
        P1 = G.window_stack(xb[..., None], G.POS1, R.W1_WIN).reshape(
            n * len(G.POS1), -1)
        Xh, means1, norms1, ok1 = G.center_norm_rows(P1)
        means1 = means1.reshape(n, -1)
        norms1 = norms1.reshape(n, -1)
        ok1m = ok1.reshape(n, -1)
        C1 = xp.maximum(Xh @ W1.T, 0.0) * ok1[:, None]
        M1map = C1.reshape(n, R.G1, R.G1, R.K1)

        # ---- from-L1: greedy vs resolved, R=1..6 ---------------------------
        w1r, cf1, _ = G.residual_code(Xh, W1, 6)
        conf1 = xp.maximum(cf1[:, 0], 1e-6).reshape(n, -1)
        conf1 = xp.where(ok1m, conf1, 1e-6)
        ests_res, coefs = resolve_prefix(Xh, W1, w1r, 6)
        coef_stats["L1"].append(
            (float(xp.abs(coefs - cf1).mean()), float((coefs < 0).mean())))
        est_g = xp.zeros_like(Xh)
        for t in range(6):
            est_g = est_g + cf1[:, t, None] * W1[w1r[:, t]]
            for key, e in ((f"L1_greedy_R{t + 1}", est_g),
                           (f"L1_resolved_R{t + 1}", ests_res[t])):
                ew = e.reshape(n, len(G.POS1), -1)
                num = xp.zeros((n, R.SIDE, R.SIDE), dtype=G.DTYPE)
                den = xp.zeros((n, R.SIDE, R.SIDE), dtype=G.DTYPE)
                for p, (r, c) in enumerate(G.POS1):
                    win = xp.where(
                        ok1m[:, p, None],
                        means1[:, p, None] + norms1[:, p, None] * ew[:, p],
                        means1[:, p, None] * xp.ones_like(ew[:, p]))
                    w2d = win.reshape(n, R.W1_WIN, R.W1_WIN)
                    num[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += \
                        conf1[:, p, None, None] * w2d
                    den[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += \
                        conf1[:, p, None, None]
                score(key, num / xp.maximum(den, 1e-6), xb)

        # ---- deep rungs ----------------------------------------------------
        B2 = G.window_stack(M1map, G.POS2, R.W2_WIN).reshape(
            n * len(G.POS2), -1)
        Bh2, bm2, bn2, _ = G.center_norm_rows(B2)
        seg2 = xp.maximum(Bh2 @ W2.T, 0.0).reshape(n, len(G.POS2), -1)
        bm2 = bm2.reshape(n, -1)
        bn2 = bn2.reshape(n, -1)
        w2r, cf2, _ = G.residual_code(Bh2, W2, 4)
        conf2 = xp.maximum(cf2[:, 0], 1e-6).reshape(n, -1)
        ests2, coefs2 = resolve_prefix(Bh2, W2, w2r, 4)
        coef_stats["L2"].append(
            (float(xp.abs(coefs2 - cf2).mean()), float((coefs2 < 0).mean())))
        est2_g = xp.zeros_like(Bh2)
        for t in range(4):
            est2_g = est2_g + cf2[:, t, None] * W2[w2r[:, t]]
        for key, e2 in (("L2_greedy", est2_g), ("L2_resolved", ests2[-1])):
            c1r = G.expand_level((e2.reshape(n, len(G.POS2), -1), conf2),
                                 G.POS2, R.W2_WIN, R.G1, R.K1, W2, "sc",
                                 bm2, bn2)
            score(key, G.decode_pixels_graded(
                c1r.reshape(n, len(G.POS1), -1), means1, norms1, W1), xb)

        M2map = seg2.reshape(n, R.G2, R.G2, R.K2)
        B3 = G.window_stack(M2map, G.POS3, R.W3_WIN).reshape(
            n * len(G.POS3), -1)
        Bh3, bm3, bn3, _ = G.center_norm_rows(B3)
        bm3 = bm3.reshape(n, -1)
        bn3 = bn3.reshape(n, -1)
        w3r, cf3, _ = G.residual_code(Bh3, W3, 4)
        conf3 = xp.maximum(cf3[:, 0], 1e-6).reshape(n, -1)
        ests3, coefs3 = resolve_prefix(Bh3, W3, w3r, 4)
        coef_stats["L3"].append(
            (float(xp.abs(coefs3 - cf3).mean()), float((coefs3 < 0).mean())))
        est3_g = xp.zeros_like(Bh3)
        for t in range(4):
            est3_g = est3_g + cf3[:, t, None] * W3[w3r[:, t]]
        for key, e3 in (("L3_greedy", est3_g), ("L3_resolved", ests3[-1])):
            c2r = G.expand_level((e3.reshape(n, len(G.POS3), -1), conf3),
                                 G.POS3, R.W3_WIN, R.G2, R.K2, W3, "sc",
                                 bm3, bn3)
            c1r = G.expand_level(c2r.reshape(n, len(G.POS2), -1), G.POS2,
                                 R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)
            score(key, G.decode_pixels_graded(
                c1r.reshape(n, len(G.POS1), -1), means1, norms1, W1), xb)

    res = {"test_n": len(Xte)}
    for key, d in mets.items():
        res[f"{key}_mse"] = round(float(np.concatenate(d["mse"]).mean()), 5)
        res[f"{key}_corr"] = round(float(np.concatenate(d["corr"]).mean()), 3)
    for lvl, st in coef_stats.items():
        res[f"{lvl}_mean_coef_shift"] = round(float(np.mean([a for a, _ in st])), 4)
        res[f"{lvl}_neg_coef_frac"] = round(float(np.mean([b for _, b in st])), 4)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
