"""Exp 8 — CPU-online full-stack sequential residual learning.

Kills Exp 5's confound: full-stack residual LEARNING was refuted only
under GPU mini-batch training, which independently dulls residual
structure. Here the same full-stack rule runs in the CPU online regime
that produced the good Exp 4 weights: L1 sequential residual on dense
windows, L2/L3 sequential residual on their SKELETON views (bootstrap
adopts skeleton, then rounds), R_TRAIN=4 everywhere, speech dense relu.
Eval = the verified GPU pipeline (residual reads at every rung).

If mini-batch was the confound: fullseq >= exp4seq on the deep residual
reads. If it still degrades: the L1-recipe-does-not-transfer verdict is
regime-independent (off-manifold leftovers / view mismatch).

Run:  .venv/bin/python experiments/2026_08_28/fullseq_cpu/run_fullseq_cpu.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_learning"))
sys.path.insert(0, str(HERE.parent / "residual_full_gpu"))
import run_recon3 as R                    # noqa: E402
import run_residual_learning as RL        # noqa: E402
import run_residual_full_gpu as G         # noqa: E402

OUTPUT_DIR = HERE / "results"
R_TRAIN = 4


def encode_over_seq(prev, dic, win, stride, g_out, rounds):
    """L2/L3 pass: dense relu speech; skeleton bootstrap-adopt, then
    sequential residual learning on the skeleton view."""
    out = np.zeros((g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            block = prev[gi * stride:gi * stride + win,
                         gj * stride:gj * stride + win, :]
            v_hat, n = R.center_norm(block.ravel())
            if n < R.NORM_FLOOR:
                continue
            c = dic.forward(v_hat)
            s_hat, sn = R.center_norm(R._pp_top1(block).ravel())
            if sn > R.NORM_FLOOR:
                if dic.n_boot < dic.k:
                    dic.learn(s_hat, dic.forward(s_hat))
                else:
                    RL.seq_learn(dic, s_hat, rounds)
            out[gi, gj, :] = np.maximum(c, 0.0)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, _, Xte, _ = R.load()
    rng = np.random.default_rng(R.SEED)
    d1 = R.Layer(R.K1, R.W1_WIN * R.W1_WIN, R.ETA1, rng)
    d2 = R.Layer(R.K2, R.W2_WIN * R.W2_WIN * R.K1, R.ETA2, rng)
    d3 = R.Layer(R.K3, R.W3_WIN * R.W3_WIN * R.K2, R.ETA3, rng)

    for i, x in enumerate(Xtr):
        m1 = RL.encode_pixels_arm(x, d1, "seq")
        m2 = encode_over_seq(m1, d2, R.W2_WIN, R.W2_STR, R.G2, R_TRAIN)
        encode_over_seq(m2, d3, R.W3_WIN, R.W3_STR, R.G3, R_TRAIN)
        if (i + 1) % 5000 == 0:
            print(f"[fullseq_cpu] {i + 1}/{len(Xtr)}", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz", W1=d1.W, W2=d2.W, W3=d3.W)

    xp = G.xp
    W1 = xp.asarray(d1.W, dtype=G.DTYPE)
    W2 = xp.asarray(d2.W, dtype=G.DTYPE)
    W3 = xp.asarray(d3.W, dtype=G.DTYPE)
    Xte_x = xp.asarray(Xte, dtype=G.DTYPE)
    res = {"train_n": R.TRAIN_N, "test_n": len(Xte), "r_train": R_TRAIN,
           "units_used": [int((d.win_counts > 0).sum()) for d in (d1, d2, d3)]}
    gal = G.eval_arm("fullseq_cpu", W1, W2, W3, Xte_x, res)

    cols = ["orig", "L1_res_R6", "L2_sc_graded", "L2_res_R4",
            "L3_parity", "L3_res"]
    imgs, titles = [], []
    for i in range(8):
        for cn in cols:
            imgs.append(gal[cn][i])
            titles.append(cn if i == 0 else None)
    R.gallery(imgs, titles, OUTPUT_DIR / "ladder_fullseq_cpu.png",
              "fullseq_cpu: orig | L1 R6 | L2 sc | L2 res | L3 parity | L3 res",
              len(cols))
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
