"""Exp 12 — committee-LOO residual learning (CPU-online, full stack).

The principled middle between sequential greedy and the team code:
SELECTION stays sequential (strongest first, subtract, next — the
soloist-preserving order), but ATTRIBUTION becomes exact: the chosen
committee's coefficients are solved jointly via the Gram inverse
(partial correlations — the user's CSHL operation), and each member's
geodesic target is the window minus what the OTHER members explain
(the user's learning3 LOO formula, restricted to the committee of 4).

Everything else identical to run_fullseq_cpu (the standing recipe):
all levels, L1 dense view / L2-L3 skeleton views, R=4, dense relu
speech, 55k online. Eval = the verified GPU pipeline.

Risk under test: does exact blame-sharing within the committee
re-blur ownership at the margins?

Run:  .venv/bin/python experiments/2026_08_28/committee_learn/run_committee_learn.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_full_gpu"))
import run_recon3 as R                    # noqa: E402
import run_residual_full_gpu as G         # noqa: E402

OUTPUT_DIR = HERE / "results"
R_TRAIN = 4
RES_STOP = 0.02


def committee_learn(dic, x_hat, rounds):
    """Sequential selection, joint Gram-solve attribution, LOO targets."""
    W = dic.W
    cur = x_hat.copy()
    members = []
    for _ in range(rounds):
        cn = float(np.linalg.norm(cur))
        if cn < RES_STOP:
            break
        c = W @ (cur / cn)
        i = int(np.argmax(c))
        if c[i] <= 0.0:
            break
        if i not in members:
            members.append(i)
        cur = cur - (W[i] @ cur) * W[i]
    if not members:
        return
    S = np.array(members)
    Ws = W[S]
    Gm = Ws @ Ws.T + 1e-6 * np.eye(len(S))
    coef = np.linalg.solve(Gm, Ws @ x_hat)
    recon = coef @ Ws
    for m, i in enumerate(S):
        t = x_hat - (recon - coef[m] * W[i])      # minus what OTHERS explain
        tn = float(np.linalg.norm(t))
        if tn < R.EPS:
            continue
        t_hat = t / tn
        ci = float(W[i] @ t_hat)
        if ci <= 0.0:
            continue
        w = W[i].copy()
        tau = t_hat - ci * w
        taun = float(np.linalg.norm(tau))
        if taun > R.EPS:
            th = dic.eta * ci
            dic.W[i] = w * np.cos(th) + (tau / taun) * np.sin(th)
        dic.win_counts[i] += 1


def encode_pixels_cl(x2d, dic):
    out = np.zeros((R.G1, R.G1, dic.k))
    for gi in range(R.G1):
        for gj in range(R.G1):
            p = x2d[gi * R.W1_STR:gi * R.W1_STR + R.W1_WIN,
                    gj * R.W1_STR:gj * R.W1_STR + R.W1_WIN].ravel()
            p_hat, n = R.center_norm(p)
            if n < R.NORM_FLOOR:
                continue
            c = dic.forward(p_hat)
            if dic.n_boot < dic.k:
                dic.learn(p_hat, c)
            else:
                committee_learn(dic, p_hat, R_TRAIN)
            out[gi, gj, :] = np.maximum(c, 0.0)
    return out


def encode_over_cl(prev, dic, win, stride, g_out):
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
                    committee_learn(dic, s_hat, R_TRAIN)
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
        m1 = encode_pixels_cl(x, d1)
        m2 = encode_over_cl(m1, d2, R.W2_WIN, R.W2_STR, R.G2)
        encode_over_cl(m2, d3, R.W3_WIN, R.W3_STR, R.G3)
        if (i + 1) % 5000 == 0:
            print(f"[committee] {i + 1}/{len(Xtr)}", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz", W1=d1.W, W2=d2.W, W3=d3.W)
    xp = G.xp
    res = {"train_n": R.TRAIN_N, "test_n": len(Xte), "r_train": R_TRAIN,
           "units_used": [int((d.win_counts > 0).sum()) for d in (d1, d2, d3)]}
    gal = G.eval_arm("committee", xp.asarray(d1.W, dtype=G.DTYPE),
                     xp.asarray(d2.W, dtype=G.DTYPE),
                     xp.asarray(d3.W, dtype=G.DTYPE),
                     xp.asarray(Xte, dtype=G.DTYPE), res)

    l1s = R.Layer(R.K1, d1.W.shape[1], 0.0, np.random.default_rng(0))
    l1s.W = d1.W
    R.gallery([np.maximum(d1.W[i], 0).reshape(R.W1_WIN, R.W1_WIN)
               for i in range(R.K1)], None,
              OUTPUT_DIR / "templates_L1_committee.png",
              "L1 after committee-LOO learning", 16, cmap="inferno")
    cols = ["orig", "L1_res_R6", "L2_res_R4", "L3_res"]
    imgs, titles = [], []
    for i in range(8):
        for cn in cols:
            imgs.append(gal[cn][i])
            titles.append(cn if i == 0 else None)
    R.gallery(imgs, titles, OUTPUT_DIR / "ladder_committee.png",
              "committee: orig | L1 R6 | L2 res | L3 res", len(cols))
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
