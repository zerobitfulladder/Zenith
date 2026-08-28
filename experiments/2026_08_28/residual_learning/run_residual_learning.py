"""Exp 4 — residual LEARNING at L1, two shapes, vs the frozen baseline.

Exp 3 proved residual READING works but runs on a whole-window-shaped
vocabulary (round-2 match quality 0.47 vs 0.81). Today's question: does a
leftover-shaped vocabulary close that gap? Two shapes of the same
principle, both retraining L1 from scratch (L2/L3 retrained on top with
unchanged rules; upward speech stays dense relu of the original window):

  seq   sequential greedy (the Exp 3 read, now learning): per window up
        to R_TRAIN rounds — the round's winner takes its geodesic step
        toward the CURRENT LEFTOVER, its projection is subtracted, the
        next round's winner competes over the remainder. Round 1 is
        exactly the old rule (the leftover is the whole window).

  loo   the user's months-old parallel form (experiments/2026_03_16/crg/learning3.py,
        adapted to the current unit): every active template learns AT
        ONCE, each toward what the OTHERS fail to explain (reconstruct
        with everyone except me — the "shadow" — and learn toward the
        unexplained part). No rounds, no single winner.

Evaluation per arm = Exp 3's instrument on the new weights: residual
read R=1..6 (corr/MSE, per-round match quality and leftover norm), plus
the plain graded-read from-L1 and the deep rungs (from-L2/L3, graded
avg) to see what the new dictionaries cost elsewhere.

Run:  .venv/bin/python experiments/2026_08_28/residual_learning/run_residual_learning.py
Env:  RC_TRAIN/RC_TEST (shared), RL_RTRAIN, RL_ARMS ("seq,loo"), RL_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_read"))
import run_recon3 as R          # noqa: E402
import run_residual as RR       # noqa: E402

OUTPUT_DIR = HERE / "results" / os.environ.get('RL_TAG', '').lstrip('_')
R_TRAIN = int(os.environ.get("RL_RTRAIN", "4"))
ARMS = os.environ.get("RL_ARMS", "seq,loo").split(",")
RMAX = 6
RES_STOP = 0.02


def seq_learn(dic, x_hat, rounds):
    """Sequential greedy residual learning on one window."""
    cur = x_hat.copy()
    for _ in range(rounds):
        cn = float(np.linalg.norm(cur))
        if cn < RES_STOP:
            break
        cur_hat = cur / cn
        c = dic.W @ cur_hat
        i = int(np.argmax(c))
        ci = float(c[i])
        if ci <= 0.0:
            break
        w = dic.W[i].copy()
        tau = cur_hat - ci * w
        tn = float(np.linalg.norm(tau))
        if tn > R.EPS:
            th = dic.eta * ci
            dic.W[i] = w * np.cos(th) + (tau / tn) * np.sin(th)
        dic.win_counts[i] += 1
        cur = cur - (w @ cur) * w          # subtract what the matcher explained


def loo_learn(dic, x_hat):
    """Parallel leave-one-out residual learning (learning3.py, adapted:
    relu-active templates, geodesic step, same eta)."""
    W = dic.W
    s = W @ x_hat
    a = np.maximum(s, 0.0)
    active = a > 0
    if not active.any():
        return
    recon = a @ W
    shadow = recon[None, :] - a[:, None] * W
    sh_n = np.linalg.norm(shadow, axis=1, keepdims=True)
    shadow_hat = shadow / (sh_n + R.EPS)
    r = x_hat[None, :] - shadow_hat
    tau = r - np.sum(r * W, axis=1, keepdims=True) * W
    tn = np.linalg.norm(tau, axis=1, keepdims=True)
    tau_hat = np.where(tn > R.EPS, tau / (tn + R.EPS), 0.0)
    th = dic.eta * tn
    W_new = W * np.cos(th) + tau_hat * np.sin(th)
    W_new /= np.linalg.norm(W_new, axis=1, keepdims=True) + R.EPS
    dic.W = np.where(active[:, None], W_new, W)
    dic.win_counts[active] += 1


def encode_pixels_arm(x2d, dic, arm):
    """L1 pass during training: dense relu speech of the ORIGINAL window
    (communication unchanged); learning per arm."""
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
                dic.learn(p_hat, c)                    # Forgy adopt
            elif arm == "seq":
                seq_learn(dic, p_hat, R_TRAIN)
            else:
                loo_learn(dic, p_hat)
            out[gi, gj, :] = np.maximum(c, 0.0)
    return out


def train_arm(arm, Xtr):
    rng = np.random.default_rng(R.SEED)
    d1 = R.Layer(R.K1, R.W1_WIN * R.W1_WIN, R.ETA1, rng)
    d2 = R.Layer(R.K2, R.W2_WIN * R.W2_WIN * R.K1, R.ETA2, rng)
    d3 = R.Layer(R.K3, R.W3_WIN * R.W3_WIN * R.K2, R.ETA3, rng)
    for ep in range(R.EPOCHS):
        for i, x in enumerate(Xtr):
            m1 = encode_pixels_arm(x, d1, arm)
            m2 = R.encode_over(m1, d2, R.W2_WIN, R.W2_STR, R.G2, True)
            R.encode_over(m2, d3, R.W3_WIN, R.W3_STR, R.G3, True)
            if (i + 1) % 5000 == 0:
                print(f"[{arm}] epoch {ep + 1}/{R.EPOCHS} {i + 1}/{len(Xtr)}",
                      flush=True)
    return d1, d2, d3


def eval_arm(arm, d1, d2, d3, Xte, res):
    n = len(Xte)
    # residual read, R=1..6
    ms = np.zeros((RMAX, n))
    cs = np.zeros((RMAX, n))
    all_rn, all_mq = [], []
    for t in range(n):
        out, stats = RR.recon_l1_residual(Xte[t], d1.W, RMAX)
        for r in range(RMAX):
            ms[r, t] = float(((out[r] - Xte[t]) ** 2).mean())
            cs[r, t] = R.shape_corr(out[r], Xte[t])
        for rn, mq in stats:
            all_rn.append(rn)
            all_mq.append(mq)
    for r in range(RMAX):
        res[f"{arm}_L1_R{r + 1}_mse"] = round(float(ms[r].mean()), 5)
        res[f"{arm}_L1_R{r + 1}_corr"] = round(float(cs[r].mean()), 3)
    rn = np.array(all_rn)
    mq = np.array(all_mq)
    res[f"{arm}_residual_norm_per_round"] = [
        round(float(np.nanmean(rn[:, r])), 3) for r in range(RMAX)]
    res[f"{arm}_match_quality_per_round"] = [
        round(float(np.nanmean(mq[:, r])), 3) for r in range(RMAX)]
    res[f"{arm}_units_used"] = int((d1.win_counts > 0).sum())
    print(f"[{arm}] residual read done", flush=True)

    # graded read from L1 + deep rungs (Exp 1 forms, avg blending)
    M1, means, norms = R.encode_pixels_batch(Xte, d1)
    M2 = R.encode_over_batch(M1, d2, R.W2_WIN, R.W2_STR, R.G2)
    M3 = R.encode_over_batch(M2, d3, R.W3_WIN, R.W3_STR, R.G3)
    codes = {1: M1, 2: M2, 3: M3}
    for level in (1, 2, 3):
        gm, gc = [], []
        for t in range(n):
            rec = R.recon_from(level, codes[level][t], means[t], norms[t],
                               d1, d2, d3, top1=False)
            gm.append(float(((rec - Xte[t]) ** 2).mean()))
            gc.append(R.shape_corr(rec, Xte[t]))
        res[f"{arm}_L{level}_graded_mse"] = round(float(np.mean(gm)), 5)
        res[f"{arm}_L{level}_graded_corr"] = round(float(np.mean(gc)), 3)
    print(f"[{arm}] deep rungs done", flush=True)

    R.gallery([np.maximum(d1.W[i], 0).reshape(R.W1_WIN, R.W1_WIN)
               for i in range(R.K1)], None,
              OUTPUT_DIR / f"templates_L1_{arm}.png",
              f"L1 after {arm} residual learning", 16, cmap="inferno")
    imgs, titles = [], []
    for t in range(6):
        out, _ = RR.recon_l1_residual(Xte[t], d1.W, RMAX)
        imgs.append(Xte[t]); titles.append("orig")
        for r in (0, 1, 2, RMAX - 1):
            imgs.append(out[r]); titles.append(f"R={r + 1}")
    R.gallery(imgs, titles, OUTPUT_DIR / f"residual_ladder_{arm}.png",
              f"{arm}: orig | R=1 | R=2 | R=3 | R=6", 5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, _, Xte, _ = R.load()
    res = {"train_n": R.TRAIN_N, "test_n": len(Xte), "r_train": R_TRAIN,
           "arms": ARMS}
    for arm in ARMS:
        d1, d2, d3 = train_arm(arm, Xtr)
        np.savez(OUTPUT_DIR / f"weights_{arm}.npz",
                 W1=d1.W, W2=d2.W, W3=d3.W, wins1=d1.win_counts)
        eval_arm(arm, d1, d2, d3, Xte, res)
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
