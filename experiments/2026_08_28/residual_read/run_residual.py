"""Exp 3 — residual coding at L1, pure read, frozen dictionary.

The winner already won and learning already happened (this morning's
weights, untouched). At encode time each window gets extra READ rounds:
subtract what the winner explained, match the leftover against the same
frozen bank, add that second voice; repeat up to R rounds. Stop early if
nothing in the bank matches the leftover (best dot <= 0) or the leftover
is negligible.

Window estimate after R rounds (projection form):
    mean + norm * (c1*w1 + c2*w2 + ... + cR*wR)
where c_r is the raw projection of the current leftover on that round's
winner. Each round can only shrink the leftover. R=1 in the old
renormalized form reproduces Exp 2's best from-L1 arm (top-1 read,
confidence-weighted blend) as the bridge/verification.

Applies to the from-L1 rung only: residual rounds need the real window
to subtract from; the deeper rungs reconstruct without seeing the input.

Run:  .venv/bin/python experiments/2026_08_28/residual_read/run_residual.py
Env:  RC_TEST (shared with run_recon3), RS_TAG, RS_RMAX
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
import run_recon3 as R  # noqa: E402

OUTPUT_DIR = HERE / "results" / os.environ.get('RS_TAG', '').lstrip('_')
WEIGHTS = HERE.parent / "recon_ladder" / "results" / "weights.npz"
RMAX = int(os.environ.get("RS_RMAX", "6"))
CONF_EPS = 1e-6
RES_FLOOR = 0.02          # leftover this small (vs unit window) = done


def residual_rounds(x_hat, W1, rmax):
    """Greedy rounds on one centered-normalized window.

    Returns per-round cumulative estimates [rmax, dim], the confidence
    (round-1 correlation), per-round residual norms and per-round
    normalized match quality (how well round r's winner matched its own
    leftover, on the leftover's scale)."""
    dim = x_hat.shape[0]
    ests = np.zeros((rmax, dim))
    res_norms = np.full(rmax, np.nan)
    match_q = np.full(rmax, np.nan)
    est = np.zeros(dim)
    r = x_hat.copy()
    conf = 0.0
    for t in range(rmax):
        c = W1 @ r
        i = int(np.argmax(c))
        ci = float(c[i])
        rn_before = float(np.linalg.norm(r))
        if ci <= 0.0 or rn_before < RES_FLOOR:
            ests[t:] = est
            break
        if t == 0:
            conf = ci
        match_q[t] = ci / rn_before
        est = est + ci * W1[i]
        r = r - ci * W1[i]
        ests[t] = est
        res_norms[t] = float(np.linalg.norm(r))
    else:
        return ests, conf, res_norms, match_q
    return ests, conf, res_norms, match_q


def recon_l1_residual(x2d, W1, rmax, renorm_r1=False):
    """From-L1 reconstructions for R = 1..rmax on one image,
    confidence-weighted blending (Exp 2's winner for sparse reads)."""
    num = np.zeros((rmax, R.SIDE, R.SIDE))
    den = np.zeros((R.SIDE, R.SIDE))
    stats = []
    for gi in range(R.G1):
        for gj in range(R.G1):
            p = x2d[gi * R.W1_STR:gi * R.W1_STR + R.W1_WIN,
                    gj * R.W1_STR:gj * R.W1_STR + R.W1_WIN].ravel()
            mean = p.mean()
            p_hat, n = R.center_norm(p)
            if n < R.NORM_FLOOR:
                wins = np.repeat(np.full((1, R.W1_WIN, R.W1_WIN), mean),
                                 rmax, axis=0)
                conf = CONF_EPS
            else:
                ests, conf, res_norms, match_q = residual_rounds(
                    p_hat, W1, rmax)
                stats.append((res_norms, match_q))
                conf = max(conf, CONF_EPS)
                if renorm_r1:            # old form, R=1 bridge arm only
                    e = ests[0]
                    en = float(np.linalg.norm(e))
                    if en > R.EPS:
                        ests = np.repeat((e / en)[None], rmax, axis=0)
                wins = (mean + n * ests).reshape(rmax, R.W1_WIN, R.W1_WIN)
            r0, c0 = gi * R.W1_STR, gj * R.W1_STR
            num[:, r0:r0 + R.W1_WIN, c0:c0 + R.W1_WIN] += conf * wins
            den[r0:r0 + R.W1_WIN, c0:c0 + R.W1_WIN] += conf
    out = num / np.maximum(den, CONF_EPS)[None]
    return out, stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    W1 = np.load(WEIGHTS)["W1"]
    _, _, Xte, _ = R.load()
    n = len(Xte)

    res = {"test_n": n, "rmax": RMAX, "res_floor": RES_FLOOR}

    # bridge arm: R=1 with the old renormalized direction (= Exp 2 best)
    ms, cs = [], []
    for t in range(n):
        out, _ = recon_l1_residual(Xte[t], W1, 1, renorm_r1=True)
        ms.append(float(((out[0] - Xte[t]) ** 2).mean()))
        cs.append(R.shape_corr(out[0], Xte[t]))
    res["L1_R1_renorm_mse"] = round(float(np.mean(ms)), 5)
    res["L1_R1_renorm_corr"] = round(float(np.mean(cs)), 3)
    print("bridge arm done", flush=True)

    ms = np.zeros((RMAX, n))
    cs = np.zeros((RMAX, n))
    all_res_norms, all_match_q = [], []
    for t in range(n):
        out, stats = recon_l1_residual(Xte[t], W1, RMAX)
        for r in range(RMAX):
            ms[r, t] = float(((out[r] - Xte[t]) ** 2).mean())
            cs[r, t] = R.shape_corr(out[r], Xte[t])
        for rn, mq in stats:
            all_res_norms.append(rn)
            all_match_q.append(mq)
        if (t + 1) % 1000 == 0:
            print(f"{t + 1}/{n}", flush=True)
    for r in range(RMAX):
        res[f"L1_R{r + 1}_mse"] = round(float(ms[r].mean()), 5)
        res[f"L1_R{r + 1}_corr"] = round(float(cs[r].mean()), 3)
    rn = np.array(all_res_norms)
    mq = np.array(all_match_q)
    res["mean_residual_norm_per_round"] = [
        round(float(np.nanmean(rn[:, r])), 3) for r in range(RMAX)]
    res["mean_match_quality_per_round"] = [
        round(float(np.nanmean(mq[:, r])), 3) for r in range(RMAX)]
    res["frac_windows_reaching_round"] = [
        round(float(np.mean(~np.isnan(mq[:, r]))), 3) for r in range(RMAX)]

    imgs, titles = [], []
    for t in range(8):
        out, _ = recon_l1_residual(Xte[t], W1, RMAX)
        imgs.append(Xte[t]); titles.append("orig")
        for r in (0, 1, 2, RMAX - 1):
            imgs.append(out[r]); titles.append(f"R={r + 1}")
    R.gallery(imgs, titles, OUTPUT_DIR / "residual_ladder.png",
              "From L1 with residual rounds: orig | R=1 | R=2 | R=3 | R=6", 5)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
