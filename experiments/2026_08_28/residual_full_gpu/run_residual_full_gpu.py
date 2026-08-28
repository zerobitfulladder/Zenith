"""Exp 5 — FULL-STACK sequential residual learning + residual reads, on GPU.

Residual learning extends from L1 to every layer: each layer's learning
view stays what it was (L1 = dense centered window, L2/L3 = per-position
top-1 skeleton of the block), and up to R_TRAIN sequential rounds run on
that view — the round's winner steps toward the current leftover, its
projection is subtracted, the next winner competes over the remainder.
Round 1 with R_TRAIN=1 is exactly the old rule, so plain layers are the
rounds=1 special case. Upward speech everywhere stays dense relu of the
ORIGINAL window.

Residual READS extend to the deep rungs: reconstructing from L2/L3, each
position's centered block is residual-coded against that layer's bank
(few voices, each covering different content) instead of the old dense
sum (blurry) or top-1 (lossy) — plus the block's own mean/norm side
channels restored at expansion, which Exp 1's expand never had. Controls
decompose the gains: parity read (Exp 1's exact expand) / side-channel
graded / side-channel residual.

GPU: CuPy mini-batch per the verified run_gpu_minibatch.py pattern (f32,
one vectorized geodesic step per template per batch toward the normalized
mean of its targets, theta = clip(eta*sum(c), 0.3)).

Arms:
  exp4seq  no training — GPU eval of Exp 4's CPU weights_seq.npz
           (eval-parity check against known CPU numbers)
  l1only   GPU training, residual rounds at L1 only (GPU training parity
           check against Exp 4 seq)
  full     residual rounds at L1+L2+L3 (the new thing)

Run:  .venv/bin/python experiments/2026_08_28/residual_full_gpu/run_residual_full_gpu.py
Env:  RF_XP=cupy|numpy, RF_TRAIN, RF_TEST, RF_B, RF_RTRAIN, RF_ARMS, RF_TAG
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
import run_recon3 as R  # noqa: E402  (geometry, gallery, CPU decode shims)

XP_NAME = os.environ.get("RF_XP", "cupy")
if XP_NAME == "cupy":
    import cupy as xp
    import cupyx

    def scatter_add(t, i, v):
        cupyx.scatter_add(t, i, v)
else:
    xp = np

    def scatter_add(t, i, v):
        np.add.at(t, i, v)

DTYPE = xp.float32
OUTPUT_DIR = HERE / "results" / os.environ.get('RF_TAG', '').lstrip('_')
TRAIN_N = int(os.environ.get("RF_TRAIN", "55000"))
TEST_N = int(os.environ.get("RF_TEST", "5000"))
B = int(os.environ.get("RF_B", "128"))
R_TRAIN = int(os.environ.get("RF_RTRAIN", "4"))
RMAX1, RMAX_DEEP = 6, 4
RES_STOP, THETA_CAP, EPS = 0.02, 0.3, 1e-9
ARMS = os.environ.get("RF_ARMS", "exp4seq,l1only,full").split(",")
EB = 500                                   # eval chunk of images
SEED = R.SEED

POS1 = [(r, c) for r in range(0, R.SIDE - R.W1_WIN + 1, R.W1_STR)
        for c in range(0, R.SIDE - R.W1_WIN + 1, R.W1_STR)]
POS2 = [(r, c) for r in range(0, R.G1 - R.W2_WIN + 1, R.W2_STR)
        for c in range(0, R.G1 - R.W2_WIN + 1, R.W2_STR)]
POS3 = [(r, c) for r in range(0, R.G2 - R.W3_WIN + 1, R.W3_STR)
        for c in range(0, R.G2 - R.W3_WIN + 1, R.W3_STR)]


class Dict:
    """Bank on xp; bootstrap-adopt then batched geodesic updates."""

    def __init__(self, k, dim, eta):
        self.k, self.dim, self.eta = k, dim, eta
        self.W = xp.zeros((k, dim), dtype=DTYPE)
        self.n_boot = 0
        self.win_counts = xp.zeros(k, dtype=xp.int64)
        self._noise = np.random.default_rng(SEED)

    def bootstrap(self, V_hat):
        need = self.k - self.n_boot
        if need <= 0 or len(V_hat) == 0:
            return
        take = min(need, len(V_hat))
        w = V_hat[:take] + xp.asarray(
            ((0.05 / np.sqrt(self.dim))
             * self._noise.standard_normal((take, self.dim))).astype(np.float32))
        w = w - w.mean(axis=1, keepdims=True)
        w = w / (xp.linalg.norm(w, axis=1, keepdims=True) + EPS)
        self.W[self.n_boot:self.n_boot + take] = w
        self.win_counts[self.n_boot:self.n_boot + take] += 1
        self.n_boot += take

    def update(self, T_hat, winners, cvals):
        if len(T_hat) == 0:
            return
        S = xp.zeros((self.k, self.dim), dtype=DTYPE)
        Csum = xp.zeros(self.k, dtype=DTYPE)
        scatter_add(S, winners, T_hat)
        scatter_add(Csum, winners, cvals)
        hit = Csum > 0
        if not bool(hit.any()):
            return
        mu = S[hit]
        mu = mu / (xp.linalg.norm(mu, axis=1, keepdims=True) + EPS)
        w = self.W[hit]
        cw = xp.sum(w * mu, axis=1, keepdims=True)
        tau = mu - cw * w
        tn = xp.linalg.norm(tau, axis=1, keepdims=True)
        theta = xp.minimum(self.eta * Csum[hit], THETA_CAP)[:, None]
        w_new = xp.where(tn > EPS,
                         w * xp.cos(theta) + (tau / xp.maximum(tn, EPS)) * xp.sin(theta),
                         w)
        w_new = w_new - w_new.mean(axis=1, keepdims=True)
        w_new = w_new / (xp.linalg.norm(w_new, axis=1, keepdims=True) + EPS)
        self.W[hit] = w_new
        self.win_counts += (Csum > 0).astype(xp.int64)


def center_norm_rows(V):
    """Rows -> (centered+normalized, mean, norm, valid)."""
    mean = V.mean(axis=1, keepdims=True)
    V = V - mean
    n = xp.linalg.norm(V, axis=1)
    ok = n > R.NORM_FLOOR
    V = V / xp.maximum(n, EPS)[:, None]
    return V, mean[:, 0], n * ok, ok


def window_stack(maps, positions, win):
    parts = [maps[:, r:r + win, c:c + win, :].reshape(len(maps), -1)
             for r, c in positions]
    return xp.stack(parts, axis=1)


def skeleton_rows(blocks):
    """(M, P, Kprev) -> per-position top-1 view, flattened."""
    am = xp.argmax(blocks, axis=2)
    vals = xp.take_along_axis(blocks, am[:, :, None], axis=2)
    sk = xp.zeros_like(blocks)
    xp.put_along_axis(sk, am[:, :, None], xp.maximum(vals, 0.0), axis=2)
    return sk.reshape(len(blocks), -1)


def residual_learn(dic, T_hat, ok, rounds):
    """Sequential residual learning, batched: collect (target, winner,
    cosine) over rounds on the learning view, one update per batch."""
    cur = T_hat * ok[:, None]
    targets, winners, cvals = [], [], []
    for _ in range(rounds):
        rn = xp.linalg.norm(cur, axis=1)
        alive = ok & (rn > RES_STOP)
        cur_hat = cur / xp.maximum(rn, EPS)[:, None]
        C = cur_hat @ dic.W.T
        wnr = xp.argmax(C, axis=1)
        cm = xp.max(C, axis=1)
        alive = alive & (cm > 0)
        if not bool(alive.any()):
            break
        targets.append(cur_hat[alive])
        winners.append(wnr[alive])
        cvals.append(cm[alive])
        proj = xp.sum(cur * dic.W[wnr], axis=1) * alive
        cur = cur - proj[:, None] * dic.W[wnr]
    if targets:
        dic.update(xp.concatenate(targets), xp.concatenate(winners),
                   xp.concatenate(cvals))


def residual_code(V_hat, W, rmax):
    """Greedy residual coding of unit rows (Exp 3's read, batched):
    raw-dot coefficients on the unnormalized leftover.
    Returns winners (M, rmax), coefs (M, rmax), matchq (M, rmax)."""
    M = len(V_hat)
    cur = V_hat.copy()
    winners = xp.zeros((M, rmax), dtype=xp.int64)
    coefs = xp.zeros((M, rmax), dtype=DTYPE)
    matchq = xp.zeros((M, rmax), dtype=DTYPE)
    for t in range(rmax):
        rn = xp.linalg.norm(cur, axis=1)
        C = cur @ W.T
        wnr = xp.argmax(C, axis=1)
        cm = xp.max(C, axis=1)
        alive = (cm > 0) & (rn > RES_STOP)
        cm = cm * alive
        winners[:, t] = wnr
        coefs[:, t] = cm
        matchq[:, t] = xp.where(alive, cm / xp.maximum(rn, EPS), xp.nan)
        cur = cur - cm[:, None] * W[wnr]
    return winners, coefs, matchq


def level_pass(prev_maps, dic, positions, win, kprev, g_out, view,
               learning, rounds):
    N = len(prev_maps)
    Vf = window_stack(prev_maps, positions, win)
    V_hat, _, _, ok = center_norm_rows(Vf.reshape(N * len(positions), -1))
    if learning and dic.n_boot < dic.k:
        dic.bootstrap(V_hat[ok])
    C = V_hat @ dic.W.T
    code = xp.maximum(C, 0.0) * ok[:, None]
    if learning and dic.n_boot >= dic.k:
        if view == "skeleton":
            blocks = Vf.reshape(N * len(positions), win * win, kprev)
            T_hat, _, _, ok_t = center_norm_rows(skeleton_rows(blocks))
        else:
            T_hat, ok_t = V_hat, ok
        residual_learn(dic, T_hat, ok_t, rounds)
    return code.reshape(N, g_out, g_out, dic.k)


def train_arm(arm, Xtr_x):
    d1 = Dict(R.K1, R.W1_WIN * R.W1_WIN, R.ETA1)
    d2 = Dict(R.K2, R.W2_WIN * R.W2_WIN * R.K1, R.ETA2)
    d3 = Dict(R.K3, R.W3_WIN * R.W3_WIN * R.K2, R.ETA3)
    r_deep = R_TRAIN if arm == "full" else 1
    t0 = time.time()
    for s in range(0, len(Xtr_x), B):
        xb = Xtr_x[s:s + B]
        m1 = level_pass(xb[..., None], d1, POS1, R.W1_WIN, 1, R.G1,
                        "dense", True, R_TRAIN)
        m2 = level_pass(m1, d2, POS2, R.W2_WIN, R.K1, R.G2,
                        "skeleton", True, r_deep)
        level_pass(m2, d3, POS3, R.W3_WIN, R.K2, R.G3,
                   "skeleton", True, r_deep)
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    print(f"[{arm}] trained in {time.time() - t0:.1f}s", flush=True)
    return d1, d2, d3


# ------------------------------------------------------------------- eval ---

def _corr_rows(A, T):
    A = A.reshape(len(A), -1)
    T = T.reshape(len(T), -1)
    A = A - A.mean(axis=1, keepdims=True)
    T = T - T.mean(axis=1, keepdims=True)
    A = A / (xp.linalg.norm(A, axis=1, keepdims=True) + EPS)
    T = T / (xp.linalg.norm(T, axis=1, keepdims=True) + EPS)
    return xp.sum(A * T, axis=1)


def decode_pixels_graded(code1, means1, norms1, W1):
    """Estimated L1 code -> pixels: per-window graded direction + side
    channels, plain-average overlap (Exp 1's decode, batched)."""
    N = len(code1)
    num = xp.zeros((N, R.SIDE, R.SIDE), dtype=DTYPE)
    den = xp.zeros((R.SIDE, R.SIDE), dtype=DTYPE)
    for p, (r, c) in enumerate(POS1):
        seg = code1[:, p, :]
        v = seg @ W1
        v = v - v.mean(axis=1, keepdims=True)
        vn = xp.linalg.norm(v, axis=1)
        okp = (seg.max(axis=1) > 0) & (vn > EPS)
        v = v / xp.maximum(vn, EPS)[:, None]
        win = xp.where(okp[:, None],
                       means1[:, p, None] + norms1[:, p, None] * v,
                       means1[:, p, None] * xp.ones_like(v))
        num[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += win.reshape(N, R.W1_WIN, R.W1_WIN)
        den[r:r + R.W1_WIN, c:c + R.W1_WIN] += 1.0
    return num / xp.maximum(den, 1.0)[None]


def recon_l1_residual_gpu(Xh, means1, norms1, ok1, W1, rmax):
    """From-L1 residual read: per-round cumulative recon images."""
    N, P = Xh.shape[:2]
    flat = Xh.reshape(N * P, -1)
    winners, coefs, matchq = residual_code(flat, W1, rmax)
    conf = xp.maximum(coefs[:, 0], 1e-6).reshape(N, P)
    conf = xp.where(ok1, conf, 1e-6)
    est = xp.zeros((N * P, W1.shape[1]), dtype=DTYPE)
    outs = []
    num = xp.zeros((N, R.SIDE, R.SIDE), dtype=DTYPE)
    den = xp.zeros((N, R.SIDE, R.SIDE), dtype=DTYPE)
    for t in range(rmax):
        est = est + coefs[:, t, None] * W1[winners[:, t]]
        e = est.reshape(N, P, -1)
        num[:] = 0.0
        den[:] = 0.0
        for p, (r, c) in enumerate(POS1):
            win = xp.where(ok1[:, p, None],
                           means1[:, p, None] + norms1[:, p, None] * e[:, p],
                           means1[:, p, None] * xp.ones((N, W1.shape[1]), dtype=DTYPE))
            w2d = win.reshape(N, R.W1_WIN, R.W1_WIN)
            num[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += conf[:, p, None, None] * w2d
            den[:, r:r + R.W1_WIN, c:c + R.W1_WIN] += conf[:, p, None, None]
        outs.append(num / xp.maximum(den, 1e-6))
    return outs, matchq


def expand_level(voices, positions, win, g_prev, kprev, W, mode,
                 means=None, norms=None):
    """Blocks' estimates -> previous-level code map.
    mode 'parity':  seg @ W, plain average, no side channels (Exp 1).
    mode 'sc':      side-channel restore (mean + norm * unit direction),
                    confidence-weighted overlap.
    voices: for graded arms (N, P, K) relu code; for residual arms a
    tuple (winners, coefs) already turned into est rows (N, P, D)."""
    N = voices[0].shape[0] if isinstance(voices, tuple) else voices.shape[0]
    acc = xp.zeros((N, g_prev, g_prev, kprev), dtype=DTYPE)
    den = xp.zeros((N, g_prev, g_prev), dtype=DTYPE)
    for p, (r, c) in enumerate(positions):
        if isinstance(voices, tuple):                      # residual est rows
            est, conf = voices
            contrib = est[:, p]
            cw = conf[:, p]
        else:
            seg = voices[:, p]
            contrib = seg @ W
            cw = xp.maximum(seg.max(axis=1), 1e-6)
        if mode == "sc":
            d = contrib - contrib.mean(axis=1, keepdims=True) \
                if not isinstance(voices, tuple) else contrib
            if not isinstance(voices, tuple):              # graded: renorm dir
                dn = xp.linalg.norm(d, axis=1, keepdims=True)
                d = d / xp.maximum(dn, EPS)
            contrib = means[:, p, None] + norms[:, p, None] * d
            wgt = cw[:, None, None, None]
        else:
            wgt = xp.ones((N, 1, 1, 1), dtype=DTYPE)
        blk = contrib.reshape(N, win, win, kprev)
        acc[:, r:r + win, c:c + win, :] += wgt * blk
        den[:, r:r + win, c:c + win] += wgt[:, :, :, 0]
    acc = acc / xp.maximum(den, 1e-6)[..., None]
    return xp.maximum(acc, 0.0)


def eval_arm(arm, W1, W2, W3, Xte_x, res):
    N = len(Xte_x)
    mets = {}

    def add(name, imgs_by_key, sl, target):
        for key, img in imgs_by_key.items():
            mets.setdefault(key, {"mse": [], "corr": []})
            mets[key]["mse"].append(
                xp.asnumpy(((img - target) ** 2).mean(axis=(1, 2)))
                if XP_NAME == "cupy" else ((img - target) ** 2).mean(axis=(1, 2)))
            cc = _corr_rows(img, target)
            mets[key]["corr"].append(xp.asnumpy(cc) if XP_NAME == "cupy" else cc)

    mq1_all, mq2_all, mq3_all = [], [], []
    keep_gallery = {}
    for s in range(0, N, EB):
        xb = Xte_x[s:s + EB]
        n = len(xb)
        P1 = window_stack(xb[..., None], POS1, R.W1_WIN).reshape(n, len(POS1), -1)
        Xh, means1, norms1, ok1 = center_norm_rows(P1.reshape(n * len(POS1), -1))
        Xh = Xh.reshape(n, len(POS1), -1)
        means1 = means1.reshape(n, len(POS1))
        norms1 = norms1.reshape(n, len(POS1))
        ok1 = ok1.reshape(n, len(POS1))
        C1 = xp.maximum(Xh.reshape(n * len(POS1), -1) @ W1.T, 0.0) \
            * ok1.reshape(-1)[:, None]
        C1 = C1.reshape(n, len(POS1), -1)
        out = {}

        # from-L1: residual read R=1..RMAX1 + graded baseline
        recons, mq1 = recon_l1_residual_gpu(Xh, means1, norms1, ok1, W1, RMAX1)
        mq1_all.append(mq1)
        for t, img in enumerate(recons):
            out[f"L1_res_R{t + 1}"] = img
        out["L1_graded"] = decode_pixels_graded(C1, means1, norms1, W1)

        # from-L2
        M1map = C1.reshape(n, R.G1, R.G1, R.K1)
        B2 = window_stack(M1map, POS2, R.W2_WIN).reshape(n * len(POS2), -1)
        Bh2, bm2, bn2, ok2 = center_norm_rows(B2)
        seg2 = (xp.maximum(Bh2 @ W2.T, 0.0) * ok2[:, None]).reshape(n, len(POS2), -1)
        bm2 = bm2.reshape(n, len(POS2))
        bn2 = bn2.reshape(n, len(POS2))
        c1_par = expand_level(seg2, POS2, R.W2_WIN, R.G1, R.K1, W2, "parity")
        out["L2_parity"] = decode_pixels_graded(
            c1_par.reshape(n, len(POS1), -1), means1, norms1, W1)
        c1_sc = expand_level(seg2, POS2, R.W2_WIN, R.G1, R.K1, W2, "sc",
                             bm2, bn2)
        out["L2_sc_graded"] = decode_pixels_graded(
            c1_sc.reshape(n, len(POS1), -1), means1, norms1, W1)
        w2r, cf2, mq2 = residual_code(Bh2, W2, RMAX_DEEP)
        mq2_all.append(mq2)
        est2 = xp.zeros((n * len(POS2), W2.shape[1]), dtype=DTYPE)
        for t in range(RMAX_DEEP):
            est2 = est2 + cf2[:, t, None] * W2[w2r[:, t]]
            conf2 = xp.maximum(cf2[:, 0], 1e-6).reshape(n, len(POS2))
            c1_res = expand_level(
                (est2.reshape(n, len(POS2), -1), conf2),
                POS2, R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)
            out[f"L2_res_R{t + 1}"] = decode_pixels_graded(
                c1_res.reshape(n, len(POS1), -1), means1, norms1, W1)

        # from-L3
        M2map = seg2.reshape(n, R.G2, R.G2, R.K2)
        B3 = window_stack(M2map, POS3, R.W3_WIN).reshape(n * len(POS3), -1)
        Bh3, bm3, bn3, ok3 = center_norm_rows(B3)
        seg3 = (xp.maximum(Bh3 @ W3.T, 0.0) * ok3[:, None]).reshape(n, len(POS3), -1)
        bm3 = bm3.reshape(n, len(POS3))
        bn3 = bn3.reshape(n, len(POS3))
        c2_par = expand_level(seg3, POS3, R.W3_WIN, R.G2, R.K2, W3, "parity")
        c1_par3 = expand_level(c2_par.reshape(n, len(POS2), -1), POS2,
                               R.W2_WIN, R.G1, R.K1, W2, "parity")
        out["L3_parity"] = decode_pixels_graded(
            c1_par3.reshape(n, len(POS1), -1), means1, norms1, W1)
        w3r, cf3, mq3 = residual_code(Bh3, W3, RMAX_DEEP)
        mq3_all.append(mq3)
        est3 = xp.zeros((n * len(POS3), W3.shape[1]), dtype=DTYPE)
        for t in range(RMAX_DEEP):
            est3 = est3 + cf3[:, t, None] * W3[w3r[:, t]]
        conf3 = xp.maximum(cf3[:, 0], 1e-6).reshape(n, len(POS3))
        c2_res = expand_level((est3.reshape(n, len(POS3), -1), conf3),
                              POS3, R.W3_WIN, R.G2, R.K2, W3, "sc", bm3, bn3)
        c1_res3 = expand_level(c2_res.reshape(n, len(POS2), -1), POS2,
                               R.W2_WIN, R.G1, R.K1, W2, "sc", bm2, bn2)
        out["L3_res"] = decode_pixels_graded(
            c1_res3.reshape(n, len(POS1), -1), means1, norms1, W1)

        add(arm, out, s, xb)
        if s == 0:
            keep_gallery = {k: (xp.asnumpy(v[:8]) if XP_NAME == "cupy"
                                else v[:8].copy()) for k, v in out.items()}
            keep_gallery["orig"] = (xp.asnumpy(xb[:8]) if XP_NAME == "cupy"
                                    else xb[:8].copy())

    for key, d in mets.items():
        res[f"{arm}_{key}_mse"] = round(float(np.concatenate(d["mse"]).mean()), 5)
        res[f"{arm}_{key}_corr"] = round(float(np.concatenate(d["corr"]).mean()), 3)
    for nm, mq in (("L1", mq1_all), ("L2", mq2_all), ("L3", mq3_all)):
        m = xp.concatenate(mq)
        m = xp.asnumpy(m) if XP_NAME == "cupy" else m
        res[f"{arm}_matchq_{nm}"] = [round(float(np.nanmean(m[:, t])), 3)
                                     for t in range(m.shape[1])]
    return keep_gallery


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, _, Xte, _ = R.load()
    Xtr_x = xp.asarray(Xtr[:TRAIN_N], dtype=DTYPE)
    Xte_x = xp.asarray(Xte[:TEST_N], dtype=DTYPE)
    res = {"xp": XP_NAME, "train_n": TRAIN_N, "test_n": int(len(Xte_x)),
           "r_train": R_TRAIN, "arms": ARMS}

    for arm in ARMS:
        if arm == "exp4seq":
            w = np.load(HERE.parent / "residual_learning" / "results" / "weights_seq.npz")
            W1 = xp.asarray(w["W1"], dtype=DTYPE)
            W2 = xp.asarray(w["W2"], dtype=DTYPE)
            W3 = xp.asarray(w["W3"], dtype=DTYPE)
        else:
            d1, d2, d3 = train_arm(arm, Xtr_x)
            W1, W2, W3 = d1.W, d2.W, d3.W
            res[f"{arm}_units_used"] = [
                int((d.win_counts > 0).sum()) for d in (d1, d2, d3)]
            np.savez(OUTPUT_DIR / f"weights_{arm}.npz",
                     **{k: (v.get() if XP_NAME == "cupy" else v)
                        for k, v in (("W1", W1), ("W2", W2), ("W3", W3))})
        t0 = time.time()
        gal = eval_arm(arm, W1, W2, W3, Xte_x, res)
        print(f"[{arm}] eval {time.time() - t0:.1f}s", flush=True)

        cols = ["orig", "L1_res_R6", "L2_sc_graded", f"L2_res_R{RMAX_DEEP}",
                "L3_parity", "L3_res"]
        imgs, titles = [], []
        for i in range(8):
            for cname in cols:
                imgs.append(gal[cname][i])
                titles.append(cname if i == 0 else None)
        R.gallery(imgs, titles, OUTPUT_DIR / f"ladder_{arm}.png",
                  f"{arm}: orig | L1 res R6 | L2 sc | L2 res | L3 parity | L3 res",
                  len(cols))
        if arm == "full":
            W1n = W1.get() if XP_NAME == "cupy" else np.asarray(W1)
            l1 = R.Layer(R.K1, W1n.shape[1], 0.0, np.random.default_rng(0))
            l2 = R.Layer(R.K2, W2.shape[1], 0.0, np.random.default_rng(0))
            l3 = R.Layer(R.K3, W3.shape[1], 0.0, np.random.default_rng(0))
            l1.W = W1n.astype(np.float64)
            l2.W = (W2.get() if XP_NAME == "cupy" else np.asarray(W2)).astype(np.float64)
            l3.W = (W3.get() if XP_NAME == "cupy" else np.asarray(W3)).astype(np.float64)
            for lvl, kk, ncol in ((1, R.K1, 16), (2, R.K2, 16), (3, R.K3, 20)):
                R.gallery([R.unit_to_pixels(lvl, u, l1, l2, l3)
                           for u in range(kk)], None,
                          OUTPUT_DIR / f"templates_L{lvl}_full.png",
                          f"L{lvl} after full-stack residual learning ({kk})",
                          ncol, cmap="inferno")
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
