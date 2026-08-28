"""Reconstruction-first 3-layer convolutional stack. No labels, no top memory.

The objective changes today: RECONSTRUCTION ERROR is the target. The network
is the classic stack, untouched; everything downstream of the code is rebuilt
fresh and simple (no squelch, no feathering, no confidence weighting — the
old renderer was tuned to make label-drawings pretty, not to minimize error).

  L1   8x8 pixel windows, stride 2  -> 11x11 grid, K=64   learns DENSE
  L2   3x3 windows over L1, stride 2 ->  5x5  grid, K=64   learns SKELETON
  L3   3x3 windows over L2, stride 1 ->  3x3  grid, K=100  learns SKELETON

Every layer: mean-center + L2-normalize the window, correlate against the
bank, top-1 winner rotates toward the input (geodesic step). Output is dense
positive correlations (relu). Only the top-1 template learns.

Side channels (bookkeeping only, never enter matching or learning): each L1
window's mean and norm are stored at encode time and re-applied at decode,
so the reconstruction lives in real pixel space and pixel MSE is honest.

The instrument is a LADDER: reconstruct from the L1 code, the L2 code, the
L3 code — each rung prices that layer's bottleneck — in two read modes
(graded = dense relu all the way down; top1 = hardened per position).

Run:  .venv/bin/python experiments/2026_08_28/recon_ladder/run_recon3.py
Env:  RC_TRAIN, RC_EPOCHS, RC_K1, RC_K2, RC_K3, RC_TAG
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (Path(__file__).resolve().parent / "results"
              / os.environ.get('RC_TAG', '').lstrip('_'))

SIDE = 28
W1_WIN, W1_STR = 8, 2
W2_WIN, W2_STR = 3, 2
W3_WIN, W3_STR = 3, 1
K1 = int(os.environ.get("RC_K1", "64"))
K2 = int(os.environ.get("RC_K2", "64"))
K3 = int(os.environ.get("RC_K3", "100"))
ETA1, ETA2, ETA3 = 0.02, 0.03, 0.03
EPS, NORM_FLOOR = 1e-8, 0.15
EPOCHS = int(os.environ.get("RC_EPOCHS", "1"))
TRAIN_N = int(os.environ.get("RC_TRAIN", "55000"))
TEST_N = int(os.environ.get("RC_TEST", "5000"))
SEED = 42

G1 = (SIDE - W1_WIN) // W1_STR + 1        # 11
G2 = (G1 - W2_WIN) // W2_STR + 1          # 5
G3 = (G2 - W3_WIN) // W3_STR + 1          # 3


def center_norm(v):
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return (np.zeros_like(v), 0.0) if n < EPS else (v / n, n)


class Layer:
    """Top-1 WTA spherical quantizer; bootstrap-adopt then geodesic rotate."""

    def __init__(self, k, dim, eta, rng):
        self.k, self.dim, self.eta, self.rng = k, dim, eta, rng
        self.W = np.zeros((k, dim))
        self.n_boot = 0
        self.win_counts = np.zeros(k, dtype=np.int64)

    def forward(self, x_hat):
        return self.W @ x_hat

    def learn(self, x_hat, c):
        if self.n_boot < self.k:                      # adopt (Forgy init)
            w = x_hat + 0.01 * self.rng.standard_normal(self.dim)
            w -= w.mean()
            w /= np.linalg.norm(w) + EPS
            self.W[self.n_boot] = w
            self.win_counts[self.n_boot] += 1
            self.n_boot += 1
            return
        i = int(np.argmax(c))
        c_i = float(c[i])
        if c_i <= 0.0:
            return
        w = self.W[i]
        tau = x_hat - c_i * w
        tn = float(np.linalg.norm(tau))
        if tn > EPS:
            th = self.eta * c_i
            self.W[i] = w * np.cos(th) + (tau / tn) * np.sin(th)
        self.win_counts[i] += 1


# ---------------------------------------------------------------- upward ---

def _pp_top1(block):
    """Per-position top-1 skeleton of a code window (the learning view)."""
    out = np.zeros_like(block)
    for a in range(block.shape[0]):
        for b in range(block.shape[1]):
            seg = block[a, b]
            m = int(np.argmax(seg))
            if seg[m] > 0:
                out[a, b, m] = seg[m]
    return out


def encode_pixels(x2d, dic, learning):
    """L1 code + the side channels (per-window mean and norm)."""
    out = np.zeros((G1, G1, dic.k))
    means = np.zeros((G1, G1))
    norms = np.zeros((G1, G1))
    for gi in range(G1):
        for gj in range(G1):
            p = x2d[gi * W1_STR:gi * W1_STR + W1_WIN,
                    gj * W1_STR:gj * W1_STR + W1_WIN].ravel()
            means[gi, gj] = p.mean()
            p_hat, n = center_norm(p)
            if n < NORM_FLOOR:
                continue
            norms[gi, gj] = n
            c = dic.forward(p_hat)
            if learning:
                dic.learn(p_hat, c)              # L1 learns dense
            out[gi, gj, :] = np.maximum(c, 0.0)  # dense speech
    return out, means, norms


def encode_over(prev, dic, win, stride, g_out, learning):
    out = np.zeros((g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            block = prev[gi * stride:gi * stride + win,
                         gj * stride:gj * stride + win, :]
            v_hat, n = center_norm(block.ravel())
            if n < NORM_FLOOR:
                continue
            c = dic.forward(v_hat)
            if learning:                          # learn on the skeleton
                s_hat, sn = center_norm(_pp_top1(block).ravel())
                if sn > NORM_FLOOR:
                    dic.learn(s_hat, dic.forward(s_hat))
            out[gi, gj, :] = np.maximum(c, 0.0)   # speak dense
    return out


def encode_pixels_batch(X, dic):
    out = np.zeros((len(X), G1, G1, dic.k))
    means = np.zeros((len(X), G1, G1))
    norms = np.zeros((len(X), G1, G1))
    for gi in range(G1):
        for gj in range(G1):
            P = X[:, gi * W1_STR:gi * W1_STR + W1_WIN,
                  gj * W1_STR:gj * W1_STR + W1_WIN].reshape(len(X), -1).copy()
            means[:, gi, gj] = P.mean(axis=1)
            P -= P.mean(axis=1, keepdims=True)
            nr = np.linalg.norm(P, axis=1)
            ok = nr > NORM_FLOOR
            norms[:, gi, gj] = nr * ok
            P[ok] /= nr[ok, None]
            out[:, gi, gj, :] = np.maximum(P @ dic.W.T, 0.0) * ok[:, None]
    return out, means, norms


def encode_over_batch(prev, dic, win, stride, g_out):
    out = np.zeros((len(prev), g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            V = prev[:, gi * stride:gi * stride + win,
                     gj * stride:gj * stride + win, :].reshape(len(prev), -1).copy()
            V -= V.mean(axis=1, keepdims=True)
            nr = np.linalg.norm(V, axis=1)
            ok = nr > NORM_FLOOR
            V[ok] /= nr[ok, None]
            out[:, gi, gj, :] = np.maximum(V @ dic.W.T, 0.0) * ok[:, None]
    return out


# -------------------------------------------------------------- downward ---
# Fresh and simple: no squelch, no feathering, no confidence weighting.
# Every window contributes its full estimate; overlaps are plain averages.

def harden(map3d):
    """Per position keep only the strongest entry (top-1 read)."""
    out = np.zeros_like(map3d)
    for gi in range(map3d.shape[0]):
        for gj in range(map3d.shape[1]):
            seg = np.maximum(map3d[gi, gj], 0.0)
            m = int(np.argmax(seg))
            if seg[m] > 0:
                out[gi, gj, m] = seg[m]
    return out


def expand_plain(code3d, dic, win, stride, prev_shape):
    """One level down: overlap-add the templates named by the code, average
    where footprints overlap, relu. No squelch."""
    acc = np.zeros(prev_shape)
    den = np.zeros(prev_shape[:2])
    for gi in range(code3d.shape[0]):
        for gj in range(code3d.shape[1]):
            seg = np.maximum(code3d[gi, gj], 0.0)
            if seg.max() <= 0.0:
                continue
            acc[gi * stride:gi * stride + win,
                gj * stride:gj * stride + win, :] += (seg @ dic.W).reshape(
                    win, win, prev_shape[2])
            den[gi * stride:gi * stride + win,
                gj * stride:gj * stride + win] += 1.0
    acc = np.where(den[:, :, None] > 0, acc / np.maximum(den[:, :, None], 1.0), 0.0)
    return np.maximum(acc, 0.0)


def decode_l1(code1, means, norms, dic, top1=False, skip_empty=False):
    """L1 code -> pixels. Each window is rebuilt as
    mean + norm * (unit direction voted by the code); overlaps average."""
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    for gi in range(G1):
        for gj in range(G1):
            seg = np.maximum(code1[gi, gj], 0.0)
            if top1 and seg.max() > 0:
                m = int(np.argmax(seg))
                v = seg[m] * dic.W[m]
            else:
                v = seg @ dic.W
            vn = float(np.linalg.norm(v - v.mean()))
            if seg.max() <= 0.0 or vn < EPS:
                if skip_empty:
                    continue
                win = np.full((W1_WIN, W1_WIN), means[gi, gj])
            else:
                v_hat = (v - v.mean()) / vn
                win = (means[gi, gj] + norms[gi, gj] * v_hat).reshape(
                    W1_WIN, W1_WIN)
            r, c = gi * W1_STR, gj * W1_STR
            num[r:r + W1_WIN, c:c + W1_WIN] += win
            den[r:r + W1_WIN, c:c + W1_WIN] += 1.0
    return np.where(den > 0, num / np.maximum(den, 1.0), 0.0)


def recon_from(level, code, means, norms, d1, d2, d3, top1=False):
    """The ladder: rebuild pixels from the code at a given level."""
    if level == 3:
        c3 = harden(code) if top1 else code
        code = expand_plain(c3, d3, W3_WIN, W3_STR, (G2, G2, K2))
        level = 2
    if level == 2:
        c2 = harden(code) if top1 else code
        code = expand_plain(c2, d2, W2_WIN, W2_STR, (G1, G1, K1))
        level = 1
    return decode_l1(code, means, norms, d1, top1=top1)


# ---------------------------------------------------------------- metrics ---

def shape_corr(a, b):
    a_hat, an = center_norm(a.ravel())
    b_hat, bn = center_norm(b.ravel())
    if an < EPS or bn < EPS:
        return 0.0
    return float(a_hat @ b_hat)


def gallery(imgs, titles, path, suptitle, ncol, cmap="gray"):
    nrow = int(np.ceil(len(imgs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.15, nrow * 1.32))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes[i // ncol, i % ncol]
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i], cmap=cmap)
            if titles is not None and titles[i] is not None:
                ax.set_title(str(titles[i]), fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def unit_to_pixels(level, unit, d1, d2, d3):
    """One unit, one central position, rendered down — the template gallery."""
    if level == 1:
        return np.maximum(d1.W[unit], 0.0).reshape(W1_WIN, W1_WIN)
    zero_means = np.zeros((G1, G1))
    unit_norms = np.ones((G1, G1))
    if level == 2:
        c2 = np.zeros((G2, G2, K2))
        c2[G2 // 2, G2 // 2, unit] = 1.0
        c1 = expand_plain(c2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    else:
        c3 = np.zeros((G3, G3, K3))
        c3[G3 // 2, G3 // 2, unit] = 1.0
        c2 = expand_plain(c3, d3, W3_WIN, W3_STR, (G2, G2, K2))
        c1 = expand_plain(c2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    return decode_l1(c1, zero_means, unit_norms, d1, skip_empty=True)


# ------------------------------------------------------------------ main ---

def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (X[:TRAIN_N], y[:TRAIN_N],
            X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load()
    rng = np.random.default_rng(SEED)
    d1 = Layer(K1, W1_WIN * W1_WIN, ETA1, rng)
    d2 = Layer(K2, W2_WIN * W2_WIN * K1, ETA2, rng)
    d3 = Layer(K3, W3_WIN * W3_WIN * K2, ETA3, rng)

    for ep in range(EPOCHS):
        for i, x in enumerate(Xtr):
            m1, _, _ = encode_pixels(x, d1, True)
            m2 = encode_over(m1, d2, W2_WIN, W2_STR, G2, True)
            encode_over(m2, d3, W3_WIN, W3_STR, G3, True)
            if (i + 1) % 5000 == 0:
                print(f"epoch {ep + 1}/{EPOCHS}  {i + 1}/{len(Xtr)}", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz", W1=d1.W, W2=d2.W, W3=d3.W,
             wins1=d1.win_counts, wins2=d2.win_counts, wins3=d3.win_counts)

    # ---- encode the test set ----------------------------------------------
    M1, means, norms = encode_pixels_batch(Xte, d1)
    M2 = encode_over_batch(M1, d2, W2_WIN, W2_STR, G2)
    M3 = encode_over_batch(M2, d3, W3_WIN, W3_STR, G3)

    res = {"train_n": TRAIN_N, "epochs": EPOCHS, "test_n": TEST_N,
           "l1_units_used": int((d1.win_counts > 0).sum()),
           "l2_units_used": int((d2.win_counts > 0).sum()),
           "l3_units_used": int((d3.win_counts > 0).sum())}

    # ---- per-window quantization residual (what top-1 learning optimizes) --
    # For unit vectors, ||window - c*w||^2 = 1 - c^2, so mean top-1 corr per
    # window IS the layer's own local reconstruction quality.
    for name, M in (("L1", M1), ("L2", M2), ("L3", M3)):
        tops = M.max(axis=3).ravel()
        tops = tops[tops > 0]
        res[f"window_top1_corr_{name}"] = round(float(tops.mean()), 4)
        res[f"window_residual_{name}"] = round(float((1 - tops ** 2).mean()), 4)

    # ---- the reconstruction ladder ----------------------------------------
    mean_img = Xtr.mean(axis=0)
    base_mse = float(((Xte - mean_img[None]) ** 2).mean())
    res["baseline_mean_image_mse"] = round(base_mse, 5)
    res["baseline_mean_image_corr"] = round(
        float(np.mean([shape_corr(mean_img, Xte[t]) for t in range(TEST_N)])), 3)

    # side channels alone (every window flat at its mean) — the DC control
    sc_mse, sc_corr = [], []
    for t in range(TEST_N):
        r = decode_l1(np.zeros((G1, G1, K1)), means[t], norms[t], d1)
        sc_mse.append(float(((r - Xte[t]) ** 2).mean()))
        sc_corr.append(shape_corr(r, Xte[t]))
    res["sidechannel_only_mse"] = round(float(np.mean(sc_mse)), 5)
    res["sidechannel_only_corr"] = round(float(np.mean(sc_corr)), 3)

    codes = {1: M1, 2: M2, 3: M3}
    for level in (1, 2, 3):
        for mode, t1 in (("graded", False), ("top1", True)):
            ms, cs = [], []
            for t in range(TEST_N):
                r = recon_from(level, codes[level][t], means[t], norms[t],
                               d1, d2, d3, top1=t1)
                ms.append(float(((r - Xte[t]) ** 2).mean()))
                cs.append(shape_corr(r, Xte[t]))
            res[f"recon_L{level}_{mode}_mse"] = round(float(np.mean(ms)), 5)
            res[f"recon_L{level}_{mode}_corr"] = round(float(np.mean(cs)), 3)
        print(f"ladder L{level} done", flush=True)

    # ---- galleries ---------------------------------------------------------
    gallery([unit_to_pixels(1, i, d1, d2, d3) for i in range(K1)], None,
            OUTPUT_DIR / "templates_L1.png",
            f"L1: {K1} stroke units (8x8 windows)", 16, cmap="inferno")
    gallery([unit_to_pixels(2, i, d1, d2, d3) for i in range(K2)], None,
            OUTPUT_DIR / "templates_L2.png",
            f"L2: {K2} motif units, rendered to pixels", 16, cmap="inferno")
    gallery([unit_to_pixels(3, i, d1, d2, d3) for i in range(K3)], None,
            OUTPUT_DIR / "templates_L3.png",
            f"L3: {K3} part units, rendered to pixels", 20, cmap="inferno")

    rt = []
    for t in range(10):
        rt.append(Xte[t])
        for level in (1, 2, 3):
            rt.append(recon_from(level, codes[level][t], means[t], norms[t],
                                 d1, d2, d3, top1=False))
    gallery(rt, ["orig", "from L1", "from L2", "from L3"] * 10,
            OUTPUT_DIR / "roundtrip_graded.png",
            "The ladder, graded read: original | from L1 | from L2 | from L3", 4)

    rt = []
    for t in range(10):
        rt.append(Xte[t])
        for level in (1, 2, 3):
            rt.append(recon_from(level, codes[level][t], means[t], norms[t],
                                 d1, d2, d3, top1=True))
    gallery(rt, ["orig", "from L1", "from L2", "from L3"] * 10,
            OUTPUT_DIR / "roundtrip_top1.png",
            "The ladder, top-1 read: original | from L1 | from L2 | from L3", 4)

    # label-free sampling: a random top-1 L3 code rendered down (no top
    # layer exists, so this is the only generation this rig can do)
    samp = []
    for _ in range(20):
        c3 = np.zeros((G3, G3, K3))
        for gi in range(G3):
            for gj in range(G3):
                c3[gi, gj, rng.integers(K3)] = 1.0
        c2 = expand_plain(c3, d3, W3_WIN, W3_STR, (G2, G2, K2))
        c1 = expand_plain(c2, d2, W2_WIN, W2_STR, (G1, G1, K1))
        samp.append(decode_l1(c1, np.zeros((G1, G1)), np.ones((G1, G1)), d1,
                              skip_empty=True))
    gallery(samp, None, OUTPUT_DIR / "samples_random_code.png",
            "Random one-hot L3 codes rendered down (label-free sampling)", 10)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    lines = ["# recon3 — reconstruction ladder\n"]
    for k, v in res.items():
        lines.append(f"- {k}: {v}")
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
