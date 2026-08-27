"""Static 3-layer convolutional Zenith with a label memory on top.

No trails, no time, no arrows — the pre-temporal rig, rebuilt clean.

  L1   8x8 pixel windows, stride 2  -> 11x11 grid, K=64   strokes
  L2   3x3 windows over L1, stride 2 ->  5x5  grid, K=64   motifs
  L3   3x3 windows over L2, stride 1 ->  3x3  grid, K=100  parts
  TOP  memory over [L3 code ; 0.5 * one-hot(10)], K=200

Every layer is the same unit: mean-center and L2-normalize the window,
correlate against the bank, top-1 winner rotates toward the input on the
unit sphere (geodesic step). Nothing else learns.

The three settled choices, all previously validated:
  - DENSE COMMUNICATION: a layer speaks every positive correlation at full
    magnitude (relu-all), not a top-k selection.
  - SKELETON LEARNING at L2/L3: they learn from a per-position top-1 view
    of their input window, while still speaking densely. (Three-regime
    law: the shared mass of a dense target poisons a layer only where it
    is task-irrelevant — zero at L1 after centering, haze at L2/L3,
    class-relevant at the top. So L1 and TOP learn dense, L2/L3 skeleton.)
  - TOP-1 GENERATION: the label queries the top memory with the code half
    blank; the retrieved code is hardened to one unit per position at
    every level on the way down, then rendered with squelch + feathered
    overlap-add.

Run:  .venv/bin/python experiments/2026_08_27/static/run_static3.py
Env:  ST_TRAIN, ST_EPOCHS, ST_K1, ST_K2, ST_K3, ST_KTOP, ST_TAG
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (ROOT / "experiments" / "2026_08_27" / "static" / "results"
              / (os.environ.get("ST_TAG", "").lstrip("_") or "w8"))

SIDE = 28
W1_WIN, W1_STR = 8, 2
W2_WIN, W2_STR = 3, 2
W3_WIN, W3_STR = 3, 1
K1 = int(os.environ.get("ST_K1", "64"))
K2 = int(os.environ.get("ST_K2", "64"))
K3 = int(os.environ.get("ST_K3", "100"))
KTOP = int(os.environ.get("ST_KTOP", "200"))
ETA1, ETA2, ETA3, ETATOP = 0.02, 0.03, 0.03, 0.04
LAM = 0.5
EPS, NORM_FLOOR, RENDER_SQUELCH = 1e-8, 0.15, 0.10
EPOCHS = int(os.environ.get("ST_EPOCHS", "2"))
TRAIN_N = int(os.environ.get("ST_TRAIN", "20000"))
TEST_N, PROBE_N = 5000, 5000
SEED = 42

G1 = (SIDE - W1_WIN) // W1_STR + 1        # 11
G2 = (G1 - W2_WIN) // W2_STR + 1          # 5
G3 = (G2 - W3_WIN) // W3_STR + 1          # 3
CODE3_DIM = G3 * G3 * K3
_t = np.ones(W1_WIN)
_t[0] = _t[-1] = 0.5
FEATHER = np.outer(_t, _t)


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
    out = np.zeros((G1, G1, dic.k))
    for gi in range(G1):
        for gj in range(G1):
            p = x2d[gi * W1_STR:gi * W1_STR + W1_WIN,
                    gj * W1_STR:gj * W1_STR + W1_WIN].ravel()
            p_hat, n = center_norm(p)
            if n < NORM_FLOOR:
                continue
            c = dic.forward(p_hat)
            if learning:
                dic.learn(p_hat, c)              # L1 learns dense
            out[gi, gj, :] = np.maximum(c, 0.0)  # dense speech
    return out


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
    for gi in range(G1):
        for gj in range(G1):
            P = X[:, gi * W1_STR:gi * W1_STR + W1_WIN,
                  gj * W1_STR:gj * W1_STR + W1_WIN].reshape(len(X), -1).copy()
            P -= P.mean(axis=1, keepdims=True)
            nr = np.linalg.norm(P, axis=1)
            ok = nr > NORM_FLOOR
            P[ok] /= nr[ok, None]
            out[:, gi, gj, :] = np.maximum(P @ dic.W.T, 0.0) * ok[:, None]
    return out


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

def _squelch_pos(seg):
    seg = np.maximum(seg, 0.0)
    m = seg.max()
    if m <= 0.0:
        return None
    seg = seg.copy()
    seg[seg < RENDER_SQUELCH * m] = 0.0
    return seg


def harden(map3d, k_keep=1):
    """Per position keep only the k_keep strongest entries."""
    out = np.zeros_like(map3d)
    for gi in range(map3d.shape[0]):
        for gj in range(map3d.shape[1]):
            seg = np.maximum(map3d[gi, gj], 0.0)
            if seg.max() <= 0:
                continue
            order = np.argsort(seg)[::-1][:k_keep]
            out[gi, gj, order] = seg[order]
    return out


def expand(code3d, dic, win, stride, prev_shape):
    acc = np.zeros(prev_shape)
    kp = prev_shape[2]
    for gi in range(code3d.shape[0]):
        for gj in range(code3d.shape[1]):
            seg = _squelch_pos(code3d[gi, gj])
            if seg is None:
                continue
            acc[gi * stride:gi * stride + win,
                gj * stride:gj * stride + win, :] += (seg @ dic.W).reshape(win, win, kp)
    acc = np.maximum(acc, 0.0)
    for gi in range(prev_shape[0]):
        for gj in range(prev_shape[1]):
            m = acc[gi, gj].max()
            if m > 0:
                acc[gi, gj][acc[gi, gj] < RENDER_SQUELCH * m] = 0.0
    return acc


def render_pixels(code1, dic):
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = code1.max()
    for gi in range(G1):
        for gj in range(G1):
            seg = _squelch_pos(code1[gi, gj])
            if seg is None or seg.max() < RENDER_SQUELCH * peak:
                continue
            conf = float(seg.max())
            r, c = gi * W1_STR, gj * W1_STR
            num[r:r + W1_WIN, c:c + W1_WIN] += (seg @ dic.W).reshape(
                W1_WIN, W1_WIN) * FEATHER * conf
            den[r:r + W1_WIN, c:c + W1_WIN] += FEATHER * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def render_from_l3(code3, d1, d2, d3, top1=True):
    m3 = harden(code3) if top1 else code3
    m2 = expand(m3, d3, W3_WIN, W3_STR, (G2, G2, K2))
    if top1:
        m2 = harden(m2)
    m1 = expand(m2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    if top1:
        m1 = harden(m1)
    return render_pixels(m1, d1)


# ------------------------------------------------------------------ main ---

def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (X[:TRAIN_N], y[:TRAIN_N],
            X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N])


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load()
    rng = np.random.default_rng(SEED)
    d1 = Layer(K1, W1_WIN * W1_WIN, ETA1, rng)
    d2 = Layer(K2, W2_WIN * W2_WIN * K1, ETA2, rng)
    d3 = Layer(K3, W3_WIN * W3_WIN * K2, ETA3, rng)
    top = Layer(KTOP, CODE3_DIM + 10, ETATOP, rng)

    for ep in range(EPOCHS):
        for i, (x, yy) in enumerate(zip(Xtr, ytr)):
            m1 = encode_pixels(x, d1, True)
            m2 = encode_over(m1, d2, W2_WIN, W2_STR, G2, True)
            m3 = encode_over(m2, d3, W3_WIN, W3_STR, G3, True)
            h_hat, hn = center_norm(m3.ravel())
            if hn < EPS:
                continue
            lab = np.zeros(10)
            lab[yy] = LAM
            z_hat, zn = center_norm(np.concatenate([h_hat, lab]))
            if zn > EPS:
                top.learn(z_hat, top.forward(z_hat))   # top learns dense
            if (i + 1) % 5000 == 0:
                print(f"epoch {ep + 1}/{EPOCHS}  {i + 1}/{len(Xtr)}", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz",
             W1=d1.W, W2=d2.W, W3=d3.W, Wtop=top.W)

    # ---- how well does each level separate the classes? --------------------
    M1te = encode_pixels_batch(Xte, d1)
    M2te = encode_over_batch(M1te, d2, W2_WIN, W2_STR, G2)
    M3te = encode_over_batch(M2te, d3, W3_WIN, W3_STR, G3)
    M1tr = encode_pixels_batch(Xtr[:PROBE_N], d1)
    M2tr = encode_over_batch(M1tr, d2, W2_WIN, W2_STR, G2)
    M3tr = encode_over_batch(M2tr, d3, W3_WIN, W3_STR, G3)
    res = {}
    for name, tr, te in (("L1", M1tr, M1te), ("L2", M2tr, M2te), ("L3", M3tr, M3te)):
        clf = LogisticRegression(max_iter=1000).fit(
            tr.reshape(len(tr), -1), ytr[:PROBE_N])
        res[f"probe_{name}"] = round(
            float(clf.score(te.reshape(len(te), -1), yte)), 4)

    label_half = top.W[:, CODE3_DIM:]
    owner = label_half.argmax(axis=1)
    H = M3te.reshape(len(M3te), -1)
    H = H - H.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    Z = np.concatenate([H, np.zeros((len(H), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ top.W.T
    res["hard_readout"] = round(float((owner[C2.argmax(axis=1)] == yte).mean()), 4)
    res["memories_per_label"] = {str(j): int((owner == j).sum()) for j in range(10)}
    res["l1_units_used"] = int((d1.win_counts > 0).sum())
    res["l2_units_used"] = int((d2.win_counts > 0).sum())
    res["l3_units_used"] = int((d3.win_counts > 0).sum())
    res["top_units_used"] = int((top.win_counts > 0).sum())

    # ---- generation: LABEL IN, PICTURE OUT ---------------------------------
    gen, consistent = [], 0
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), lab]))
        w = int(np.argmax(top.forward(z_hat)))
        consistent += int(owner[w] == j)
        code3 = np.maximum(top.W[w, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        gen.append(render_from_l3(code3, d1, d2, d3))
    res["label_query_hits_own_memory"] = f"{consistent}/10"
    gallery(gen, list(range(10)), OUTPUT_DIR / "generation_labels.png",
            "Label in, picture out — top-1 generation through 3 layers", 10)

    # several memories per label, so the variety is visible
    var_imgs, var_titles = [], []
    for j in range(10):
        rows = np.where(owner == j)[0]
        rows = rows[np.argsort(-label_half[rows, j])][:6]
        for r in rows:
            code3 = np.maximum(top.W[r, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
            var_imgs.append(render_from_l3(code3, d1, d2, d3))
            var_titles.append(j)
        for _ in range(6 - len(rows)):
            var_imgs.append(np.zeros((SIDE, SIDE)))
            var_titles.append(None)
    gallery(var_imgs, var_titles, OUTPUT_DIR / "generation_variants.png",
            "Six memories per label (rows = 0..9)", 6)

    # ---- round trip, as a sanity check on the drawing path -----------------
    rt = []
    for k in range(10):
        rt.append(Xte[k])
        rt.append(render_from_l3(M3te[k], d1, d2, d3))
    gallery(rt, None, OUTPUT_DIR / "roundtrip.png",
            "Real image (left of each pair) vs its round trip through L1-L3", 10)
    rc = [float(center_norm(render_from_l3(M3te[k], d1, d2, d3).ravel())[0]
                @ center_norm(Xte[k].ravel())[0]) for k in range(50)]
    res["roundtrip_corr"] = round(float(np.mean(rc)), 3)

    # ---- templates, one gallery per layer ----------------------------------
    gallery([np.maximum(d1.W[i], 0).reshape(W1_WIN, W1_WIN) for i in range(K1)],
            None, OUTPUT_DIR / "templates_L1.png",
            f"L1: {K1} stroke units ({W1_WIN}x{W1_WIN} pixel windows)", 16,
            cmap="inferno")
    l2 = []
    for u in range(K2):
        c2 = np.zeros((G2, G2, K2))
        c2[G2 // 2, G2 // 2, u] = 1.0
        l2.append(render_pixels(expand(c2, d2, W2_WIN, W2_STR, (G1, G1, K1)), d1))
    gallery(l2, None, OUTPUT_DIR / "templates_L2.png",
            f"L2: {K2} motif units, expanded down to pixels", 16)
    l3 = []
    for u in range(K3):
        c3 = np.zeros((G3, G3, K3))
        c3[G3 // 2, G3 // 2, u] = 1.0
        l3.append(render_from_l3(c3, d1, d2, d3, top1=False))
    gallery(l3, None, OUTPUT_DIR / "templates_L3.png",
            f"L3: {K3} part units, expanded down to pixels", 16)

    print(json.dumps(res, indent=1), flush=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(res, indent=2))
    (OUTPUT_DIR / "report.md").write_text(
        "# Static 3-layer conv Zenith + label memory\n\n"
        f"L1 {W1_WIN}x{W1_WIN}/s{W1_STR} K={K1} -> {G1}x{G1} | "
        f"L2 {W2_WIN}x{W2_WIN}/s{W2_STR} K={K2} -> {G2}x{G2} | "
        f"L3 {W3_WIN}x{W3_WIN}/s{W3_STR} K={K3} -> {G3}x{G3} | TOP K={KTOP}\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in res.items()) + "\n")
    print("artifacts in", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
