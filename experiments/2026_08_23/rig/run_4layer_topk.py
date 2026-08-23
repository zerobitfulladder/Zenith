"""4-layer hierarchy: three stacked shared dictionaries + concat memory top.

L1: 4x4 pixel windows, stride 2, K=36  -> grid 13x13   (strokes)
L2: 3x3 windows over L1 grid, stride 2, K=64 -> 6x6    (motifs, sees 8x8 px)
L3: 3x3 windows over L2 grid, stride 1, K=100 -> 4x4   (parts, sees 16x16 px)
Top: K=200 concat memory over [L3 code ; lam * onehot]

Depth-fix variant: LEARNING stays top-1 everywhere, but the upward CODE
carries multiple winners per window — k_out = 3 (L1), 2 (L2), 1 (L3),
the sparsity gradient (denser low, sparser high). Everything else is
identical to run_4layer.py, so this is a one-variable comparison against
the naive stack's binding failure.

Generation runs the stack backwards: label -> top constellation over parts
-> each part expands (with squelch) into motifs -> motifs into strokes ->
feathered pixel render. Weights are saved (weights.npz) for later play.

Run:  .venv/bin/python experiments/2026_08_23/rig/run_4layer_topk.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from gain_feedback import EPS, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000
SIDE = int(os.environ.get("GF_SIDE", "28"))
K1 = int(os.environ.get("GF_K1", "36"))
K2 = int(os.environ.get("GF_K2L", "64"))
K3 = int(os.environ.get("GF_K3", "100"))
KTOP = int(os.environ.get("GF_KTOP", "200"))
W1_WIN, W1_STR = 4, int(os.environ.get("GF_W1_STR", "2"))  # over pixels; stride 1 -> grid 25
W2_WIN, W2_STR = 3, 2      # over L1 grid -> grid 6
W3_WIN, W3_STR = 3, 1      # over L2 grid -> grid 4
ETA1, ETA2, ETA3, ETATOP = 0.02, 0.03, 0.03, 0.04
K_OUT1, K_OUT2, K_OUT3 = 3, 2, 1   # winners REPORTED per window, per level
# Reporting mode (learning is top-1 in all modes):
#   "topk"     — the sparsity gradient above
#   "reluall"  — every positive correlation, full magnitude, no output inhibition
#   "contrast" — relu(c - mean(c)): graded margins over the field average,
#                inhibition = subtracting what the competition explains
#   "signed"   — the full raw correlation vector, negatives included
#   "sparselearn" — communicate like reluall, but L2/L3 LEARN from a
#                per-position-top-1 sparsified view of their window
#                (learn on the skeleton, speak with the full voice)
REPORT = os.environ.get("GF_REPORT", "topk")
EPOCHS = 2
LAM = 0.5
SEED = 42
NORM_FLOOR = 0.15
RENDER_SQUELCH = 0.10
_SUFFIX = "topk" if REPORT == "topk" else REPORT
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "four_layer" / "results" / _SUFFIX
# ---------------------------------------------------------------------------

G1 = (SIDE - W1_WIN) // W1_STR + 1          # 13
G2 = (G1 - W2_WIN) // W2_STR + 1            # 6
G3 = (G2 - W3_WIN) // W3_STR + 1            # 4
CODE3_DIM = G3 * G3 * K3                    # 1600
FEATHER = np.outer([0.5, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 0.5])


def load_data():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return (
        X[:TRAIN_N], y[:TRAIN_N],
        X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N],
    )


# ---- Encoding (single image, with optional learning) ----------------------

def _report_topk(out, gi, gj, c, k_out):
    """Write the reported activations into the code, per REPORT mode."""
    if REPORT == "signed":
        out[gi, gj, :] = c
        return
    if REPORT in ("reluall", "sparselearn"):
        out[gi, gj, :] = np.maximum(c, 0.0)
        return
    if REPORT == "contrast":
        out[gi, gj, :] = np.maximum(c - c.mean(), 0.0)
        return
    order = np.argsort(c)[::-1][:k_out]
    for w in order:
        if c[w] <= 0.0:
            break
        out[gi, gj, w] = float(c[w])


def encode_pixels(x2d, dic, learning, k_out):
    out = np.zeros((G1, G1, dic.k))
    for gi in range(G1):
        for gj in range(G1):
            p = x2d[gi * W1_STR:gi * W1_STR + W1_WIN,
                    gj * W1_STR:gj * W1_STR + W1_WIN].ravel()
            p = p - p.mean()
            n = np.linalg.norm(p)
            if n < NORM_FLOOR:
                continue
            p_hat = p / n
            c = dic.forward(p_hat)
            if learning:
                dic.learn(p_hat, c)   # top-1 plasticity, always
            _report_topk(out, gi, gj, c, k_out)
    return out


def _pp_top1(w3d):
    """Per-position top-1 skeleton of a code window."""
    out = np.zeros_like(w3d)
    for a in range(w3d.shape[0]):
        for b in range(w3d.shape[1]):
            seg = w3d[a, b]
            m = int(np.argmax(seg))
            if seg[m] > 0:
                out[a, b, m] = seg[m]
    return out


def encode_over(prev, dic, win, stride, g_out, learning, k_out):
    out = np.zeros((g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            block = prev[gi * stride:gi * stride + win,
                         gj * stride:gj * stride + win, :]
            v = block.ravel()
            v = v - v.mean()
            n = np.linalg.norm(v)
            if n < NORM_FLOOR:
                continue
            v_hat = v / n
            c = dic.forward(v_hat)
            if learning:
                if REPORT == "sparselearn":
                    # Learn from the skeleton view: peaked, near-disjoint
                    # targets restore differentiation pressure; the dense
                    # message upward is untouched.
                    vs = _pp_top1(block).ravel()
                    vs = vs - vs.mean()
                    ns = np.linalg.norm(vs)
                    if ns > NORM_FLOOR:
                        vs_hat = vs / ns
                        dic.learn(vs_hat, dic.forward(vs_hat))
                else:
                    dic.learn(v_hat, c)   # top-1 plasticity, always
            _report_topk(out, gi, gj, c, k_out)
    return out


# ---- Batch encoding (inference only, vectorized per position) --------------

def _scatter_topk_batch(out_slice, C, ok, k_out):
    N = len(C)
    if REPORT == "signed":
        out_slice[:] = C * ok[:, None]
        return
    if REPORT in ("reluall", "sparselearn"):
        out_slice[:] = np.maximum(C, 0.0) * ok[:, None]
        return
    if REPORT == "contrast":
        out_slice[:] = np.maximum(C - C.mean(axis=1, keepdims=True), 0.0) * ok[:, None]
        return
    kk = min(k_out, C.shape[1])
    idx = np.argpartition(-C, kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(C, idx, axis=1)
    vals = np.maximum(vals, 0.0) * ok[:, None]
    for j in range(kk):
        out_slice[np.arange(N), idx[:, j]] = vals[:, j]


def encode_pixels_batch(X2d, dic, k_out):
    N = len(X2d)
    out = np.zeros((N, G1, G1, dic.k))
    for gi in range(G1):
        for gj in range(G1):
            P = X2d[:, gi * W1_STR:gi * W1_STR + W1_WIN,
                    gj * W1_STR:gj * W1_STR + W1_WIN].reshape(N, -1)
            P = P - P.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(P, axis=1)
            ok = norms > NORM_FLOOR
            P[ok] /= norms[ok, None]
            C = P @ dic.W.T
            _scatter_topk_batch(out[:, gi, gj, :], C, ok, k_out)
    return out


def encode_over_batch(prev, dic, win, stride, g_out, k_out):
    N = len(prev)
    out = np.zeros((N, g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            V = prev[:, gi * stride:gi * stride + win,
                     gj * stride:gj * stride + win, :].reshape(N, -1)
            V = V - V.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(V, axis=1)
            ok = norms > NORM_FLOOR
            V[ok] /= norms[ok, None]
            C = V @ dic.W.T
            _scatter_topk_batch(out[:, gi, gj, :], C, ok, k_out)
    return out


# ---- Downward expansion (generation / reconstruction) ----------------------

def _squelch_pos(seg):
    seg = np.maximum(seg, 0.0)
    m = seg.max()
    if m <= 0.0:
        return None
    seg = seg.copy()
    seg[seg < RENDER_SQUELCH * m] = 0.0
    return seg


def expand(code3d, dic, win, stride, prev_shape):
    """One level down: constellation over this level's units -> coefficient
    canvas over the level below, accumulated across overlapping windows."""
    acc = np.zeros(prev_shape)
    g = code3d.shape[0]
    kp = prev_shape[2]
    for gi in range(g):
        for gj in range(g):
            seg = _squelch_pos(code3d[gi, gj])
            if seg is None:
                continue
            contrib = (seg @ dic.W).reshape(win, win, kp)
            acc[gi * stride:gi * stride + win,
                gj * stride:gj * stride + win, :] += contrib
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
            patch = (seg @ dic.W).reshape(W1_WIN, W1_WIN)
            conf = float(seg.max())
            r, c = gi * W1_STR, gj * W1_STR
            num[r:r + W1_WIN, c:c + W1_WIN] += patch * FEATHER * conf
            den[r:r + W1_WIN, c:c + W1_WIN] += FEATHER * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def render_from_l3(code3, d1, d2, d3):
    m2 = expand(code3, d3, W3_WIN, W3_STR, (G2, G2, K2))
    m1 = expand(m2, d2, W2_WIN, W2_STR, (G1, G1, K1))
    return render_pixels(m1, d1)


# ---- Main ------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()

    rng = np.random.default_rng(SEED)
    d1 = ZenithLayer(K1, W1_WIN * W1_WIN, ETA1, rng)
    d2 = ZenithLayer(K2, W2_WIN * W2_WIN * K1, ETA2, rng)
    d3 = ZenithLayer(K3, W3_WIN * W3_WIN * K2, ETA3, rng)
    top = ZenithLayer(KTOP, CODE3_DIM + 10, ETATOP, rng)

    for _ in range(EPOCHS):
        for x, y in tqdm(list(zip(Xtr, ytr)), desc="4layer", ncols=80):
            m1 = encode_pixels(x, d1, learning=True, k_out=K_OUT1)
            m2 = encode_over(m1, d2, W2_WIN, W2_STR, G2, learning=True, k_out=K_OUT2)
            m3 = encode_over(m2, d3, W3_WIN, W3_STR, G3, learning=True, k_out=K_OUT3)
            h_hat, h_norm = center_norm(m3.ravel())
            if h_norm < EPS:
                continue
            label = np.zeros(10)
            label[y] = LAM
            z_hat, z_norm = center_norm(np.concatenate([h_hat, label]))
            if z_norm > EPS:
                top.learn(z_hat, top.forward(z_hat))

    np.savez(OUTPUT_DIR / "weights.npz", W1=d1.W, W2=d2.W, W3=d3.W, Wtop=top.W)

    # ---- Codes at every level (test + probe subsets) -----------------------
    M1te = encode_pixels_batch(Xte, d1, K_OUT1)
    M2te = encode_over_batch(M1te, d2, W2_WIN, W2_STR, G2, K_OUT2)
    M3te = encode_over_batch(M2te, d3, W3_WIN, W3_STR, G3, K_OUT3)
    M1tr = encode_pixels_batch(Xtr[:PROBE_N], d1, K_OUT1)
    M2tr = encode_over_batch(M1tr, d2, W2_WIN, W2_STR, G2, K_OUT2)
    M3tr = encode_over_batch(M2tr, d3, W3_WIN, W3_STR, G3, K_OUT3)

    probes = {}
    for name, tr, te in [("L1", M1tr, M1te), ("L2", M2tr, M2te), ("L3", M3tr, M3te)]:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(tr.reshape(len(tr), -1), ytr[:PROBE_N])
        probes[name] = float(clf.score(te.reshape(len(te), -1), yte))

    # ---- Top readouts and retrieval ---------------------------------------
    label_half = top.W[:, CODE3_DIM:]
    owner = label_half.argmax(axis=1)

    H = M3te.reshape(len(M3te), -1)
    H = H - H.mean(axis=1, keepdims=True)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + EPS
    Z = np.concatenate([H, np.zeros((len(H), 10))], axis=1)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + EPS
    C2 = Z @ top.W.T
    acc_hard = float((owner[C2.argmax(axis=1)] == yte).mean())
    kv = 10
    part = np.argpartition(-C2, kv, axis=1)[:, :kv]
    mask = np.zeros_like(C2)
    mask[np.arange(len(C2))[:, None], part] = 1.0
    acc_topk = float((((np.maximum(C2, 0.0) * mask) @ label_half).argmax(axis=1) == yte).mean())

    gen_imgs, consistent = [], 0
    for j in range(10):
        label = np.zeros(10)
        label[j] = LAM
        z_hat, _ = center_norm(np.concatenate([np.zeros(CODE3_DIM), label]))
        winner = int(np.argmax(top.forward(z_hat)))
        consistent += int(owner[winner] == j)
        code3 = np.maximum(top.W[winner, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        gen_imgs.append(render_from_l3(code3, d1, d2, d3))

    def _peak(W, win, kprev):
        r = []
        for t in range(len(W)):
            w3 = np.maximum(W[t], 0.0).reshape(win * win, kprev)
            for p in range(win * win):
                s = w3[p].sum()
                if s > 0:
                    r.append(w3[p].max() / s)
        return float(np.mean(r))

    print(f"RESULT probeL1={probes['L1']:.4f} probeL2={probes['L2']:.4f} "
          f"probeL3={probes['L3']:.4f} hard={acc_hard:.4f} top10={acc_topk:.4f} "
          f"consistent={consistent}/10 "
          f"peak2={_peak(d2.W, W2_WIN, K1):.3f} peak3={_peak(d3.W, W3_WIN, K2):.3f}")

    # ---- Figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for j, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[j], cmap="gray")
        ax.set_title(str(j), fontsize=9)
        ax.axis("off")
    fig.suptitle("Label-only generation through 4 layers (parts -> motifs -> strokes)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generation_labelonly.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(5, 5, figsize=(7, 7.4))
    for u, ax in enumerate(axes.flat):
        c2 = np.zeros((G2, G2, K2))
        c2[2, 2, u] = 1.0
        ax.imshow(render_pixels(expand(c2, d2, W2_WIN, W2_STR, (G1, G1, K1)), d1), cmap="gray")
        ax.axis("off")
    fig.suptitle("L2 motif units (first 25), rendered to pixels")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l2_motifs.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(6.5, 7))
    for u, ax in enumerate(axes.flat):
        c3 = np.zeros((G3, G3, K3))
        c3[1, 1, u] = 1.0
        ax.imshow(render_from_l3(c3, d1, d2, d3), cmap="gray")
        ax.axis("off")
    fig.suptitle("L3 part units (first 16), rendered to pixels")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l3_parts.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for k in range(8):
        axes[0, k].imshow(Xte[k], cmap="gray")
        axes[1, k].imshow(render_from_l3(M3te[k], d1, d2, d3), cmap="gray")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("Test images (top) vs full round-trip reconstruction pixels->L3->pixels (bottom)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recon_roundtrip.png", dpi=110)
    plt.close(fig)

    # ---- Report -----------------------------------------------------------
    (OUTPUT_DIR / "report.md").write_text("\n".join([
        f"# 4-layer hierarchy, reporting mode: {REPORT} (MNIST)",
        "",
        f"k_out per level (topk mode): L1={K_OUT1}, L2={K_OUT2}, L3={K_OUT3}; learning top-1 everywhere.",
        "Naive-stack baselines: probes 0.943/0.919/0.855, hard 0.463, top10 0.553.",
        "",
        f"Config: L1 {W1_WIN}x{W1_WIN}s{W1_STR} K={K1} (grid {G1}); "
        f"L2 {W2_WIN}x{W2_WIN}s{W2_STR} K={K2} (grid {G2}); "
        f"L3 {W3_WIN}x{W3_WIN}s{W3_STR} K={K3} (grid {G3}); top K={KTOP}, "
        f"lam={LAM}, EPOCHS={EPOCHS}, TRAIN_N={TRAIN_N}, SEED={SEED}.",
        "Batch-2 baselines: probe(L1 code) 0.9146-0.947, hard 0.743, top10 0.782.",
        "",
        "| probe L1 | probe L2 | probe L3 | hard | top10 | label-only consistent |",
        "|---|---|---|---|---|---|",
        f"| {probes['L1']:.4f} | {probes['L2']:.4f} | {probes['L3']:.4f} | "
        f"{acc_hard:.4f} | {acc_topk:.4f} | {consistent}/10 |",
        "",
        "Figures: generation_labelonly.png, l2_motifs.png, l3_parts.png, "
        "recon_roundtrip.png. Weights: weights.npz",
    ]) + "\n")
    print(f"Report written to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
