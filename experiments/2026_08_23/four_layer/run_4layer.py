"""4-layer hierarchy: three stacked shared dictionaries + concat memory top.

L1: 4x4 pixel windows, stride 2, K=36  -> grid 13x13   (strokes)
L2: 3x3 windows over L1 grid, stride 2, K=64 -> 6x6    (motifs, sees 8x8 px)
L3: 3x3 windows over L2 grid, stride 1, K=100 -> 4x4   (parts, sees 16x16 px)
Top: K=200 concat memory over [L3 code ; lam * onehot]

All levels learn simultaneously, online, top-1 WTA, bootstrap init, no
feedback, no pooling — the naive-stack arm of the depth question, with a
linear probe at every level to watch class information flow upward.

Generation runs the stack backwards: label -> top constellation over parts
-> each part expands (with squelch) into motifs -> motifs into strokes ->
feathered pixel render. Weights are saved (weights.npz) for later play.

Run:  .venv/bin/python experiments/2026_08_23/four_layer/run_4layer.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from gain_feedback import EPS, ZenithLayer, center_norm

ROOT = Path(__file__).resolve().parents[3]

# ---- Configuration -------------------------------------------------------
TRAIN_N = 20000
TEST_N = 5000
PROBE_N = 5000
SIDE = 28
K1, K2, K3, KTOP = 36, 64, 100, 200
W1_WIN, W1_STR = 4, 2      # over pixels  -> grid 13
W2_WIN, W2_STR = 3, 2      # over L1 grid -> grid 6
W3_WIN, W3_STR = 3, 1      # over L2 grid -> grid 4
ETA1, ETA2, ETA3, ETATOP = 0.02, 0.03, 0.03, 0.04
EPOCHS = 2
LAM = 0.5
SEED = 42
NORM_FLOOR = 0.15
RENDER_SQUELCH = 0.10
OUTPUT_DIR = ROOT / "experiments" / "2026_08_23" / "four_layer" / "results" / "naive"
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

def encode_pixels(x2d, dic, learning):
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
            w = dic.learn(p_hat, c) if learning else int(np.argmax(c))
            out[gi, gj, w] = max(float(c[w]), 0.0)
    return out


def encode_over(prev, dic, win, stride, g_out, learning):
    out = np.zeros((g_out, g_out, dic.k))
    for gi in range(g_out):
        for gj in range(g_out):
            v = prev[gi * stride:gi * stride + win,
                     gj * stride:gj * stride + win, :].ravel()
            v = v - v.mean()
            n = np.linalg.norm(v)
            if n < NORM_FLOOR:
                continue
            v_hat = v / n
            c = dic.forward(v_hat)
            w = dic.learn(v_hat, c) if learning else int(np.argmax(c))
            out[gi, gj, w] = max(float(c[w]), 0.0)
    return out


# ---- Batch encoding (inference only, vectorized per position) --------------

def encode_pixels_batch(X2d, dic):
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
            wins = C.argmax(axis=1)
            out[np.arange(N), gi, gj, wins] = np.maximum(C[np.arange(N), wins], 0.0) * ok
    return out


def encode_over_batch(prev, dic, win, stride, g_out):
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
            wins = C.argmax(axis=1)
            out[np.arange(N), gi, gj, wins] = np.maximum(C[np.arange(N), wins], 0.0) * ok
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
            m1 = encode_pixels(x, d1, learning=True)
            m2 = encode_over(m1, d2, W2_WIN, W2_STR, G2, learning=True)
            m3 = encode_over(m2, d3, W3_WIN, W3_STR, G3, learning=True)
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
    M1te = encode_pixels_batch(Xte, d1)
    M2te = encode_over_batch(M1te, d2, W2_WIN, W2_STR, G2)
    M3te = encode_over_batch(M2te, d3, W3_WIN, W3_STR, G3)
    M1tr = encode_pixels_batch(Xtr[:PROBE_N], d1)
    M2tr = encode_over_batch(M1tr, d2, W2_WIN, W2_STR, G2)
    M3tr = encode_over_batch(M2tr, d3, W3_WIN, W3_STR, G3)

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

    print(f"RESULT probeL1={probes['L1']:.4f} probeL2={probes['L2']:.4f} "
          f"probeL3={probes['L3']:.4f} hard={acc_hard:.4f} top10={acc_topk:.4f} "
          f"consistent={consistent}/10")

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
        "# 4-layer hierarchy (MNIST)",
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
