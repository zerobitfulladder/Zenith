"""Exp 9 — label-conditioned generation on the residual-trained banks.

L1/L2 FROZEN from Exp 4's seq weights (rounds-at-L1, the best banks).
A label-bearing last layer is trained on top, learning DENSE from the
joint [content ; lam*one-hot] (attaching the label makes this layer the
top; three-regime law: class-relevant shared mass learns dense).
Per-pathway convention: center+normalize content first, then the concat.

Two arms (the user's sentence admits both; both tested):
  global  one position over the whole L2 map (5x5x64=1600 dims + 10),
          K=200 — the static3-top pattern on today's banks.
  conv    L3 keeps its 9 positions, each block gets the label
          (576 + 10 dims), K=200 shared. Risk stated in advance: the
          generation query [empty ; label] is IDENTICAL at all nine
          positions over a shared bank -> every position ranks units the
          same way; tested is whether "lighting up enough units" +
          overlap blending rescues a drawable digit.

Generation: query [zeros ; lam*label], winner(s) light up, their stored
content halves travel down through frozen W2, W1 to pixels (no side
channels exist without an input; render is shape-only, norms=1).

Run:  .venv/bin/python experiments/2026_08_28/labelgen/run_labelgen.py
Env:  LG_TRAIN (55000), LG_TEST (5000), LG_K (200), LG_TAG
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
OUTPUT_DIR = HERE / "results" / os.environ.get('LG_TAG', '').lstrip('_')
TRAIN_N = int(os.environ.get("LG_TRAIN", "55000"))
TEST_N = int(os.environ.get("LG_TEST", "5000"))
KTOP = int(os.environ.get("LG_K", "200"))
LAM = 0.5
ETA_TOP = 0.04
DIM_G = R.G2 * R.G2 * R.K2                 # 1600
DIM_C = R.W3_WIN * R.W3_WIN * R.K2         # 576
EB = 1000


def encode_m2(X, W1, W2):
    """Images -> L2 code maps (numpy), via the frozen GPU banks."""
    out = np.zeros((len(X), R.G2, R.G2, R.K2), dtype=np.float64)
    for s in range(0, len(X), EB):
        xb = xp.asarray(X[s:s + EB], dtype=G.DTYPE)
        n = len(xb)
        P1 = G.window_stack(xb[..., None], G.POS1, R.W1_WIN).reshape(
            n * len(G.POS1), -1)
        Xh, _, _, ok1 = G.center_norm_rows(P1)
        C1 = xp.maximum(Xh @ W1.T, 0.0) * ok1[:, None]
        M1map = C1.reshape(n, R.G1, R.G1, R.K1)
        B2 = G.window_stack(M1map, G.POS2, R.W2_WIN).reshape(
            n * len(G.POS2), -1)
        Bh2, _, _, ok2 = G.center_norm_rows(B2)
        seg2 = xp.maximum(Bh2 @ W2.T, 0.0) * ok2[:, None]
        m = seg2.reshape(n, R.G2, R.G2, R.K2)
        out[s:s + EB] = xp.asnumpy(m) if G.XP_NAME == "cupy" else m
    return out


def joint(content, lab_vec):
    h_hat, hn = R.center_norm(content.ravel())
    if hn < R.EPS:
        return None
    z_hat, zn = R.center_norm(np.concatenate([h_hat, lab_vec]))
    return z_hat if zn > R.EPS else None


def render(code2map, l1, l2, top1=False):
    """L2 code map -> pixels through the frozen banks (shape-only)."""
    m = R.harden(code2map) if top1 else code2map
    m1 = R.expand_plain(m, l2, R.W2_WIN, R.W2_STR, (R.G1, R.G1, R.K1))
    if top1:
        m1 = R.harden(m1)
    return R.decode_l1(m1, np.zeros((R.G1, R.G1)), np.ones((R.G1, R.G1)),
                       l1, skip_empty=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    w = np.load(HERE.parent / "residual_learning" / "results" / "weights_seq.npz")
    W1x = xp.asarray(w["W1"], dtype=G.DTYPE)
    W2x = xp.asarray(w["W2"], dtype=G.DTYPE)
    l1 = R.Layer(R.K1, w["W1"].shape[1], 0.0, np.random.default_rng(0))
    l2 = R.Layer(R.K2, w["W2"].shape[1], 0.0, np.random.default_rng(0))
    l1.W, l2.W = w["W1"].astype(np.float64), w["W2"].astype(np.float64)

    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr = X[:TRAIN_N], y[:TRAIN_N]
    Xte, yte = X[TRAIN_N:TRAIN_N + TEST_N], y[TRAIN_N:TRAIN_N + TEST_N]

    print("encoding L2 maps...", flush=True)
    M2tr = encode_m2(Xtr, W1x, W2x)
    M2te = encode_m2(Xte, W1x, W2x)

    rng = np.random.default_rng(R.SEED)
    dg = R.Layer(KTOP, DIM_G + 10, ETA_TOP, rng)
    dc = R.Layer(KTOP, DIM_C + 10, ETA_TOP, rng)
    res = {"train_n": TRAIN_N, "test_n": TEST_N, "k": KTOP, "lam": LAM}

    # ---- train both label layers (online, dense, top-1) --------------------
    for i in range(TRAIN_N):
        lab = np.zeros(10)
        lab[ytr[i]] = LAM
        z = joint(M2tr[i], lab)
        if z is not None:
            dg.learn(z, dg.forward(z))
        for gi in range(R.G3):
            for gj in range(R.G3):
                blk = M2tr[i][gi:gi + R.W3_WIN, gj:gj + R.W3_WIN, :]
                zc = joint(blk, lab)
                if zc is not None:
                    dc.learn(zc, dc.forward(zc))
        if (i + 1) % 10000 == 0:
            print(f"top layers {i + 1}/{TRAIN_N}", flush=True)
    np.savez(OUTPUT_DIR / "weights_top.npz", Wg=dg.W, Wc=dc.W)
    res["units_used"] = [int((d.win_counts > 0).sum()) for d in (dg, dc)]

    # ---- hard readouts -----------------------------------------------------
    own_g = dg.W[:, DIM_G:].argmax(axis=1)
    own_c = dc.W[:, DIM_C:].argmax(axis=1)
    hits_g = hits_c = 0
    for i in range(TEST_N):
        z = joint(M2te[i], np.zeros(10))
        if z is not None:
            hits_g += int(own_g[int(np.argmax(dg.forward(z)))] == yte[i])
        votes = np.zeros(10)
        for gi in range(R.G3):
            for gj in range(R.G3):
                zc = joint(M2te[i][gi:gi + R.W3_WIN, gj:gj + R.W3_WIN, :],
                           np.zeros(10))
                if zc is not None:
                    votes += dc.W[int(np.argmax(dc.forward(zc))), DIM_C:]
        hits_c += int(int(np.argmax(votes)) == yte[i])
    res["hard_readout_global"] = round(hits_g / TEST_N, 4)
    res["hard_readout_conv_vote"] = round(hits_c / TEST_N, 4)

    # ---- generation: GLOBAL arm -------------------------------------------
    cons = 0
    imgs, titles = [], []
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        zq, _ = R.center_norm(np.concatenate([np.zeros(DIM_G), lab]))
        wnr = int(np.argmax(dg.forward(zq)))
        cons += int(own_g[wnr] == j)
        code2 = np.maximum(dg.W[wnr, :DIM_G], 0.0).reshape(R.G2, R.G2, R.K2)
        imgs.append(render(code2, l1, l2, top1=True))
        titles.append(f"{j} t1")
        imgs.append(render(code2, l1, l2, top1=False))
        titles.append(f"{j} gr")
    res["global_label_hits_own_memory"] = f"{cons}/10"
    R.gallery(imgs, titles, OUTPUT_DIR / "generation_global.png",
              "GLOBAL arm: label -> winner memory -> pixels "
              "(t1 = hardened, gr = graded descent)", 10)

    var_imgs, var_titles = [], []
    for j in range(10):
        rows = np.where(own_g == j)[0]
        rows = rows[np.argsort(-dg.W[rows, DIM_G + j])][:6]
        for rr in rows:
            code2 = np.maximum(dg.W[rr, :DIM_G], 0.0).reshape(R.G2, R.G2, R.K2)
            var_imgs.append(render(code2, l1, l2, top1=True))
            var_titles.append(j)
        for _ in range(6 - len(rows)):
            var_imgs.append(np.zeros((R.SIDE, R.SIDE)))
            var_titles.append(None)
    res["global_memories_per_label"] = {
        str(j): int((own_g == j).sum()) for j in range(10)}
    R.gallery(var_imgs, var_titles, OUTPUT_DIR / "generation_global_variants.png",
              "GLOBAL arm: six memories per label (hardened descent)", 6)

    # ---- generation: CONV arm ---------------------------------------------
    zqc, _ = R.center_norm(np.concatenate([np.zeros(DIM_C), np.zeros(10)]))
    imgs, titles = [], []
    for j in range(10):
        lab = np.zeros(10)
        lab[j] = LAM
        zq, _ = R.center_norm(np.concatenate([np.zeros(DIM_C), lab]))
        c = dc.forward(zq)                       # identical at all 9 positions
        for mode, kk in (("top1", 1), ("top5", 5), ("graded", 0)):
            code2acc = np.zeros((R.G2, R.G2, R.K2))
            den = np.zeros((R.G2, R.G2))
            act = np.maximum(c, 0.0)
            if kk:
                keep = np.argsort(act)[::-1][:kk]
                mask = np.zeros_like(act)
                mask[keep] = act[keep]
                act = mask
            for gi in range(R.G3):
                for gj in range(R.G3):
                    blk = (act @ np.maximum(dc.W[:, :DIM_C], 0.0)).reshape(
                        R.W3_WIN, R.W3_WIN, R.K2)
                    code2acc[gi:gi + R.W3_WIN, gj:gj + R.W3_WIN, :] += blk
                    den[gi:gi + R.W3_WIN, gj:gj + R.W3_WIN] += 1.0
            code2 = code2acc / np.maximum(den, 1.0)[:, :, None]
            if mode == "top1":
                imgs.append(render(code2, l1, l2, top1=True))
            else:
                imgs.append(render(code2, l1, l2, top1=False))
            titles.append(f"{j} {mode}")
    R.gallery(imgs, titles, OUTPUT_DIR / "generation_conv.png",
              "CONV arm: label at every position (top1 / top5 / graded reads)", 6)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
