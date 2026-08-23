"""8-level pyramid with luminance channel and drift-based early stopping.

Seven consolidated-unit dictionaries + a 1000-memory archive, ~2M params.
L1 appends each window's raw mean brightness as an extra channel — outside
the Pearson metric, inside the message and the memories, re-lit at render.
Skeleton learning excludes the luminance channel (structure teaches,
shading speaks). Training stops when per-epoch template drift < DRIFT_STOP.

Run:  .venv/bin/python experiments/2026_08_23/pyramid/run_pyramid.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("GF_REPORT", "sparselearn")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402

from run_gpu_minibatch import B, Dict, XP_NAME, _center_norm_rows, xp  # noqa: E402
from run_4layer_topk import expand, render_pixels  # noqa: E402  (generic helpers)
from gain_feedback import center_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = (ROOT / "experiments" / "2026_08_23"
       / os.environ.get("GF_OUT", "pyramid/results"))
SIDE = 48
LUM_W = 0.5
LAM = 0.5
KTOP = int(os.environ.get("GF_KTOP2", "1000"))
ETAS = [0.02] + [0.03] * 8
ETATOP = 0.04
MAX_EPOCHS = int(os.environ.get("GF_MAX_EPOCHS", "12"))
TIME_BUDGET_S = float(os.environ.get("GF_TIME_S", "1e9"))
DRIFT_STOP = 0.003
NORM_FLOOR = 0.15

# (win, stride, K) per dictionary level; level 0 reads pixels.
# Overridable: GF_LEVELS='[[4,1,64],[5,2,256],[5,1,256]]' etc.
LEVELS = [tuple(t) for t in json.loads(os.environ.get(
    "GF_LEVELS",
    "[[4,1,48],[3,2,64],[3,1,96],[3,2,128],[3,1,128],[3,2,160],[2,1,192]]"))]

GRIDS = []
g = SIDE
for win, stride, _ in LEVELS:
    g = (g - win) // stride + 1
    GRIDS.append(g)
CH = [LEVELS[0][2] + 1] + [k for _, _, k in LEVELS[1:]]   # channels per output map
CODE_DIM = GRIDS[-1] * GRIDS[-1] * LEVELS[-1][2]
ATTR_NAMES = json.loads((ROOT / "data/celeba/attr_names.json").read_text())
A_IDX = {n: i for i, n in enumerate(ATTR_NAMES)}
TOP_DIM = CODE_DIM + len(ATTR_NAMES)


def windows_x(maps, g_in, win, stride):
    """(N, g_in, g_in, C) -> (N, P, win*win*C) on xp."""
    pos = [(r, c) for r in range(0, g_in - win + 1, stride)
           for c in range(0, g_in - win + 1, stride)]
    parts = [maps[:, r:r + win, c:c + win, :].reshape(len(maps), -1) for r, c in pos]
    return xp.stack(parts, axis=1)


def pixel_pass(xb, dic, learning):
    """L1: stroke channels + appended raw-mean luminance channel."""
    N = len(xb)
    g = GRIDS[0]
    win = LEVELS[0][0]
    W = windows_x(xb[..., None], SIDE, win, LEVELS[0][1])       # (N,P,16)
    V = W.reshape(-1, win * win)
    mu = V.mean(axis=1)
    Vc = V - mu[:, None]
    n = xp.linalg.norm(Vc, axis=1)
    ok = n > NORM_FLOOR
    Vh = Vc / xp.maximum(n, 1e-9)[:, None]
    if learning and dic.n_boot < dic.k:
        dic.bootstrap(Vh[ok])
    C = Vh @ dic.W.T
    code = xp.maximum(C, 0.0) * ok[:, None]
    if learning and dic.n_boot >= dic.k:
        winners = xp.argmax(C, axis=1)
        cv = xp.max(C, axis=1) * ok
        keep = cv > 0
        dic.update(Vh[keep], winners[keep], cv[keep])
    lum = (LUM_W * mu)[:, None]
    out = xp.concatenate([code, lum], axis=1)
    return out.reshape(N, g, g, dic.k + 1)


def code_pass(prev, dic, li, learning):
    """Level li >= 1: dense matching/message, skeleton learning (lum excluded)."""
    win, stride, _ = LEVELS[li]
    g_in, g_out = GRIDS[li - 1], GRIDS[li]
    ch = CH[li - 1]
    Wf = windows_x(prev, g_in, win, stride)                     # (N,P,win*win*ch)
    V = Wf.reshape(-1, win * win * ch)
    Vh, ok = _center_norm_rows(V)
    if learning and dic.n_boot < dic.k:
        dic.bootstrap(Vh[ok])
    C = Vh @ dic.W.T
    code = xp.maximum(C, 0.0) * ok[:, None]
    if learning and dic.n_boot >= dic.k:
        blocks = Wf.reshape(-1, win * win, ch)
        strokes = blocks[:, :, :ch - 1] if li == 1 else blocks  # lum only exists at L1 maps
        am = xp.argmax(strokes, axis=2)
        vals = xp.take_along_axis(strokes, am[:, :, None], axis=2)
        sk = xp.zeros_like(blocks)
        if li == 1:
            skv = xp.zeros_like(strokes)
            xp.put_along_axis(skv, am[:, :, None], xp.maximum(vals, 0.0), axis=2)
            sk[:, :, :ch - 1] = skv
        else:
            xp.put_along_axis(sk, am[:, :, None], xp.maximum(vals, 0.0), axis=2)
        T = sk.reshape(len(blocks), -1)
        Th, oks = _center_norm_rows(T)
        Cs = Th @ dic.W.T
        winners = xp.argmax(Cs, axis=1)
        cv = xp.max(Cs, axis=1) * oks
        keep = cv > 0
        dic.update(Th[keep], winners[keep], cv[keep])
    return code.reshape(len(prev), g_out, g_out, dic.k)


def encode_stack(xb, dics, learning):
    m = pixel_pass(xb, dics[0], learning)
    maps = [m]
    for li in range(1, len(LEVELS)):
        m = code_pass(m, dics[li], li, learning)
        maps.append(m)
    return maps


def harden_np(m, lum_ch=False, k=1):
    o = np.zeros_like(m)
    ch = m.shape[2] - (1 if lum_ch else 0)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            seg = np.maximum(m[a, b, :ch], 0.0)
            if seg.max() > 0:
                idx = np.argsort(seg)[::-1][:k]
                o[a, b, idx] = seg[idx]
            if lum_ch:
                o[a, b, ch] = m[a, b, ch]
    return o


def render_down(code_top, dics_np):
    """Expand from the top dictionary's grid to pixels, with luminance."""
    m = code_top
    for li in range(len(LEVELS) - 1, 0, -1):
        win, stride, _ = LEVELS[li]
        g_prev = GRIDS[li - 1]
        m = expand(harden_np(m, lum_ch=False), dics_np[li], win, stride,
                   (g_prev, g_prev, CH[li - 1]))
        m = harden_np(m, lum_ch=(li - 1 == 0))
    # m is now the L1 canvas (G0, G0, K1+1): strokes + luminance channel.
    k1 = LEVELS[0][2]
    win, stride = LEVELS[0][0], LEVELS[0][1]
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    F = np.outer([0.5, 1, 1, 0.5][:win], [0.5, 1, 1, 0.5][:win])
    for gi in range(GRIDS[0]):
        for gj in range(GRIDS[0]):
            seg = np.maximum(m[gi, gj, :k1], 0.0)
            conf = float(seg.max())
            if conf <= 0:
                continue
            idx = int(np.argmax(seg))
            lum = m[gi, gj, k1] / LUM_W
            patch = lum + seg[idx] * dics_np[0].W[idx].reshape(win, win) / max(seg[idx], 1e-9) * seg[idx]
            patch = lum + dics_np[0].W[idx].reshape(win, win) * seg[idx]
            r, c = gi * stride, gj * stride
            num[r:r + win, c:c + win] += patch * F * conf
            den[r:r + win, c:c + win] += F * conf
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / "data/celeba/images_48.npy")
    A = np.load(ROOT / "data/celeba/attrs_48.npy").astype(np.float64)
    N = len(X)
    dims = [16] + [LEVELS[li][0] ** 2 * CH[li - 1] for li in range(1, len(LEVELS))]
    dics = [Dict(LEVELS[i][2], dims[i], ETAS[i]) for i in range(len(LEVELS))]
    top = Dict(KTOP, TOP_DIM, ETATOP)
    total = sum(d.W.size for d in dics) + top.W.size
    print(f"grids={GRIDS} code={CODE_DIM} params={total:,}", flush=True)

    Xx = xp.asarray(X.astype(np.float32))
    Lx = xp.asarray((A * LAM).astype(np.float32))
    t0 = time.time()
    prev_snap = None
    for ep in range(MAX_EPOCHS):
        for s in range(0, N, B):
            maps = encode_stack(Xx[s:s + B], dics, learning=True)
            H = maps[-1].reshape(len(maps[-1]), -1)
            H, okh = _center_norm_rows(H)
            Z = xp.concatenate([H, Lx[s:s + B]], axis=1)
            Z, okz = _center_norm_rows(Z)
            okz = okz & okh
            if top.n_boot < top.k:
                top.bootstrap(Z[okz])
            if top.n_boot >= top.k:
                Ct = Z @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cv = xp.max(Ct, axis=1) * okz
                keep = cv > 0
                top.update(Z[keep], winners[keep], cv[keep])
        snap = [xp.asnumpy(d.W.copy()) if XP_NAME == "cupy" else d.W.copy() for d in dics]
        if prev_snap is not None:
            drifts = []
            for a, b in zip(snap, prev_snap):
                cos = np.clip(np.sum(a * b, axis=1), -1, 1)
                drifts.append(float(np.mean(np.arccos(np.clip(cos, -1, 1)))))
            worst = max(drifts)
            print(f"epoch {ep + 1}: {time.time() - t0:.0f}s drift={['%.4f' % d for d in drifts]}",
                  flush=True)
            if worst < DRIFT_STOP:
                print(f"EARLY STOP at epoch {ep + 1} (max drift {worst:.4f})", flush=True)
                prev_snap = snap
                break
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"TIME BUDGET reached at epoch {ep + 1}", flush=True)
            break
        else:
            print(f"epoch {ep + 1}: {time.time() - t0:.0f}s (baseline)", flush=True)
        prev_snap = snap
    train_s = time.time() - t0

    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    dics_np = []
    for d in dics:
        class Bank:
            pass
        b = Bank()
        b.W = to_np(d.W)
        b.k = d.k
        dics_np.append(b)
    Wtn = to_np(top.W)
    wins = to_np(top.win_counts)
    np.savez(OUT / "weights.npz", **{f"W{i+1}": b.W for i, b in enumerate(dics_np)},
             Wtop=Wtn, top_wins=wins)
    if os.environ.get("GF_SKIP_EVAL"):
        print(f"RESULT train_s={train_s:.0f} params={total:,} (eval skipped)", flush=True)
        return

    # ---- Eval: encode subset on GPU without learning, pull codes ----------
    EV = 3000
    maps = encode_stack(Xx[:EV], dics, learning=False)
    codes = {li: to_np(maps[li].reshape(EV, -1)) for li in range(3, len(LEVELS))}
    male = A[:EV, A_IDX["Male"]]
    probes = {}
    for li, C in codes.items():
        clf = LogisticRegression(max_iter=500)
        cut = 2000
        clf.fit(C[:cut], male[:cut])
        probes[li] = float(clf.score(C[cut:], male[cut:]))
    print("male-probes:", {f"L{li+1}": round(p, 4) for li, p in probes.items()}, flush=True)

    # ---- Galleries at the new scales --------------------------------------
    for li, count in [(4, 48), (5, 60)]:
        g = GRIDS[li]
        k = LEVELS[li][2]
        cols = 12
        rows = int(np.ceil(count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
        for u, ax in enumerate(np.asarray(axes).flat):
            if u < count:
                c = np.zeros((g, g, k))
                c[g // 2, g // 2, u % k] = 1.0
                img = render_down(c, dics_np) if li == len(LEVELS) - 1 else None
                # expand from level li down:
                m = c
                for lj in range(li, 0, -1):
                    win, stride, _ = LEVELS[lj]
                    gp = GRIDS[lj - 1]
                    m = expand(harden_np(m), dics_np[lj], win, stride, (gp, gp, CH[lj - 1]))
                    m = harden_np(m, lum_ch=(lj - 1 == 0))
                # render L1 canvas
                mm = m
                k1 = LEVELS[0][2]
                num = np.zeros((SIDE, SIDE)); den = np.zeros((SIDE, SIDE))
                F = np.outer([0.5, 1, 1, 0.5], [0.5, 1, 1, 0.5])
                for gi in range(GRIDS[0]):
                    for gj in range(GRIDS[0]):
                        seg = np.maximum(mm[gi, gj, :k1], 0.0)
                        if seg.max() <= 0:
                            continue
                        idx = int(np.argmax(seg))
                        patch = dics_np[0].W[idx].reshape(4, 4) * seg[idx]
                        num[gi:gi + 4, gj:gj + 4] += patch * F * seg[idx]
                        den[gi:gi + 4, gj:gj + 4] += F * seg[idx]
                img = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
                a2 = np.abs(img)
                if a2.max() > 0:
                    ys, xs = np.where(a2 > 0.15 * a2.max())
                    img = img[max(ys.min()-2, 0):ys.max()+3, max(xs.min()-2, 0):xs.max()+3]
                ax.imshow(img, cmap="gray")
            ax.axis("off")
        fig.suptitle(f"L{li+1} units (RF ~{4 + sum((LEVELS[j][0]-1)*np.prod([LEVELS[i][1] for i in range(j)]) for j in range(1, li+1))}px)")
        fig.tight_layout()
        fig.savefig(OUT / f"units_L{li+1}.png", dpi=110)
        plt.close(fig)

    # ---- Roundtrip + archive sample (with luminance) ----------------------
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    top_maps = to_np(maps[-1][:8])
    for i in range(8):
        axes[0, i].imshow(X[i], cmap="gray")
        axes[1, i].imshow(render_down(top_maps[i], dics_np), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Roundtrip through 7 levels (with luminance)")
    fig.tight_layout()
    fig.savefig(OUT / "roundtrip.png", dpi=110)
    plt.close(fig)

    order = np.argsort(-wins)[:12]
    g = GRIDS[-1]
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for ax, mi in zip(axes.flat, order):
        code = np.maximum(Wtn[mi, :CODE_DIM], 0.0).reshape(g, g, LEVELS[-1][2])
        ax.imshow(render_down(code, dics_np), cmap="gray")
        ax.set_title(f"m{mi} ({int(wins[mi])})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Archive sample (8-level, luminance)")
    fig.tight_layout()
    fig.savefig(OUT / "archive_sample.png", dpi=110)
    plt.close(fig)

    print(f"RESULT train_s={train_s:.0f} params={total:,} probes={probes}", flush=True)
    (OUT / "report.md").write_text(
        f"# 8-level pyramid + luminance\n\ngrids {GRIDS}, params {total:,}, "
        f"train {train_s:.0f}s.\nMale-probes: {probes}\n"
        "Figures: units_L5.png, units_L6.png, roundtrip.png, archive_sample.png\n")


if __name__ == "__main__":
    main()
