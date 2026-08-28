"""Exp 2 — how should overlapping proposals blend on the way down?

Decode-only sweep on Exp 1's saved weights (no retraining). At every
blending step of the descent (L3->L2, L2->L1, L1->pixels) the overlapping
sources' proposals are combined by one of three rules:

  avg     plain average                    (Exp 1's rule — the baseline)
  conf    confidence-weighted average      (the old feathering, minus feather)
  winner  the most confident source WRITES (user's proposal: per target
          location, the source that matched its own template best wins;
          no blending at all)

Confidence of a source position = the max of its code segment (how well
that window matched its best template). Flat/empty windows get a tiny
confidence so the background is still covered.

Run:  .venv/bin/python experiments/2026_08_28/blend/run_blend.py
Env:  RC_TEST (shared with run_recon3), BL_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
import run_recon3 as R  # noqa: E402  (constants, encoders, gallery, load)

OUTPUT_DIR = HERE / "results" / os.environ.get('BL_TAG', '').lstrip('_')
WEIGHTS = HERE.parent / "recon_ladder" / "results" / "weights.npz"
CONF_EPS = 1e-6


def decode_l1_blend(code1, means, norms, W1, top1, blend):
    """L1 code -> pixels under one of the three blending rules."""
    num = np.zeros((R.SIDE, R.SIDE))
    den = np.zeros((R.SIDE, R.SIDE))
    best = np.full((R.SIDE, R.SIDE), -1.0)
    out = np.zeros((R.SIDE, R.SIDE))
    for gi in range(R.G1):
        for gj in range(R.G1):
            seg = np.maximum(code1[gi, gj], 0.0)
            conf = float(seg.max())
            if conf > 0.0:
                if top1:
                    m = int(np.argmax(seg))
                    v = seg[m] * W1[m]
                else:
                    v = seg @ W1
                vn = float(np.linalg.norm(v - v.mean()))
            else:
                vn = 0.0
            if conf <= 0.0 or vn < R.EPS:
                win = np.full((R.W1_WIN, R.W1_WIN), means[gi, gj])
                conf = CONF_EPS
            else:
                v_hat = (v - v.mean()) / vn
                win = (means[gi, gj] + norms[gi, gj] * v_hat).reshape(
                    R.W1_WIN, R.W1_WIN)
            r, c = gi * R.W1_STR, gj * R.W1_STR
            sl = (slice(r, r + R.W1_WIN), slice(c, c + R.W1_WIN))
            if blend == "avg":
                num[sl] += win
                den[sl] += 1.0
            elif blend == "conf":
                num[sl] += conf * win
                den[sl] += conf
            else:                                     # winner
                mask = conf > best[sl]
                out[sl][mask] = win[mask]
                best[sl][mask] = conf
    if blend == "winner":
        return out
    return np.where(den > CONF_EPS, num / np.maximum(den, CONF_EPS), 0.0)


def expand_blend(code3d, W, win, stride, prev_shape, blend):
    """One level down under one of the three blending rules."""
    acc = np.zeros(prev_shape)
    den = np.zeros(prev_shape[:2])
    best = np.full(prev_shape[:2], -1.0)
    out = np.zeros(prev_shape)
    for gi in range(code3d.shape[0]):
        for gj in range(code3d.shape[1]):
            seg = np.maximum(code3d[gi, gj], 0.0)
            conf = float(seg.max())
            if conf <= 0.0:
                continue
            block = (seg @ W).reshape(win, win, prev_shape[2])
            sl = (slice(gi * stride, gi * stride + win),
                  slice(gj * stride, gj * stride + win))
            if blend == "avg":
                acc[sl] += block
                den[sl] += 1.0
            elif blend == "conf":
                acc[sl] += conf * block
                den[sl] += conf
            else:                                     # winner
                mask = conf > best[sl]
                out[sl][mask] = block[mask]
                best[sl][mask] = conf
    if blend == "winner":
        return np.maximum(out, 0.0)
    acc = np.where(den[:, :, None] > CONF_EPS,
                   acc / np.maximum(den[:, :, None], CONF_EPS), 0.0)
    return np.maximum(acc, 0.0)


def recon_blend(level, code, means, norms, W1, W2, W3, top1, blend):
    if level == 3:
        c3 = R.harden(code) if top1 else code
        code = expand_blend(c3, W3, R.W3_WIN, R.W3_STR, (R.G2, R.G2, R.K2), blend)
        level = 2
    if level == 2:
        c2 = R.harden(code) if top1 else code
        code = expand_blend(c2, W2, R.W2_WIN, R.W2_STR, (R.G1, R.G1, R.K1), blend)
        level = 1
    return decode_l1_blend(code, means, norms, W1, top1, blend)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    w = np.load(WEIGHTS)
    W1, W2, W3 = w["W1"], w["W2"], w["W3"]

    _, _, Xte, _ = R.load()
    d1 = R.Layer(R.K1, W1.shape[1], 0.0, np.random.default_rng(0))
    d2 = R.Layer(R.K2, W2.shape[1], 0.0, np.random.default_rng(0))
    d3 = R.Layer(R.K3, W3.shape[1], 0.0, np.random.default_rng(0))
    d1.W, d2.W, d3.W = W1, W2, W3
    M1, means, norms = R.encode_pixels_batch(Xte, d1)
    M2 = R.encode_over_batch(M1, d2, R.W2_WIN, R.W2_STR, R.G2)
    M3 = R.encode_over_batch(M2, d3, R.W3_WIN, R.W3_STR, R.G3)
    codes = {1: M1, 2: M2, 3: M3}
    n = len(Xte)

    res = {"test_n": n}
    for level in (1, 2, 3):
        for mode, t1 in (("graded", False), ("top1", True)):
            for blend in ("avg", "conf", "winner"):
                ms, cs = [], []
                for t in range(n):
                    r = recon_blend(level, codes[level][t], means[t], norms[t],
                                    W1, W2, W3, t1, blend)
                    ms.append(float(((r - Xte[t]) ** 2).mean()))
                    cs.append(R.shape_corr(r, Xte[t]))
                res[f"L{level}_{mode}_{blend}_mse"] = round(float(np.mean(ms)), 5)
                res[f"L{level}_{mode}_{blend}_corr"] = round(float(np.mean(cs)), 3)
            print(f"L{level} {mode} done", flush=True)

    imgs, titles = [], []
    for t in range(6):
        imgs.append(Xte[t]); titles.append("orig")
        for blend in ("avg", "conf", "winner"):
            imgs.append(recon_blend(3, M3[t], means[t], norms[t],
                                    W1, W2, W3, False, blend))
            titles.append(f"g-{blend}")
        imgs.append(recon_blend(3, M3[t], means[t], norms[t],
                                W1, W2, W3, True, "winner"))
        titles.append("t1-winner")
    R.gallery(imgs, titles, OUTPUT_DIR / "blend_L3.png",
              "From L3: orig | graded avg/conf/winner | top1 winner", 5)

    imgs, titles = [], []
    for t in range(6):
        imgs.append(Xte[t]); titles.append("orig")
        for blend in ("avg", "conf", "winner"):
            imgs.append(recon_blend(1, M1[t], means[t], norms[t],
                                    W1, W2, W3, True, blend))
            titles.append(f"t1-{blend}")
    R.gallery(imgs, titles, OUTPUT_DIR / "blend_L1_top1.png",
              "From L1, top-1 read: orig | avg | conf | winner", 4)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
