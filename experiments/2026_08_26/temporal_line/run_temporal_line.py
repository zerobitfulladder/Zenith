"""Temporal v2: rotating line, two layers — spatial L1 + temporal L2.

Stimulus: 16x16 canvas, anti-aliased line of radius 7 through the
center, rotating 4 deg/frame (45-frame loop; theta and theta+180 are
identical). No labels.

L1 (spatial): 8x8 windows, stride 4 (3x3 grid), K1=64, standard dense
pixel-window learning -> dense relu code (dim 9*K1), L2-normalized.
L2 (temporal): the two-timescale mechanism over L1's code — learns
[fast trace ; g_s*slow trace ; g_n*NEXT code], top-1 geodesic.
Trained sequentially (L1 first, then frozen) to avoid nonstationarity.

Playback: prime traces on true frames, then query with next-half
empty; the winner's stored next-code advances the state; each
predicted code renders to pixels through L1 (top-1 per position,
feathered overlap-add) for the GIF.

Run:  .venv/bin/python experiments/2026_08_26/temporal_line/run_temporal_line.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_gpu_minibatch import Dict  # noqa: E402  (numpy mode via GF_XP)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "temporal_line" / "results"
H = 16
DEG_STEP = 4
PERIOD = 180 // DEG_STEP            # 45 frames
RADIUS, WIDTH = 7.0, 1.2
WIN, STR = 8, 4
POS = [(r, c) for r in range(0, H - WIN + 1, STR) for c in range(0, H - WIN + 1, STR)]
G = int(np.sqrt(len(POS)))          # 3
K1, ETA1 = 64, 0.05
CODE_DIM = len(POS) * K1            # 576
K2, ETA2 = 96, 0.05
ALPHA_F, ALPHA_S = 0.5, 0.06
G_S, G_N = 0.5, 0.5
L1_FRAMES = 1500
EPOCHS2, TRAIN_FRAMES = 3, 2000
PRIME, FREE_RUN = 60, 120
NORM_FLOOR = 1e-3
_f = np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5])
FEATHER = np.outer(_f, _f)


def frame_at(t):
    th = np.deg2rad((t * DEG_STEP) % 180)
    d = np.array([np.cos(th), np.sin(th)])
    yy, xx = np.mgrid[0:H, 0:H]
    p = np.stack([xx - (H - 1) / 2, yy - (H - 1) / 2], axis=-1)
    along = p @ d
    perp = np.abs(p[..., 0] * d[1] - p[..., 1] * d[0])
    img = np.clip(1.0 - perp / WIDTH, 0, 1) * (np.abs(along) <= RADIUS)
    return img.astype(np.float32), (t * DEG_STEP) % 180


def cn_rows(V):
    V = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(V, axis=1)
    return V / np.maximum(n, 1e-9)[:, None], n > NORM_FLOOR


def encode(img, W1):
    V = np.stack([img[r:r + WIN, c:c + WIN].ravel() for r, c in POS])
    Vh, ok = cn_rows(V)
    c = np.maximum(Vh @ W1.T, 0.0) * ok[:, None]
    f = c.ravel()
    return f / (np.linalg.norm(f) + 1e-9)


def render(code, W1):
    c = code.reshape(len(POS), K1)
    num = np.zeros((H, H))
    den = np.zeros((H, H))
    for i, (r, cc) in enumerate(POS):
        seg = np.maximum(c[i], 0.0)
        if seg.max() <= 0:
            continue
        one = np.zeros(K1)
        one[int(np.argmax(seg))] = seg.max()
        patch = (one @ W1).reshape(WIN, WIN)
        conf = float(seg.max())
        num[r:r + WIN, cc:cc + WIN] += patch * FEATHER * conf
        den[r:r + WIN, cc:cc + WIN] += FEATHER * conf
    img = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    v = np.maximum(img, 0.0)
    return v / (v.max() + 1e-9)


def angle_of(img):
    yy, xx = np.mgrid[0:H, 0:H]
    m = img.sum()
    if m <= 0:
        return None
    cx, cy = (img * xx).sum() / m, (img * yy).sum() / m
    mu20 = (img * (xx - cx) ** 2).sum()
    mu02 = (img * (yy - cy) ** 2).sum()
    mu11 = (img * (xx - cx) * (yy - cy)).sum()
    return np.rad2deg(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % 180


def ang_err(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def save_gif(frames, path, scale=14, ms=80):
    imgs = [Image.fromarray((np.kron(np.clip(f, 0, 1),
                                     np.ones((scale, scale))) * 255
                             ).astype(np.uint8)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_gif([frame_at(t)[0] for t in range(2 * PERIOD)],
             OUTPUT_DIR / "input.gif")

    # ---- Phase 1: spatial L1 ---------------------------------------------
    d1 = Dict(K1, WIN * WIN, ETA1)
    for t in range(L1_FRAMES):
        img, _ = frame_at(t)
        V = np.stack([img[r:r + WIN, c:c + WIN].ravel() for r, c in POS])
        Vh, ok = cn_rows(V)
        Vh = Vh[ok]
        if d1.n_boot < d1.k:
            d1.bootstrap(Vh)
            continue
        C = Vh @ d1.W.T
        w = np.argmax(C, axis=1)
        cv = np.maximum(C[np.arange(len(w)), w], 0.0).astype(np.float32)
        d1.update(Vh, w, cv)
    W1 = d1.W

    # ---- Phase 2: temporal L2 over frozen L1 codes ------------------------
    d2 = Dict(K2, 3 * CODE_DIM, ETA2)
    for _ in range(EPOCHS2):
        F = np.zeros(CODE_DIM, dtype=np.float32)
        S = np.zeros(CODE_DIM, dtype=np.float32)
        for t in range(TRAIN_FRAMES):
            c_now = encode(frame_at(t)[0], W1)
            F = ALPHA_F * c_now + (1 - ALPHA_F) * F
            S = ALPHA_S * c_now + (1 - ALPHA_S) * S
            c_next = encode(frame_at(t + 1)[0], W1)
            z, _ = cn_rows(np.concatenate([F, G_S * S, G_N * c_next])[None, :])
            if d2.n_boot < d2.k:
                d2.bootstrap(z)
                continue
            C = (z @ d2.W.T)[0]
            w = int(np.argmax(C))
            d2.update(z, np.array([w]),
                      np.array([max(C[w], 0.0)], dtype=np.float32))

    # ---- Prime, then free-run --------------------------------------------
    F = np.zeros(CODE_DIM, dtype=np.float32)
    S = np.zeros(CODE_DIM, dtype=np.float32)
    for t in range(PRIME):
        c_now = encode(frame_at(t)[0], W1)
        F = ALPHA_F * c_now + (1 - ALPHA_F) * F
        S = ALPHA_S * c_now + (1 - ALPHA_S) * S

    gen, errs = [], []
    for i in range(FREE_RUN):
        q, _ = cn_rows(np.concatenate(
            [F, G_S * S, np.zeros(CODE_DIM, dtype=np.float32)])[None, :])
        winner = int(np.argmax(d2.W @ q[0]))
        c_hat = np.maximum(d2.W[winner, 2 * CODE_DIM:], 0.0)
        c_hat = c_hat / (np.linalg.norm(c_hat) + 1e-9)
        img = render(c_hat, W1)
        gen.append(img)
        a = angle_of(img)
        true_a = frame_at(PRIME + i)[1]
        if a is not None:
            errs.append(ang_err(a, true_a))
        F = ALPHA_F * c_hat + (1 - ALPHA_F) * F
        S = ALPHA_S * c_hat + (1 - ALPHA_S) * S

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    errs = np.array(errs)
    print(f"free-run {FREE_RUN} frames: mean angle error={errs.mean():.1f} deg, "
          f"max={errs.max():.1f} deg", flush=True)

    # ---- Filmstrip + L1 gallery ------------------------------------------
    fig, axes = plt.subplots(4, 15, figsize=(14, 4.6))
    for i in range(15):
        axes[0, i].imshow(frame_at(PRIME + 3 * i)[0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(gen[3 * i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(frame_at(PRIME + 45 + 3 * i)[0], cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(gen[45 + 3 * i], cmap="gray", vmin=0, vmax=1)
        for r in range(4):
            axes[r, i].axis("off")
    fig.suptitle("Rotating line: true vs free-run (every 3rd frame, 2 stretches)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filmstrip.png", dpi=110)
    plt.close(fig)

    order = np.argsort(d1.win_counts)[::-1]
    fig, axes = plt.subplots(8, 8, figsize=(8, 8.3))
    for ax, u in zip(axes.flat, order):
        ax.imshow(W1[u].reshape(WIN, WIN), cmap="gray")
        ax.axis("off")
    fig.suptitle("L1 spatial templates (8x8), most-rehearsed first")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l1_templates.png", dpi=110)
    plt.close(fig)

    order2 = np.argsort(d2.win_counts)[::-1][:48]
    fig, axes = plt.subplots(6, 8, figsize=(11, 9))
    for ax, u in zip(axes.flat, order2):
        row = d2.W[u]
        fast = render(np.maximum(row[:CODE_DIM], 0.0), W1)
        nxt = render(np.maximum(row[2 * CODE_DIM:], 0.0), W1)
        tile = np.hstack([fast, np.full((H, 1), np.nan), nxt])
        ax.imshow(np.ma.masked_invalid(tile), cmap="gray")
        ax.axis("off")
    fig.suptitle("L2 temporal units, most-rehearsed first — "
                 "left: fast-trail half (rendered), right: emitted next code")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l2_units.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join([
        "# Temporal v2: rotating line, spatial L1 + temporal L2",
        "",
        f"K1={K1} (8x8 s4), K2={K2}, period {PERIOD} frames, no labels.",
        f"Free-run {FREE_RUN} frames: mean angle error {errs.mean():.1f} deg, "
        f"max {errs.max():.1f} deg.",
        "",
        "Files: input.gif, generated.gif, filmstrip.png, l1_templates.png, "
        "l2_units.png",
    ]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
