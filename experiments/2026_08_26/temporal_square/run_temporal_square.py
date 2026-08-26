"""Temporal v1: a bouncing square, learned and replayed with EMA traces.

Stimulus: 8x24 canvas, 4x4 square sweeping right then left (40-frame
loop). No labels. Each timestep's learning vector is the joint state
[fast EMA trace ; g_s * slow EMA trace ; g_n * NEXT frame] — the user's
two-timescale mechanism: either trace alone is ambiguous, the pair
plus trail disambiguates direction and phase. Standard top-1 geodesic
learning on the centered/normalized triple (the label-concat pattern,
aimed at time).

Playback: prime traces with 1.5 true cycles (no learning), then query
[fast ; g_s*slow ; empty], emit the winner's stored next-half as the
next frame, update traces with the emission, repeat — auto-advance.

Outputs: input.gif, generated.gif, filmstrip.png, report.md.

Run:  .venv/bin/python experiments/2026_08_26/temporal_square/run_temporal_square.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "temporal_square" / "results"
H, W, SQ = 8, 24, 4
XMAX = W - SQ                       # 20
PERIOD = 2 * XMAX                   # 40 frames per loop
ALPHA_F, ALPHA_S = 0.5, 0.06        # fast / slow trace rates
G_S, G_N = 0.5, 0.5                 # slow-trace and next-frame gains
K, ETA = 64, 0.05
EPOCHS = 3
TRAIN_FRAMES = 2000
PRIME = 60
FREE_RUN = 100
D = H * W


def frame_at(t):
    ph = t % PERIOD
    x = ph if ph <= XMAX else PERIOD - ph
    f = np.zeros((H, W), dtype=np.float32)
    f[2:2 + SQ, x:x + SQ] = 1.0
    return f, x


def center_norm_row(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v), n


def joint(F, S, nxt):
    z, _ = center_norm_row(np.concatenate(
        [F.ravel(), G_S * S.ravel(), G_N * nxt.ravel()]))
    return z


def save_gif(frames, path, scale=12, ms=80):
    imgs = []
    for f in frames:
        g = np.clip(f, 0, 1)
        big = np.kron(g, np.ones((scale, scale)))
        imgs.append(Image.fromarray((big * 255).astype(np.uint8)))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_gif([frame_at(t)[0] for t in range(2 * PERIOD)],
             OUTPUT_DIR / "input.gif")

    dic = Dict(K, 3 * D, ETA)
    for _ in range(EPOCHS):
        F = np.zeros((H, W), dtype=np.float32)
        S = np.zeros((H, W), dtype=np.float32)
        for t in range(TRAIN_FRAMES):
            x, _ = frame_at(t)
            F = ALPHA_F * x + (1 - ALPHA_F) * F
            S = ALPHA_S * x + (1 - ALPHA_S) * S
            nxt, _ = frame_at(t + 1)
            z = joint(F, S, nxt)[None, :]
            if dic.n_boot < dic.k:
                dic.bootstrap(z)
                continue
            C = (z @ dic.W.T)[0]
            w = int(np.argmax(C))
            dic.update(z, np.array([w]), np.array([max(C[w], 0.0)],
                                                  dtype=np.float32))

    # ---- Prime with the true sequence, then free-run ----------------------
    F = np.zeros((H, W), dtype=np.float32)
    S = np.zeros((H, W), dtype=np.float32)
    for t in range(PRIME):
        x, _ = frame_at(t)
        F = ALPHA_F * x + (1 - ALPHA_F) * F
        S = ALPHA_S * x + (1 - ALPHA_S) * S

    gen, true_x, gen_x = [], [], []
    for i in range(FREE_RUN):
        q, _ = center_norm_row(np.concatenate(
            [F.ravel(), G_S * S.ravel(), np.zeros(D, dtype=np.float32)]))
        winner = int(np.argmax(dic.W @ q))
        pred = np.maximum(dic.W[winner, 2 * D:], 0.0)
        if pred.max() > 0:
            pred = pred / pred.max()
        nxt = pred.reshape(H, W)
        gen.append(nxt)
        tx = frame_at(PRIME + i)[1]
        true_x.append(tx)
        col = nxt.sum(axis=0)
        gen_x.append(int(np.argmax(np.convolve(col, np.ones(SQ), "valid"))))
        F = ALPHA_F * nxt + (1 - ALPHA_F) * F
        S = ALPHA_S * nxt + (1 - ALPHA_S) * S

    save_gif(gen, OUTPUT_DIR / "generated.gif")

    err = np.abs(np.array(true_x) - np.array(gen_x))
    reversals = np.sign(np.diff(gen_x))
    n_rev = int(np.sum(np.abs(np.diff(reversals[reversals != 0])) > 0))
    print(f"free-run {FREE_RUN} frames: mean |x error|={err.mean():.2f}, "
          f"max={err.max()}, direction reversals in generation={n_rev}",
          flush=True)

    fig, axes = plt.subplots(4, 13, figsize=(13, 4.4))
    for i in range(13):
        axes[0, i].imshow(frame_at(PRIME + 2 * i)[0], cmap="gray",
                          vmin=0, vmax=1)
        axes[1, i].imshow(gen[2 * i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(frame_at(PRIME + 26 + 2 * i)[0], cmap="gray",
                          vmin=0, vmax=1)
        axes[3, i].imshow(gen[26 + 2 * i], cmap="gray", vmin=0, vmax=1)
        for r in range(4):
            axes[r, i].axis("off")
    for r, lbl in enumerate(["true", "gen", "true", "gen"]):
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(False)
        axes[r, 0].set_ylabel(lbl, fontsize=8, rotation=0, ha="right",
                              va="center")
    fig.suptitle("Bouncing square: true continuation vs free-run generation "
                 "(every 2nd frame)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filmstrip.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join([
        "# Temporal v1: bouncing square, EMA-trace auto-advance",
        "",
        f"K={K}, alphas fast/slow {ALPHA_F}/{ALPHA_S}, gains slow/next "
        f"{G_S}/{G_N}, {EPOCHS} epochs x {TRAIN_FRAMES} frames, no labels.",
        "",
        f"Free-run {FREE_RUN} frames after {PRIME}-frame prime: "
        f"mean |x err| {err.mean():.2f}, max {err.max()}, "
        f"reversals {n_rev}.",
        "",
        "Files: input.gif, generated.gif, filmstrip.png",
    ]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
