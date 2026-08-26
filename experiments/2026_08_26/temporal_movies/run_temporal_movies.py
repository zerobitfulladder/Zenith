"""Temporal v3: THREE labeled movies in one network, cold-start playback.

Movies (16x16, anti-aliased, looping): rot = center line rotating
4 deg/frame (period 45); tb = horizontal line scanning top->bottom
(period 16); lr = vertical line scanning left->right (period 16).
Shared frames across movies (rot@90 ~ lr mid, rot@0 ~ tb mid) make
single frames ambiguous — label + trail must separate the movies.

L1 spatial (shared, 8x8 s4, K1=64) trained on interleaved frames, then
frozen. L2 temporal joint = [fast ; g_s*slow ; g_lab*onehot(3) ;
g_n*next code], K2=128, episodes per movie with traces reset.

Playback: COLD START — zero traces, query [0; 0; label; 0]; the label
alone picks the entry unit; emissions build the traces; auto-advance.

Metrics per label: movie purity (each generated frame matched to the
nearest true frame across all movies) and phase-advance fraction.

Run:  .venv/bin/python experiments/2026_08_26/temporal_movies/run_temporal_movies.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_line"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_gpu_minibatch import Dict  # noqa: E402
from run_temporal_line import (  # noqa: E402
    H,
    WIN,
    K1,
    ETA1,
    cn_rows,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "temporal_movies" / "results"
WIDTH = 1.2
# Stride 2 (vs 4 in the rotating-line rig): scan movies translate by 1px
# per frame and a stride-4 code is too translation-tolerant — adjacent
# phases collapse onto shared units (crowding -> superposition fixed
# points). Finer stride restores positional resolution in the code.
STR = 2
POS = [(r, c) for r in range(0, H - WIN + 1, STR) for c in range(0, H - WIN + 1, STR)]
CODE_DIM = len(POS) * K1            # 5*5*64 = 1600
NORM_FLOOR = 1e-3
_f = np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5])
FEATHER = np.outer(_f, _f)
K2, ETA2 = 160, 0.05
ALPHA_F, ALPHA_S = 0.35, 0.06       # longer fast trail: stronger phase signal
G_S, G_LAB, G_N = 0.5, 0.5, 0.5


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
L1_STEPS = 600
EPOCHS2, EP_FRAMES = 3, 600
FREE_RUN = 100
JDIM = 3 * CODE_DIM + 3


def frame_rot(t):
    th = np.deg2rad((t * 4) % 180)
    d = np.array([np.cos(th), np.sin(th)])
    yy, xx = np.mgrid[0:H, 0:H]
    p = np.stack([xx - (H - 1) / 2, yy - (H - 1) / 2], axis=-1)
    along = p @ d
    perp = np.abs(p[..., 0] * d[1] - p[..., 1] * d[0])
    return (np.clip(1.0 - perp / WIDTH, 0, 1)
            * (np.abs(along) <= 7.0)).astype(np.float32)


def frame_tb(t):
    y = t % H
    yy = np.arange(H)[:, None] * np.ones((1, H))
    return np.clip(1.0 - np.abs(yy - y) / WIDTH, 0, 1).astype(np.float32)


def frame_lr(t):
    x = t % H
    xx = np.ones((H, 1)) * np.arange(H)[None, :]
    return np.clip(1.0 - np.abs(xx - x) / WIDTH, 0, 1).astype(np.float32)


MOVIES = [("rot", frame_rot, 45), ("tb", frame_tb, 16), ("lr", frame_lr, 16)]


def joint(F, S, lab, nxt):
    z, _ = cn_rows(np.concatenate(
        [F, G_S * S, G_LAB * lab, G_N * nxt])[None, :])
    return z


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn, period in MOVIES:
        save_gif([fn(t) for t in range(2 * period)],
                 OUTPUT_DIR / f"input_{name}.gif")

    # ---- L1 on interleaved frames ----------------------------------------
    d1 = Dict(K1, WIN * WIN, ETA1)
    for step in range(L1_STEPS):
        for _, fn, _ in MOVIES:
            img = fn(step)
            V = np.stack([img[r:r + WIN, c:c + WIN].ravel() for r, c in POS])
            Vh, ok = cn_rows(V)
            Vh = Vh[ok]
            if len(Vh) == 0:
                continue
            if d1.n_boot < d1.k:
                d1.bootstrap(Vh)
                continue
            C = Vh @ d1.W.T
            w = np.argmax(C, axis=1)
            cv = np.maximum(C[np.arange(len(w)), w], 0.0).astype(np.float32)
            d1.update(Vh, w, cv)
    W1 = d1.W

    # ---- L2: all three movies INTERLEAVED --------------------------------
    # Sequential episodes starve later movies at bootstrap (the first
    # movie's frames adopt every unit; tb/lr then collapse onto 1 unit
    # each — measured). Interleaving gives bootstrap a fair mix.
    d2 = Dict(K2, JDIM, ETA2)
    labs = np.eye(3, dtype=np.float32)
    for _ in range(EPOCHS2):
        Fs = [np.zeros(CODE_DIM, dtype=np.float32) for _ in MOVIES]
        Ss = [np.zeros(CODE_DIM, dtype=np.float32) for _ in MOVIES]
        for t in range(EP_FRAMES):
            for mi, (_, fn, _) in enumerate(MOVIES):
                c_now = encode(fn(t), W1)
                Fs[mi] = ALPHA_F * c_now + (1 - ALPHA_F) * Fs[mi]
                Ss[mi] = ALPHA_S * c_now + (1 - ALPHA_S) * Ss[mi]
                z = joint(Fs[mi], Ss[mi], labs[mi], encode(fn(t + 1), W1))
                if d2.n_boot < d2.k:
                    d2.bootstrap(z)
                    continue
                C = (z @ d2.W.T)[0]
                w = int(np.argmax(C))
                d2.update(z, np.array([w]),
                          np.array([max(C[w], 0.0)], dtype=np.float32))

    # ---- True-frame library for purity metric ----------------------------
    lib_imgs, lib_movie, lib_phase = [], [], []
    for mi, (_, fn, period) in enumerate(MOVIES):
        for t in range(period):
            f = fn(t)
            lib_imgs.append(f.ravel() / (np.linalg.norm(f) + 1e-9))
            lib_movie.append(mi)
            lib_phase.append(t)
    LIB = np.stack(lib_imgs)
    lib_movie, lib_phase = np.array(lib_movie), np.array(lib_phase)

    # ---- Cold-start free-run per label -----------------------------------
    rows = []
    strips = []
    for mi, (name, fn, period) in enumerate(MOVIES):
        lab = np.zeros(3, dtype=np.float32)
        lab[mi] = 1.0
        F = np.zeros(CODE_DIM, dtype=np.float32)
        S = np.zeros(CODE_DIM, dtype=np.float32)
        gen, match_m, match_p, winners = [], [], [], set()
        for _ in range(FREE_RUN):
            q = joint(F, S, lab, np.zeros(CODE_DIM, dtype=np.float32))
            winner = int(np.argmax(d2.W @ q[0]))
            winners.add(winner)
            c_hat = np.maximum(d2.W[winner, 2 * CODE_DIM + 3:], 0.0)
            c_hat = c_hat / (np.linalg.norm(c_hat) + 1e-9)
            img = render(c_hat, W1)
            gen.append(img)
            v = img.ravel() / (np.linalg.norm(img) + 1e-9)
            b = int(np.argmax(LIB @ v))
            match_m.append(lib_movie[b])
            match_p.append(lib_phase[b])
            F = ALPHA_F * c_hat + (1 - ALPHA_F) * F
            S = ALPHA_S * c_hat + (1 - ALPHA_S) * S
        save_gif(gen, OUTPUT_DIR / f"gen_{name}.gif")
        strips.append((name, fn, gen))

        match_m, match_p = np.array(match_m), np.array(match_p)
        purity = float((match_m == mi).mean())
        own = np.where(match_m == mi)[0]
        step1 = np.diff(own) == 1
        adv = np.diff(match_p[own]) % period
        moved = (adv >= 1) & (adv <= 3)
        advance = float(moved[step1].mean()) if step1.any() else 0.0
        rows.append((name, purity, advance))
        print(f"LABEL {name}: purity={purity:.2f} advance={advance:.2f} "
              f"distinct_units={len(winners)}", flush=True)

    # ---- Two labels at once: superposed cue ------------------------------
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        na, nb = MOVIES[a][0], MOVIES[b][0]
        lab = (np.eye(3, dtype=np.float32)[a]
               + np.eye(3, dtype=np.float32)[b])
        F = np.zeros(CODE_DIM, dtype=np.float32)
        S = np.zeros(CODE_DIM, dtype=np.float32)
        gen2, mm = [], []
        for _ in range(FREE_RUN):
            q = joint(F, S, lab, np.zeros(CODE_DIM, dtype=np.float32))
            winner = int(np.argmax(d2.W @ q[0]))
            c_hat = np.maximum(d2.W[winner, 2 * CODE_DIM + 3:], 0.0)
            c_hat = c_hat / (np.linalg.norm(c_hat) + 1e-9)
            img = render(c_hat, W1)
            gen2.append(img)
            v = img.ravel() / (np.linalg.norm(img) + 1e-9)
            mm.append(lib_movie[int(np.argmax(LIB @ v))])
            F = ALPHA_F * c_hat + (1 - ALPHA_F) * F
            S = ALPHA_S * c_hat + (1 - ALPHA_S) * S
        save_gif(gen2, OUTPUT_DIR / f"gen_pair_{na}_{nb}.gif")
        mm = np.array(mm)
        fr = [float((mm == i).mean()) for i in range(3)]
        switches = int((np.diff(mm) != 0).sum())
        print(f"PAIR {na}+{nb}: frames rot/tb/lr = "
              f"{fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}, switches={switches}",
              flush=True)

    fig, axes = plt.subplots(6, 15, figsize=(14, 6.6))
    for s, (name, fn, gen) in enumerate(strips):
        for i in range(15):
            axes[2 * s, i].imshow(fn(i * 2), cmap="gray", vmin=0, vmax=1)
            axes[2 * s + 1, i].imshow(gen[i * 2], cmap="gray", vmin=0, vmax=1)
            axes[2 * s, i].axis("off")
            axes[2 * s + 1, i].axis("off")
        axes[2 * s, 0].axis("on")
        axes[2 * s, 0].set_xticks([])
        axes[2 * s, 0].set_yticks([])
        for sp in axes[2 * s, 0].spines.values():
            sp.set_visible(False)
        axes[2 * s, 0].set_ylabel(f"{name}\ntrue/gen", fontsize=7,
                                  rotation=0, ha="right", va="center")
    fig.suptitle("Three labeled movies — true vs cold-start free-run "
                 "(every 2nd frame; gen phase-offset is free)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filmstrip.png", dpi=110)
    plt.close(fig)

    lines = [
        "# Temporal v3: three labeled movies, cold-start playback",
        "",
        f"K1={K1}, K2={K2}, labels 3, no priming — label alone is the cue.",
        "",
        "| movie | purity | advance |",
        "|---|---|---|",
    ]
    for name, purity, advance in rows:
        lines.append(f"| {name} | {purity:.2f} | {advance:.2f} |")
    lines += ["", "Files: input_*.gif, gen_*.gif, filmstrip.png"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
