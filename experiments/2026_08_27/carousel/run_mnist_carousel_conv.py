"""MNIST carousel with the CONVOLUTIONAL uniform unit (user's design).

One unit type everywhere: bounded in space by a receptive field,
in time by a leaky trail, with the delayed-write arrow.
  L1: 8x8 windows, stride 1 (21x21 positions), SHARED dictionary
      K1=64, per-position trails (gamma .5), rows
      [GC*arriving window ; lagged trail]. Blank windows skipped
      (zero-symbol law). Output: per-position winner -> code map.
  L2: the same unit OVER THE CODE (receptive field = whole code map
      for v1; conv L2 comes later), slower trail (gamma .9), rows
      [GC*code map ; lagged code trail]. The carousel's temporal
      structure lives HERE — over the code, not the pixels.

Task: class sequence 0..9 looping, fresh exemplar per class per lap.
Generation bounce: L2 successor-read (cargo-empty) -> cargo = next
code map -> graded reconstruction through L1's cargo windows ->
overlap-add render -> re-encode own emission -> bounce.

Run: .venv/bin/python experiments/2026_08_27/carousel/run_mnist_carousel_conv.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_square_completion import cn1  # noqa: E402
from run_temporal_square import save_gif  # noqa: E402
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "carousel" / "results" / "mnist_carousel_conv"
SIDE, WIN = 28, 8
NP1 = SIDE - WIN + 1                    # 21 positions per axis
NPOS = NP1 * NP1                        # 441 windows
WD = WIN * WIN                          # 64 dims per window
K1, K2 = 64, 32
G1, G2 = 0.5, float(os.environ.get("CC_G2", "0.5"))
GC = 0.5
EPS = 1e-6
EPOCHS = 2
TRAIN_TICKS = 2000                      # 200 laps, fresh exemplars
PRIME = 50
FREE = 100
CODE_D = NPOS * K1

X = np.load(ROOT / "data/mnist/digits/train_images.npy")
y = np.load(ROOT / "data/mnist/digits/train_labels.npy")
BYCLASS = [X[y == c] for c in range(10)]
CM_N = np.stack([cn1(b.mean(axis=0).ravel()) for b in BYCLASS])


def frame_for(t):
    c, lap = t % 10, t // 10
    return BYCLASS[c][lap % len(BYCLASS[c])].astype(np.float32)


def classify(frame):
    return int(np.argmax(CM_N @ cn1(frame.ravel())))


def win_view(frame):
    """(NPOS, WD) view of all 8x8 windows."""
    s = np.lib.stride_tricks.sliding_window_view(frame, (WIN, WIN))
    return s.reshape(NPOS, WD)


def cn_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.maximum(n, 1e-9)).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b1 = Dict(K1, 2 * WD, 0.05)
    b2 = Dict(K2, 2 * CODE_D, 0.05)

    def encode(frame, T1, learning):
        """Per-position L1 pass. Returns code map (NPOS,K1 one-hots)
        and updates trails (delayed write: learn/match first)."""
        wins = win_view(frame)
        live = np.where((wins.max(axis=1) > EPS)
                        | (T1.max(axis=1) > EPS))[0]
        code = np.zeros((NPOS, K1), np.float32)
        if len(live):
            Z = np.concatenate([GC * cn_rows(wins[live]),
                                cn_rows(T1[live])], axis=1)
            Z = cn_rows(Z)
            if learning and b1.n_boot < K1:
                b1.bootstrap(Z[: K1 - b1.n_boot])
            if b1.n_boot >= K1:
                C = Z @ b1.W.T
                ws = np.argmax(C, axis=1)
                if learning:
                    cv = np.maximum(C[np.arange(len(live)), ws], 0.0)
                    b1.update(Z, ws.astype(np.int64),
                              cv.astype(np.float32))
                if os.environ.get("CC_VIEW", "graded") == "skeleton":
                    code[live, ws] = np.maximum(
                        C[np.arange(len(live)), ws], 0.0)
                else:
                    code[live] = np.maximum(C, 0.0)   # graded view
        T1 *= G1
        T1 += wins
        return code

    def l2_vec(code_flat, T2):
        return cn1(np.concatenate([GC * cn1(code_flat), cn1(T2)]))

    # ---- Train ------------------------------------------------------------
    for _ in range(EPOCHS):
        T1 = np.zeros((NPOS, WD), np.float32)
        T2 = np.zeros(CODE_D, np.float32)
        for t in range(TRAIN_TICKS):
            code = encode(frame_for(t), T1, True).ravel()
            if b1.n_boot >= K1:
                z = l2_vec(code, T2)[None, :]
                if b2.n_boot < K2:
                    b2.bootstrap(z)
                else:
                    C = (z @ b2.W.T)[0]
                    w = int(np.argmax(C))
                    b2.update(z, np.array([w]),
                              np.array([max(C[w], 0.0)], dtype=np.float32))
            T2 = code + G2 * T2

    # ---- Phase B: FRESH L2 on the now-stable code (staged training) -------
    b2 = Dict(K2, 2 * CODE_D, 0.05)
    wins_hist = np.zeros(K2)
    for _ in range(2):
        T1 = np.zeros((NPOS, WD), np.float32)
        T2 = np.zeros(CODE_D, np.float32)
        for t in range(TRAIN_TICKS):
            code = encode(frame_for(t), T1, False).ravel()   # L1 frozen
            z = l2_vec(code, T2)[None, :]
            if b2.n_boot < K2:
                b2.bootstrap(z)
            else:
                C = (z @ b2.W.T)[0]
                w = int(np.argmax(C))
                b2.update(z, np.array([w]),
                          np.array([max(C[w], 0.0)], dtype=np.float32))
                wins_hist[w] += 1
            T2 = code + G2 * T2
    top_share = wins_hist.max() / max(wins_hist.sum(), 1)
    n_active = int((wins_hist > 0).sum())
    print(f"Phase-B L2 training: {n_active}/{K2} rows ever win, "
          f"top row share {top_share:.2f}", flush=True)

    np.savez(OUTPUT_DIR / "weights.npz", W1=b1.W, W2=b2.W)

    cargo_basis = np.maximum(b1.W[:, :WD], 0.0)     # (K1, WD)

    def render(prof, mode):
        canvas = np.zeros((SIDE, SIDE), np.float32)
        weight = np.zeros((SIDE, SIDE), np.float32)
        mass = prof.sum(axis=1)
        thr = 0.1 * mass.max()
        if mode == "graded":
            pred_wins = prof @ cargo_basis
        else:                                        # committed per position
            pred_wins = cargo_basis[np.argmax(prof, axis=1)]
        for p in range(NPOS):
            if mass[p] <= thr:
                continue
            r, c = divmod(p, NP1)
            canvas[r:r + WIN, c:c + WIN] += pred_wins[p].reshape(WIN, WIN)
            weight[r:r + WIN, c:c + WIN] += 1.0
        canvas /= np.maximum(weight, 1.0)
        return canvas / canvas.max() if canvas.max() > 0 else canvas

    def successor_prof(T2):
        q = cn1(np.concatenate([np.zeros(CODE_D, np.float32), cn1(T2)]))
        w2 = int(np.argmax(b2.W @ q))
        return w2, np.maximum(b2.W[w2, :CODE_D], 0.0).reshape(NPOS, K1)

    # ---- Teacher-forced: is the code-level successor read itself OK? ------
    T1 = np.zeros((NPOS, WD), np.float32)
    T2 = np.zeros(CODE_D, np.float32)
    tf_ok, tf_ws = [], set()
    for t in range(PRIME + 100):
        if t >= PRIME:
            w2, prof = successor_prof(T2)
            tf_ws.add(w2)
            tf_ok.append(classify(render(prof, "graded")) == t % 10)
        code = encode(frame_for(t), T1, False).ravel()
        T2 = code + G2 * T2
    tf_line = (f"TEACHER-FORCED next-class via code successor read: "
               f"{np.mean(tf_ok):.2f} correct, {len(tf_ws)} distinct "
               f"L2 winners over 100 reads")

    lines = [tf_line]
    for mode in ("graded", "committed"):
        T1 = np.zeros((NPOS, WD), np.float32)
        T2 = np.zeros(CODE_D, np.float32)
        for t in range(PRIME):
            code = encode(frame_for(t), T1, False).ravel()
            T2 = code + G2 * T2
        gen, cls = [], []
        for i in range(FREE):
            _, prof = successor_prof(T2)
            emission = render(prof, mode)
            gen.append(emission)
            cls.append(classify(emission))
            code = encode(emission, T1, False).ravel()
            T2 = code + G2 * T2
        cls = np.array(cls)
        intended = np.arange(PRIME, PRIME + FREE) % 10
        adv = float(np.mean((np.diff(cls) % 10) == 1))
        timeline = float(np.mean(cls == intended))
        save_gif(gen, OUTPUT_DIR / f"generated_{mode}.gif", ms=350)
        lines.append(f"BOUNCE ({mode:9s}): advance {adv:.2f}, timeline "
                     f"{timeline:.2f}, first 30: "
                     + "".join(map(str, cls[:30])))
    report = "\n".join(
        lines + ["(whole-frame refs: TOP1 1.00/1.00, TOP2+ frozen 8s)"])
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# MNIST carousel, convolutional uniform unit, code-level bounce\n\n"
        + report + "\n")

    fig, axes = plt.subplots(4, 16, figsize=(16 * 1.1, 4 * 1.35))
    for i in range(K1):
        ax = axes[i // 16, i % 16]
        keys = np.maximum(b1.W[i, WD:], 0.0).reshape(WIN, WIN)
        cargo = np.maximum(b1.W[i, :WD], 0.0).reshape(WIN, WIN)
        stack = np.vstack([keys / max(keys.max(), 1e-9),
                           np.full((1, WIN), 0.5, np.float32),
                           cargo / max(cargo.max(), 1e-9)])
        ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("L1 shared 8x8 stroke units: KEYS (window trail, top) / "
                 "CARGO (arriving window, bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L1.png", dpi=140)
    print("wrote templates_L1.png")


if __name__ == "__main__":
    main()
