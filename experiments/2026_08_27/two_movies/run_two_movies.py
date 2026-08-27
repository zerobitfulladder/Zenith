"""Two labeled digit sequences over the SAME three digits (user's spec).

  label bit 0  ->  1,2,3,1,2,3, ...
  label bit 1  ->  3,2,1,3,2,1, ...

Fresh MNIST exemplar every time, so pixels never repeat and the ORDER is
the only stable structure. The two streams have identical long-run
statistics (each digit equally often), so direction cannot live in a slow
average — it lives in short-range order (which digit preceded this one)
and in the label. That is what makes this the fair two-scale test.

Three layers, label at the top (user's spec):
  L1  4x4 windows stride 1, K=36, row [2.0*window ; window trail]
      — the Exp 9 winner (best codec AND best generator of the sweep).
      Reports a skeleton code map (per-position top-1).
  L2  over the code map, row [0.5*code ; lagged code trail], delayed
      write. This is where the sequence arrow lives. Speaks a dense relu
      profile upward.
  L3  over L2's speech, row [0.5*msg ; lagged msg trail ; LAM*label].
      The label is a constant tag on every row of its stream.

Staged training (the Exp 8 fix for bootstrap monopoly): L1 alone, then a
fresh L2 on frozen L1, then a fresh L3 on both frozen.
Streams are INTERLEAVED tick by tick, each carrying its own trails
(08-26 lesson: with adopt-until-full bootstrap, curriculum order IS
allocation).

Generation: COLD START FROM THE LABEL ALONE — all trails zero, no
priming, no frames shown. Three read modes compared on one network:
  top     L3 names an L2 unit; its cargo is the next code map (descent)
  l2      L2's own successor read, L3 ignored (does the trail alone
          carry direction?)
  refine  L2 picks within the candidate pool L3 names (gate-then-match,
          the untested lever from Exp 7)

Judged by a probe trained on renders of real frames (Exp 9b's law: never
judge generated output with a gate calibrated on real input).

Run:  .venv/bin/python experiments/2026_08_27/two_movies/run_two_movies.py
"""

import json
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
from sklearn.linear_model import LogisticRegression

from run_square_completion import cn1            # noqa: E402
from run_temporal_square import save_gif         # noqa: E402
from run_gpu_minibatch import Dict               # noqa: E402

SUFFIX = os.environ.get("TM_TAG", "")
OUTPUT_DIR = (ROOT / "experiments" / "2026_08_27" / "two_movies" / "results"
              / (SUFFIX.lstrip("_") or "base"))

SIDE, WIN = 28, 4
NP1 = SIDE - WIN + 1
NPOS, WD = NP1 * NP1, WIN * WIN
K1, K2, K3 = 36, 64, 32
CODE_D = NPOS * K1
GC1, GC2, GC3 = 2.0, 0.5, 0.5      # arriving-half gain per layer
G1, G2, G3 = 0.5, 0.5, 0.9         # trail decay per layer
LAM = 0.5                          # label weight in the L3 row
EPS, SQ = 1e-6, 0.10
EPOCHS = int(os.environ.get("TM_EPOCHS", "2"))
TICKS = int(os.environ.get("TM_TICKS", "3000"))    # per epoch, both streams
FREE = 60
# Monopoly guard (classic "conscience" from competitive learning, never
# tried in this project): during LEARNING only, bias the winner choice
# against rows that have been winning more than their fair share. Reads
# and stored content are untouched — this changes allocation, not memory.
CONSC = float(os.environ.get("TM_CONSCIENCE", "0"))

ORDERS = {0: [1, 2, 3], 1: [3, 2, 1]}
CLASSES = [1, 2, 3]
_t = np.ones(WIN)
_t[0] = _t[-1] = 0.5
FEATHER = np.outer(_t, _t)

X = np.load(ROOT / "data/mnist/digits/train_images.npy")
y = np.load(ROOT / "data/mnist/digits/train_labels.npy")
BYCLASS = {c: X[y == c] for c in CLASSES}


def frame_for(stream, step):
    """Fresh exemplar every lap; the two streams never share an image."""
    c = ORDERS[stream][step % 3]
    pool = BYCLASS[c]
    return pool[(step // 3 * 2 + stream) % len(pool)].astype(np.float32)


def label_vec(stream):
    v = np.zeros(2, np.float32)
    v[stream] = 1.0
    return v


def cn_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    return (M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True),
                           1e-9)).astype(np.float32)


def win_view(frame):
    s = np.lib.stride_tricks.sliding_window_view(frame, (WIN, WIN))
    return s.reshape(NPOS, WD)


# ---------------------------------------------------------------- layers ---
b1 = Dict(K1, 2 * WD, 0.05)
b2 = Dict(K2, 2 * CODE_D, 0.05)
b3 = Dict(K3, 2 * K2 + 2, 0.05)


def encode(frame, T1, learning):
    """L1: per-position winner -> skeleton code map. Delayed write."""
    wins = win_view(frame)
    live = np.where((wins.max(axis=1) > EPS) | (T1.max(axis=1) > EPS))[0]
    code = np.zeros((NPOS, K1), np.float32)
    if len(live):
        Z = cn_rows(np.concatenate(
            [GC1 * cn_rows(wins[live]), cn_rows(T1[live])], axis=1))
        if learning and b1.n_boot < K1:
            b1.bootstrap(Z[: K1 - b1.n_boot])
        if b1.n_boot >= K1:
            C = Z @ b1.W.T
            ws = np.argmax(C, axis=1)
            cv = np.maximum(C[np.arange(len(live)), ws], 0.0)
            if learning:
                b1.update(Z, ws.astype(np.int64), cv.astype(np.float32))
            code[live, ws] = cv
    T1 *= G1
    T1 += wins
    return code.ravel()


def l2_row(code, T2):
    return cn1(np.concatenate([GC2 * cn1(code), cn1(T2)]))


def l3_row(msg, T3, lab):
    return cn1(np.concatenate([GC3 * cn1(msg), cn1(T3), LAM * lab]))


def render(prof, mode="committed"):
    """Code map -> pixels. Validated recipe (squelch, peak gate, feather)."""
    prof = prof.reshape(NPOS, K1)
    basis = np.maximum(b1.W[:, :WD], 0.0)
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = prof.max()
    if peak <= 0:
        return num
    for p in range(NPOS):
        seg = np.maximum(prof[p], 0.0)
        m = seg.max()
        if m <= 0 or m < SQ * peak:
            continue
        if mode == "committed":
            s = np.zeros_like(seg)
            s[int(np.argmax(seg))] = m
            seg = s
        else:
            seg = np.where(seg < SQ * m, 0.0, seg)
        r, c = divmod(p, NP1)
        num[r:r + WIN, c:c + WIN] += (seg @ basis).reshape(WIN, WIN) * FEATHER * m
        den[r:r + WIN, c:c + WIN] += FEATHER * m
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return out / out.max() if out.max() > 0 else out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"conscience": CONSC}

    # ---- Phase A: L1 alone -------------------------------------------------
    for ep in range(EPOCHS):
        T1 = [np.zeros((NPOS, WD), np.float32) for _ in range(2)]
        for t in range(TICKS):
            s = t % 2
            encode(frame_for(s, t // 2), T1[s], True)
        print(f"L1 epoch {ep + 1}/{EPOCHS}", flush=True)

    # ---- Phase B: fresh L2 on frozen L1 ------------------------------------
    w2h = np.zeros(K2)
    for ep in range(EPOCHS):
        T1 = [np.zeros((NPOS, WD), np.float32) for _ in range(2)]
        T2 = [np.zeros(CODE_D, np.float32) for _ in range(2)]
        for t in range(TICKS):
            s = t % 2
            code = encode(frame_for(s, t // 2), T1[s], False)
            z = l2_row(code, T2[s])[None, :]
            if b2.n_boot < K2:
                b2.bootstrap(z)
            else:
                C = (z @ b2.W.T)[0]
                bias = CONSC * (w2h / max(w2h.sum(), 1.0) - 1.0 / K2)
                w = int(np.argmax(C - bias))
                b2.update(z, np.array([w]),
                          np.array([max(C[w], 0.0)], np.float32))
                w2h[w] += 1
            T2[s] = code + G2 * T2[s]
        print(f"L2 epoch {ep + 1}/{EPOCHS}", flush=True)
    stats["l2_rows_used"] = int((w2h > 0).sum())
    stats["l2_top_share"] = round(float(w2h.max() / max(w2h.sum(), 1)), 3)

    # ---- Phase C: fresh L3 on frozen L1+L2 ---------------------------------
    w3h = np.zeros(K3)
    for ep in range(EPOCHS):
        T1 = [np.zeros((NPOS, WD), np.float32) for _ in range(2)]
        T2 = [np.zeros(CODE_D, np.float32) for _ in range(2)]
        T3 = [np.zeros(K2, np.float32) for _ in range(2)]
        for t in range(TICKS):
            s = t % 2
            code = encode(frame_for(s, t // 2), T1[s], False)
            msg = np.maximum(b2.W @ l2_row(code, T2[s]), 0.0)
            T2[s] = code + G2 * T2[s]
            z = l3_row(msg, T3[s], label_vec(s))[None, :]
            if b3.n_boot < K3:
                b3.bootstrap(z)
            else:
                C = (z @ b3.W.T)[0]
                bias = CONSC * (w3h / max(w3h.sum(), 1.0) - 1.0 / K3)
                w = int(np.argmax(C - bias))
                b3.update(z, np.array([w]),
                          np.array([max(C[w], 0.0)], np.float32))
                w3h[w] += 1
            T3[s] = msg + G3 * T3[s]
        print(f"L3 epoch {ep + 1}/{EPOCHS}", flush=True)
    stats["l3_rows_used"] = int((w3h > 0).sum())
    stats["l3_top_share"] = round(float(w3h.max() / max(w3h.sum(), 1)), 3)

    np.savez(OUTPUT_DIR / "weights.npz", W1=b1.W, W2=b2.W, W3=b3.W)

    # ---- matched judge: probe on renders of REAL frames --------------------
    T1 = [np.zeros((NPOS, WD), np.float32) for _ in range(2)]
    Xr, yr = [], []
    for t in range(1200):
        s = t % 2
        step = t // 2
        code = encode(frame_for(s, step), T1[s], False)
        if t >= 20:
            Xr.append(render(code).ravel())
            yr.append(ORDERS[s][step % 3])
    Xr, yr = np.array(Xr), np.array(yr)
    cut = int(0.8 * len(Xr))
    probe = LogisticRegression(max_iter=2000).fit(Xr[:cut], yr[:cut])
    stats["judge_val_acc"] = round(float(probe.score(Xr[cut:], yr[cut:])), 3)

    # ---- Teacher-forced: does the network hold the transitions? -----------
    T1 = [np.zeros((NPOS, WD), np.float32) for _ in range(2)]
    T2 = [np.zeros(CODE_D, np.float32) for _ in range(2)]
    tf = []
    for t in range(400):
        s, step = t % 2, t // 2
        if t >= 40:
            q = cn1(np.concatenate([np.zeros(CODE_D, np.float32), cn1(T2[s])]))
            w = int(np.argmax(b2.W @ q))
            pred = np.maximum(b2.W[w, :CODE_D], 0.0)
            tf.append(int(probe.predict(render(pred).ravel()[None, :])[0])
                      == ORDERS[s][step % 3])
        code = encode(frame_for(s, step), T1[s], False)
        T2[s] = code + G2 * T2[s]
    stats["tf_next_class_l2"] = round(float(np.mean(tf)), 3)

    # ---- Generation: COLD START FROM THE LABEL ALONE -----------------------
    def generate(stream, mode, render_mode="committed"):
        lab = label_vec(stream)
        T1 = np.zeros((NPOS, WD), np.float32)
        T2 = np.zeros(CODE_D, np.float32)
        T3 = np.zeros(K2, np.float32)
        gen, w2s, w3s = [], [], []
        for _ in range(FREE):
            prof2 = None
            if mode in ("top", "refine"):
                q3 = cn1(np.concatenate(
                    [np.zeros(K2, np.float32), cn1(T3), LAM * lab]))
                w3 = int(np.argmax(b3.W @ q3))
                w3s.append(w3)
                prof2 = np.maximum(b3.W[w3, :K2], 0.0)
            if mode == "top":
                w2 = int(np.argmax(prof2))
            elif mode == "l2":
                q2 = cn1(np.concatenate(
                    [np.zeros(CODE_D, np.float32), cn1(T2)]))
                w2 = int(np.argmax(b2.W @ q2))
            else:                                   # refine: gate then match
                q2 = cn1(np.concatenate(
                    [np.zeros(CODE_D, np.float32), cn1(T2)]))
                sc = b2.W @ q2
                pool = np.argsort(prof2)[::-1][:5]
                if prof2[pool].max() <= 0:
                    pool = np.arange(K2)
                w2 = int(pool[np.argmax(sc[pool])])
            w2s.append(w2)
            code = np.maximum(b2.W[w2, :CODE_D], 0.0)
            em = render(code, render_mode)
            gen.append(em)
            # re-see own emission, climb, advance every trail
            code_in = encode(em.astype(np.float32), T1, False)
            msg = np.maximum(b2.W @ l2_row(code_in, T2), 0.0)
            T2 = code_in + G2 * T2
            T3 = msg + G3 * T3
        cls = probe.predict(np.array([g.ravel() for g in gen]))
        order = ORDERS[stream]
        nxt = {order[i]: order[(i + 1) % 3] for i in range(3)}
        adv = float(np.mean([nxt[a] == b for a, b in zip(cls[:-1], cls[1:])]))
        return gen, cls, adv, len(set(w2s)), len(set(w3s))

    panels = {}
    for mode in ("top", "l2", "refine"):
        for stream in (0, 1):
            gen, cls, adv, n2, n3 = generate(stream, mode)
            key = f"{mode}_label{stream}"
            stats[f"{key}_direction_acc"] = round(adv, 3)
            stats[f"{key}_seq"] = "".join(map(str, cls[:30]))
            stats[f"{key}_distinct_L2"] = n2
            stats[f"{key}_distinct_L3"] = n3
            save_gif(gen, OUTPUT_DIR / f"gen_{key}.gif", scale=8, ms=400)
            panels[key] = (gen, cls)

    print(json.dumps(stats, indent=1), flush=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(stats, indent=2))

    # ---- the money picture: both labels, side by side ----------------------
    for mode in ("top", "l2", "refine"):
        fig, axes = plt.subplots(2, 15, figsize=(15, 2.9))
        for r, stream in enumerate((0, 1)):
            gen, cls = panels[f"{mode}_label{stream}"]
            for i in range(15):
                axes[r, i].imshow(gen[i], cmap="gray")
                axes[r, i].set_title(str(cls[i]), fontsize=9)
                axes[r, i].axis("off")
            axes[r, 0].text(-0.6, 0.5, f"label {stream}\n{ORDERS[stream]}",
                            transform=axes[r, 0].transAxes, ha="right",
                            va="center", fontsize=9)
        fig.suptitle(
            f"read={mode}: cold start from the LABEL ALONE — "
            f"up {stats[f'{mode}_label0_direction_acc']} / "
            f"down {stats[f'{mode}_label1_direction_acc']}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"panel_{mode}.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(3, 12, figsize=(12, 3.4))
    for i in range(K1):
        ax = axes[i // 12, i % 12]
        c = np.maximum(b1.W[i, :WD], 0.0).reshape(WIN, WIN)
        ax.imshow(c / max(c.max(), 1e-9), cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("L1 stroke templates (4x4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L1.png", dpi=140)
    plt.close(fig)
    print("artifacts written to", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
