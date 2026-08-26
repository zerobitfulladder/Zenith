"""Digit carousel, PURE form — the user's architecture with no slots.

L1 stores cn([leaky-integrated input ; G*down]) — two channels, nothing
else. L2 stores cn([era activity profile ; G*label]) — the era keyed by
itself, label at the final layer. No next-slots, no arrows, no
present/tail splits, no empty-channel queries: everything queries with
everything. The arrow of time is the INTER-LAYER LAG: the down channel
accompanying a digit's early frames still describes the PREVIOUS era
(L2 outputs trail their input by construction), so at playback the
down-match pulls retrieval forward across transitions while T-match
sustains holds — hold and advance are one gain competition (GD).

Dwell 8 / era stride 4: the image reaches the top mid-display (the
propagation requirement). Emission = the retrieved row's T-part.

Run:  .venv/bin/python experiments/2026_08_26/pure_form/run_digits_pure.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from run_4layer_topk import load_data  # noqa: E402
from run_temporal_digits import (  # noqa: E402
    CODE_DIM,
    POS,
    WIN,
    SeqDict,
    cn_rows,
    encode,
    render,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "pure_form" / "results" / "digits"
GAMMA1 = 0.8
GAMMA2 = 0.9   # L2 = LEAKY integration of L1 activity (user's spec; the
               # window-sum-reset version made down jump to "now" mid-digit,
               # killing the lag that drives advance -> measured freeze)
GD, GL = 1.0, 0.5
K1T, K2T = 256, 64
KSTRIDE, DWELL = 4, 8
DIM1 = CODE_DIM + K1T
DIM2 = K1T + 10
L1_DIGITS = 1000
N_CYCLES = 1200
FREE_RUN = 240


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _, _ = load_data()
    pools = [np.where(ytr == c)[0] for c in range(10)]

    def exemplar(cyc, c):
        return Xtr[pools[c][cyc % len(pools[c])]]

    v1 = SeqDict(64, WIN * WIN, 0.05, seed=3)
    rng = np.random.default_rng(0)
    for i in rng.permutation(L1_DIGITS):
        V = np.stack([Xtr[i][r:r + WIN, c:c + WIN].ravel() for r, c in POS])
        Vh, ok = cn_rows(V)
        for v in Vh[ok]:
            v1.step(v.astype(np.float32))
    W1 = v1.W
    print("vocabulary trained", flush=True)
    save_gif([exemplar(cyc, c) for cyc in range(2) for c in range(10)
              for _ in range(DWELL)], OUTPUT_DIR / "input.gif")

    b1 = SeqDict(K1T, DIM1, 0.05, seed=1)
    b2 = SeqDict(K2T, DIM2, 0.05, seed=2)
    eye10 = np.eye(10, dtype=np.float32)
    cache = {}

    def code_of(cyc, c):
        key = (cyc, c)
        if key not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[key] = encode(exemplar(cyc, c), W1)
        return cache[key]

    T1 = np.zeros(CODE_DIM, np.float32)
    down = np.zeros(K1T, np.float32)
    acc = np.zeros(K1T, np.float32)
    tick = 0
    for cyc in range(N_CYCLES):
        for c in range(10):
            code = code_of(cyc, c)
            for _ in range(DWELL):
                T1 = code + GAMMA1 * T1
                z1 = cn1(np.concatenate([unit(T1), GD * down]))
                u1 = b1.step(z1)
                if b1.n_boot >= K1T:
                    a1 = np.zeros(K1T, np.float32)
                    a1[u1] = 1.0
                    acc[:] = a1 + GAMMA2 * acc     # leaky, never reset
                tick += 1
                if tick % KSTRIDE == 0 and acc.sum() > 0:
                    cur = unit(acc)
                    if b1.n_boot >= K1T:
                        b2.step(cn1(np.concatenate([cur, GL * eye10[c]])))
                    if b2.n_boot > 0:
                        q2 = cn1(np.concatenate([cur, GL * eye10[c]]))
                        w = int(np.argmax(b2.W @ q2))
                        down = unit(np.maximum(b2.W[w, :K1T], 0.0))
    print("temporal trained", flush=True)

    judge = LogisticRegression(max_iter=1000)
    judge.fit(Xtr[:10000].reshape(10000, -1), ytr[:10000])

    # ---- Cold start: label belief 0 + two-digit hum, then free-run -------
    T1 = np.zeros(CODE_DIM, np.float32)
    down = np.zeros(K1T, np.float32)
    acc = np.zeros(K1T, np.float32)
    tick = 0
    belief = eye10[0].copy()

    def l2_playback():
        nonlocal down, belief
        if acc.sum() <= 0:
            return
        cur = unit(acc)
        if b2.n_boot > 0:
            q2 = cn1(np.concatenate([cur, GL * belief]))
            w = int(np.argmax(b2.W @ q2))
            down = unit(np.maximum(b2.W[w, :K1T], 0.0))
            belief = eye10[int(np.argmax(b2.W[w, K1T:]))].copy()

    for c in (0, 1):
        code = encode(exemplar(N_CYCLES + 1, c), W1)
        belief = eye10[c].copy()
        for _ in range(DWELL):
            T1 = code + GAMMA1 * T1
            q = cn1(np.concatenate([unit(T1), GD * down]))
            u1 = int(np.argmax(b1.W @ q))
            a1 = np.zeros(K1T, np.float32)
            a1[u1] = 1.0
            acc[:] = a1 + GAMMA2 * acc
            tick += 1
            if tick % KSTRIDE == 0:
                l2_playback()

    gen, jlab = [], []
    for _ in range(FREE_RUN):
        q = cn1(np.concatenate([unit(T1), GD * down]))
        u1 = int(np.argmax(b1.W @ q))
        e = np.maximum(b1.W[u1, :CODE_DIM], 0.0)
        e = (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32)
        img = render(e, W1)
        gen.append(img)
        jlab.append(int(judge.predict(img.reshape(1, -1))[0]))
        T1 = e + GAMMA1 * T1
        a1 = np.zeros(K1T, np.float32)
        a1[u1] = 1.0
        acc[:] = a1 + GAMMA2 * acc
        tick += 1
        if tick % KSTRIDE == 0:
            l2_playback()
    print("judged labels (first 120):",
          "".join(str(l) for l in jlab[:120]), flush=True)
    save_gif(gen, OUTPUT_DIR / "generated.gif")
    np.savez(OUTPUT_DIR / "weights.npz", W1=W1, Wb1=b1.W, Wb2=b2.W)

    class_means = [Xtr[pools[c][:N_CYCLES]].mean(axis=0) for c in range(10)]
    picks = {}
    for i, l in enumerate(jlab):
        if i % DWELL == DWELL // 2:
            picks.setdefault(l, []).append(gen[i])
    n_show = min(5, max((len(v) for v in picks.values()), default=0))
    fig, axes = plt.subplots(n_show + 1, 10,
                             figsize=(10.5, 1.1 * (n_show + 1) + 0.6))
    for c in range(10):
        axes[0, c].imshow(class_means[c], cmap="gray")
        axes[0, c].axis("off")
        for r in range(n_show):
            ax = axes[r + 1, c]
            if c in picks and r < len(picks[c]):
                ax.imshow(picks[c][r], cmap="gray")
            ax.axis("off")
    fig.suptitle("PURE-form carousel — [leaky input ; down] only, no slots; "
                 "row 0: class means")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cycles.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Pure-form digit carousel (no slots anywhere)",
         "",
         f"Judged labels (first 120): {''.join(str(l) for l in jlab[:120])}",
         "",
         "Figure: cycles.png; input.gif / generated.gif; weights.npz"])
        + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
