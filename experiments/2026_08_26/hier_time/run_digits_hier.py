"""Digit carousel under HIERARCHICAL TIME — no labels anywhere.

The user's claim, tested: the original carousel's label channel (and
next-label cargo) hand-fed the sequence-position context that a slower
layer should supply structurally. Here: L1 per tick = [leaky-summed
code ; L2 down-projection ; next-code arrow]; L2 samples KSTRIDE=HOLD
ticks per output (one era = one digit exposure), stores era-arrows,
projects the PREDICTED next era down. No label input, no next-label.
Shared vocabulary (as the original temporal_digits/results).

Free-run after an 8-frame hum (no label to cue with, by design); the
pixel judge reads the emitted digits; success = the judged stream
counts 0..9 in order with proper dwells, timed purely by the
two-layer clock.

Run:  .venv/bin/python experiments/2026_08_26/hier_time/run_digits_hier.py
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
    HOLD,
    SeqDict,
    cn,
    encode,
    render,
    save_gif,
)
OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "hier_time" / "results" / "digits"
GAMMA1 = 0.8   # dwell clock: matches the validated carousel's alpha=0.2
               # (tail:present 4:1; gamma .5's 6%-residue froze it — the
               # held-input clock disease, third appearance)
GD, GN = 1.0, 0.5   # era hand needs authority: 3-digit orbit at 0.5
K1T, K2T = 256, 64  # more era memories: 10 eras x style variety
KSTRIDE = HOLD                       # one era = one digit exposure
DIM1 = CODE_DIM + K1T + CODE_DIM
L1_DIGITS = 1000
N_CYCLES = 1200
PRIME_FRAMES = 8
FREE_RUN = 200


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

    # Shared spatial vocabulary, as the original carousel.
    from run_temporal_digits import cn_rows, POS, WIN
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
              for _ in range(HOLD)], OUTPUT_DIR / "input.gif")

    b1 = SeqDict(K1T, DIM1, 0.05, seed=1)
    b2 = SeqDict(K2T, 2 * K1T, 0.05, seed=2)
    T1 = np.zeros(CODE_DIM, np.float32)
    down = np.zeros(K1T, np.float32)
    acc = np.zeros(K1T, np.float32)
    prev = [None]
    tick = [0]
    cache = {}

    def code_of(cyc, c):
        key = (cyc, c)
        if key not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[key] = encode(exemplar(cyc, c), W1)
        return cache[key]

    def l2_pass(learning):
        nonlocal down
        if acc.sum() <= 0:
            return
        cur = unit(acc.copy())
        if learning and prev[0] is not None and b1.n_boot >= K1T:
            z2 = cn1(np.concatenate([prev[0], 0.5 * cur]))
            b2.step(z2)
        if b2.n_boot > 0:
            q2 = cn1(np.concatenate([cur, np.zeros(K1T, np.float32)]))
            w = int(np.argmax(b2.W @ q2))
            down = unit(np.maximum(b2.W[w, K1T:], 0.0))
        prev[0] = cur
        acc[:] = 0.0

    for cyc in range(N_CYCLES):
        for c in range(10):
            code = code_of(cyc, c)
            for h in range(HOLD):
                T1 = code + GAMMA1 * T1
                last = h == HOLD - 1
                nc = (c + 1) % 10 if last else c
                ncyc = cyc + 1 if (last and c == 9) else cyc
                nxt = code_of(ncyc, nc)
                z1 = cn1(np.concatenate(
                    [unit(T1), GD * down, GN * nxt]))
                u1 = b1.step(z1)
                if b1.n_boot >= K1T:
                    acc[u1] += 1.0
                tick[0] += 1
                if tick[0] % KSTRIDE == 0:
                    l2_pass(learning=True)
    print("temporal trained", flush=True)

    judge = LogisticRegression(max_iter=1000)
    judge.fit(Xtr[:10000].reshape(10000, -1), ytr[:10000])

    # ---- Cold start: 8-frame hum from UNSEEN exemplars, then free-run ----
    T1 = np.zeros(CODE_DIM, np.float32)
    down = np.zeros(K1T, np.float32)
    acc[:] = 0.0
    prev[0] = None
    tick[0] = 0
    for c in (0, 1):
        code = encode(exemplar(N_CYCLES + 1, c), W1)
        for _ in range(HOLD):
            T1 = code + GAMMA1 * T1
            q = cn1(np.concatenate(
                [unit(T1), GD * down, np.zeros(CODE_DIM, np.float32)]))
            u1 = int(np.argmax(b1.W @ q))
            acc[u1] += 1.0
            tick[0] += 1
            if tick[0] % KSTRIDE == 0:
                l2_pass(learning=False)

    gen, jlab = [], []
    for _ in range(FREE_RUN):
        q = cn1(np.concatenate(
            [unit(T1), GD * down, np.zeros(CODE_DIM, np.float32)]))
        u1 = int(np.argmax(b1.W @ q))
        c_hat = np.maximum(b1.W[u1, CODE_DIM + K1T:], 0.0)
        c_hat = (c_hat / (np.linalg.norm(c_hat) + 1e-9)).astype(np.float32)
        img = render(c_hat, W1)
        gen.append(img)
        jlab.append(int(judge.predict(img.reshape(1, -1))[0]))
        T1 = c_hat + GAMMA1 * T1
        acc[u1] += 1.0
        tick[0] += 1
        if tick[0] % KSTRIDE == 0:
            l2_pass(learning=False)
    print("judged labels (first 80):",
          "".join(str(l) for l in jlab[:80]), flush=True)
    save_gif(gen, OUTPUT_DIR / "generated.gif")
    np.savez(OUTPUT_DIR / "weights.npz", W1=W1, Wb1=b1.W, Wb2=b2.W)

    class_means = [Xtr[pools[c][:N_CYCLES]].mean(axis=0) for c in range(10)]
    picks = {}
    for i, l in enumerate(jlab):
        if i % HOLD == HOLD // 2:
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
    fig.suptitle("Label-free hierarchical carousel — row 0: class means; "
                 "rows 1+: emissions per generated cycle (judge-sorted)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cycles.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Label-free hierarchical digit carousel",
         "",
         f"Judged labels (first 80): {''.join(str(l) for l in jlab[:80])}",
         "",
         "Reference (flat + label scaffolding): loops all 10 in order,",
         "holds 1-6 frames. Figure: cycles.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
