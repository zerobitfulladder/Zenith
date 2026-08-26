"""Three labeled movies under HIERARCHICAL TIME (retest of v3).

Temporal L1 (every tick): query = [leaky-summed spatial code ; label ;
L2's DOWN-PROJECTION ; next-code slot]. Temporal L2: samples KSTRIDE=5
L1 ticks per output (structural timescale), stores era-arrows
[this window ; NEXT window] and projects the PREDICTED next era down.
Interleaved training (bootstrap law); cold start from label alone.

Flat-rig reference: purity 1.00/1.00/1.00, advance 1.00 tb / 1.00 lr
/ 0.72-strict rot.

Run:  .venv/bin/python experiments/2026_08_26/hier_time/run_movies_hier.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_line"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_movies"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_gpu_minibatch import Dict  # noqa: E402
from run_temporal_line import H, WIN, K1, ETA1, cn_rows, save_gif  # noqa: E402
from run_temporal_movies import (  # noqa: E402
    CODE_DIM,
    MOVIES,
    POS,
    encode,
    render,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "hier_time" / "results" / "movies"
GAMMA1 = 0.5
GLAB, GD, GN = 0.5, 0.5, 0.5
K1T, K2T = 160, 32
KSTRIDE = 5
DIM1 = CODE_DIM + 3 + K1T + CODE_DIM
EPOCHS, EP_FRAMES = 3, 600
FREE_RUN = 100


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


class MovieState:
    def __init__(self):
        self.T1 = np.zeros(CODE_DIM, np.float32)
        self.down = np.zeros(K1T, np.float32)
        self.acc = np.zeros(K1T, np.float32)
        self.prev = None
        self.tick = 0


def l2_pass(b2, st, learning):
    if st.acc.sum() <= 0:
        return
    cur = unit(st.acc.copy())
    if learning and st.prev is not None:
        z2 = cn1(np.concatenate([st.prev, 0.5 * cur]))[None, :]
        if b2.n_boot < b2.k:
            b2.bootstrap(z2)
        else:
            C = (z2 @ b2.W.T)[0]
            w = int(np.argmax(C))
            b2.update(z2, np.array([w]),
                      np.array([max(C[w], 0.0)], dtype=np.float32))
    if b2.n_boot > 0:
        q2 = cn1(np.concatenate([cur, np.zeros(K1T, np.float32)]))
        w = int(np.argmax(b2.W @ q2))
        st.down = unit(np.maximum(b2.W[w, K1T:], 0.0))
    st.prev = cur
    st.acc[:] = 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn, period in MOVIES:
        save_gif([fn(t) for t in range(2 * period)],
                 OUTPUT_DIR / f"input_{name}.gif")

    # ---- Spatial L1 (interleaved, as in v3) ------------------------------
    d1 = Dict(K1, WIN * WIN, ETA1)
    for step in range(600):
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
    print("spatial trained", flush=True)

    # ---- Temporal banks, interleaved with per-movie states ----------------
    b1 = Dict(K1T, DIM1, 0.05)
    b2 = Dict(K2T, 2 * K1T, 0.05)
    labs = np.eye(3, dtype=np.float32)
    for _ in range(EPOCHS):
        sts = {mi: MovieState() for mi in range(3)}
        for t in range(EP_FRAMES):
            for mi, (_, fn, _) in enumerate(MOVIES):
                st = sts[mi]
                code = encode(fn(t), W1)
                st.T1 = code + GAMMA1 * st.T1
                nxt = encode(fn(t + 1), W1)
                z1 = cn1(np.concatenate(
                    [unit(st.T1), GLAB * labs[mi], GD * st.down,
                     GN * nxt]))[None, :]
                u1 = -1
                if b1.n_boot < b1.k:
                    b1.bootstrap(z1)
                else:
                    C = (z1 @ b1.W.T)[0]
                    u1 = int(np.argmax(C))
                    b1.update(z1, np.array([u1]),
                              np.array([max(C[u1], 0.0)], dtype=np.float32))
                if u1 >= 0:
                    st.acc[u1] += 1.0
                st.tick += 1
                if st.tick % KSTRIDE == 0 and b1.n_boot >= K1T:
                    l2_pass(b2, st, learning=True)
    print("temporal trained", flush=True)

    # ---- Frame library for purity/advance metrics -------------------------
    lib_imgs, lib_movie, lib_phase = [], [], []
    for mi, (_, fn, period) in enumerate(MOVIES):
        for t in range(period):
            f = fn(t)
            lib_imgs.append(f.ravel() / (np.linalg.norm(f) + 1e-9))
            lib_movie.append(mi)
            lib_phase.append(t)
    LIB = np.stack(lib_imgs)
    lib_movie, lib_phase = np.array(lib_movie), np.array(lib_phase)

    rows = []
    strips = []
    for mi, (name, fn, period) in enumerate(MOVIES):
        st = MovieState()
        gen, mm, mp = [], [], []
        for _ in range(FREE_RUN):
            q = cn1(np.concatenate(
                [unit(st.T1), GLAB * labs[mi], GD * st.down,
                 np.zeros(CODE_DIM, np.float32)]))
            u1 = int(np.argmax(b1.W @ q))
            c_hat = np.maximum(b1.W[u1, CODE_DIM + 3 + K1T:], 0.0)
            c_hat = (c_hat / (np.linalg.norm(c_hat) + 1e-9)).astype(np.float32)
            img = render(c_hat, W1)
            gen.append(img)
            v = img.ravel() / (np.linalg.norm(img) + 1e-9)
            b = int(np.argmax(LIB @ v))
            mm.append(lib_movie[b])
            mp.append(lib_phase[b])
            st.T1 = c_hat + GAMMA1 * st.T1
            st.acc[u1] += 1.0
            st.tick += 1
            if st.tick % KSTRIDE == 0:
                l2_pass(b2, st, learning=False)
        save_gif(gen, OUTPUT_DIR / f"gen_{name}.gif")
        strips.append((name, fn, gen))
        mm, mp = np.array(mm), np.array(mp)
        purity = float((mm == mi).mean())
        own = np.where(mm == mi)[0]
        step1 = np.diff(own) == 1
        adv = np.diff(mp[own]) % period
        moved = (adv >= 1) & (adv <= 3)
        advance = float(moved[step1].mean()) if step1.any() else 0.0
        rows.append((name, purity, advance))
        print(f"LABEL {name}: purity={purity:.2f} advance={advance:.2f}",
              flush=True)

    fig, axes = plt.subplots(6, 15, figsize=(14, 6.6))
    for si, (name, fn, gen) in enumerate(strips):
        for i in range(15):
            axes[2 * si, i].imshow(fn(i * 2), cmap="gray", vmin=0, vmax=1)
            axes[2 * si + 1, i].imshow(gen[i * 2], cmap="gray", vmin=0, vmax=1)
            axes[2 * si, i].axis("off")
            axes[2 * si + 1, i].axis("off")
        axes[2 * si, 0].axis("on")
        axes[2 * si, 0].set_xticks([])
        axes[2 * si, 0].set_yticks([])
        for sp in axes[2 * si, 0].spines.values():
            sp.set_visible(False)
        axes[2 * si, 0].set_ylabel(f"{name}\ntrue/gen", fontsize=7,
                                   rotation=0, ha="right", va="center")
    fig.suptitle("Three movies, HIERARCHICAL time — true vs cold-start "
                 "free-run (every 2nd frame)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filmstrip.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Three movies, hierarchical time",
         "",
         "| movie | purity | advance |",
         "|---|---|---|"]
        + [f"| {n} | {p:.2f} | {a:.2f} |" for n, p, a in rows]
        + ["", "Flat reference: purity 1.00 all; advance 1.00/1.00/0.72."])
        + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
