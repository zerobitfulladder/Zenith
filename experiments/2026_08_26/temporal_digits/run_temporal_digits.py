"""Temporal v4: the digit carousel — memorization vs abstraction in time.

Stimulus: MNIST digits 0->1->...->9->0..., each held HOLD frames, and
EVERY cycle uses fresh exemplars of each class. The transition
structure ("after 7 comes some 0") is stable; the pixels are not.
A memorizer would need exemplar chains (unlearnable — exemplars never
repeat); an abstractor distills class prototypes.

L1 spatial (8x8 s2 over 28x28, K1=64, trained first, frozen). L2
temporal joint = [fast ; g*slow ; g*label_now ; g*next_code ;
g*next_label] — the NEXT LABEL is part of the emission, so playback
advances its own labels. Cold start: zero traces + label 0 only.

Predictions (before running):
1. Generation cycles 0..9 in order with holds, indefinitely.
2. Emitted digits are class PROTOTYPES: closer to the class pixel-mean
   than to any single training exemplar (the geodesic step averages
   varied futures — abstraction is forced by exemplar variety).
3. Same drawing per class every loop (deterministic attractor):
   cross-cycle correlation ~1.
4. Watch-item: blur at class-switch frames.

Run:  .venv/bin/python experiments/2026_08_26/temporal_digits/run_temporal_digits.py
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

from run_4layer_topk import load_data  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "temporal_digits" / "results"
H = 28
WIN, STR = 8, 2
POS = [(r, c) for r in range(0, H - WIN + 1, STR) for c in range(0, H - WIN + 1, STR)]
GRID = int(np.sqrt(len(POS)))       # 11
K1, ETA1 = 64, 0.05
CODE_DIM = len(POS) * K1            # 7744
HOLD = 4
CYCLE = 10 * HOLD                   # 40 frames
K2, ETA2 = 256, 0.05
ALPHA_F, ALPHA_S = 0.2, 0.06        # slow fast-trace: under a HELD input
                                    # the previous digit's residue is the
                                    # only dwell clock (.8^4=.41 at hold end
                                    # vs .65^4=.18, which froze the carousel)
GG = 0.5                            # gain for slow/labels/next halves
JDIM = 3 * CODE_DIM + 20
THETA_CAP = 0.3
NORM_FLOOR = 1e-3
L1_DIGITS = 1000
N_CYCLES = 1200                     # 12,000 digits consumed
FREE_RUN = 5 * CYCLE                # 200 frames = 5 generated cycles
_f = np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5])
FEATHER = np.outer(_f, _f)


def cn(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v)


def cn_rows(V):
    V = V - V.mean(axis=1, keepdims=True)
    n = np.linalg.norm(V, axis=1)
    return V / np.maximum(n, 1e-9)[:, None], n > NORM_FLOOR


class SeqDict:
    """Single-sample bank: adopt-until-full bootstrap + inline geodesic."""

    def __init__(self, k, dim, eta, seed=0):
        self.k, self.dim, self.eta = k, dim, eta
        self.W = np.zeros((k, dim), dtype=np.float32)
        self.n_boot = 0
        self.win_counts = np.zeros(k, dtype=np.int64)
        self._noise = np.random.default_rng(seed)

    def step(self, z):
        if self.n_boot < self.k:
            w = z + (0.05 / np.sqrt(self.dim)) * self._noise.standard_normal(
                self.dim).astype(np.float32)
            self.W[self.n_boot] = cn(w)
            self.n_boot += 1
            self.win_counts[self.n_boot - 1] += 1
            return self.n_boot - 1
        C = self.W @ z
        u = int(np.argmax(C))
        c = max(float(C[u]), 0.0)
        if c > 0:
            w = self.W[u]
            cw = float(w @ z)
            tau = z - cw * w
            tn = np.linalg.norm(tau)
            if tn > 1e-9:
                th = min(self.eta * c, THETA_CAP)
                self.W[u] = cn(w * np.cos(th) + (tau / tn) * np.sin(th))
            self.win_counts[u] += 1
        return u


def encode(img, W1):
    V = np.stack([img[r:r + WIN, c:c + WIN].ravel() for r, c in POS])
    Vh, ok = cn_rows(V)
    code = np.maximum(Vh @ W1.T, 0.0) * ok[:, None]
    f = code.ravel()
    return (f / (np.linalg.norm(f) + 1e-9)).astype(np.float32)


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


def joint(F, S, lab, nc, nl):
    return cn(np.concatenate([F, GG * S, GG * lab, GG * nc, GG * nl])).astype(
        np.float32)


def save_gif(frames, path, scale=8, ms=100):
    imgs = [Image.fromarray((np.kron(np.clip(f, 0, 1),
                                     np.ones((scale, scale))) * 255
                             ).astype(np.uint8)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _, _ = load_data()
    pools = [np.where(ytr == c)[0] for c in range(10)]

    def exemplar(cyc, c):
        return Xtr[pools[c][cyc % len(pools[c])]]

    save_gif([exemplar(cyc, c) for cyc in range(2) for c in range(10)
              for _ in range(HOLD)], OUTPUT_DIR / "input.gif")

    # ---- L1 spatial -------------------------------------------------------
    d1 = SeqDict(K1, WIN * WIN, ETA1)
    rng = np.random.default_rng(0)
    for i in rng.permutation(L1_DIGITS):
        V = np.stack([Xtr[i][r:r + WIN, c:c + WIN].ravel() for r, c in POS])
        Vh, ok = cn_rows(V)
        for v in Vh[ok]:
            d1.step(v.astype(np.float32))
    W1 = d1.W

    # ---- L2 temporal stream ----------------------------------------------
    d2 = SeqDict(K2, JDIM, ETA2, seed=1)
    eye = np.eye(10, dtype=np.float32)
    F = np.zeros(CODE_DIM, dtype=np.float32)
    S = np.zeros(CODE_DIM, dtype=np.float32)
    codes_cache = {}

    def code_of(cyc, c):
        key = (cyc, c)
        if key not in codes_cache:
            if len(codes_cache) > 40:
                codes_cache.clear()
            codes_cache[key] = encode(exemplar(cyc, c), W1)
        return codes_cache[key]

    for cyc in range(N_CYCLES):
        for c in range(10):
            c_now = code_of(cyc, c)
            for h in range(HOLD):
                F = ALPHA_F * c_now + (1 - ALPHA_F) * F
                S = ALPHA_S * c_now + (1 - ALPHA_S) * S
                last = h == HOLD - 1
                nc_class = (c + 1) % 10 if last else c
                nc_cyc = cyc + 1 if (last and c == 9) else cyc
                nxt = code_of(nc_cyc, nc_class)
                d2.step(joint(F, S, eye[c], nxt, eye[nc_class]))

    # ---- Cold-start free-run ---------------------------------------------
    F = np.zeros(CODE_DIM, dtype=np.float32)
    S = np.zeros(CODE_DIM, dtype=np.float32)
    lab = eye[0].copy()
    gen, gen_lab = [], []
    for _ in range(FREE_RUN):
        q = joint(F, S, lab, np.zeros(CODE_DIM, dtype=np.float32),
                  np.zeros(10, dtype=np.float32))
        u = int(np.argmax(d2.W @ q))
        row = d2.W[u]
        nc = np.maximum(row[2 * CODE_DIM + 10:3 * CODE_DIM + 10], 0.0)
        nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
        nl = int(np.argmax(row[3 * CODE_DIM + 10:]))
        gen.append(render(nc, W1))
        gen_lab.append(nl)
        F = ALPHA_F * nc + (1 - ALPHA_F) * F
        S = ALPHA_S * nc + (1 - ALPHA_S) * S
        lab = eye[nl].copy()
    save_gif(gen, OUTPUT_DIR / "generated.gif")
    print("emitted labels (first 80):", "".join(str(l) for l in gen_lab[:80]),
          flush=True)

    np.savez(OUTPUT_DIR / "run.npz", gen=np.stack(gen),
             labels=np.array(gen_lab), W1=W1, W2=d2.W)

    # ---- Provenance audit: nearest exemplar among ALL shown, raw and
    # ---- re-rendered (render caps fidelity — apples to apples) -----------
    class_means = [Xtr[pools[c][:N_CYCLES]].mean(axis=0) for c in range(10)]
    picks = {}
    for i, l in enumerate(gen_lab):
        if i % HOLD == HOLD // 2:
            picks.setdefault(l, []).append(gen[i])
    lines_m = []
    gallery = []
    for c in range(10):
        if c not in picks or len(picks[c]) < 2:
            continue
        g = picks[c][1]
        gv = cn(g.ravel())
        corr_mean = float(gv @ cn(class_means[c].ravel()))
        pool = Xtr[pools[c][:N_CYCLES]]
        P = np.stack([cn(e.ravel()) for e in pool])
        corrs = P @ gv
        top = np.argsort(corrs)[::-1][:3]
        corr_ex = float(corrs[top[0]])
        rend = render(encode(pool[top[0]], W1), W1)
        corr_rend = float(gv @ cn(rend.ravel()))
        self_rend = float(cn(pool[top[0]].ravel()) @ cn(rend.ravel()))
        stab = float(cn(picks[c][1].ravel()) @ cn(picks[c][-1].ravel()))
        lines_m.append((c, corr_mean, corr_ex, corr_rend, stab))
        gallery.append((c, g, [pool[t] for t in top],
                        [float(corrs[t]) for t in top], rend, corr_rend,
                        self_rend))
        print(f"class {c}: corr(mean)={corr_mean:.3f} "
              f"max raw corr(all {len(pool)} exemplars)={corr_ex:.3f} "
              f"corr(vs RENDERED nearest)={corr_rend:.3f} "
              f"(render self-fidelity {self_rend:.3f}) cross-cycle={stab:.3f}",
              flush=True)

    fig, axes = plt.subplots(len(gallery), 5,
                             figsize=(7.5, 1.4 * len(gallery) + 0.6))
    for gi, (c, g, near, ncorr, rend, corr_rend, _) in enumerate(gallery):
        row = [("emitted", g), (f"n1 {ncorr[0]:.2f}", near[0]),
               (f"n2 {ncorr[1]:.2f}", near[1]),
               (f"n3 {ncorr[2]:.2f}", near[2]),
               (f"n1 rendered {corr_rend:.2f}", rend)]
        for j, (ttl, im) in enumerate(row):
            ax = axes[gi, j]
            ax.imshow(im, cmap="gray")
            ax.set_title(ttl, fontsize=6)
            ax.axis("off")
    fig.suptitle("Provenance: emitted digit vs nearest training exemplars")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nearest.png", dpi=110)
    plt.close(fig)

    n_show = min(5, max(len(v) for v in picks.values()) if picks else 0)
    fig, axes = plt.subplots(n_show + 1, 10, figsize=(10.5, 1.1 * (n_show + 1) + 0.6))
    for c in range(10):
        axes[0, c].imshow(class_means[c], cmap="gray")
        axes[0, c].axis("off")
        for r in range(n_show):
            ax = axes[r + 1, c]
            if c in picks and r < len(picks[c]):
                ax.imshow(picks[c][r], cmap="gray")
            ax.axis("off")
    fig.suptitle("Digit carousel — row 0: class pixel-means; "
                 "rows 1+: emitted digit per generated cycle")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cycles.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Digit carousel: memorization vs abstraction",
         "",
         f"{N_CYCLES} cycles x 10 fresh exemplars, HOLD={HOLD}, K2={K2}.",
         f"Emitted labels (first 80): {''.join(str(l) for l in gen_lab[:80])}",
         "",
         "| class | corr mean | max raw corr | corr vs rendered n1 | cross-cycle |",
         "|---|---|---|---|---|"]
        + [f"| {c} | {cm:.3f} | {ce:.3f} | {cr:.3f} | {st:.3f} |"
           for c, cm, ce, cr, st in lines_m]
        + ["", "Files: input.gif, generated.gif, cycles.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
