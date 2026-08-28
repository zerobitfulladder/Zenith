"""Animated GIFs of the temporal free-runs (Exp 15 configurations).

Rebuilds the two tops deterministically (same seeds/procedure as
run_antifreeze.py), runs the arms worth watching, writes looping GIFs
(5x nearest-neighbor upscale, ~8 fps) into results/gifs/:

  truth_*            the real movie
  twotrack_pixel_*   two-track baseline (partial advance)
  twotrack_sub_*     two-track + subtractive read (the champion),
                     trained combos AND the three held-out combos
  joint_frozen_*     the ep20 joint stack, frozen solid
  joint_adapt_*      the same rig unfrozen by the adaptation channel

Run:  .venv/bin/python experiments/2026_08_28/antifreeze/run_make_gifs.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "two_pathways"))
sys.path.insert(0, str(HERE.parent / "joint_stack"))
import run_recon3 as R                    # noqa: E402
import run_two_pathways as TP             # noqa: E402
import run_joint_stack as JS              # noqa: E402

GIF_DIR = HERE / "results" / "gifs"
T = 32
TP.T = T
EP_TOP, K_TOP, ETA_TOP, GC = 20, 512, 0.04, 0.3
SUB, HGAM, A_INC, A_DEC = 0.5, 0.5, 0.3, 0.7
FREE_STEPS = 2 * T
C1J, C2J, CF = JS.C1_DIM, JS.C2_DIM, TP.CODE_DIM


def save_gif(frames, path, scale=5, ms=125):
    mx = max(float(np.max(f)) for f in frames) + 1e-9
    imgs = []
    for f in frames:
        a = (np.clip(f / mx, 0, 1) * 255).astype(np.uint8)
        a = np.kron(a, np.ones((scale, scale), dtype=np.uint8))
        imgs.append(Image.fromarray(a, mode="L"))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0)
    print(f"wrote {path.name}", flush=True)


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    exemplar = {d: X[np.where(y == d)[0][0]] for d in TP.DIGITS}
    combos = [(d, m) for d in TP.DIGITS for m in TP.MOTIONS]
    train_combos = [c for c in combos if c not in TP.HELD_OUT]
    laps = {c: TP.make_lap(exemplar[c[0]], c[1]) for c in combos}
    dlaps = {c: TP.lap_diffs(laps[c]) for c in combos}

    wt = np.load(HERE.parent / "two_pathways" / "results" / "weights_fronts.npz")
    Wf, Wm = wt["Wf"], wt["Wm"]
    wj = np.load(HERE.parent / "joint_stack" / "results" / "weights_fronts.npz")
    WjJ, W2J = wj["Wj"], wj["W2"]
    l1f = R.Layer(64, 64, 0.0, np.random.default_rng(0))
    l1f.W = Wf
    l1jJ = R.Layer(JS.K1J, 64, 0.0, np.random.default_rng(0))
    l1jJ.W = WjJ.reshape(JS.K1J, 8, 8, 2)[..., 0].reshape(JS.K1J, -1)

    codesF = {c: TP.encode_front(laps[c][..., None], Wf) for c in combos}
    codesM = {c: TP.encode_front(dlaps[c][..., None], Wm) for c in combos}

    def jf(c):
        return np.stack([laps[c], np.roll(laps[c], 1, axis=0)], axis=-1)

    codes1J = {c: TP.encode_front(jf(c), WjJ) for c in combos}
    codes2J = {c: JS.encode_l2(codes1J[c], W2J) for c in combos}

    print("retraining tops...", flush=True)
    top2 = R.Layer(K_TOP, CF * 3 + 8, ETA_TOP, np.random.default_rng(R.SEED))
    topj = R.Layer(K_TOP, C1J * 2 + C2J + 8, ETA_TOP,
                   np.random.default_rng(R.SEED))
    for _ in range(EP_TOP):
        for t in range(T):
            for (d, m) in train_combos:
                lv = TP.labels_vec(d, m)
                r2 = TP.top_row(codesF[(d, m)][t],
                                [TP.cn(codesF[(d, m)][t - 1]),
                                 TP.cn(codesM[(d, m)][t - 1])] + lv, CF)
                if r2 is not None:
                    top2.learn(r2, top2.forward(r2))
                cargo = GC * TP.cn(codes1J[(d, m)][t])
                zj, nj = R.center_norm(np.concatenate(
                    [cargo, TP.cn(codes1J[(d, m)][t - 1]),
                     TP.cn(codes2J[(d, m)][t - 1])] + lv))
                if nj > R.EPS:
                    topj.learn(zj, topj.forward(zj))

    def run(rig, mode, d, m):
        sub = "sub" in mode
        adapt = "adapt" in mode
        top = topj if rig == "joint" else top2
        hist = np.zeros(top.dim)
        a = np.zeros(top.k)
        frames = []
        for t in range(FREE_STEPS):
            if rig == "joint":
                if t == 0:
                    k1, k2 = np.zeros(C1J), np.zeros(C2J)
                else:
                    prev2 = frames[-2] if t >= 2 else np.zeros_like(frames[-1])
                    st = np.stack([frames[-1], prev2], axis=-1)
                    c1m = TP.encode_front(st[None], WjJ)
                    k1, k2 = c1m.ravel(), JS.encode_l2(c1m, W2J).ravel()
                q, _ = R.center_norm(np.concatenate(
                    [np.zeros(C1J), TP.cn(k1), TP.cn(k2)]
                    + TP.labels_vec(d, m)))
                cargo_dim, shim, kk = C1J, l1jJ, JS.K1J
            else:
                if t == 0:
                    kf, km = np.zeros(CF), np.zeros(CF)
                else:
                    kf = TP.encode_front(frames[-1][None, ..., None], Wf).ravel()
                    dt = frames[-1] - (frames[-2] if t >= 2
                                       else np.zeros_like(frames[-1]))
                    km = TP.encode_front(dt[None, ..., None], Wm).ravel()
                q, _ = R.center_norm(np.concatenate(
                    [np.zeros(CF), TP.cn(kf), TP.cn(km)]
                    + TP.labels_vec(d, m)))
                cargo_dim, shim, kk = CF, l1f, 64
            scores = top.forward(q)
            if sub and t > 0:
                hh = hist / (np.linalg.norm(hist) + R.EPS)
                scores = scores - SUB * (top.W @ hh)
            if adapt:
                scores = scores - a
            w = int(np.argmax(scores))
            hist = HGAM * hist + (1 - HGAM) * top.W[w]
            a *= A_DEC
            a[w] += A_INC
            frames.append(TP.render_code(top.W[w, :cargo_dim], shim, kk))
        return frames

    for (d, m) in [(0, "right"), (1, "rot")]:
        save_gif(list(laps[(d, m)]), GIF_DIR / f"truth_{d}_{m}.gif")
    save_gif(run("twotrack", "pixel", 0, "right"),
             GIF_DIR / "twotrack_pixel_0_right.gif")
    for (d, m) in [(0, "right"), (1, "rot"), (0, "down")] + TP.HELD_OUT:
        held = "_HELDOUT" if (d, m) in TP.HELD_OUT else ""
        save_gif(run("twotrack", "pixel_sub", d, m),
                 GIF_DIR / f"twotrack_sub_{d}_{m}{held}.gif")
    save_gif(run("joint", "pixel", 0, "right"),
             GIF_DIR / "joint_frozen_0_right.gif")
    for (d, m) in [(0, "right"), (1, "rot")]:
        save_gif(run("joint", "pixel_adapt", d, m),
                 GIF_DIR / f"joint_adapt_{d}_{m}.gif")
    print("all gifs written to", GIF_DIR, flush=True)


if __name__ == "__main__":
    main()
