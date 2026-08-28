"""Exp 14 — single-track JOINT stack, bottom to top (user request).

One track only: L1 sees [frame_t ; frame_t-1] as 2-channel windows
(transition units), L2 convolves over L1's code (transition MOTIFS,
skeleton learning), and the lag-advance top keys on the lagged L1+L2
codes plus labels. No form/motion split anywhere.

  L1   8x8/s2 x 2ch, K=128, dense learn      (joint transition parts)
  L2   3x3/s2 over code, K=128, skeleton     (transition motifs)
  TOP  [GC*present-L1-code ; lagged L1 code ; lagged L2 code ;
        digit + motion labels], K=512, lag-advance (cargo=present)

Arms: EP_TOP in {2, 20} — Exp 13's lesson made primary (overtrained
tops freeze free-run; joint-shallow scored 0.83 advance undertrained
vs 0.00 overtrained). Same world/held-outs as Exp 13 for comparison.

Predictions (before running):
1. Depth helps teacher-forced: lagged L2 motifs disambiguate phase,
   tf > Exp 13 joint-shallow's 0.703.
2. EP_TOP=2 free-runs well (>0.8 advance); EP_TOP=20 freezes (<0.1) —
   the freeze is training-depth, not joint-track, in origin.
3. Held-out combos still recite (enumerating top unchanged).

Run:  .venv/bin/python experiments/2026_08_28/joint_stack/run_joint_stack.py
Env:  JS_T (32), JS_KTOP (512), JS_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_full_gpu"))
sys.path.insert(0, str(HERE.parent / "two_pathways"))
import run_recon3 as R                    # noqa: E402
import run_residual_full_gpu as G         # noqa: E402
import run_two_pathways as TP             # noqa: E402

xp = G.xp
OUTPUT_DIR = HERE / "results" / os.environ.get('JS_TAG', '').lstrip('_')
T = int(os.environ.get("JS_T", "32"))
TP.T = T                                   # keep helpers consistent
K1J, K2J = 128, 128
K_TOP = int(os.environ.get("JS_KTOP", "512"))
EP_TOPS = [2, 20]
GC, LAB, ETA_TOP = 0.3, 0.5, 0.04
G2 = (R.G1 - R.W2_WIN) // R.W2_STR + 1     # 5
C1_DIM = R.G1 * R.G1 * K1J
C2_DIM = G2 * G2 * K2J
FREE_STEPS = 2 * T


def train_l2(code_maps, k2):
    d = G.Dict(k2, R.W2_WIN * R.W2_WIN * K1J, R.ETA2)
    Xx = xp.asarray(code_maps, dtype=G.DTYPE)
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(code_maps))
        for s in range(0, len(code_maps), 8):
            xb = Xx[xp.asarray(order[s:s + 8])]
            G.level_pass(xb, d, G.POS2, R.W2_WIN, K1J, G2,
                         "skeleton", True, 1)
    return (d.W.get() if G.XP_NAME == "cupy" else np.asarray(d.W)).astype(
        np.float64)


def encode_l2(code_maps, W2):
    Xx = xp.asarray(code_maps, dtype=G.DTYPE)
    P = G.window_stack(Xx, G.POS2, R.W2_WIN).reshape(
        len(code_maps) * len(G.POS2), -1)
    Ph, _, _, ok = G.center_norm_rows(P)
    C = xp.maximum(Ph @ xp.asarray(W2, dtype=G.DTYPE).T, 0.0) * ok[:, None]
    C = C.reshape(len(code_maps), G2, G2, W2.shape[0])
    return (C.get() if G.XP_NAME == "cupy" else np.asarray(C)).astype(
        np.float64)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    exemplar = {d: X[np.where(y == d)[0][0]] for d in TP.DIGITS}

    combos = [(d, m) for d in TP.DIGITS for m in TP.MOTIONS]
    train_combos = [c for c in combos if c not in TP.HELD_OUT]
    laps = {c: TP.make_lap(exemplar[c[0]], c[1]) for c in combos}

    def joint_frames(c):
        return np.stack([laps[c], np.roll(laps[c], 1, axis=0)], axis=-1)

    trJ = np.concatenate([joint_frames(c) for c in train_combos])
    print("training L1 (joint 2ch)...", flush=True)
    Wj = TP.train_front(trJ, 2, K1J)
    l1j = R.Layer(K1J, R.W1_WIN ** 2, 0.0, np.random.default_rng(0))
    l1j.W = Wj.reshape(K1J, R.W1_WIN, R.W1_WIN, 2)[..., 0].reshape(K1J, -1)

    codes1 = {c: TP.encode_front(joint_frames(c), Wj) for c in combos}
    print("training L2 over code...", flush=True)
    W2 = train_l2(np.concatenate([codes1[c] for c in train_combos]), K2J)
    codes2 = {c: encode_l2(codes1[c], W2) for c in combos}
    np.savez(OUTPUT_DIR / "weights_fronts.npz", Wj=Wj, W2=W2)

    res = {"T": T, "k_top": K_TOP, "held_out": [list(c) for c in TP.HELD_OUT]}
    for name, codes, k in (("L1joint", codes1, K1J), ("L2joint", codes2, K2J)):
        wins_m = np.zeros((k, len(TP.MOTIONS)))
        wins_d = np.zeros((k, len(TP.DIGITS)))
        for (d, m) in train_combos:
            flat = codes[(d, m)].reshape(-1, k)
            wn = flat.argmax(axis=1)
            act = flat.max(axis=1) > 0
            for u in wn[act]:
                wins_m[u, TP.MOTIONS.index(m)] += 1
                wins_d[u, TP.DIGITS.index(d)] += 1
        tot = wins_m.sum(axis=1) + 1e-9
        res[f"{name}_motion_selectivity"] = round(
            float((wins_m.max(axis=1) / tot)[tot > 1].mean()), 3)
        res[f"{name}_digit_selectivity"] = round(
            float((wins_d.max(axis=1) / (wins_d.sum(axis=1) + 1e-9))[tot > 1].mean()), 3)

    dim = C1_DIM * 2 + C2_DIM + len(TP.DIGITS) + len(TP.MOTIONS)

    def build_row(c1_t, c1_p, c2_p, d, m, empty_cargo=False):
        cargo = (np.zeros(C1_DIM) if empty_cargo else GC * TP.cn(c1_t))
        parts = [cargo, TP.cn(c1_p), TP.cn(c2_p)] + TP.labels_vec(d, m)
        z, n = R.center_norm(np.concatenate(parts))
        return z if n > R.EPS else None

    strips, strip_titles = [], []
    for ep_top in EP_TOPS:
        rng = np.random.default_rng(R.SEED)
        top = R.Layer(K_TOP, dim, ETA_TOP, rng)
        print(f"training top (EP={ep_top})...", flush=True)
        for _ in range(ep_top):
            for t in range(T):
                for (d, m) in train_combos:
                    row = build_row(codes1[(d, m)][t], codes1[(d, m)][t - 1],
                                    codes2[(d, m)][t - 1], d, m)
                    if row is not None:
                        top.learn(row, top.forward(row))
        tag = f"ep{ep_top}"
        res[f"{tag}_top_units_used"] = int((top.win_counts > 0).sum())

        def emit(c1_p, c2_p, d, m):
            q = build_row(None, c1_p, c2_p, d, m, empty_cargo=True)
            w = int(np.argmax(top.forward(q)))
            return TP.render_code(top.W[w, :C1_DIM], l1j, K1J)

        for gtag, group in (("train", train_combos), ("held", TP.HELD_OUT)):
            tf = []
            for (d, m) in group:
                for t in range(T):
                    f = emit(codes1[(d, m)][t - 1], codes2[(d, m)][t - 1], d, m)
                    tf.append(float(TP.cn(f) @ TP.cn(laps[(d, m)][t])))
            res[f"{tag}_tf_{gtag}"] = round(float(np.mean(tf)), 3)

        def free_run(d, m):
            frames = []
            for t in range(FREE_STEPS):
                if t == 0:
                    c1_p, c2_p = np.zeros(C1_DIM), np.zeros(C2_DIM)
                else:
                    prev2 = frames[-2] if t >= 2 else np.zeros_like(frames[-1])
                    st = np.stack([frames[-1], prev2], axis=-1)
                    c1m = TP.encode_front(st[None], Wj)
                    c1_p = c1m.ravel()
                    c2_p = encode_l2(c1m, W2).ravel()
                frames.append(emit(c1_p, c2_p, d, m))
            return frames

        for gtag, group in (("train", train_combos), ("held", TP.HELD_OUT)):
            advs, mms = [], []
            for gi, (d, m) in enumerate(group):
                fr = free_run(d, m)
                adv, mm = TP.phase_track(fr[T // 2:], laps[(d, m)])
                advs.append(adv)
                mms.append(mm)
                if gi < 2 or gtag == "held":
                    for i in range(0, T, 2):
                        strips.append(fr[i])
                        strip_titles.append(f"{tag} {d}{m[:2]}" if i == 0 else None)
            res[f"{tag}_free_{gtag}_advance"] = round(float(np.mean(advs)), 3)
            res[f"{tag}_free_{gtag}_match"] = round(float(np.mean(mms)), 3)
        print(f"{tag} done", flush=True)
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)

    R.gallery(strips, strip_titles, OUTPUT_DIR / "free_runs_joint_stack.png",
              "Joint single-track stack free runs (rows per arm; held-out last)",
              16)
    R.gallery([Wj[i].reshape(8, 8, 2)[..., 0] for i in range(K1J)], None,
              OUTPUT_DIR / "templates_L1_framehalf.png",
              "Joint L1, frame_t half (signed)", 16, cmap="bwr")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
