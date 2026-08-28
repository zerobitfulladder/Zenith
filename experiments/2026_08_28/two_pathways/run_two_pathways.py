"""Exp 13 — two-pathway temporal rig: form + motion, unified, lag-advance.

The user's design (from direction-preferring neurons — units receiving
current + delayed input; biology: the Reichardt correlator, ventral/
dorsal pathways): IDENTITY and MOTION are different things.

  form track    conv L1 over the current frame        (what)
  motion track  conv L1 over the SIGNED frame diff    (where it's going)
  unification   global lag-advance memory over
                [GC*present-form ; lagged form ; lagged motion ;
                 digit label ; motion label]
                — cargo = present (small gain), keys = lagged context.
                NO next slots; the write lags perception one tick.

Control arm: JOINT front — one bank over stacked [frame_t ; frame_t-1]
2-channel windows (transition units; enumerates identity x motion).

World: MNIST exemplars (digits 0-4) x motions {right, down, rot},
T=32 circular laps. HELD-OUT combos (2,right),(3,down),(4,rot) test
composition. Staged training (fronts on shuffled frames via GPU
mini-batch B=8; frozen fronts; top online, combos interleaved).

Predictions (recorded before running):
1. Two-track free-runs lap correctly on trained combos INCLUDING
   rotation (motion track cracks the 0.39 advance of the hier rig).
2. Motion units are direction-selective and largely form-neutral;
   form units motion-neutral.
3. Held-out combos FAIL at the top under top-1 cold start (v5 law:
   a global bank recites the nearest trained combo) — the factored
   FRONT alone cannot compose; the top read must be factored next.
4. Joint front also laps trained combos (small world, capacity fine)
   but its units are transition-tuned (digit x motion specific), and
   it fails held-out combos the same way.

Run:  .venv/bin/python experiments/2026_08_28/two_pathways/run_two_pathways.py
Env:  TP_T (32), TP_KTOP (512), TP_EPTOP (20), TP_TAG
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.ndimage import rotate as nd_rotate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "residual_full_gpu"))
import run_recon3 as R                    # noqa: E402
import run_residual_full_gpu as G         # noqa: E402

xp = G.xp
OUTPUT_DIR = HERE / "results" / os.environ.get('TP_TAG', '').lstrip('_')
DIGITS = [0, 1, 2, 3, 4]
MOTIONS = ["right", "down", "rot"]
HELD_OUT = [(2, "right"), (3, "down"), (4, "rot")]
T = int(os.environ.get("TP_T", "32"))
K_FORM, K_MOT, K_JOINT = 64, 64, 128
K_TOP = int(os.environ.get("TP_KTOP", "512"))
EP_FRONT, EP_TOP = 3, int(os.environ.get("TP_EPTOP", "20"))
GC, LAB, ETA_TOP = 0.3, 0.5, 0.04
CODE_DIM = R.G1 * R.G1 * K_FORM
JCODE_DIM = R.G1 * R.G1 * K_JOINT
FREE_STEPS = 2 * T


def make_lap(img, motion):
    frames = np.zeros((T, R.SIDE, R.SIDE))
    for t in range(T):
        if motion == "right":
            frames[t] = np.roll(img, t, axis=1)
        elif motion == "down":
            frames[t] = np.roll(img, t, axis=0)
        else:
            frames[t] = np.clip(nd_rotate(img, 360.0 * t / T, reshape=False,
                                          order=1, mode="constant"), 0, 1)
    return frames


def lap_diffs(frames):
    return frames - np.roll(frames, 1, axis=0)      # circular: t minus t-1


def train_front(Xn, channels, k):
    d = G.Dict(k, R.W1_WIN * R.W1_WIN * channels, R.ETA1)
    Xx = xp.asarray(Xn, dtype=G.DTYPE)
    rng = np.random.default_rng(0)
    for _ in range(EP_FRONT):
        order = rng.permutation(len(Xn))
        for s in range(0, len(Xn), 8):
            xb = Xx[xp.asarray(order[s:s + 8])]
            G.level_pass(xb, d, G.POS1, R.W1_WIN, channels, R.G1,
                         "dense", True, 1)
    return (d.W.get() if G.XP_NAME == "cupy" else np.asarray(d.W)).astype(
        np.float64)


def encode_front(Xn, W):
    Xx = xp.asarray(Xn, dtype=G.DTYPE)
    P = G.window_stack(Xx, G.POS1, R.W1_WIN).reshape(
        len(Xn) * len(G.POS1), -1)
    Ph, _, _, ok = G.center_norm_rows(P)
    C = xp.maximum(Ph @ xp.asarray(W, dtype=G.DTYPE).T, 0.0) * ok[:, None]
    C = C.reshape(len(Xn), R.G1, R.G1, W.shape[0])
    return (C.get() if G.XP_NAME == "cupy" else np.asarray(C)).astype(
        np.float64)


def cn(v):
    return R.center_norm(np.asarray(v, dtype=np.float64).ravel())[0]


def top_row(cargo, key_parts, cargo_dim, empty_cargo=False):
    c = np.zeros(cargo_dim) if empty_cargo else GC * cn(cargo)
    z, n = R.center_norm(np.concatenate([c] + key_parts))
    return z if n > R.EPS else None


def labels_vec(d, m):
    dv = np.zeros(len(DIGITS))
    dv[DIGITS.index(d)] = LAB
    mv = np.zeros(len(MOTIONS))
    mv[MOTIONS.index(m)] = LAB
    return [dv, mv]


def render_code(code_flat, l1shim, k):
    code = np.maximum(code_flat, 0.0).reshape(R.G1, R.G1, k)
    return R.decode_l1(code, np.zeros((R.G1, R.G1)), np.ones((R.G1, R.G1)),
                       l1shim, skip_empty=True)


def phase_track(run_frames, lap):
    lapn = np.stack([cn(f) for f in lap])
    best, corrs = [], []
    for f in run_frames:
        c = lapn @ cn(f)
        best.append(int(np.argmax(c)))
        corrs.append(float(np.max(c)))
    adv = np.mean([(best[i + 1] - best[i]) % T == 1
                   for i in range(len(best) - 1)])
    return float(adv), float(np.mean(corrs))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    exemplar = {d: X[np.where(y == d)[0][0]] for d in DIGITS}

    combos = [(d, m) for d in DIGITS for m in MOTIONS]
    train_combos = [c for c in combos if c not in HELD_OUT]
    laps = {c: make_lap(exemplar[c[0]], c[1]) for c in combos}
    dlaps = {c: lap_diffs(laps[c]) for c in combos}

    trF = np.concatenate([laps[c] for c in train_combos])
    trD = np.concatenate([dlaps[c] for c in train_combos])
    trJ = np.stack([np.concatenate([laps[c] for c in train_combos]),
                    np.concatenate([np.roll(laps[c], 1, axis=0)
                                    for c in train_combos])], axis=-1)

    print("training fronts...", flush=True)
    Wf = train_front(trF[..., None], 1, K_FORM)
    Wm = train_front(trD[..., None], 1, K_MOT)
    Wj = train_front(trJ, 2, K_JOINT)
    np.savez(OUTPUT_DIR / "weights_fronts.npz", Wf=Wf, Wm=Wm, Wj=Wj)

    l1f = R.Layer(K_FORM, R.W1_WIN ** 2, 0.0, np.random.default_rng(0))
    l1f.W = Wf
    # frame-half shim for rendering the joint arm (channel 0 = frame_t)
    l1j = R.Layer(K_JOINT, R.W1_WIN ** 2, 0.0, np.random.default_rng(0))
    l1j.W = Wj.reshape(K_JOINT, R.W1_WIN, R.W1_WIN, 2)[..., 0].reshape(
        K_JOINT, -1)

    codesF = {c: encode_front(laps[c][..., None], Wf) for c in combos}
    codesM = {c: encode_front(dlaps[c][..., None], Wm) for c in combos}
    codesJ = {c: encode_front(
        np.stack([laps[c], np.roll(laps[c], 1, axis=0)], axis=-1), Wj)
        for c in combos}

    # ---- selectivity: who wins where -------------------------------------
    res = {"T": T, "k_top": K_TOP, "held_out": [list(c) for c in HELD_OUT]}
    for name, codes, k in (("form", codesF, K_FORM), ("motion", codesM, K_MOT),
                           ("joint", codesJ, K_JOINT)):
        wins_m = np.zeros((k, len(MOTIONS)))
        wins_d = np.zeros((k, len(DIGITS)))
        for (d, m) in train_combos:
            cmap = codes[(d, m)]
            wn = cmap.reshape(-1, k).argmax(axis=1)
            act = cmap.reshape(-1, k).max(axis=1) > 0
            for u in wn[act]:
                wins_m[u, MOTIONS.index(m)] += 1
                wins_d[u, DIGITS.index(d)] += 1
        tot = wins_m.sum(axis=1) + 1e-9
        res[f"{name}_motion_selectivity"] = round(
            float((wins_m.max(axis=1) / tot)[tot > 1].mean()), 3)
        res[f"{name}_digit_selectivity"] = round(
            float((wins_d.max(axis=1) / (wins_d.sum(axis=1) + 1e-9))[tot > 1].mean()), 3)

    # ---- top banks: two-track and joint ----------------------------------
    rng = np.random.default_rng(R.SEED)
    dim2 = CODE_DIM * 3 + len(DIGITS) + len(MOTIONS)
    dimj = JCODE_DIM * 2 + len(DIGITS) + len(MOTIONS)
    top2 = R.Layer(K_TOP, dim2, ETA_TOP, rng)
    topj = R.Layer(K_TOP, dimj, ETA_TOP, rng)
    print("training tops...", flush=True)
    for ep in range(EP_TOP):
        for t in range(T):
            for (d, m) in train_combos:
                lv = labels_vec(d, m)
                r2 = top_row(codesF[(d, m)][t],
                             [cn(codesF[(d, m)][t - 1]),
                              cn(codesM[(d, m)][t - 1])] + lv, CODE_DIM)
                if r2 is not None:
                    top2.learn(r2, top2.forward(r2))
                rj = top_row(codesJ[(d, m)][t],
                             [cn(codesJ[(d, m)][t - 1])] + lv, JCODE_DIM)
                if rj is not None:
                    topj.learn(rj, topj.forward(rj))
    res["top_units_used"] = [int((t.win_counts > 0).sum()) for t in (top2, topj)]

    # ---- teacher-forced next-frame ---------------------------------------
    def emit2(cf_p, cm_p, d, m):
        q = top_row(np.zeros(CODE_DIM),
                    [cn(cf_p), cn(cm_p)] + labels_vec(d, m), CODE_DIM,
                    empty_cargo=True)
        w = int(np.argmax(top2.forward(q)))
        return render_code(top2.W[w, :CODE_DIM], l1f, K_FORM)

    def emitj(cj_p, d, m):
        q = top_row(np.zeros(JCODE_DIM), [cn(cj_p)] + labels_vec(d, m),
                    JCODE_DIM, empty_cargo=True)
        w = int(np.argmax(topj.forward(q)))
        return render_code(topj.W[w, :JCODE_DIM], l1j, K_JOINT)

    for tag, group in (("train", train_combos), ("held", HELD_OUT)):
        tf2, tfj = [], []
        for (d, m) in group:
            for t in range(T):
                f2 = emit2(codesF[(d, m)][t - 1], codesM[(d, m)][t - 1], d, m)
                tf2.append(float(cn(f2) @ cn(laps[(d, m)][t])))
                fj = emitj(codesJ[(d, m)][t - 1], d, m)
                tfj.append(float(cn(fj) @ cn(laps[(d, m)][t])))
        res[f"tf_{tag}_twotrack"] = round(float(np.mean(tf2)), 3)
        res[f"tf_{tag}_joint"] = round(float(np.mean(tfj)), 3)

    # ---- free run (cold start from labels, re-see loop) -------------------
    def free_run2(d, m):
        frames = []
        for t in range(FREE_STEPS):
            if t == 0:
                cf_p, cm_p = np.zeros(CODE_DIM), np.zeros(CODE_DIM)
            else:
                cf_p = encode_front(frames[-1][None, ..., None], Wf).ravel()
                dt = frames[-1] - (frames[-2] if t >= 2 else np.zeros_like(frames[-1]))
                cm_p = encode_front(dt[None, ..., None], Wm).ravel()
            frames.append(emit2(cf_p, cm_p, d, m))
        return frames

    def free_runj(d, m):
        frames = []
        for t in range(FREE_STEPS):
            if t == 0:
                cj_p = np.zeros(JCODE_DIM)
            else:
                prev2 = frames[-2] if t >= 2 else np.zeros_like(frames[-1])
                st = np.stack([frames[-1], prev2], axis=-1)
                cj_p = encode_front(st[None], Wj).ravel()
            frames.append(emitj(cj_p, d, m))
        return frames

    strips, strip_titles = [], []
    for tag, group in (("train", train_combos[:4] + train_combos[-2:]),
                       ("held", HELD_OUT)):
        a2, aj, m2, mj = [], [], [], []
        for (d, m) in group:
            fr2 = free_run2(d, m)
            frj = free_runj(d, m)
            adv2, mm2 = phase_track(fr2[T // 2:], laps[(d, m)])
            advj, mmj = phase_track(frj[T // 2:], laps[(d, m)])
            a2.append(adv2); aj.append(advj); m2.append(mm2); mj.append(mmj)
            if (d, m) in (group[:2] if tag == "train" else group):
                for i in range(0, T, 2):
                    strips.append(fr2[i])
                    strip_titles.append(f"{d}{m[:2]}" if i == 0 else None)
        res[f"free_{tag}_twotrack_advance"] = round(float(np.mean(a2)), 3)
        res[f"free_{tag}_twotrack_match"] = round(float(np.mean(m2)), 3)
        res[f"free_{tag}_joint_advance"] = round(float(np.mean(aj)), 3)
        res[f"free_{tag}_joint_match"] = round(float(np.mean(mj)), 3)

    R.gallery(strips, strip_titles, OUTPUT_DIR / "free_runs_twotrack.png",
              "Two-track free runs (every 2nd frame; last rows = held-out combos)",
              16)
    R.gallery([np.maximum(Wf[i], 0).reshape(8, 8) for i in range(K_FORM)],
              None, OUTPUT_DIR / "templates_form.png",
              "Form track L1", 16, cmap="inferno")
    R.gallery([Wm[i].reshape(8, 8) for i in range(K_MOT)],
              None, OUTPUT_DIR / "templates_motion.png",
              "Motion track L1 (signed diffs)", 16, cmap="bwr")
    R.gallery([Wj[i].reshape(8, 8, 2)[..., 0] for i in range(K_JOINT)],
              None, OUTPUT_DIR / "templates_joint_framehalf.png",
              "Joint front, frame_t half", 16, cmap="bwr")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
