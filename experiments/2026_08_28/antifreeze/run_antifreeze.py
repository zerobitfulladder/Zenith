"""Exp 15 — the anti-freeze trio, read-time only, on the frozen rigs.

Three fixes from the freeze diagnosis (pixel-loop drift + hold-is-
self-reinforcing), applied WITHOUT retraining fronts (tops rebuilt
deterministically, same seeds/procedures as Exps 13/14):

  code    feed back the retrieved CODE instead of re-seeing the
          rendered drawing (the square rigs' self-cleaning loop;
          render only for display)
  sub     subtractive read at the top: scores minus SUB * overlap with
          a decaying history of retrieved rows (explaining-away, the
          08-27 anti-freeze; content-aware, graded — not hard
          refractory). Rows carry the arrow, so once the repeat is
          suppressed the lagged keys should choose FORWARD (last
          night's 50/50 was on an arrow-less rig).
  adapt   adaptation channel: per-unit fatigue a_u (+A_INC on win,
          *A_DEC per step) subtracted from scores — response decays
          under constant use; attacks period-1 directly.

Rigs: Exp 14 joint stack, ep20 top (baseline advance 0.000, crisp) —
the clean testbed; Exp 13 two-track (baseline 0.362). Two-track "code"
arm is a hybrid: form key from the retrieved cargo, motion key still
from rendered-frame diffs (motion has no stored cargo).

Run:  .venv/bin/python experiments/2026_08_28/antifreeze/run_antifreeze.py
Env:  AF_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "two_pathways"))
sys.path.insert(0, str(HERE.parent / "joint_stack"))
import run_recon3 as R                    # noqa: E402
import run_two_pathways as TP             # noqa: E402
import run_joint_stack as JS              # noqa: E402

OUTPUT_DIR = HERE / "results" / os.environ.get('AF_TAG', '').lstrip('_')
T = 32
TP.T = T
EP_TOP, K_TOP, ETA_TOP, GC = 20, 512, 0.04, 0.3
SUB, HGAM = 0.5, 0.5
A_INC, A_DEC = 0.3, 0.7
FREE_STEPS = 2 * T
C1J, C2J = JS.C1_DIM, JS.C2_DIM
CF = TP.CODE_DIM


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    exemplar = {d: X[np.where(y == d)[0][0]] for d in TP.DIGITS}
    combos = [(d, m) for d in TP.DIGITS for m in TP.MOTIONS]
    train_combos = [c for c in combos if c not in TP.HELD_OUT]
    laps = {c: TP.make_lap(exemplar[c[0]], c[1]) for c in combos}
    dlaps = {c: TP.lap_diffs(laps[c]) for c in combos}

    # ---- rebuild rigs from saved fronts -----------------------------------
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

    # ---- free-run engine ---------------------------------------------------
    def run(rig, mode, d, m):
        sub = "sub" in mode
        adapt = "adapt" in mode
        codeloop = mode.startswith("code")
        top = topj if rig == "joint" else top2
        hist = np.zeros(top.dim)
        a = np.zeros(top.k)
        frames, cargo_prev = [], None
        for t in range(FREE_STEPS):
            if rig == "joint":
                if t == 0:
                    k1, k2 = np.zeros(C1J), np.zeros(C2J)
                elif codeloop:
                    c1map = np.maximum(cargo_prev, 0.0).reshape(
                        R.G1, R.G1, JS.K1J)
                    k1 = c1map.ravel()
                    k2 = JS.encode_l2(c1map[None], W2J).ravel()
                else:
                    prev2 = frames[-2] if t >= 2 else np.zeros_like(frames[-1])
                    st = np.stack([frames[-1], prev2], axis=-1)
                    c1m = TP.encode_front(st[None], WjJ)
                    k1, k2 = c1m.ravel(), JS.encode_l2(c1m, W2J).ravel()
                q, nq = R.center_norm(np.concatenate(
                    [np.zeros(C1J), TP.cn(k1), TP.cn(k2)]
                    + TP.labels_vec(d, m)))
                cargo_dim = C1J
                shim, kk = l1jJ, JS.K1J
            else:
                if t == 0:
                    kf, km = np.zeros(CF), np.zeros(CF)
                else:
                    if codeloop:
                        kf = np.maximum(cargo_prev, 0.0)
                    else:
                        kf = TP.encode_front(frames[-1][None, ..., None],
                                             Wf).ravel()
                    dt = frames[-1] - (frames[-2] if t >= 2
                                       else np.zeros_like(frames[-1]))
                    km = TP.encode_front(dt[None, ..., None], Wm).ravel()
                q, nq = R.center_norm(np.concatenate(
                    [np.zeros(CF), TP.cn(kf), TP.cn(km)]
                    + TP.labels_vec(d, m)))
                cargo_dim = CF
                shim, kk = l1f, 64
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
            cargo_prev = top.W[w, :cargo_dim]
            frames.append(TP.render_code(cargo_prev, shim, kk))
        return frames

    res = {"T": T, "sub": SUB, "hgam": HGAM, "a_inc": A_INC, "a_dec": A_DEC}
    strips, titles = [], []
    ARMS = {"joint": ["pixel", "code", "pixel_sub", "pixel_adapt", "code_sub"],
            "twotrack": ["pixel", "code", "pixel_sub", "pixel_adapt"]}
    for rig, arms in ARMS.items():
        for mode in arms:
            for gtag, group in (("train", train_combos), ("held", TP.HELD_OUT)):
                advs, mms = [], []
                for gi, (d, m) in enumerate(group):
                    fr = run(rig, mode, d, m)
                    adv, mm = TP.phase_track(fr[T // 2:], laps[(d, m)])
                    advs.append(adv)
                    mms.append(mm)
                    if gtag == "train" and gi == 0:
                        for i in range(0, T, 2):
                            strips.append(fr[i])
                            titles.append(f"{rig[:2]}-{mode}" if i == 0 else None)
                res[f"{rig}_{mode}_{gtag}_advance"] = round(float(np.mean(advs)), 3)
                res[f"{rig}_{mode}_{gtag}_match"] = round(float(np.mean(mms)), 3)
            print(f"{rig} {mode} done", flush=True)
            with open(OUTPUT_DIR / "metrics.json", "w") as f:
                json.dump(res, f, indent=2)

    R.gallery(strips, titles, OUTPUT_DIR / "antifreeze_runs.png",
              "Anti-freeze arms, first trained combo (every 2nd frame)", 16)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
