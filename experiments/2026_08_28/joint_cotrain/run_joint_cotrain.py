"""Exp 16 — the whole temporal stack learning TOGETHER (user request).

No stages: L1 (joint [t;t-1] 2ch windows), L2 (over L1's code, skeleton
view), and the lag-advance top all learn simultaneously, online, from
one interleaved stream of the training movies. The top's keys are the
LAGGED codes as they actually streamed (carried across the stream, not
recomputed) — everything plastic at once, the biologically honest
regime.

Tension under test: the 08-27 monopoly lesson (co-training a small
global bank on a forming code -> one-unit monopoly; staged fixed it)
vs the spatial stack's standing practice (all layers co-train online,
always have, no monopoly). Which regime does the temporal rig follow?

Same world as Exps 13/14 (digits 0-4 x {right,down,rot}, T=32,
held-outs). Eval mirrors Exp 14 (staged ep20 = the bar: tf 0.696,
free pixel advance 0.000) plus the code-loop read from Exp 15.

Predictions (before running):
1. No L1/L2 monopoly (online + interleaved differs from the failed
   case's minibatch-on-forming-code); top usage may be thinner early.
2. Teacher-forced lands near staged (within a few points) — early top
   rows written against immature codes get re-written by rehearsal.
3. Free-run behaves like staged (the freeze is a read/loop property,
   not a training-schedule property).

Run:  .venv/bin/python experiments/2026_08_28/joint_cotrain/run_joint_cotrain.py
Env:  JC_EPOCHS (20), JC_TAG
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

OUTPUT_DIR = HERE / "results" / os.environ.get('JC_TAG', '').lstrip('_')
T = 32
TP.T = T
K1, K2, K_TOP = 128, 128, 512
EPOCHS = int(os.environ.get("JC_EPOCHS", "20"))
GC, ETA_TOP = 0.3, 0.04
C1_DIM = R.G1 * R.G1 * K1
G2 = (R.G1 - R.W2_WIN) // R.W2_STR + 1
C2_DIM = G2 * G2 * K2
FREE_STEPS = 2 * T


def encode_joint(x2ch, dic, learning):
    """2-channel 8x8 windows, dense learning, relu speech (online)."""
    out = np.zeros((R.G1, R.G1, dic.k))
    for gi in range(R.G1):
        for gj in range(R.G1):
            p = x2ch[gi * R.W1_STR:gi * R.W1_STR + R.W1_WIN,
                     gj * R.W1_STR:gj * R.W1_STR + R.W1_WIN, :].ravel()
            p_hat, n = R.center_norm(p)
            if n < R.NORM_FLOOR:
                continue
            c = dic.forward(p_hat)
            if learning:
                dic.learn(p_hat, c)
            out[gi, gj, :] = np.maximum(c, 0.0)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(R.ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(R.ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    exemplar = {d: X[np.where(y == d)[0][0]] for d in TP.DIGITS}
    combos = [(d, m) for d in TP.DIGITS for m in TP.MOTIONS]
    train_combos = [c for c in combos if c not in TP.HELD_OUT]
    laps = {c: TP.make_lap(exemplar[c[0]], c[1]) for c in combos}

    rng = np.random.default_rng(R.SEED)
    d1 = R.Layer(K1, R.W1_WIN * R.W1_WIN * 2, R.ETA1, rng)
    d2 = R.Layer(K2, R.W2_WIN * R.W2_WIN * K1, R.ETA2, rng)
    top = R.Layer(K_TOP, C1_DIM * 2 + C2_DIM + 8, ETA_TOP, rng)

    def build_row(c1_t, c1_p, c2_p, d, m, empty_cargo=False):
        cargo = np.zeros(C1_DIM) if empty_cargo else GC * TP.cn(c1_t)
        parts = [cargo, TP.cn(c1_p), TP.cn(c2_p)] + TP.labels_vec(d, m)
        z, n = R.center_norm(np.concatenate(parts))
        return z if n > R.EPS else None

    # ---- ONE stream, everything learning at once --------------------------
    prev = {c: None for c in train_combos}       # (c1, c2) carried lagged
    for ep in range(EPOCHS):
        for t in range(T):
            for (d, m) in train_combos:
                fr = laps[(d, m)][t]
                fp = laps[(d, m)][t - 1]
                m1 = encode_joint(np.stack([fr, fp], axis=-1), d1, True)
                m2 = R.encode_over(m1, d2, R.W2_WIN, R.W2_STR, G2, True)
                if prev[(d, m)] is not None:
                    c1p, c2p = prev[(d, m)]
                    row = build_row(m1.ravel(), c1p, c2p, d, m)
                    if row is not None:
                        top.learn(row, top.forward(row))
                prev[(d, m)] = (m1.ravel(), m2.ravel())
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep + 1}/{EPOCHS}", flush=True)
    np.savez(OUTPUT_DIR / "weights.npz", W1=d1.W, W2=d2.W, Wtop=top.W)

    res = {"T": T, "epochs": EPOCHS,
           "units_used": [int((b.win_counts > 0).sum())
                          for b in (d1, d2, top)],
           "top_max_win_share": round(float(
               top.win_counts.max() / max(top.win_counts.sum(), 1)), 4)}

    # ---- eval on the frozen result ----------------------------------------
    l1shim = R.Layer(K1, R.W1_WIN ** 2, 0.0, np.random.default_rng(0))
    l1shim.W = d1.W.reshape(K1, R.W1_WIN, R.W1_WIN, 2)[..., 0].reshape(K1, -1)

    def jf(c):
        return np.stack([laps[c], np.roll(laps[c], 1, axis=0)], axis=-1)

    codes1 = {c: TP.encode_front(jf(c), d1.W) for c in combos}
    codes2 = {c: JS.encode_l2(codes1[c], d2.W) for c in combos}

    def emit(c1_p, c2_p, d, m):
        q = build_row(None, c1_p, c2_p, d, m, empty_cargo=True)
        w = int(np.argmax(top.forward(q)))
        return top.W[w, :C1_DIM], TP.render_code(top.W[w, :C1_DIM], l1shim, K1)

    for gtag, group in (("train", train_combos), ("held", TP.HELD_OUT)):
        tf = []
        for (d, m) in group:
            for t in range(T):
                _, f = emit(codes1[(d, m)][t - 1].ravel(),
                            codes2[(d, m)][t - 1].ravel(), d, m)
                tf.append(float(TP.cn(f) @ TP.cn(laps[(d, m)][t])))
        res[f"tf_{gtag}"] = round(float(np.mean(tf)), 3)

    strips, titles = [], []
    for mode in ("pixel", "code"):
        for gtag, group in (("train", train_combos), ("held", TP.HELD_OUT)):
            advs, mms = [], []
            for gi, (d, m) in enumerate(group):
                frames, cargo = [], None
                for t in range(FREE_STEPS):
                    if t == 0:
                        k1, k2 = np.zeros(C1_DIM), np.zeros(C2_DIM)
                    elif mode == "code":
                        c1map = np.maximum(cargo, 0.0).reshape(R.G1, R.G1, K1)
                        k1 = c1map.ravel()
                        k2 = JS.encode_l2(c1map[None], d2.W).ravel()
                    else:
                        prev2 = frames[-2] if t >= 2 else np.zeros_like(frames[-1])
                        st = np.stack([frames[-1], prev2], axis=-1)
                        c1m = TP.encode_front(st[None], d1.W)
                        k1 = c1m.ravel()
                        k2 = JS.encode_l2(c1m, d2.W).ravel()
                    cargo, f = emit(k1, k2, d, m)
                    frames.append(f)
                adv, mm = TP.phase_track(frames[T // 2:], laps[(d, m)])
                advs.append(adv)
                mms.append(mm)
                if gtag == "train" and gi == 0:
                    for i in range(0, T, 2):
                        strips.append(frames[i])
                        titles.append(f"{mode}" if i == 0 else None)
            res[f"free_{mode}_{gtag}_advance"] = round(float(np.mean(advs)), 3)
            res[f"free_{mode}_{gtag}_match"] = round(float(np.mean(mms)), 3)
        print(f"free-run {mode} done", flush=True)

    R.gallery(strips, titles, OUTPUT_DIR / "cotrain_runs.png",
              "Co-trained stack free runs (pixel loop / code loop)", 16)
    R.gallery([d1.W[i].reshape(8, 8, 2)[..., 0] for i in range(K1)], None,
              OUTPUT_DIR / "templates_L1_framehalf.png",
              "Co-trained joint L1, frame_t half", 16, cmap="bwr")
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
