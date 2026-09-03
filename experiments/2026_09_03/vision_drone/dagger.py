"""Teach on the pupil's own states. DAgger by counting.

Round r: fly the current pupil (the teacher takes a share BETA[r] of the ticks
so early rounds stay near the goal), and at EVERY visited state record what the
PID would have commanded. Add those ticks to the data. Strokes, message
statistics and track statistics stay fixed from the checkpoint; the object
layer and the tally are retrained on the union. Then fly the pure pupil.

Usage:  uv run python -u dagger.py [--rounds 3] [--fly 40] [--test 20]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cupy as cp
import box_world as B
import vrig as V
import vision_pupil as VP

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
ROUNDS, FLY, TEST = arg("--rounds", 4), arg("--fly", 40), arg("--test", 20)
BETA = [0.5, 0.25, 0.0, 0.0, 0.0, 0.0]
NL = V.NLEV ** 2


def features(npz, cfg, rig, X, S, ep, tick):
    W1, mu, sd = cp.asarray(npz["W1"]), cp.asarray(npz["mu"]), cp.asarray(npz["sd"])
    C = V.encode_contrast(rig, W1, cp.asarray(X))
    if cfg.get("track"):
        tmu, tsd, tw = cp.asarray(npz["tmu"]), cp.asarray(npz["tsd"]), float(cfg["track_w"])
        K = cp.asarray(V.encode_track(V.track_feats(S, ep, tick)))
        C = cp.concatenate([(C - mu) / sd, tw * (K - tmu) / tsd], 1)
        return C / (cp.linalg.norm(C, axis=1, keepdims=True) + V.EPS)
    return V.standardize(C, mu, sd)


def fly(act, pid, rng, episodes, beta, record):
    """Fly with the pupil (teacher on a share beta of ticks). If record, return every
    visited state with the teacher's label. Returns success rate, median ticks, median final dist."""
    ok, lens, dists = 0, [], []
    S, T, L, E, K = [], [], [], [], []
    for e in range(episodes):
        s, tgt = B.init(rng); VP.reset(act); hold = 0; lv = (V.NLEV // 2, V.NLEV // 2)
        for t in range(B.W.EP_CAP):
            teach = pid(s, tgt)
            lv_p = act(s, tgt, lv)
            lv = teach if rng.random() < beta else lv_p
            if record:
                S.append(s.copy()); T.append(tgt.copy()); L.append(teach); E.append(e); K.append(t)
            s = B.W.physics(s, lv)
            s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
            hold = hold + 1 if B.W.at_goal(s, tgt) else 0
            if hold >= 10:
                ok += 1; break
        lens.append(t + 1); dists.append(float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])))
    stats = (ok / episodes, int(np.median(lens)), float(np.median(dists)))
    if record:
        return stats, (np.array(S), np.array(T), np.array(L), np.array(E), np.array(K))
    return stats


def main():
    cfg = json.load(open(OUT / "pupil.json")); npz = dict(np.load(OUT / "pupil.npz"))
    C_ch = int(cfg["frames"]); assert C_ch == 1, "dagger.py assumes the single-frame + track pupil"
    rig = V.VRig(48, int(cfg["ps"]), 1, int(cfg["grid"]), int(cfg["k1"]))
    K2 = int(cfg["k2"])
    d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
    S0, T0, L0, ep0, tick0 = d["states"], d["targets"], d["levels"], d["ep"], d["tick"]
    rng = np.random.default_rng(0)
    base = rng.choice(len(F), 20000, replace=False)
    # the aggregate: (features, joint label), starting from 20k of the teacher's ticks.
    # track features need episode context: computed on the full arrays, then indexed.
    Fb = F[base].astype(np.float32).reshape(len(base), -1) / 255.0
    W1, mu, sd = cp.asarray(npz["W1"]), cp.asarray(npz["mu"]), cp.asarray(npz["sd"])
    tmu, tsd, tw = cp.asarray(npz["tmu"]), cp.asarray(npz["tsd"]), float(cfg["track_w"])
    Kall = V.encode_track(V.track_feats(S0, ep0, tick0))
    Cimg = V.encode_contrast(rig, W1, cp.asarray(Fb))
    Xagg = cp.concatenate([(Cimg - mu) / sd, tw * (cp.asarray(Kall[base]) - tmu) / tsd], 1)
    Xagg /= cp.linalg.norm(Xagg, axis=1, keepdims=True) + V.EPS
    yagg = cp.asarray(L0[base, 0] * V.NLEV + L0[base, 1])
    pid = B.make_pid()
    print(f"start: {len(yagg)} teacher ticks. rounds {ROUNDS}, fly {FLY}/round, test {TEST}, teacher share {BETA[:ROUNDS]}", flush=True)
    act = VP.load_policy(np.load(OUT / "pupil.npz"), cfg)
    # held-out teacher episodes, for the offline score per round (no flights until the end)
    te = ep0 >= int(0.8 * (ep0.max() + 1)); i_te = np.where(te)[0][::4]
    Fte = F[i_te].astype(np.float32).reshape(len(i_te), -1) / 255.0
    Cte = cp.concatenate([(V.encode_contrast(rig, W1, cp.asarray(Fte)) - mu) / sd,
                          tw * (cp.asarray(Kall[i_te]) - tmu) / tsd], 1)
    Cte /= cp.linalg.norm(Cte, axis=1, keepdims=True) + V.EPS
    Lte = L0[i_te]

    def offline(npz):
        W2, TJ, TL, TR = (cp.asarray(npz[k]) for k in ("W2", "TJ", "TL", "TR"))
        NLc, NRc = cp.asarray(npz["NL"]), cp.asarray(npz["NR"])
        pl, pr = V.read(W2, TJ, TL, TR, Cte, mode="mean", NL=NLc, NR=NRc)
        pl, pr = cp.asnumpy(pl), cp.asnumpy(pr)
        return (np.mean((np.abs(pl - Lte[:, 0]) <= 1) & (np.abs(pr - Lte[:, 1]) <= 1)),
                np.mean(np.abs(pl - Lte[:, 0]) + np.abs(pr - Lte[:, 1])) / 2)
    w1, e1 = offline(npz)
    print(f"round 0  offline (teacher's held-out states): within 1 level {w1:.3f}, mean |err| {e1:.2f}", flush=True)
    for r in range(ROUNDS):
        t0 = time.time()
        stats, (S, T, L, E, Kt) = fly(act, pid, np.random.default_rng(1000 + r), FLY, BETA[r], True)
        Fn = np.stack([B.render(s, t) for s, t in zip(S, T)]).reshape(len(S), -1).astype(np.float32)
        Kn = V.encode_track(V.track_feats(S, E, Kt))
        Cn = V.encode_contrast(rig, W1, cp.asarray(Fn))
        Xn = cp.concatenate([(Cn - mu) / sd, tw * (cp.asarray(Kn) - tmu) / tsd], 1)
        Xn /= cp.linalg.norm(Xn, axis=1, keepdims=True) + V.EPS
        Xagg = cp.concatenate([Xagg, Xn]); yagg = cp.concatenate([yagg, cp.asarray(L[:, 0] * V.NLEV + L[:, 1])])
        W2, N2, hires = V.train_l2(Xagg, yagg, K2, NL, seed=7, hire_free_only=True, tol=1)
        TJ, TL, TR, NLc, NRc = V.motor_tables(N2, K2)
        npz.update(W2=cp.asnumpy(W2), TJ=cp.asnumpy(TJ), TL=cp.asnumpy(TL), TR=cp.asnumpy(TR),
                   NL=cp.asnumpy(NLc), NR=cp.asnumpy(NRc))
        np.savez(OUT / "pupil.npz", **npz); np.savez(OUT / "pupil_dagger.npz", **npz)
        json.dump(cfg, open(OUT / "pupil_dagger.json", "w"))
        act = VP.load_policy(np.load(OUT / "pupil.npz"), cfg)
        w1, e1 = offline(npz)
        print(f"round {r+1}  flew {FLY} with teacher share {BETA[r]:.2f} (success {stats[0]:.2f}, "
              f"median {stats[1]} ticks), +{len(S)} ticks -> {len(yagg)} total; retrained (hires {hires}, "
              f"dead {int((N2.sum(1) == 0).sum())}); offline within 1 level {w1:.3f}, mean |err| {e1:.2f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
    st = fly(act, pid, np.random.default_rng(123), TEST, 0.0, False)
    print(f"FINAL pure pupil, {TEST} flights: success {st[0]:.2f}  median ticks {st[1]}  final dist {st[2]:.2f}", flush=True)


if __name__ == "__main__":
    main()
