"""Image in, tally counts strokes-and-objects against the teacher's commands.

    frames (48x48, 1 or 2 channels: the frame, and the frame minus the previous one)
    -> L1 strokes 9x9 x400, top-1, no pressure (trained without labels)
    -> contrast message pooled 4x4 (6400), standardised
    -> L2 object templates x400 (purity step, label + belief, hire on error)
    -> ONE tally over the joint command (left level, right level: 13x13 = 169),
       read two ways: the joint argmax, or the two marginals (one per motor).

Held-out episodes: how often the pupil names the teacher's level. Then the
real test: the pupil flies from the centre to fresh targets, frozen.

Usage:  uv run python train.py [--frames 1|2] [--k2 400] [--episodes-test 50]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cupy as cp
import box_world as B
import vrig as V

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
FRAMES, K1, K2, PS, GRID = arg("--frames", 1), 400, arg("--k2", 400), 9, 4
TRACK = "--track" in sys.argv
TAG = arg("--tag", f"f{FRAMES}" + ("_track" if TRACK else ""), str)
NTRAIN_L1, NTRAIN_L2, EP_TEST = 6000, 20000, arg("--episodes-test", 50)
NL = V.NLEV ** 2


def channels(F, ep, tick):
    """(N, C, 48, 48) float32 in [0,1]; channel 2 = frame minus previous frame of the episode."""
    F = F.astype(np.float32) / 255.0
    if FRAMES == 1:
        return F[:, None]
    prev = np.roll(F, 1, axis=0); prev[tick == 0] = F[tick == 0]
    # signed difference on a ZERO background: with a 0.5 offset the joint mean-centring of a
    # two-channel patch turned the whole difference channel into a constant block (see
    # templates_L1_2ch.png of the first run) and no motion survived into the templates.
    return np.stack([F, F - prev], 1)


def main():
    d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
    S, T, L, ep, tick = d["states"], d["targets"], d["levels"], d["ep"], d["tick"]
    y = L[:, 0] * V.NLEV + L[:, 1]
    X = channels(F, ep, tick).reshape(len(F), -1)
    n_ep = ep.max() + 1; test_eps = np.arange(n_ep) >= int(0.8 * n_ep)
    te = test_eps[ep]; tr = ~te
    rng = np.random.default_rng(0)
    i_l1 = rng.choice(np.where(tr)[0], NTRAIN_L1, replace=False)
    i_l2 = rng.choice(np.where(tr)[0], min(NTRAIN_L2, tr.sum()), replace=False)
    i_te = np.where(te)[0]
    rig = V.VRig(48, PS, FRAMES, GRID, K1)
    print(f"{len(F)} ticks, {n_ep} episodes ({te.sum()} test ticks), frames={FRAMES}, "
          f"L1 {K1}x{PS}x{PS}x{FRAMES}ch over {rig.npos} positions, L2 x{K2}, joint commands {NL}")

    t0 = time.time()
    W1 = V.train_l1(rig, cp.asarray(X[i_l1]), seed=7)
    print(f"L1 trained ({time.time()-t0:.0f}s)")
    Ctr = V.encode_contrast(rig, W1, cp.asarray(X[i_l2]))
    Cte = V.encode_contrast(rig, W1, cp.asarray(X[i_te]))
    mu, sd = Ctr.mean(0, keepdims=True), Ctr.std(0, keepdims=True) + 1e-6
    tmu = tsd = tw = None
    if TRACK:
        # the motion track, standardised and weighted so its block carries the same
        # expected squared norm as the image block, then joined before the L2 normalisation
        Ktr_np = V.encode_track(V.track_feats(S, ep, tick)[i_l2]); Kte_np = V.encode_track(V.track_feats(S, ep, tick)[i_te])
        Ktr, Kte = cp.asarray(Ktr_np), cp.asarray(Kte_np)
        tmu, tsd = Ktr.mean(0, keepdims=True), Ktr.std(0, keepdims=True) + 1e-6
        tw = float(np.sqrt(Ctr.shape[1] / Ktr.shape[1]))
        Ctr = cp.concatenate([(Ctr - mu) / sd, tw * (Ktr - tmu) / tsd], 1)
        Cte = cp.concatenate([(Cte - mu) / sd, tw * (Kte - tmu) / tsd], 1)
        Ctr /= cp.linalg.norm(Ctr, axis=1, keepdims=True) + V.EPS
        Cte /= cp.linalg.norm(Cte, axis=1, keepdims=True) + V.EPS
        print(f"track: {Ktr.shape[1]} numbers, block weight {tw:.2f}")
    else:
        Ctr, Cte = V.standardize(Ctr, mu, sd), V.standardize(Cte, mu, sd)
    print(f"encoded ({time.time()-t0:.0f}s), message {Ctr.shape[1]} numbers, "
          f"{float((Ctr != 0).sum(1).mean()):.0f} nonzero")
    ytr, yte = cp.asarray(y[i_l2]), cp.asarray(y[i_te])
    W2, N2, hires = V.train_l2(Ctr, ytr, K2, NL, seed=7)
    TJ, TL, TR, NLc, NRc = V.motor_tables(N2, K2)
    print(f"L2 trained ({time.time()-t0:.0f}s), hires {hires}, dead {int((N2.sum(1)==0).sum())}")

    # ---- offline: does the pupil name the teacher's levels on held-out episodes?
    Lte = L[i_te]
    for name in ("joint", "marginal", "mean"):
        pl, pr = V.read(W2, TJ, TL, TR, Cte, mode=name, NL=NLc, NR=NRc)
        pl, pr = cp.asnumpy(pl), cp.asnumpy(pr)
        exact = np.mean((pl == Lte[:, 0]) & (pr == Lte[:, 1]))
        within = np.mean((np.abs(pl - Lte[:, 0]) <= 1) & (np.abs(pr - Lte[:, 1]) <= 1))
        mae = np.mean(np.abs(pl - Lte[:, 0]) + np.abs(pr - Lte[:, 1])) / 2
        print(f"  {name + ' read':<14} exact pair {exact:.3f}   both within 1 level {within:.3f}   mean |level error| {mae:.2f}")

    # ---- save the pupil for the viewer
    cfg = {"kind": "module", "module": "vision_pupil", "dir": "experiments/2026_09_03/vision_drone",
           "frames": FRAMES, "k1": K1, "k2": K2, "ps": PS, "grid": GRID, "read": "mean", "diff": "signed",
           "track": TRACK, "track_w": tw}
    arrays = dict(W1=cp.asnumpy(W1), mu=cp.asnumpy(mu), sd=cp.asnumpy(sd), W2=cp.asnumpy(W2),
                  TJ=cp.asnumpy(TJ), TL=cp.asnumpy(TL), TR=cp.asnumpy(TR), NL=cp.asnumpy(NLc), NR=cp.asnumpy(NRc))
    if TRACK:
        arrays.update(tmu=cp.asnumpy(tmu), tsd=cp.asnumpy(tsd))
    for name in ("pupil", f"pupil_{TAG}"):
        np.savez(OUT / f"{name}.npz", **arrays); json.dump(cfg, open(OUT / f"{name}.json", "w"))

    # ---- closed loop: fly, frozen
    import vision_pupil as VP
    npz = np.load(OUT / "pupil.npz")
    for mode in ("mean", "marginal"):
        cfg["read"] = mode
        act = VP.load_policy(npz, cfg)
        rng = np.random.default_rng(123); ok, lens, dists = 0, [], []
        t1 = time.time()
        for e in range(EP_TEST):
            s, tgt = B.init(rng); VP.reset(act); hold = 0; lv = (V.NLEV // 2, V.NLEV // 2)
            for t in range(B.W.EP_CAP):
                lv = act(s, tgt, lv)
                s = B.W.physics(s, lv)
                s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
                hold = hold + 1 if B.W.at_goal(s, tgt) else 0
                if hold >= 10:
                    ok += 1; break
            lens.append(t + 1); dists.append(float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])))
        print(f"  FLIGHT {mode:<9} success {ok/EP_TEST:.2f}   median ticks {int(np.median(lens))}   "
              f"median final distance {np.median(dists):.2f}   (PID: 1.00, 109)   ({time.time()-t1:.0f}s)")
    cfg["read"] = "mean"
    for name in ("pupil", f"pupil_{TAG}"):
        json.dump(cfg, open(OUT / f"{name}.json", "w"))


if __name__ == "__main__":
    main()
