"""DAgger by adding counts. No retraining: the pupil flies short episodes, the
teacher labels every state it visits, the two tallies gain those counts.

Usage:  uv run python -u l1_dagger.py [--rounds 6] [--fly 40] [--cap 150] [--test 20]
"""
import sys, json, time
import numpy as np, cupy as cp
from pathlib import Path
import box_world as B, vrig as V, vision_pupil as VP
import l1_tally as T

OUT = Path(__file__).resolve().parent / "results"
def arg(f, d, c=int): return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
ROUNDS, FLY, CAP, TEST = arg("--rounds", 6), arg("--fly", 40), arg("--cap", 150), arg("--test", 20)
JOINT = "--joint" in sys.argv
NLEV = V.NLEV; NJ = NLEV * NLEV
if JOINT:
    T.NLEV = NJ
if "--wide" in sys.argv:
    V.TRACK_RANGES = ((-5.0, 5.0), (-5.0, 5.0), (-1.5, 1.5), (-6.0, 6.0))
BETA = [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def main():
    cfg = json.load(open(OUT / ("pupil_l1joint.json" if JOINT else "pupil_l1.json"))); T.TRACKW = float(cfg["trackw"])
    NAME = "pupil_l1joint" if JOINT else "pupil_l1"
    npz = np.load(OUT / "pupil_f1_track.npz")
    rig = V.VRig(48, 9, 1, int(cfg["grid"]), 400); W1 = cp.asarray(npz["W1"])
    d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
    S0, L0, ep0, tick0 = d["states"], d["levels"], d["ep"], d["tick"]
    Kall = V.encode_track(V.track_feats(S0, ep0, tick0))
    te = ep0 >= int(0.8 * (ep0.max() + 1)); tr = np.where(~te)[0]; i_te = np.where(te)[0][::2]
    X = F.astype(np.float32).reshape(len(F), -1) / 255.0
    idx, keep = T.winners(rig, W1, X)
    N = {}
    if JOINT:
        N["J"] = list(T.counts(rig, idx[tr], keep[tr], Kall[tr], L0[tr, 0] * NLEV + L0[tr, 1]))
    else:
        for name, col in (("L", 0), ("R", 1)):
            N[name] = list(T.counts(rig, idx[tr], keep[tr], Kall[tr], L0[tr, col]))
    img_votes, trk_votes = float(keep[tr].sum(1).mean()), float(Kall[tr].sum(1).mean())
    pid = B.make_pid()

    def save():
        if JOINT:
            arrays = dict(W1=npz["W1"], Ti=cp.asnumpy(T.table3(N["J"][0])), Tt=cp.asnumpy(T.table3(N["J"][1])),
                          img_votes=img_votes, trk_votes=trk_votes)
        else:
            arrays = dict(W1=npz["W1"], TiL=cp.asnumpy(T.table3(N["L"][0])), TtL=cp.asnumpy(T.table3(N["L"][1])),
                          TiR=cp.asnumpy(T.table3(N["R"][0])), TtR=cp.asnumpy(T.table3(N["R"][1])), img_votes=img_votes, trk_votes=trk_votes)
        for nm in (NAME, "pupil"):
            np.savez(OUT / f"{nm}.npz", **arrays); json.dump(cfg, open(OUT / f"{nm}.json", "w"))
        return VP.load_policy(np.load(OUT / f"{NAME}.npz"), cfg)

    def offline():
        Lte = L0[i_te]; out = []
        if JOINT:
            ev = T.evidence(rig, idx[i_te], keep[i_te], Kall[i_te], T.table3(N["J"][0]), T.table3(N["J"][1]))
            p = cp.exp(ev - ev.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            P3 = p.reshape(len(i_te), NLEV, NLEV); lv = cp.arange(NLEV, dtype=cp.float32)
            out = [cp.asnumpy(cp.rint((P3.sum(2) * lv).sum(1))).astype(int), cp.asnumpy(cp.rint((P3.sum(1) * lv).sum(1))).astype(int)]
        else:
            for name, col in (("L", 0), ("R", 1)):
                ev = T.evidence(rig, idx[i_te], keep[i_te], Kall[i_te], T.table3(N[name][0]), T.table3(N[name][1]))
                out.append(cp.asnumpy(T.read(ev)))
        pl, pr = out
        return (np.mean((np.abs(pl - Lte[:, 0]) <= 1) & (np.abs(pr - Lte[:, 1]) <= 1)),
                np.abs((pl + pr) - (Lte[:, 0] + Lte[:, 1])).mean() / 2, np.abs((pl - pr) - (Lte[:, 0] - Lte[:, 1])).mean() / 2)

    act = save(); w1, ec, ed = offline()
    print(f"round 0: {len(tr)} teacher ticks; offline within-1 {w1:.3f}, |coll err| {ec:.2f}, |diff err| {ed:.2f}", flush=True)
    for r in range(ROUNDS):
        t0 = time.time(); rng = np.random.default_rng(500 + r)
        S, Tg, L, E, K, ok = [], [], [], [], [], 0
        for e in range(FLY):
            s, tgt = B.init(rng); VP.reset(act); hold = 0; lv = (6, 6)
            for t in range(CAP):
                teach = pid(s, tgt); lv_p = act(s, tgt, lv)
                lv = teach if rng.random() < BETA[r] else lv_p
                S.append(s.copy()); Tg.append(tgt.copy()); L.append(teach); E.append(e); K.append(t)
                s = B.W.physics(s, lv); s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
                hold = hold + 1 if B.W.at_goal(s, tgt) else 0
                if hold >= 10:
                    ok += 1; break
        S, Tg, L, E, K = map(np.array, (S, Tg, L, E, K))
        Fn = np.stack([B.render(a, b) for a, b in zip(S, Tg)]).reshape(len(S), -1).astype(np.float32)
        Kn = V.encode_track(V.track_feats(S, E, K))
        idn, kpn = T.winners(rig, W1, Fn)
        if JOINT:
            Ni, Nt = T.counts(rig, idn, kpn, Kn, L[:, 0] * NLEV + L[:, 1]); N["J"][0] += Ni; N["J"][1] += Nt
        else:
            for name, col in (("L", 0), ("R", 1)):
                Ni, Nt = T.counts(rig, idn, kpn, Kn, L[:, col]); N[name][0] += Ni; N[name][1] += Nt
        act = save(); w1, ec, ed = offline()
        print(f"round {r+1}: flew {FLY}x{CAP} ticks, teacher share {BETA[r]:.2f}, success {ok/FLY:.2f}, +{len(S)} ticks; "
              f"offline within-1 {w1:.3f}, |coll err| {ec:.2f}, |diff err| {ed:.2f}   ({time.time()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(123); ok, lens, dists = 0, [], []
    for e in range(TEST):
        s, tgt = B.init(rng); VP.reset(act); hold = 0; lv = (6, 6)
        for t in range(B.W.EP_CAP):
            lv = act(s, tgt, lv); s = B.W.physics(s, lv)
            s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
            hold = hold + 1 if B.W.at_goal(s, tgt) else 0
            if hold >= 10:
                ok += 1; break
        lens.append(t + 1); dists.append(float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])))
    print(f"FINAL pure pupil, {TEST} flights: success {ok/TEST:.2f}  median ticks {int(np.median(lens))}  final dist {np.median(dists):.2f}   (PID 1.00, 109)", flush=True)


if __name__ == "__main__":
    main()
