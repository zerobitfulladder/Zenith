"""Same votes as l1_tally.py, ONE tally over the command pair (13x13 = 169 columns).
Read: sum the rows over positions and track bins, then the best pair, or the
count-weighted mean per motor from the pair distribution."""
import sys, json, time
import numpy as np, cupy as cp
from pathlib import Path
import box_world as B, vrig as V, vision_pupil as VP
import l1_tally as T

OUT = Path(__file__).resolve().parent / "results"
NLEV = V.NLEV; NJ = NLEV * NLEV
T.NLEV = NJ                                   # tables over the joint pair
GRID = int(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else 6
T.TRACKW = float(sys.argv[sys.argv.index("--trackw") + 1]) if "--trackw" in sys.argv else 1.0
TAG = "pupil_l1joint" + (f"_g{GRID}" if GRID != 6 else "") + (f"_tw{T.TRACKW:g}" if T.TRACKW != 1.0 else "")


def main():
    npz = np.load(OUT / "pupil_f1_track.npz")
    rig = V.VRig(48, 9, 1, GRID, 400); W1 = cp.asarray(npz["W1"])
    d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
    S0, L0, ep0, tick0 = d["states"], d["levels"], d["ep"], d["tick"]
    y = L0[:, 0] * NLEV + L0[:, 1]
    Kall = V.encode_track(V.track_feats(S0, ep0, tick0))
    te = ep0 >= int(0.8 * (ep0.max() + 1)); tr = np.where(~te)[0]; i_te = np.where(te)[0][::2]
    X = F.astype(np.float32).reshape(len(F), -1) / 255.0
    idx, keep = T.winners(rig, W1, X)
    Ni, Nt = T.counts(rig, idx[tr], keep[tr], Kall[tr], y[tr])
    Ti, Tt = T.table3(Ni), T.table3(Nt)
    ev = T.evidence(rig, idx[i_te], keep[i_te], Kall[i_te], Ti, Tt)             # (n, 169)
    Lte = L0[i_te]
    p = cp.exp(ev - ev.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    P3 = p.reshape(len(i_te), NLEV, NLEV)
    lv = cp.arange(NLEV, dtype=cp.float32)
    reads = {"best pair": (ev.argmax(1) // NLEV, ev.argmax(1) % NLEV),
             "pair mean": (cp.rint((P3.sum(2) * lv).sum(1)), cp.rint((P3.sum(1) * lv).sum(1)))}
    for name, (pl, pr) in reads.items():
        pl, pr = cp.asnumpy(pl).astype(int), cp.asnumpy(pr).astype(int)
        w1 = np.mean((np.abs(pl - Lte[:, 0]) <= 1) & (np.abs(pr - Lte[:, 1]) <= 1))
        ec = np.abs((pl + pr) - (Lte[:, 0] + Lte[:, 1])).mean() / 2; ed = np.abs((pl - pr) - (Lte[:, 0] - Lte[:, 1])).mean() / 2
        print(f"  single joint tally, {name:<9}: within 1 level (both) {w1:.3f}   |collective err| {ec:.2f}   |differential err| {ed:.2f}")
    np.savez(OUT / f"{TAG}.npz", W1=npz["W1"], Ti=cp.asnumpy(Ti), Tt=cp.asnumpy(Tt),
             img_votes=float(keep[tr].sum(1).mean()), trk_votes=float(Kall[tr].sum(1).mean()))
    json.dump({"kind": "module", "module": "vision_pupil", "dir": "experiments/2026_09_03/vision_drone",
               "frames": 1, "k1": 400, "ps": 9, "grid": GRID, "read": "l1joint", "track": True, "trackw": T.TRACKW},
              open(OUT / f"{TAG}.json", "w"))
    print(f"saved {TAG}  (grid {GRID}x{GRID})")


if __name__ == "__main__":
    main()
