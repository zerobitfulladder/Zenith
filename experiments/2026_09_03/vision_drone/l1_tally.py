"""No object layer. Strokes -> two per-cell tallies, one per motor.

Every position's winning stroke votes with its row T[stroke, cell, level] (6x6
cells over the 40x40 positions, the MNIST champion's read), and every active
bin of the motion track votes with its own row T[channel, bin, level], weighted
by its bump value. Two tables, LEFT and RIGHT. Read = sum of all votes ->
count-weighted mean level (smooth) or argmax. Additive evidence: "target left"
and "tilting fast" are separate votes that add, which is the shape of a PID.

Usage:  uv run python l1_tally.py [--grid 6] [--trackw 1.0]
"""
import sys, json, time
import numpy as np, cupy as cp
from pathlib import Path
import box_world as B, vrig as V

OUT = Path(__file__).resolve().parent / "results"
def arg(f, d, c=int): return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
GRID, TRACKW = arg("--grid", 6), arg("--trackw", 1.0, float)
NLEV, ALPHA = V.NLEV, 1.0
NCH, NB = 4 * len(V.TRACK_LAGS), V.TRACK_NB


def table3(N):
    """log P(id | cell, level) - log P(id | cell), per cell; N (K, cells, NL)."""
    K = N.shape[0]
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = (N.sum(2, keepdims=True) + ALPHA * NLEV) / (N.sum((0, 2), keepdims=True) + ALPHA * K * NLEV)
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def winners(rig, W1, X, chunk=64):
    idx = cp.zeros((len(X), rig.npos), cp.int32); keep = cp.zeros((len(X), rig.npos), bool)
    for a in range(0, len(X), chunk):
        Q, k = rig.patches(cp.asarray(X[a:a + chunk])); m = len(Q)
        idx[a:a + m] = ((Q.reshape(-1, rig.dim) @ W1.T) ** 2).argmax(1).reshape(m, -1); keep[a:a + m] = k
    return idx, keep


def counts(rig, idx, keep, Kt, lev):
    """Image counts N[stroke, cell, level] and track counts N[chan, bin, level] (bump-weighted)."""
    n = len(idx); C = cp.tile(rig.cells, n).reshape(n, rig.npos)
    lab = cp.repeat(cp.asarray(lev), rig.npos).reshape(n, rig.npos)
    flat = (idx[keep] * rig.gg + C[keep]) * NLEV + lab[keep]
    Ni = cp.bincount(flat, minlength=rig.K * rig.gg * NLEV).reshape(rig.K, rig.gg, NLEV).astype(cp.float64)
    Kt = cp.asarray(Kt).reshape(n, NCH, NB)                     # (n, chan, bin) bump values
    Nt = cp.zeros((NB, NCH, NLEV))
    for l in range(NLEV):
        m = cp.asarray(lev == l)
        Nt[:, :, l] = Kt[m].sum(0).T                              # (bin, chan)
    return Ni, Nt


def evidence(rig, idx, keep, Kt, Ti, Tt, wi=1.0, wt=None, chunk=128):
    wt = TRACKW if wt is None else wt
    n = len(idx); Kt = cp.asarray(Kt).reshape(n, NCH, NB)
    ev = cp.zeros((n, Ti.shape[2]), cp.float32)
    for a in range(0, n, chunk):
        m = min(chunk, n - a); C = cp.tile(rig.cells, m).reshape(m, rig.npos)
        ev[a:a + m] = (Ti[idx[a:a + m], C] * keep[a:a + m][..., None]).sum(1)
    evt = cp.einsum('ncb,bcl->nl', Kt, Tt)                                       # bump-weighted rows
    return wi * ev + wt * evt * (float(keep.sum(1).mean()) / float(Kt.sum((1, 2)).mean()))


def read(ev, mode="mean"):
    if mode == "argmax":
        return ev.argmax(1)
    z = ev - ev.max(1, keepdims=True); p = cp.exp(z); p /= p.sum(1, keepdims=True)
    return cp.rint((p * cp.arange(NLEV)).sum(1)).astype(cp.int64)


def main():
    npz = np.load(OUT / "pupil_f1_track.npz")
    rig = V.VRig(48, 9, 1, GRID, 400); W1 = cp.asarray(npz["W1"])
    d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
    S0, L0, ep0, tick0 = d["states"], d["levels"], d["ep"], d["tick"]
    Kall = V.encode_track(V.track_feats(S0, ep0, tick0))
    te = ep0 >= int(0.8 * (ep0.max() + 1)); tr = np.where(~te)[0]; i_te = np.where(te)[0][::2]
    t0 = time.time()
    X = F.astype(np.float32).reshape(len(F), -1) / 255.0
    idx, keep = winners(rig, W1, X)
    print(f"{len(tr)} train ticks, {len(i_te)} test ticks, {GRID}x{GRID} cells, track weight {TRACKW}   (coded in {time.time()-t0:.0f}s)")
    tabs = {}
    for name, col in (("L", 0), ("R", 1)):
        Ni, Nt = counts(rig, idx[tr], keep[tr], Kall[tr], L0[tr, col])
        tabs[name] = (table3(Ni), table3(Nt), Ni, Nt)
    Lte = L0[i_te]
    for mode in ("argmax", "mean"):
        pl = cp.asnumpy(read(evidence(rig, idx[i_te], keep[i_te], Kall[i_te], tabs["L"][0], tabs["L"][1]), mode))
        pr = cp.asnumpy(read(evidence(rig, idx[i_te], keep[i_te], Kall[i_te], tabs["R"][0], tabs["R"][1]), mode))
        w1 = np.mean((np.abs(pl - Lte[:, 0]) <= 1) & (np.abs(pr - Lte[:, 1]) <= 1))
        ec = np.abs((pl + pr) - (Lte[:, 0] + Lte[:, 1])).mean() / 2; ed = np.abs((pl - pr) - (Lte[:, 0] - Lte[:, 1])).mean() / 2
        print(f"  {mode:<7} read: within 1 level (both) {w1:.3f}   |collective err| {ec:.2f}   |differential err| {ed:.2f}")
    # image-only and track-only, mean read, to see who carries what
    for tag, wi, wt in (("image only", 1.0, 0.0), ("track only", 0.0, 1.0)):
        pl = cp.asnumpy(read(evidence(rig, idx[i_te], keep[i_te], Kall[i_te], tabs["L"][0], tabs["L"][1], wi, wt)))
        pr = cp.asnumpy(read(evidence(rig, idx[i_te], keep[i_te], Kall[i_te], tabs["R"][0], tabs["R"][1], wi, wt)))
        ec = np.abs((pl + pr) - (Lte[:, 0] + Lte[:, 1])).mean() / 2; ed = np.abs((pl - pr) - (Lte[:, 0] - Lte[:, 1])).mean() / 2
        print(f"  {tag:<11} mean read: |collective err| {ec:.2f}   |differential err| {ed:.2f}")
    np.savez(OUT / "pupil_l1.npz", W1=npz["W1"], TiL=cp.asnumpy(tabs["L"][0]), TtL=cp.asnumpy(tabs["L"][1]),
             TiR=cp.asnumpy(tabs["R"][0]), TtR=cp.asnumpy(tabs["R"][1]),
             img_votes=float(keep[tr].sum(1).mean()), trk_votes=float(Kall[tr].sum(1).mean()))
    json.dump({"kind": "module", "module": "vision_pupil", "dir": "experiments/2026_09_03/vision_drone",
               "frames": 1, "k1": 400, "ps": 9, "grid": GRID, "read": "l1", "track": True, "trackw": TRACKW},
              open(OUT / "pupil_l1.json", "w"))
    print("saved pupil_l1")


if __name__ == "__main__":
    main()
