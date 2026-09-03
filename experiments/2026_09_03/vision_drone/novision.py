"""No vision: the target offset as bump codes, straight into the joint tally.

Channels: dx, dy (64 bins over +-6, 0.19 units per bin, finer than the 0.25
goal tolerance), vx, vy, tilt, tilt rate (32 bins), each at lags 0, 2, 5.
Every active bin votes with its row T[channel, bin, pair]; read = sum -> pair
mean per motor. Counts from the teacher's 400 flights. Then: offline score,
one probe flight, 20 full flights.

Usage:  uv run python novision.py [--dagger 4]
"""
import sys, json, time
import numpy as np, cupy as cp
from pathlib import Path
import box_world as B, vrig as V

OUT = Path(__file__).resolve().parent / "results"
NLEV = V.NLEV; NJ = NLEV * NLEV; ALPHA = 1.0
LAGS = (0, 2, 5)
CH = [("dx", -6, 6, 64), ("dy", -6, 6, 64), ("vx", -5, 5, 32), ("vy", -5, 5, 32), ("tilt", -1.5, 1.5, 32), ("rate", -6, 6, 32)]
ACTIVE = np.array([0.08, 0.40, 1.0, 0.40, 0.08], np.float32)
DAGGER = int(sys.argv[sys.argv.index("--dagger") + 1]) if "--dagger" in sys.argv else 0


def raw(s, tgt):
    return np.array([tgt[0] - s[0], tgt[1] - s[1], s[2], s[3], s[4], s[5]], np.float32)


def lagged(R, ep, tick):
    cols = []
    for lag in LAGS:
        src = np.arange(len(R)) - lag; src = np.where(tick >= lag, src, np.arange(len(R)) - tick)
        cols.append(R[src])
    return np.concatenate(cols, 1)                     # (n, 6 * len(LAGS))


def encode(feats):
    """(n, 6*lags) -> list of (n, NB_j) bump codes, concatenated."""
    n = len(feats); blocks = []
    for j in range(feats.shape[1]):
        _, lo, hi, nb = CH[j % 6]
        out = np.zeros((n, nb), np.float32)
        c = np.rint((np.clip(feats[:, j], lo, hi) - lo) / (hi - lo) * (nb - 1)).astype(int)
        for k, a in enumerate(ACTIVE):
            i = c + k - 2; ok = (i >= 0) & (i < nb)
            out[np.where(ok)[0], i[ok]] = np.maximum(out[np.where(ok)[0], i[ok]], a)
        blocks.append(out)
    return np.concatenate(blocks, 1)


D = sum(CH[j % 6][3] for j in range(6 * len(LAGS)))


def count(K, y):
    """N[feature, pair]: bump-weighted counts."""
    N = cp.zeros((D, NJ)); Kc = cp.asarray(K); yc = cp.asarray(y)
    for p in cp.unique(yc).tolist():
        N[:, p] = Kc[yc == p].sum(0)
    return N


def table(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * D)
    pm = (N.sum(1, keepdims=True) + ALPHA * NJ) / (N.sum() + ALPHA * D * NJ)
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def read(T, K):
    ev = cp.asarray(K) @ T
    p = cp.exp(ev - ev.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    P3 = p.reshape(len(K), NLEV, NLEV); lv = cp.arange(NLEV, dtype=cp.float32)
    return cp.asnumpy(cp.rint((P3.sum(2) * lv).sum(1))).astype(int), cp.asnumpy(cp.rint((P3.sum(1) * lv).sum(1))).astype(int)


def score(T, K, L):
    pl, pr = read(T, K)
    return (np.mean((np.abs(pl - L[:, 0]) <= 1) & (np.abs(pr - L[:, 1]) <= 1)),
            np.abs((pl + pr) - (L[:, 0] + L[:, 1])).mean() / 2, np.abs((pl - pr) - (L[:, 0] - L[:, 1])).mean() / 2)


class Pupil:
    def __init__(self, T): self.T = T; self.hist = []
    def reset(self): self.hist = []
    def __call__(self, s, tgt):
        self.hist.append(raw(s, tgt)); h = self.hist
        f = np.concatenate([h[max(0, len(h) - 1 - lag)] for lag in LAGS])[None]
        pl, pr = read(self.T, encode(f)); return int(pl[0]), int(pr[0])


def flights(pupil, pid, rng, n, beta=0.0, cap=B.W.EP_CAP, record=False, verbose=False):
    ok, lens, dists, rec = 0, [], [], []
    for e in range(n):
        s, tgt = B.init(rng); pupil.reset(); hold = 0
        for t in range(cap):
            teach = pid(s, tgt); lv = pupil(s, tgt)
            if verbose and t % 10 == 0:
                print(f"{t:>4}{s[0]:>7.2f}{s[1]:>7.2f}{s[2]:>6.2f}{s[3]:>6.2f}{s[4]:>6.2f}"
                      f"{np.hypot(s[0]-tgt[0], s[1]-tgt[1]):>6.2f} |  {teach[0]:>2} {teach[1]:>2}    {lv[0]:>2} {lv[1]:>2}")
            if record: rec.append((s.copy(), tgt.copy(), teach, e, t))
            if rng.random() < beta: lv = teach
            s = B.W.physics(s, lv); s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
            hold = hold + 1 if B.W.at_goal(s, tgt) else 0
            if hold >= 10: ok += 1; break
        lens.append(t + 1); dists.append(float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])))
    return (ok / n, int(np.median(lens)), float(np.median(dists))), rec


def main():
    d = np.load(OUT / "flight.npz"); S0, T0, L0, ep0, tick0 = d["states"], d["targets"], d["levels"], d["ep"], d["tick"]
    R = np.stack([raw(s, t) for s, t in zip(S0, T0)])
    K = encode(lagged(R, ep0, tick0)); y = L0[:, 0] * NLEV + L0[:, 1]
    te = ep0 >= int(0.8 * (ep0.max() + 1)); tr, i_te = ~te, np.where(te)[0]
    N = count(K[tr], y[tr]); T = table(N)
    w1, ec, ed = score(T, K[i_te], L0[i_te])
    print(f"no vision, {D} bump features -> joint tally: offline within-1 {w1:.3f}, |coll err| {ec:.2f}, |diff err| {ed:.2f}")
    pid = B.make_pid(); pupil = Pupil(T)
    print(f"{'t':>4}{'x':>7}{'y':>7}{'vx':>6}{'vy':>6}{'tilt':>6}{'dist':>6} | teacher  pupil")
    st, _ = flights(pupil, pid, np.random.default_rng(5), 1, cap=200, verbose=True)
    st, _ = flights(pupil, pid, np.random.default_rng(123), 20)
    print(f"FLIGHT no vision, 20 flights: success {st[0]:.2f}  median ticks {st[1]}  final dist {st[2]:.2f}   (PID 1.00, 109)", flush=True)
    for r in range(DAGGER):
        beta = [0.5, 0.25, 0.0, 0.0][r]
        stats, rec = flights(pupil, pid, np.random.default_rng(700 + r), 40, beta=beta, cap=150, record=True)
        S = np.array([x[0] for x in rec]); Tg = np.array([x[1] for x in rec]); L = np.array([x[2] for x in rec])
        E = np.array([x[3] for x in rec]); Kt = np.array([x[4] for x in rec])
        Kn = encode(lagged(np.stack([raw(a, b) for a, b in zip(S, Tg)]), E, Kt))
        N += count(Kn, L[:, 0] * NLEV + L[:, 1]); T = table(N); pupil.T = T
        w1, ec, ed = score(T, K[i_te], L0[i_te])
        st, _ = flights(pupil, pid, np.random.default_rng(123), 20)
        print(f"dagger round {r+1} (teacher share {beta}, +{len(rec)} ticks): offline within-1 {w1:.3f}; "
              f"20 flights: success {st[0]:.2f}  median ticks {st[1]}  final dist {st[2]:.2f}", flush=True)
    np.savez(OUT / "pupil_novision.npz", T=cp.asnumpy(T))


if __name__ == "__main__":
    main()
