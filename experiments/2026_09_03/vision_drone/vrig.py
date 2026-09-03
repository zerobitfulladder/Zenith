"""The vision rig for the drone: strokes over 48x48 frames, contrast message,
an object layer read by a tally of motor commands. Shared by train.py and
vision_pupil.py so the viewer runs exactly what was trained."""
import numpy as np
import cupy as cp
import cupyx

EPS, ALPHA, FLOOR, ETA_MIN = 1e-12, 1.0, 0.05, 0.02
NLEV = 13


class VRig:
    """Patches of PS x PS over C channels of a SIDE x SIDE frame, pooled into GRID x GRID cells."""
    def __init__(self, side=48, ps=9, chans=1, grid=4, K=400):
        self.side, self.ps, self.C, self.K = side, ps, chans, K
        self.s = side - ps + 1; self.npos = self.s ** 2; self.dim = chans * ps * ps
        a, b = np.arange(self.s), np.arange(ps)
        pix = (((a[:, None, None, None] + b[None, None, :, None]) * side
                + (a[None, :, None, None] + b[None, None, None, :]))).reshape(self.npos, ps * ps)
        self.pidx = cp.asarray(np.concatenate([pix + c * side * side for c in range(chans)], axis=1))
        self.grid, self.gg = grid, grid * grid
        i = np.arange(self.npos)
        self.cells = cp.asarray(((i // self.s) * grid // self.s) * grid + ((i % self.s) * grid // self.s))
        self.min_s = 4

    def patches(self, X):
        P = X[:, self.pidx]
        Cc = P - P.mean(-1, keepdims=True)
        n = cp.linalg.norm(Cc, axis=-1)
        return Cc / cp.maximum(n, EPS)[..., None], n > FLOOR


def train_l1(rig, X, seed, epochs=3, batch=32):
    rng = np.random.default_rng(seed)
    W = cp.asarray(rng.standard_normal((rig.K, rig.dim)), cp.float32)
    W -= W.mean(1, keepdims=True); W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    n = cp.zeros(rig.K)
    for ep in range(epochs):
        order = np.arange(len(X)); rng.shuffle(order)
        for s in range(0, len(order), batch):
            ids = cp.asarray(order[s:s + batch])
            Q, keep = rig.patches(X[ids]); V = Q.reshape(-1, rig.dim); k = keep.reshape(-1)
            win = ((V @ W.T) ** 2).argmax(1)
            wk, Vk = win[k], V[k]
            cnt = cp.bincount(wk, minlength=rig.K); live = cnt >= rig.min_s
            if int(live.sum()):
                sums = cp.zeros((rig.K, rig.dim), cp.float32); cupyx.scatter_add(sums, wk, Vk)
                n[live] += cnt[live]
                eta = cp.clip(cnt[live] / n[live], ETA_MIN, 1.0)[:, None]
                W[live] += eta * (sums[live] / cnt[live, None] - W[live])
                W /= cp.linalg.norm(W, axis=1, keepdims=True) + EPS
    return W


def encode_contrast(rig, W, X, chunk=64):
    """SoftHebb's message: match minus mean over templates, rectified, max-pooled into cells."""
    n = len(X); C = cp.zeros((n, rig.K, rig.gg), cp.float32)
    for a in range(0, n, chunk):
        Q, keep = rig.patches(X[a:a + chunk]); m = len(Q)
        S = (Q.reshape(-1, rig.dim) @ W.T) ** 2
        Sc = cp.maximum(S - S.mean(1, keepdims=True), 0.0) * keep.reshape(-1)[:, None]
        Sc = Sc.reshape(m, rig.npos, rig.K)
        for c in range(rig.gg):
            C[a:a + m, :, c] = Sc[:, rig.cells == c, :].max(1)
    return C.reshape(n, -1)


def standardize(C, mu, sd):
    Z = (C - mu) / sd
    return Z / (cp.linalg.norm(Z, axis=1, keepdims=True) + EPS)


def table(N, K):
    NL = N.shape[1]
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = (N.sum(1, keepdims=True) + ALPHA * NL) / (N.sum() + ALPHA * K * NL)
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def purity(N):
    p = (N + ALPHA) / (N.sum(1, keepdims=True) + ALPHA * N.shape[1])
    return p.max(1)


def belief(T, w0):
    ev = T[w0]
    z = (ev - ev.mean(1, keepdims=True)) / (ev.std(1, keepdims=True) + 1e-9)
    q = cp.exp(z - z.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def train_l2(Ctr, ytr, K2, NL, seed, epochs=10, batch=512, beta_l=0.25, beta_b=0.5, hire=True,
             hire_free_only=False, tol=0):
    """Object layer over the standardised contrast message: purity step, label +
    belief in the competition, hire on error. Returns W2 and the counts N2 (K2, NL).
    hire_free_only: never wipe a live row, hire only into templates that have never won.
    tol: a read counts as wrong only if either motor's level is off by more than tol
    (labels are joint pairs l*13+r when tol > 0)."""
    rng = np.random.default_rng(seed)
    D = Ctr.shape[1]
    W2 = cp.asarray(rng.standard_normal((K2, D)), cp.float32)
    W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    N = cp.zeros((K2, NL)); hires = 0
    for ep in range(epochs):
        order = np.arange(len(Ctr)); rng.shuffle(order)
        for s in range(0, len(order), batch):
            ids = cp.asarray(order[s:s + batch]); Cb, yb = Ctr[ids], ytr[ids]
            T = table(N, K2)
            S = Cb @ W2.T; err = 1.0 - S
            w0 = err.argmin(1); q = belief(T, w0)
            score = err - beta_l * T[:, yb].T - beta_b * (q @ T.T)
            win = score.argmin(1)
            if hire:
                pur = purity(N); pred = q.argmax(1)
                if tol > 0:
                    off = cp.maximum(cp.abs(pred // NLEV - yb // NLEV), cp.abs(pred % NLEV - yb % NLEV))
                    wrong = (off > tol) & (pur[win] >= 0.5)
                else:
                    wrong = (pred != yb) & (pur[win] >= 0.5)
                if hire_free_only:
                    free = N.sum(1) == 0
                    if not bool(free.any()):
                        wrong[:] = False
                if int(wrong.sum()):
                    tot = float(N.sum())
                    imp = cp.zeros(K2) if tot <= 0 else ((N / tot) * T).sum(1)
                    if hire_free_only:
                        imp = cp.where(free, imp, cp.inf)
                    imp[win] = cp.inf
                    for yy in cp.unique(yb[wrong]).tolist():
                        grp = wrong & (yb == yy); t = int(imp.argmin())
                        if not bool(cp.isfinite(imp[t])):
                            break                                   # no free template left
                        imp[t] = cp.inf; win[grp] = t; N[t] = 0; hires += 1
            eta = ((1.0 - purity(N)) / (1.0 - 1.0 / NL)).astype(cp.float32)
            N += cp.bincount(win * NL + yb, minlength=K2 * NL).reshape(K2, NL)
            cnt = cp.bincount(win, minlength=K2); live = cnt > 0
            sums = cp.zeros((K2, D), cp.float32); cupyx.scatter_add(sums, win, Cb)
            W2[live] += eta[live][:, None] * (sums[live] / cnt[live, None] - W2[live])
            W2 /= cp.linalg.norm(W2, axis=1, keepdims=True) + EPS
    return W2, N, hires


def motor_tables(N2, K2):
    """Joint tally (K2, 13*13) and the two marginals (K2, 13) as log-ratio tables,
    plus the marginal COUNTS (for the mean read)."""
    N3 = N2.reshape(K2, NLEV, NLEV)
    return table(N2, K2), table(N3.sum(2), K2), table(N3.sum(1), K2), N3.sum(2), N3.sum(1)


def read(W2, TJ, TL, TR, C, mode="marginal", NL=None, NR=None):
    """joint: argmax of the joint row. marginal: argmax per motor. mean: the
    count-weighted mean level per motor, rounded -- a smooth read."""
    w0 = (C @ W2.T).argmax(1)
    if mode == "joint":
        p = TJ[w0].argmax(1)
        return p // NLEV, p % NLEV
    if mode == "mean":
        lv = cp.arange(NLEV, dtype=cp.float32)
        pl = (NL[w0] + ALPHA); pr = (NR[w0] + ALPHA)
        ml = (pl * lv).sum(1) / pl.sum(1); mr = (pr * lv).sum(1) / pr.sum(1)
        return cp.rint(ml).astype(cp.int64), cp.rint(mr).astype(cp.int64)
    return TL[w0].argmax(1), TR[w0].argmax(1)


# ------------------------------------------------------------ the motion track ---
# vx, vy, tilt, tilt rate, each at the present and at lags 2 and 5 ticks: a bump of
# ACTIVE adjacent cells over NB cells per channel, as in the single-layer rig.
# 4 channels x 3 lags x 32 cells = 384 numbers; no pixels.
TRACK_LAGS = (0, 2, 5)
TRACK_NB = 32
TRACK_RANGES = ((-3.0, 3.0), (-3.0, 3.0), (-1.2, 1.2), (-5.0, 5.0))     # vx, vy, tilt, rate
ACTIVE = np.array([0.08, 0.40, 1.0, 0.40, 0.08], np.float32)


def track_dim():
    return 4 * len(TRACK_LAGS) * TRACK_NB


def encode_track(feats):
    """feats: (n, 4 * len(TRACK_LAGS)) raw values -> (n, track_dim()) bump code."""
    n = len(feats); out = np.zeros((n, 4 * len(TRACK_LAGS), TRACK_NB), np.float32)
    for j in range(4 * len(TRACK_LAGS)):
        lo, hi = TRACK_RANGES[j % 4]
        pos = (np.clip(feats[:, j], lo, hi) - lo) / (hi - lo) * (TRACK_NB - 1)
        c = np.rint(pos).astype(int)
        for k, a in enumerate(ACTIVE):
            idx = c + k - len(ACTIVE) // 2
            ok = (idx >= 0) & (idx < TRACK_NB)
            out[np.where(ok)[0], j, idx[ok]] = np.maximum(out[np.where(ok)[0], j, idx[ok]], a)
    return out.reshape(n, -1)


def track_feats(states, ep, tick):
    """Lagged features from the stored flight, episode-aware (lag clamped at the episode start)."""
    n = len(states); cols = []
    for lag in TRACK_LAGS:
        src = np.arange(n) - lag
        src = np.where(tick >= lag, src, np.arange(n) - tick)          # clamp to the episode's first tick
        cols.append(states[src][:, [2, 3, 4, 5]])
    return np.concatenate(cols, 1)
