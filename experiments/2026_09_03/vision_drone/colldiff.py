"""Collective vs differential. (1) On held-out teacher states, the current pupil's
error split into collective (l+r) and differential (l-r). (2) Retrain the object
layer with a finer motion partition (K2, track weight x TWX) and a tally over
(collective bin, differential bin), read as two means; same split of the error;
then the single probe flight. Usage: uv run python colldiff.py [--k2 1600] [--twx 3]"""
import sys, json, time
import numpy as np, cupy as cp
from pathlib import Path
import box_world as B, vrig as V, vision_pupil as VP
OUT = Path(__file__).resolve().parent / "results"
def arg(f, d, c=int): return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
K2, TWX = arg("--k2", 1600), arg("--twx", 3.0, float)
NLEV = V.NLEV; NC = 2 * NLEV - 1                     # collective 0..24, differential -12..12 -> 0..24

cfg = json.load(open(OUT / "pupil_f1_track.json")); npz = dict(np.load(OUT / "pupil_f1_track.npz"))
rig = V.VRig(48, 9, 1, 4, 400)
d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy")
S0, L0, ep0, tick0 = d["states"], d["levels"], d["ep"], d["tick"]
W1, mu, sd = cp.asarray(npz["W1"]), cp.asarray(npz["mu"]), cp.asarray(npz["sd"])
tmu, tsd, tw0 = cp.asarray(npz["tmu"]), cp.asarray(npz["tsd"]), float(cfg["track_w"])
Kall = V.encode_track(V.track_feats(S0, ep0, tick0))
n_ep = ep0.max() + 1; te = ep0 >= int(0.8 * n_ep)
rng = np.random.default_rng(0); i_tr = rng.choice(np.where(~te)[0], 20000, replace=False); i_te = np.where(te)[0][::3]

def feats(idx, tw):
    Fb = F[idx].astype(np.float32).reshape(len(idx), -1) / 255.0
    C = cp.concatenate([(V.encode_contrast(rig, W1, cp.asarray(Fb)) - mu) / sd, tw * (cp.asarray(Kall[idx]) - tmu) / tsd], 1)
    return C / (cp.linalg.norm(C, axis=1, keepdims=True) + V.EPS)

def split_err(pl, pr, L):
    ec = np.abs((pl + pr) - (L[:, 0] + L[:, 1])) / 2; ed = np.abs((pl - pr) - (L[:, 0] - L[:, 1])) / 2
    return ec.mean(), ed.mean(), np.mean((np.abs(pl - L[:, 0]) <= 1) & (np.abs(pr - L[:, 1]) <= 1))

# (1) the current pupil
Cte = feats(i_te, tw0)
W2, TJ, TL, TR, NLc, NRc = (cp.asarray(npz[k]) for k in ("W2", "TJ", "TL", "TR", "NL", "NR"))
pl, pr = V.read(W2, TJ, TL, TR, Cte, mode="mean", NL=NLc, NR=NRc); pl, pr = cp.asnumpy(pl), cp.asnumpy(pr)
ec, ed, w1 = split_err(pl, pr, L0[i_te])
print(f"current pupil (K2 400, track w {tw0:.1f}, per-motor mean read): within-1 {w1:.3f}, "
      f"mean |collective err| {ec:.2f}, mean |differential err| {ed:.2f}  (teacher's own std: coll {np.std(L0[:,0]+L0[:,1])/2:.2f}, diff {np.std(L0[:,0]-L0[:,1])/2:.2f})")

# (2) finer motion partition, collective/differential tally
tw = tw0 * TWX
Ctr, Cte = feats(i_tr, tw), feats(i_te, tw)
yc = L0[:, 0] + L0[:, 1]; yd = L0[:, 0] - L0[:, 1] + (NLEV - 1)
ytr = cp.asarray(yc[i_tr] * NC + yd[i_tr])
t0 = time.time()
W2n, N2, hires = V.train_l2(Ctr, ytr, K2, NC * NC, seed=7, hire_free_only=True, tol=0, hire=False)
N3 = N2.reshape(K2, NC, NC); Nc, Nd = N3.sum(2), N3.sum(1)
w0 = (Cte @ W2n.T).argmax(1)
lv = cp.arange(NC, dtype=cp.float32)
mc = ((Nc[w0] + 1) * lv).sum(1) / (Nc[w0] + 1).sum(1); md = ((Nd[w0] + 1) * lv).sum(1) / (Nd[w0] + 1).sum(1) - (NLEV - 1)
pl = cp.asnumpy(cp.clip(cp.rint((mc + md) / 2), 0, NLEV - 1)); pr = cp.asnumpy(cp.clip(cp.rint((mc - md) / 2), 0, NLEV - 1))
ec, ed, w1 = split_err(pl, pr, L0[i_te])
print(f"K2 {K2}, track w x{TWX:g}, coll/diff mean read, no hire: within-1 {w1:.3f}, mean |collective err| {ec:.2f}, "
      f"mean |differential err| {ed:.2f}, dead {int((N2.sum(1)==0).sum())}   ({time.time()-t0:.0f}s)")
# save as a pupil the viewer can load: fold coll/diff back into per-motor marginal COUNTS for the mean read
# (exact for the mean: E[l] = (E[c] + E[d]) / 2) -- store the coll/diff counts and let vision_pupil know
np.savez(OUT / "pupil_colldiff.npz", W1=npz["W1"], mu=npz["mu"], sd=npz["sd"], W2=cp.asnumpy(W2n),
         TJ=cp.asnumpy(V.table(N2, K2)), TL=cp.asnumpy(V.table(Nc, K2)), TR=cp.asnumpy(V.table(Nd, K2)),
         NL=cp.asnumpy(Nc), NR=cp.asnumpy(Nd), tmu=npz["tmu"], tsd=npz["tsd"])
cfg2 = dict(cfg); cfg2.update(k2=K2, track_w=tw, read="colldiff"); json.dump(cfg2, open(OUT / "pupil_colldiff.json", "w"))
print("saved pupil_colldiff")
