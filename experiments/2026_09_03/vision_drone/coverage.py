"""Are the dead templates dead because nothing is left to learn?

L1: over 1500 frames, every non-flat patch's match to its winning stroke (cos^2).
If the vocabulary covers the world, matches are high; the mass below 0.5 is
what is left to learn. Also: how many strokes ever win.
L2: per object template, the spread of the teacher's levels among the frames it
won (count-weighted std, in levels). A wide spread means one object stands for
states that needed different commands: either missing information (velocity)
or too coarse a partition."""
import json
from pathlib import Path
import numpy as np
import cupy as cp
import vrig as V

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
import sys
NAME = sys.argv[1] if len(sys.argv) > 1 else "pupil"
npz = np.load(OUT / f"{NAME}.npz"); cfg = json.load(open(OUT / f"{NAME}.json"))
print(f"[{NAME}] track={cfg.get('track', False)}")
C = int(cfg["frames"]); rig = V.VRig(48, int(cfg["ps"]), C, int(cfg["grid"]), int(cfg["k1"]))
W1 = cp.asarray(npz["W1"])
d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy").astype(np.float32) / 255.0
tick = d["tick"]
rng = np.random.default_rng(2); idx = np.sort(rng.choice(len(F), 1500, replace=False))
if C == 2:
    prev = np.roll(F, 1, axis=0); prev[tick == 0] = F[tick == 0]
    D = (F[idx] - prev[idx]) if cfg.get("diff") == "signed" else 0.5 + 0.5 * (F[idx] - prev[idx])
    X = np.stack([F[idx], D], 1).reshape(len(idx), -1)
else:
    X = F[idx].reshape(len(idx), -1)
best, wins = [], np.zeros(rig.K, int)
for a in range(0, len(X), 50):
    Q, keep = rig.patches(cp.asarray(X[a:a + 50]))
    Vv = Q.reshape(-1, rig.dim)[keep.reshape(-1)]
    S = (Vv @ W1.T) ** 2
    w = S.argmax(1); best.append(cp.asnumpy(S.max(1))); wins += np.bincount(cp.asnumpy(w), minlength=rig.K)
best = np.concatenate(best)
flat = 1.0 - len(best) / (len(X) * rig.npos)
print(f"L1 ({C} ch): {flat:.1%} of patches are flat and skipped; of the rest, match to the winner:")
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"   {int(q*100):>3}th percentile cos^2 = {np.quantile(best, q):.3f}")
print(f"   mass below 0.5: {np.mean(best < 0.5):.1%}   below 0.8: {np.mean(best < 0.8):.1%}")
print(f"   strokes that win at all: {(wins > 0).sum()}/{rig.K}; top 20 strokes take {np.sort(wins)[::-1][:20].sum()/wins.sum():.0%} of wins")

NL, NR = npz["NL"], npz["NR"]
lv = np.arange(V.NLEV)
def spread(N):
    tot = N.sum(1); m = (N * lv).sum(1) / np.maximum(tot, 1)
    sd = np.sqrt((N * (lv - m[:, None]) ** 2).sum(1) / np.maximum(tot, 1))
    return sd, tot
sl, tot = spread(NL); sr, _ = spread(NR)
w = tot / tot.sum()
print(f"L2: usage-weighted std of the teacher's level within an object: left {np.sum(w*sl):.2f}, right {np.sum(w*sr):.2f} levels")
print(f"    objects with std > 1.5 levels (either motor): {np.mean((sl > 1.5) | (sr > 1.5)):.0%}, carrying {w[(sl > 1.5) | (sr > 1.5)].sum():.0%} of frames")
print(f"    reference: std of the teacher's level over ALL frames: left {np.sqrt(np.cov(np.repeat(lv, NL.sum(0).astype(int)))):.2f}")
