"""Is the two-track memory broken, or is (dx, dy, angle) simply not enough?

Two separable questions:
  1. open loop -- shown a state the oracle visited, does the memory read
     back the thrusts the oracle commanded? (tests the read path)
  2. ambiguity -- for one sensory state, how many DIFFERENT commands did
     the oracle give across the training set? (tests the observation set)
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_28" / "single_layer_drone"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_29" / "temporal_drone"))
sys.path.insert(0, str(HERE))
import sl_drone as W, fast_oracle as F, two_track as T   # noqa: E402

cache = Path("/tmp/claude-1000/-home-lavender-Projects-Geodesique/"
             "b740704d-75db-4cdd-bdff-0ecb9063c424/scratchpad/rollout.npz")
z = np.load(cache)
S = [dict(zip(T.SENS_CH, r)) for r in z["S"]]
M = [tuple(r) for r in z["M"]]

cfg = json.load(open(HERE / "results/two_track.json"))
m = T.TwoTrack.from_npz(np.load(HERE / "results/two_track.npz"), cfg)

rng = np.random.default_rng(3)
idx = rng.choice(len(S), 3000, replace=False)

# 1. open loop: state in, command out, compared with what the oracle said
hit = np.zeros(2); close = np.zeros(2); err = []
for j in idx:
    s = S[j]
    si, sv = m.sens_code(s)
    win = int(np.argmax(m._bids(si, sv)))
    half = m.l2.W[win][m.mot_ix]
    vec = np.clip(half, 0.0, None) @ m.l1m.W[:m.l1m.n_boot]
    got = tuple(int(np.argmin(np.abs(W.LEVELS - m.mot_track.read(vec, c))))
                for c in T.MOTOR_CH)
    hit += np.array(got) == np.array(M[j])
    close += np.abs(np.array(got) - np.array(M[j])) <= 1
    err.append(np.abs(W.LEVELS[list(got)] - W.LEVELS[list(M[j])]).mean())
print(f"open loop, on states the oracle actually visited:")
print(f"   exact level match  L {hit[0]/len(idx):.3f}  R {hit[1]/len(idx):.3f}")
print(f"   within one level   L {close[0]/len(idx):.3f}  R {close[1]/len(idx):.3f}")
print(f"   mean thrust error  {np.mean(err):.3f}  (levels span "
      f"{W.LEVELS[0]:.1f}..{W.LEVELS[-1]:.1f}, hover {W.HOVER:.1f})")

# 2. ambiguity: same sensory cell, how many different commands?
A = np.array([[d[c] for c in T.SENS_CH] for d in S])
key = np.stack([np.digitize(A[:, 0], np.linspace(-8, 8, 33)),
                np.digitize(A[:, 1], np.linspace(-8, 8, 33)),
                np.digitize(A[:, 2], np.linspace(-1, 1, 21))], 1)
key = key[:, 0] * 10000 + key[:, 1] * 100 + key[:, 2]
Marr = np.array(M)
uk, inv = np.unique(key, return_inverse=True)
spread, n_cmd = [], []
for b in range(len(uk)):
    sel = Marr[inv == b]
    if len(sel) < 20:
        continue
    spread.append(W.LEVELS[sel].std(axis=0).mean())
    n_cmd.append(len(np.unique(sel, axis=0)))
print(f"\nambiguity of the observation set ({len(spread)} well-populated "
      f"sensory cells):")
print(f"   distinct oracle commands per cell: median {np.median(n_cmd):.0f}, "
      f"90th pct {np.percentile(n_cmd, 90):.0f}")
print(f"   thrust spread within one cell: {np.mean(spread):.3f}")
print(f"   thrust spread across the whole set: "
      f"{W.LEVELS[Marr].std(axis=0).mean():.3f}")

# 3. what the oracle's command actually depends on
import itertools
V = []
rng2 = np.random.default_rng(5)
for _ in range(4000):
    s, tgt = W.any_init(rng2)
    V.append(list(s) + list(W.LEVELS[list(F.teacher(s, tgt))]) +
             [s[0]-tgt[0], s[1]-tgt[1]])
V = np.array(V)
names = ["x", "y", "vx", "vy", "tilt", "gyro"]
print("\nhow much of the oracle's command each state variable explains "
      "(|correlation|):")
for i, nm in enumerate(names):
    c = max(abs(np.corrcoef(V[:, i], V[:, 6])[0, 1]),
            abs(np.corrcoef(V[:, i], V[:, 7])[0, 1]))
    print(f"   {nm:<5} {c:.3f}")
for i, nm in ((8, "dx"), (9, "dy")):
    c = max(abs(np.corrcoef(V[:, i], V[:, 6])[0, 1]),
            abs(np.corrcoef(V[:, i], V[:, 7])[0, 1]))
    print(f"   {nm:<5} {c:.3f}")
