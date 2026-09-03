"""One layer, sensory tracks only, trained by reward. No teacher, no vision.

state   bump codes of dx, dy, vx, vy, tilt, tilt rate, at lags 0, 2, 5 (768 numbers)
layer   K templates over that vector. Winner = nearest (cos^2). A state that no
        template matches above NOVEL takes a free template (allocation on
        novelty); the winner moves toward what it wins (cnt/n step, floor).
        Reward never touches a template.
value   Q[template, action], a counted table: each update is a running mean of
        the TD target with step 1/n (floored), so early visits count fully and
        the estimate settles. 25 actions = collective offset x differential
        offset around hover.
act     greedy on Q[winner] plus exploration noise that decays.
reward  -0.1 * distance - 0.02 * speed per tick, +1 while at goal, +10 on hold.

Usage:  uv run python -u rl_tracks.py [--episodes 600] [--k 4096]
"""
import sys, json, time
import numpy as np
from pathlib import Path
import box_world as B
import novision as NV

OUT = Path(__file__).resolve().parent / "results"
def arg(f, d, c=int): return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
EPISODES, K, CAP = arg("--episodes", 600), arg("--k", 4096), 300
GAMMA, NOVEL, ETA_MIN, ALPHA_MIN = 0.98, 0.90, 0.02, 0.05
OFFS = [-2, -1, 0, 1, 2]
ACTIONS = [(c, d) for c in OFFS for d in OFFS]            # 25
HOVER = B.W.NLEV // 2


def levels(a):
    c, d = ACTIONS[a]
    return (int(np.clip(HOVER + 2 * c + d, 0, B.W.NLEV - 1)), int(np.clip(HOVER + 2 * c - d, 0, B.W.NLEV - 1)))


class Layer:
    def __init__(self, K, D):
        self.W = np.zeros((K, D), np.float32); self.live = np.zeros(K, bool); self.n = np.zeros(K)
    def winner(self, x, learn=True):
        xn = x / (np.linalg.norm(x) + 1e-9)
        if not self.live.any():
            return self._alloc(xn)
        s = (self.W[self.live] @ xn) ** 2
        j = int(np.argmax(s)); best = float(s[j]); t = int(np.where(self.live)[0][j])
        if best < NOVEL and learn and (~self.live).any():
            return self._alloc(xn)
        if learn:
            self.n[t] += 1; eta = max(1.0 / self.n[t], ETA_MIN)
            self.W[t] += eta * (xn - self.W[t]); self.W[t] /= np.linalg.norm(self.W[t]) + 1e-9
        return t
    def _alloc(self, xn):
        t = int(np.where(~self.live)[0][0]); self.W[t] = xn; self.live[t] = True; self.n[t] = 1
        return t


def reward(s, tgt):
    dist = np.hypot(s[0] - tgt[0], s[1] - tgt[1]); speed = np.hypot(s[2], s[3])
    return -0.1 * dist - 0.02 * speed + (1.0 if B.W.at_goal(s, tgt) else 0.0)


def features(hist):
    f = np.concatenate([hist[max(0, len(hist) - 1 - lag)] for lag in NV.LAGS])[None]
    return NV.encode(f)[0]


def run_episode(layer, Q, N, rng, eps, learn=True, cap=CAP):
    s, tgt = B.init(rng); hist = []; hold = 0; ret = 0.0
    hist.append(NV.raw(s, tgt)); x = features(hist); t = layer.winner(x, learn)
    for k in range(cap):
        q = Q[t]
        a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(np.argmax(q + 1e-6 * rng.standard_normal(len(q))))
        s2 = B.W.physics(s, levels(a)); s2[0] = np.clip(s2[0], -B.BOX, B.BOX); s2[1] = np.clip(s2[1], -B.BOX, B.BOX)
        r = reward(s2, tgt); hold = hold + 1 if B.W.at_goal(s2, tgt) else 0
        done = hold >= 10
        if done: r += 10.0
        hist.append(NV.raw(s2, tgt)); x2 = features(hist); t2 = layer.winner(x2, learn)
        if learn:
            target = r if done else r + GAMMA * float(Q[t2].max())
            N[t, a] += 1; alpha = max(1.0 / N[t, a], ALPHA_MIN)
            Q[t, a] += alpha * (target - Q[t, a])
        ret += r; s, t = s2, t2
        if done: break
    return done, k + 1, float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])), ret


def main():
    layer = Layer(K, NV.D); Q = np.zeros((K, len(ACTIONS)), np.float32); N = np.zeros((K, len(ACTIONS)))
    rng = np.random.default_rng(0); t0 = time.time(); rets = []
    print(f"{K} templates over {NV.D} bump numbers, {len(ACTIONS)} actions, {EPISODES} episodes, cap {CAP}", flush=True)
    for e in range(1, EPISODES + 1):
        eps = max(0.05, 0.4 * (1 - e / EPISODES))
        done, ticks, dist, ret = run_episode(layer, Q, N, rng, eps); rets.append(ret)
        if e % 100 == 0 or e == EPISODES:
            ev = [run_episode(layer, Q, N, np.random.default_rng(123 + i), 0.0, learn=False, cap=400) for i in range(10)]
            print(f"ep {e:>5}  eps {eps:.2f}  live {int(layer.live.sum()):>5}  train return (last 100) {np.mean(rets[-100:]):>7.1f}  |  "
                  f"greedy 10 flights: success {np.mean([x[0] for x in ev]):.2f}  median ticks {int(np.median([x[1] for x in ev]))}  "
                  f"final dist {np.median([x[2] for x in ev]):.2f}   ({time.time()-t0:.0f}s)", flush=True)
    np.savez(OUT / "pupil_rl.npz", W=layer.W, live=layer.live, Q=Q, N=N)
    json.dump({"kind": "module", "module": "vision_pupil", "dir": "experiments/2026_09_03/vision_drone", "read": "rl"},
              open(OUT / "pupil_rl.json", "w"))
    print("saved pupil_rl", flush=True)


if __name__ == "__main__":
    main()
