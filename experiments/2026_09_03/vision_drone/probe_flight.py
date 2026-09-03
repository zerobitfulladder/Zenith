"""One pure-pupil episode, tick by tick: pupil's levels vs the teacher's on the same
states, and the trajectory. Bias, noise, or divergence?"""
import json, sys
import numpy as np
from pathlib import Path
import box_world as B, vrig as V, vision_pupil as VP
OUT = Path(__file__).resolve().parent / "results"
name = sys.argv[1] if len(sys.argv) > 1 else "pupil"
act = VP.load_policy(np.load(OUT / f"{name}.npz"), json.load(open(OUT / f"{name}.json")))
pid = B.make_pid(); rng = np.random.default_rng(5)
s, tgt = B.init(rng); VP.reset(act); lv = (6, 6)
print(f"[{name}] target {tgt.round(2)}")
print(f"{'t':>4}{'x':>7}{'y':>7}{'vx':>6}{'vy':>6}{'tilt':>6}{'dist':>6} | teacher  pupil")
rows = []
for t in range(300):
    teach = pid(s, tgt); lv = act(s, tgt, lv)
    rows.append((t, *s[:5], np.hypot(s[0]-tgt[0], s[1]-tgt[1]), teach, lv))
    if t % 10 == 0:
        print(f"{t:>4}{s[0]:>7.2f}{s[1]:>7.2f}{s[2]:>6.2f}{s[3]:>6.2f}{s[4]:>6.2f}{rows[-1][6]:>6.2f} |  {teach[0]:>2} {teach[1]:>2}    {lv[0]:>2} {lv[1]:>2}")
    s = B.W.physics(s, lv); s[0] = np.clip(s[0], -B.BOX, B.BOX); s[1] = np.clip(s[1], -B.BOX, B.BOX)
    if B.W.at_goal(s, tgt): print("at goal"); 
d = np.array([[r[7][0]-r[8][0], r[7][1]-r[8][1]] for r in rows])
print(f"mean (teacher - pupil) level: left {d[:,0].mean():+.2f}, right {d[:,1].mean():+.2f}; mean |diff| {np.abs(d).mean():.2f}; "
      f"collective bias {(d[:,0]+d[:,1]).mean()/2:+.2f}, differential bias {(d[:,0]-d[:,1]).mean()/2:+.2f}")
