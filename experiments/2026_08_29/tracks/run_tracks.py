"""Train the per-signal track stack to clone the oracle.

The oracle flies the whole time. Every tick the five tracks advance —
dx, dy, tilt and the two thrusts — and layer three binds all five codes
into one memory. Nothing about a pupil trajectory or replay is in here:
no configuration has yet cloned the oracle at all, and until one does,
adding distribution shift only adds variables.

The only number that means anything is EVAL: fly fresh episodes with
learning OFF and see whether it gets anywhere. The in-training success
rate is measured on a trajectory being learned at every tick, and with
20 ms ticks the memory just echoes the teacher one tick late.

Run:  .venv/bin/python experiments/2026_08_29/tracks/run_tracks.py
Env:  TR_TICKS TR_REPORT TR_K1 TR_K2 TR_K3 TR_TOPM TR_EVERY TR_TAG
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "temporal_drone"))      # shared rig modules
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))

import sl_drone as W                       # noqa: E402
import fast_oracle as F                    # noqa: E402
import td_tracks as T                      # noqa: E402

OUT = HERE / "results" / os.environ.get('TR_TAG', '').lstrip("_")
TICKS = int(os.environ.get("TR_TICKS", "300000"))
REPORT = int(os.environ.get("TR_REPORT", "25000"))
K1 = int(os.environ.get("TR_K1", "64"))
K2 = int(os.environ.get("TR_K2", "128"))
K3 = int(os.environ.get("TR_K3", "8192"))
TOPM = int(os.environ.get("TR_TOPM", "8"))
EVERY = int(os.environ.get("TR_EVERY", "5"))


def evaluate(st, n=25, seed=99):
    """Fresh episodes, learning off, the stack flying itself."""
    rng = np.random.default_rng(seed)
    was = st.frozen
    st.frozen = True
    ok, reached = [], []
    for _ in range(n):
        s, tgt = W.any_init(rng)
        st.reset()
        hold, hit, best = 0, False, np.hypot(s[0] - tgt[0], s[1] - tgt[1])
        for t in range(W.EP_CAP):
            st.push_sensors(T.sense(s, tgt))
            lv, _ = st.act()
            st.commit_motor(lv)
            s = W.physics(s, lv)
            best = min(best, float(np.hypot(s[0] - tgt[0], s[1] - tgt[1])))
            if W.at_goal(s, tgt):
                hold += 1
                if hold >= 10:
                    hit = True
                    break
            else:
                hold = 0
            if abs(s[0]) > 30 or abs(s[1]) > 30:
                break
        ok.append(hit)
        reached.append(best)
    st.frozen = was
    return float(np.mean(ok)), float(np.median(reached))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = T.Stack(k1=K1, k2=K2, k3=K3, topm=TOPM, seed=0)
    st.every = EVERY
    rng = np.random.default_rng(0)
    cfg = {"kind": "module", "module": "td_tracks",
           "dir": "experiments/2026_08_29/temporal_drone",
           "k1": K1, "k2": K2, "k3": K3, "topm": TOPM, "seed": 0,
           "every": EVERY, "mode": "clone"}

    def save(tag, extra):
        d = {f"t{c}l1": st.tr[c].l1.W for c in range(T.NT)}
        d.update({f"t{c}n1": st.tr[c].l1.n for c in range(T.NT)})
        d.update({f"t{c}l2": st.tr[c].l2.W for c in range(T.NT)})
        d.update({f"t{c}n2": st.tr[c].l2.n for c in range(T.NT)})
        d["l3"] = st.l3.W
        d["l3n"] = st.l3.n
        np.savez(OUT / f"{tag}.npz", **d)
        with open(OUT / f"{tag}.json", "w") as f:
            json.dump({**cfg, **extra}, f)

    s, tgt = W.any_init(rng)
    st.reset()
    ep_tick = episodes = 0
    agree, active = [], []
    stats, best = [], {"score": -1.0, "tick": 0}
    t0 = time.time()

    for t in range(TICKS):
        st.push_sensors(T.sense(s, tgt))
        guess, _ = st.act()                    # what the pupil WOULD do
        lv = F.teacher(s, tgt)                 # the oracle flies
        agree.append(0.5 * ((guess[0] == lv[0]) + (guess[1] == lv[1])))
        st.commit_motor(lv)
        if t % EVERY == 0:
            m = st.learn()
            if m is not None:
                active.append(len(m[0]))
        s = W.physics(s, lv)
        ep_tick += 1

        if (W.at_goal(s, tgt) or abs(s[0]) > 30 or abs(s[1]) > 30
                or ep_tick >= W.EP_CAP):
            episodes += 1
            s, tgt = W.any_init(rng)
            st.reset()
            ep_tick = 0

        if (t + 1) % REPORT == 0:
            ev, near = evaluate(st)
            row = {"tick": t + 1, "episodes": episodes,
                   "EVAL": round(ev, 3),
                   "closest_approach": round(near, 2),
                   "agree_on_oracle_path": round(float(np.mean(agree)), 3),
                   "l3_input_cells": round(float(np.mean(active)), 1)
                   if active else None,
                   "l1": [t_.l1.n for t_ in st.tr],
                   "l2": [t_.l2.n for t_ in st.tr],
                   "l3": int(st.l3.n),
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree, active = [], []
            save("checkpoint", {"tick": t + 1, "score": row["EVAL"]})
            if row["EVAL"] > best["score"]:
                best = {"score": row["EVAL"], "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} @ {best['tick']}",
                      flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "config": cfg, "reports": stats},
                          f, indent=2)

    save("weights_final", {"tick": TICKS, "score": stats[-1]["EVAL"]})
    print(f"done — best {best['score']} @ {best['tick']} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
