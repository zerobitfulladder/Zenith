"""Train the two-layer temporal stack to fly the drone.

  WARM   the oracle flies. Layer one learns the vocabulary of motion
         shapes from a sane state distribution, layer two learns to bind
         them to commands, and the reservoir fills.
  then   layer one FREEZES (its vocabulary is set, and freezing keeps the
         stored codes meaning what they meant when they were stored), the
         pupil flies, and layer two keeps learning: the live moment plus
         REPLAY moments drawn uniformly from everything ever lived.

Reported every block, including the two numbers that told the
single-layer rig apart from a memory failure: retrieval quality, which
stayed flat while it decayed, and commanded thrust against the oracle's,
which is what actually drifted.

Run:  .venv/bin/python experiments/2026_08_29/temporal_stack/run_td.py
Env:  TD_TICKS TD_WARM TD_REPLAY TD_K1 TD_K2 TD_REPORT TD_ORACLE TD_TAG
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
import td_stack as S                       # noqa: E402

OUT = HERE / "results" / f"two_layer{os.environ.get('TD_TAG', '')}"
TICKS = int(os.environ.get("TD_TICKS", "150000"))
WARM = int(os.environ.get("TD_WARM", "30000"))
REPLAY = int(os.environ.get("TD_REPLAY", "2"))
K1 = int(os.environ.get("TD_K1", "128"))
K2 = int(os.environ.get("TD_K2", "2048"))
REPORT = int(os.environ.get("TD_REPORT", "10000"))
ORACLE = os.environ.get("TD_ORACLE", "fast")
teacher = F.teacher if ORACLE == "fast" else W.teacher


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = S.Stack(k1=K1, k2=K2, seed=0)
    res = S.Reservoir(cap=50000, topm=st.topm, seed=1)
    rng = np.random.default_rng(0)

    cfg = {"kind": "module", "module": "td_stack",
           "dir": "experiments/2026_08_29/temporal_drone",
           "k1": K1, "k2": K2, "topm": st.topm, "seed": 0,
           "oracle": ORACLE, "replay": REPLAY}

    def save(tag, extra):
        np.savez(OUT / f"{tag}.npz", W1=st.l1.hc.W, n1=st.l1.hc.n_boot,
                 W2=st.l2.hc.W, n2=st.l2.hc.n_boot)
        with open(OUT / f"{tag}.json", "w") as f:
            json.dump({**cfg, **extra}, f)

    s, tgt = W.any_init(rng)
    lv_prev = (W.NLEV // 2, W.NLEV // 2)
    st.push(S.norm_state(s, tgt), lv_prev)
    ep_tick = hold = episodes = 0
    recent, agree, match = [], [], []
    p_coll, p_diff, o_coll, o_diff = [], [], [], []
    stats, best = [], {"score": -1.0, "tick": 0}
    t0 = time.time()

    for t in range(TICKS):
        lv_t = teacher(s, tgt)                       # the oracle, every tick
        if t == WARM:
            st.l1_frozen = True
            print(f"--- layer one frozen at {st.l1.hc.n_boot} shapes; "
                  f"pupil takes over ---", flush=True)

        # ACT FIRST. Learning this moment before acting on it binds the
        # oracle's command to exactly the codes and state we are about to
        # query with, so the just-touched template wins and hands the
        # label straight back. That leak reported 0.61-0.79 success for a
        # policy that scores 0.00 the moment it is frozen.
        if t < WARM:
            lv_a = lv_t
        else:
            lv_a, c = st.act()
            match.append(c)

        moment = st.learn_live(lv_t)
        if moment is not None:
            res.add(*moment)
            for _ in range(REPLAY if t >= WARM else 0):
                d = res.draw()
                if d is not None:
                    st.learn_replay(d)
        agree.append(0.5 * ((lv_a[0] == lv_t[0]) + (lv_a[1] == lv_t[1])))
        p_coll.append(0.5 * (W.LEVELS[lv_a[0]] + W.LEVELS[lv_a[1]]))
        p_diff.append(abs(W.LEVELS[lv_a[0]] - W.LEVELS[lv_a[1]]))
        o_coll.append(0.5 * (W.LEVELS[lv_t[0]] + W.LEVELS[lv_t[1]]))
        o_diff.append(abs(W.LEVELS[lv_t[0]] - W.LEVELS[lv_t[1]]))

        s = W.physics(s, lv_a)
        lv_prev = lv_a
        ep_tick += 1

        done = False
        if W.at_goal(s, tgt):
            hold += 1
            if hold >= 10:
                recent.append(1)
                done = True
        else:
            hold = 0
        if abs(s[0]) > 30 or abs(s[1]) > 30 or ep_tick >= W.EP_CAP:
            recent.append(0)
            done = True
        if done:
            episodes += 1
            s, tgt = W.any_init(rng)
            lv_prev = (W.NLEV // 2, W.NLEV // 2)
            ep_tick = hold = 0
            recent = recent[-100:]
            st.reset_history()
        st.push(S.norm_state(s, tgt), lv_prev)

        if (t + 1) % REPORT == 0:
            row = {"tick": t + 1,
                   "phase": "WARM" if t < WARM else "pupil",
                   "episodes": episodes,
                   "success_last100": round(float(np.mean(recent)) if recent
                                            else 0.0, 3),
                   "agree": round(float(np.mean(agree)), 3),
                   "retrieval": round(float(np.mean(match)), 3) if match
                   else None,
                   "thrust_pupil": round(float(np.mean(p_coll)), 2),
                   "thrust_oracle": round(float(np.mean(o_coll)), 2),
                   "spread_pupil": round(float(np.mean(p_diff)), 2),
                   "spread_oracle": round(float(np.mean(o_diff)), 2),
                   "shapes": int(st.l1.hc.n_boot),
                   "commands": int(st.l2.hc.n_boot),
                   "reservoir": int(min(res.n, res.cap)),
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree, match = [], []
            p_coll, p_diff, o_coll, o_diff = [], [], [], []
            save("checkpoint", {"tick": t + 1,
                                "score": row["success_last100"]})
            if t >= WARM and row["success_last100"] > best["score"]:
                best = {"score": row["success_last100"], "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} @ {best['tick']}",
                      flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "config": cfg, "reports": stats},
                          f, indent=2)

    save("weights_final", {"tick": TICKS,
                           "score": stats[-1]["success_last100"]})
    print(f"done — best {best['score']} @ {best['tick']} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
