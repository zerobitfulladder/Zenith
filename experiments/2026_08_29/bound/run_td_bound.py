"""Train the bound stack to clone the oracle.

Staged: track L1s learn to STAGE1, L2s to STAGE2 (L1 frozen), then L3
alone on frozen tracks. The oracle flies the whole time.

The leading indicator is the on-path read — per-channel corr / error sd
/ SNR against the oracle's command — reported every block long before
EVAL can move. EVAL is frozen fresh episodes; closest approach is a
MEAN over episodes (the 25-episode median flip-flopped between 4.05 and
2.37 in run_tracks and read as a trend that wasn't there).

Run:  .venv/bin/python experiments/2026_08_29/bound/run_td_bound.py
Env:  TB_TICKS TB_REPORT TB_K1 TB_K2 TB_K3 TB_TOPM TB_EVERY TB_CMD
      TB_STAGE1 TB_STAGE2 TB_TAG
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
import td_tracks as B                      # noqa: E402
import td3_stack as M                      # noqa: E402
import td_bound as T                       # noqa: E402

OUT = HERE / "results" / os.environ.get('TB_TAG', '').lstrip("_")
TICKS = int(os.environ.get("TB_TICKS", "300000"))
REPORT = int(os.environ.get("TB_REPORT", "25000"))
K1 = int(os.environ.get("TB_K1", "64"))
K2 = int(os.environ.get("TB_K2", "128"))
K3 = int(os.environ.get("TB_K3", "8192"))
TOPM = int(os.environ.get("TB_TOPM", "8"))
EVERY = int(os.environ.get("TB_EVERY", "5"))
CMD = float(os.environ.get("TB_CMD", str(T.SHARE_CMD)))
STAGE1 = int(os.environ.get("TB_STAGE1", "15000"))
STAGE2 = int(os.environ.get("TB_STAGE2", "30000"))


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
            st.push(B.sense(s, tgt))
            lv, _, _ = st.act()
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
    return float(np.mean(ok)), float(np.mean(reached))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = T.Stack(k1=K1, k2=K2, k3=K3, topm=TOPM, seed=0, cmd_share=CMD)
    st.every = EVERY
    rng = np.random.default_rng(0)
    cfg = {"kind": "module", "module": "td_bound",
           "dir": "experiments/2026_08_29/temporal_drone",
           "k1": K1, "k2": K2, "k3": K3, "topm": TOPM, "seed": 0,
           "every": EVERY, "cmd_share": CMD,
           "stage1": STAGE1, "stage2": STAGE2}

    def save(tag, extra):
        d = {f"t{c}l1": st.tr[c].l1.W for c in range(T.NS)}
        d.update({f"t{c}n1": st.tr[c].l1.n for c in range(T.NS)})
        d.update({f"t{c}l2": st.tr[c].l2.W for c in range(T.NS)})
        d.update({f"t{c}n2": st.tr[c].l2.n for c in range(T.NS)})
        d["l3"] = st.l3.W
        d["l3n"] = st.l3.n
        np.savez(OUT / f"{tag}.npz", **d)
        with open(OUT / f"{tag}.json", "w") as f:
            json.dump({**cfg, **extra}, f)

    s, tgt = W.any_init(rng)
    st.reset()
    ep_tick = episodes = 0
    gc, gd, oc, od, agree, winners = [], [], [], [], [], set()
    stats, best = [], {"score": -1.0, "agree": -1.0, "tick": 0}
    t0 = time.time()

    for t in range(TICKS):
        st.stage = 0 if t < STAGE1 else (1 if t < STAGE2 else 2)
        st.push(B.sense(s, tgt))
        guess, cd, win = st.act()
        lv = F.teacher(s, tgt)
        if win >= 0:
            tc, td = M.levels_to_cd(lv)
            gc.append(cd[0]); gd.append(cd[1]); oc.append(tc); od.append(td)
            agree.append(0.5 * ((guess[0] == lv[0]) + (guess[1] == lv[1])))
            winners.add(win)
        st.learn(lv)
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
            row = {"tick": t + 1, "episodes": episodes, "EVAL": round(ev, 3),
                   "closest_mean": round(near, 2)}
            if gc:
                a = [np.array(x) for x in (gc, gd, oc, od)]
                for nm, p, o in (("coll", a[0], a[2]), ("diff", a[1], a[3])):
                    e = float((p - o).std())
                    row[f"{nm}_corr"] = round(
                        float(np.corrcoef(p, o)[0, 1]), 3) \
                        if p.std() > 1e-9 else None
                    row[f"{nm}_snr"] = round(float(o.std()) / max(e, 1e-9), 2)
                row["agree"] = round(float(np.mean(agree)), 3)
                row["winners"] = len(winners)
            row.update({"l3": int(st.l3.n),
                        "mins": round((time.time() - t0) / 60, 1)})
            stats.append(row)
            print(json.dumps(row), flush=True)
            gc, gd, oc, od, agree, winners = [], [], [], [], [], set()
            save("checkpoint", {"tick": t + 1, "score": row["EVAL"]})
            # EVAL first; while it ties (0.00 for a long time), the
            # on-path agreement breaks the tie so weights_best is never
            # an empty or stale memory.
            ag = row.get("agree") or 0.0
            if (row["EVAL"], ag) > (best["score"], best["agree"]):
                best = {"score": row["EVAL"], "agree": ag, "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} (agree {ag}) "
                      f"@ {best['tick']}", flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "config": cfg, "reports": stats},
                          f, indent=2)

    save("weights_final", {"tick": TICKS, "score": stats[-1]["EVAL"]})
    print(f"done — best {best['score']} @ {best['tick']} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
