"""The split rig (user's design): description and action are separate organs.

    top layer   SENSORY-ONLY templates (cmd_share = 0): present bumps
                at 50% of the cue + track codes at 50% — the winning
                read ratio, preserved. No command anywhere in the
                vector; partial-cue command regeneration is retired.
    cetele      C[template, 13*13] counts of the teacher's (l, r)
                level pairs, notched every tick and attributed by the
                SAME masked read used at act time. One partition, one
                question — the Aug-28 consistency (rows sensory, tally
                sensory), which the joint hybrid violated and paid
                0.72 -> 0.00 to teach us.
    read        expected action under duty weights (count-weighted
                mean in (coll, diff), snapped once) — the regulation
                law; value tilt comes later as a separate phase.

Neuroscience framing (user): PFC-like memory holds descriptive state;
the basal-ganglia-like cetele selects among actions; cortex learns of
the selection through the loop, not by storing commands in its own
partition. Record precedent: sensory rows + tally beat bound 61-to-3
(2026-08-28 A/B).

Tracks reused frozen from the dagger3 checkpoint (they were sensory-
only all along). L3 trains fresh: 120k teacher-only, then tight-
envelope DAgger with the pupil flying the cetele read.

Run:  .venv/bin/python experiments/2026_08_29/split/run_td_split.py
Env:  TP_TICKS TP_PHASE1 TP_REPORT TP_K3 TP_PRES TP_EVERY TP_TAG
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
import td_tracks as B                      # noqa: E402
import td3_stack as M                      # noqa: E402
import td_bound as T                       # noqa: E402
import td_balance as L                     # noqa: E402

TRACKS_CKPT = HERE.parent / "balance" / "results" / "dagger3" / "weights_best.npz"
OUT = HERE / "results" / os.environ.get('TP_TAG', '').lstrip("_")
TICKS = int(os.environ.get("TP_TICKS", "600000"))
PHASE1 = int(os.environ.get("TP_PHASE1", "120000"))
REPORT = int(os.environ.get("TP_REPORT", "40000"))
K3 = int(os.environ.get("TP_K3", "4096"))
PRES = float(os.environ.get("TP_PRES", "0.5"))
EVERY = int(os.environ.get("TP_EVERY", "5"))
NA = W.NLEV * W.NLEV
HOVER_LV = (W.NLEV // 2, W.NLEV // 2)


def a_of(lv):
    return lv[0] * W.NLEV + lv[1]


class Split:
    def __init__(self):
        cfg = json.load(open(TRACKS_CKPT.with_suffix(".json")))
        z = np.load(TRACKS_CKPT)
        self.st = T.Stack(k1=cfg["k1"], k2=cfg["k2"], k3=K3,
                          topm=cfg["topm"], seed=0, cmd_share=0.0,
                          ns=cfg["ns"], pres_share=PRES)
        for c in range(cfg["ns"]):
            self.st.tr[c].l1.W = z[f"t{c}l1"]
            self.st.tr[c].l1.n = int(z[f"t{c}n1"])
            self.st.tr[c].l2.W = z[f"t{c}l2"]
            self.st.tr[c].l2.n = int(z[f"t{c}n2"])
        self.st.stage = 2                  # tracks frozen, L3 learns
        self.st.every = EVERY
        self.sense = L.make_sense(ns=cfg["ns"], vwarp=cfg["vwarp"],
                                  fovea=cfg.get("fovea", False))
        self.C = np.zeros((K3, NA), np.float64)

    def winner(self):
        st = self.st
        if st._codes is None or st.l3.n == 0:
            return -1
        i, v = st._sensor_cells()
        if len(i) == 0:
            return -1
        return int(np.argmax(st.l3.scores(i, v)))

    def act(self, j):
        """Expected action under the duty weights of template j."""
        if j < 0:
            return HOVER_LV
        cnt = self.C[j]
        sup = np.nonzero(cnt)[0]
        if len(sup) == 0:
            return HOVER_LV
        w = cnt[sup] / cnt[sup].sum()
        ls, rs = sup // W.NLEV, sup % W.NLEV
        c = float((w * (0.5 * (W.LEVELS[ls] + W.LEVELS[rs]))).sum())
        d = float((w * (0.5 * (W.LEVELS[rs] - W.LEVELS[ls]))).sum())
        return M.cd_to_levels(c, d)


def evaluate(rig, n=25, seed=99):
    rng = np.random.default_rng(seed)
    was = rig.st.frozen
    rig.st.frozen = True
    ok, frac = [], []
    for _ in range(n):
        s = L.air_init(rng)
        rig.st.reset()
        bal = []
        for t in range(L.EP_LEN):
            rig.st.push(rig.sense(s))
            s = W.physics(s, rig.act(rig.winner()))
            bal.append(L.balanced(s))
        ok.append(all(bal[-150:]))
        frac.append(float(np.mean(bal[100:])))
    rig.st.frozen = was
    return float(np.mean(ok)), float(np.mean(frac))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = Split()
    rng = np.random.default_rng(0)
    cfg = {"kind": "raw", "k3": K3, "pres": PRES,
           "tracks_from": str(TRACKS_CKPT)}
    t0 = time.time()

    def save(tag, extra):
        d = {f"t{c}l1": rig.st.tr[c].l1.W for c in range(rig.st.ns)}
        d.update({f"t{c}n1": rig.st.tr[c].l1.n for c in range(rig.st.ns)})
        d.update({f"t{c}l2": rig.st.tr[c].l2.W for c in range(rig.st.ns)})
        d.update({f"t{c}n2": rig.st.tr[c].l2.n for c in range(rig.st.ns)})
        d["l3"] = rig.st.l3.W
        d["l3n"] = rig.st.l3.n
        d["C"] = rig.C
        np.savez(OUT / f"{tag}.npz", **d)
        with open(OUT / f"{tag}.json", "w") as f:
            json.dump({**cfg, **extra}, f)

    s = L.air_init(rng)
    rig.st.reset()
    ep_tick = episodes = 0
    pupil_ep = False
    agree, winners = [], set()
    stats, best = [], {"score": -1.0, "agree": -1.0, "tick": 0}

    for t in range(TICKS):
        rig.st.push(rig.sense(s))
        j = rig.winner()
        lab = L.teacher(s)
        if j >= 0:
            rig.C[j, a_of(lab)] += 1.0     # notch, same question as read
            winners.add(j)
            guess = rig.act(j)
            agree.append(0.5 * ((guess[0] == lab[0])
                                + (guess[1] == lab[1])))
        rig.st.learn(lab)                  # sensory-only rotation
        fly = rig.act(j) if (pupil_ep and j >= 0) else lab
        s = W.physics(s, fly)
        ep_tick += 1
        blown = pupil_ep and (np.hypot(s[2], s[3]) > 5 or abs(s[5]) > 5)
        if ep_tick >= L.EP_LEN or blown:
            episodes += 1
            s = L.air_init(rng)
            rig.st.reset()
            ep_tick = 0
            pupil_ep = (t >= PHASE1) and (episodes % 2 == 1)

        if (t + 1) % REPORT == 0:
            ev, bf = evaluate(rig)
            row = {"tick": t + 1, "phase": 1 if t < PHASE1 else 2,
                   "EVAL": round(ev, 3), "bal_frac": round(bf, 3),
                   "agree": round(float(np.mean(agree)), 3),
                   "winners": len(winners), "l3": int(rig.st.l3.n),
                   "support": round(float((rig.C.sum(1) > 0).mean()), 3),
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree, winners = [], set()
            save("checkpoint", {"tick": t + 1, "score": row["EVAL"]})
            key = (row["EVAL"], row["agree"])
            if key > (best["score"], best["agree"]):
                best = {"score": row["EVAL"], "agree": row["agree"],
                        "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} @ {best['tick']}",
                      flush=True)
            with open(OUT / "metrics.json", "w") as f:
                json.dump({"best": best, "config": cfg,
                           "reports": stats}, f, indent=2)

    print(f"done — best {best['score']} @ {best['tick']} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
