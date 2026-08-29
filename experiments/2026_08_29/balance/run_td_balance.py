"""Train the bound stack to stop and level — the balance rung.

Teacher audition first (it must balance from every init, wearing the
same blind-start handicap the pupil has). Then staged training exactly
as run_td_bound, on fixed-length episodes so hold data dominates.

Run:  .venv/bin/python experiments/2026_08_29/balance/run_td_balance.py
Env:  TL_TICKS TL_REPORT TL_K1 TL_K2 TL_K3 TL_TOPM TL_EVERY TL_CMD
      TL_STAGE1 TL_STAGE2 TL_TAG
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

OUT = HERE / "results" / os.environ.get('TL_TAG', '').lstrip("_")
TICKS = int(os.environ.get("TL_TICKS", "120000"))
REPORT = int(os.environ.get("TL_REPORT", "20000"))
K1 = int(os.environ.get("TL_K1", "64"))
K2 = int(os.environ.get("TL_K2", "128"))
K3 = int(os.environ.get("TL_K3", "4096"))
TOPM = int(os.environ.get("TL_TOPM", "8"))
EVERY = int(os.environ.get("TL_EVERY", "5"))
CMD = float(os.environ.get("TL_CMD", str(T.SHARE_CMD)))
STAGE1 = int(os.environ.get("TL_STAGE1", "10000"))
STAGE2 = int(os.environ.get("TL_STAGE2", "20000"))
NS = int(os.environ.get("TL_NS", str(L.NS)))
VWARP = os.environ.get("TL_VWARP", "1") == "1"
FOVEA = os.environ.get("TL_FOVEA", "1") == "1"
PRES = float(os.environ.get("TL_PRES", "0.375"))
DAGGER = int(os.environ.get("TL_DAGGER", "0"))
HOVER_LV = (W.NLEV // 2, W.NLEV // 2)
SENSE = L.make_sense(ns=NS, vwarp=VWARP, fovea=FOVEA)


def episode_teacher(rng, blind=0):
    """With present cells the pupil acts from tick one, so the teacher
    flies from tick one too — no handicap, distributions aligned."""
    s = L.air_init(rng)
    hold, t_bal = 0, None
    for t in range(L.EP_LEN):
        lv = HOVER_LV if t < blind else L.teacher(s)
        s = W.physics(s, lv)
        if L.balanced(s):
            hold += 1
            if hold >= 10 and t_bal is None:
                t_bal = t
        else:
            hold = 0
    return t_bal


def audition(n=50, seed=5):
    rng = np.random.default_rng(seed)
    done = [episode_teacher(rng) for _ in range(n)]
    ok = [d for d in done if d is not None]
    print(f"teacher audition: success {len(ok) / n:.2f}  "
          f"median ticks-to-balance {np.median(ok):.0f}", flush=True)
    return len(ok) / n


def evaluate(st, n=25, seed=99):
    """The pupil flies frozen from fresh tumbling inits.

    Success is STRICT: balanced through the last 150 ticks (3 s) of
    the episode — a 10-tick gate certified "hold" that lost balance at
    1.3 s and never came back. bal_frac (time balanced after the first
    2 s) is the graded view of the same thing.
    """
    rng = np.random.default_rng(seed)
    was = st.frozen
    st.frozen = True
    ok, frac, end_spd = [], [], []
    for _ in range(n):
        s = L.air_init(rng)
        st.reset()
        bal = []
        for t in range(L.EP_LEN):
            st.push(SENSE(s))
            lv, _, _ = st.act()
            s = W.physics(s, lv)
            bal.append(L.balanced(s))
        ok.append(all(bal[-150:]))
        frac.append(float(np.mean(bal[100:])))
        end_spd.append(float(np.hypot(s[2], s[3])))
    st.frozen = was
    return float(np.mean(ok)), float(np.mean(frac)), float(np.mean(end_spd))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if audition() < 0.98:
        print("teacher failed its audition — not training on it")
        return
    st = T.Stack(k1=K1, k2=K2, k3=K3, topm=TOPM, seed=0,
                 cmd_share=CMD, ns=NS, pres_share=PRES)
    st.every = EVERY
    rng = np.random.default_rng(0)
    cfg = {"kind": "module", "module": "td_balance",
           "dir": "experiments/2026_08_29/temporal_drone",
           "k1": K1, "k2": K2, "k3": K3, "topm": TOPM, "seed": 0,
           "every": EVERY, "cmd_share": CMD, "ns": NS, "vwarp": VWARP,
           "vscale": L.VSCALE, "fovea": FOVEA, "pres": PRES,
           "stage1": STAGE1, "stage2": STAGE2}

    def save(tag, extra):
        d = {f"t{c}l1": st.tr[c].l1.W for c in range(NS)}
        d.update({f"t{c}n1": st.tr[c].l1.n for c in range(NS)})
        d.update({f"t{c}l2": st.tr[c].l2.W for c in range(NS)})
        d.update({f"t{c}n2": st.tr[c].l2.n for c in range(NS)})
        d["l3"] = st.l3.W
        d["l3n"] = st.l3.n
        np.savez(OUT / f"{tag}.npz", **d)
        with open(OUT / f"{tag}.json", "w") as f:
            json.dump({**cfg, **extra}, f)

    s = L.air_init(rng)
    st.reset()
    ep_tick = episodes = 0
    pupil_ep = False
    gc, gd, oc, od, agree, winners = [], [], [], [], [], set()
    stats, best = [], {"score": -1.0, "agree": -1.0, "tick": 0}
    t0 = time.time()

    for t in range(TICKS + DAGGER):
        st.stage = 0 if t < STAGE1 else (1 if t < STAGE2 else 2)
        st.push(SENSE(s))
        guess, cd, win = st.act()
        lv = L.teacher(s)
        if win >= 0:
            tc, td = M.levels_to_cd(lv)
            gc.append(cd[0]); gd.append(cd[1]); oc.append(tc); od.append(td)
            agree.append(0.5 * ((guess[0] == lv[0]) + (guess[1] == lv[1])))
            winners.add(win)
        st.learn(lv)
        # phase 2 (DAgger): on odd episodes the PUPIL flies while the
        # teacher keeps labelling — the memory fills with the pupil's
        # own off-manifold histories, each stored with the right answer.
        fly = guess if (pupil_ep and win >= 0) else lv
        s = W.physics(s, fly)
        ep_tick += 1
        # tight envelope: a pupil episode ends the moment it leaves the
        # sane region, so its stored ticks are glide->brake pairs, not
        # the 50 m/s junk that collapsed the first DAgger attempt
        blown = pupil_ep and (np.hypot(s[2], s[3]) > 5 or abs(s[5]) > 5)
        if ep_tick >= L.EP_LEN or blown:
            episodes += 1
            s = L.air_init(rng)
            st.reset()
            ep_tick = 0
            pupil_ep = (t >= TICKS - 1) and (episodes % 2 == 1)

        if (t + 1) % REPORT == 0:
            ev, bfrac, e_spd = evaluate(st)
            row = {"tick": t + 1, "phase": 1 if t < TICKS else 2,
                   "episodes": episodes, "EVAL": round(ev, 3),
                   "bal_frac": round(bfrac, 3),
                   "end_speed": round(e_spd, 2)}
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
