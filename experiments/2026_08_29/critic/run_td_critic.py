"""Stage 2 of the settled RL design: the critic, alone.

The policy is FROZEN (the 0.72 dagger3 checkpoint). A per-template
scalar V[j] — beside the row like `wins`, never inside it — learns by
SARSA-style TD along the winner chain while the pupil flies its own
episodes:

    r_t   = 1 if balanced(next state) else 0
    delta = r_t + gamma * V[j_{t+1}] - V[j_t]
    e     decays by gamma*lambda, e[j_t] = 1;  V += alpha * delta * e

With gamma = 0.98, V[j] reads as "discounted balanced time expected
after moments like this" (0..50). Template content is untouched —
value annotates, it never edits.

Then a validation flight with V frozen, scoring the three gates the
design demands before selection bias is allowed to exist:

  A. separation — mean V of basin templates (decoded present cells
     near zero) vs tumble templates. Must separate cleanly.
  B. honesty — corr(V[winner], empirical discounted return) on
     held-out ticks. The critic must predict, not decorate.
  C. leverage — among each tick's top-8 similarity candidates, the
     spread of V and how often argmax-V disagrees with argmax-sim.
     If near-ties carry equal value, stage 3 has nothing to choose
     with, and we stop here.

Run:  .venv/bin/python experiments/2026_08_29/critic/run_td_critic.py
Env:  TC_CKPT TC_TICKS TC_GAMMA TC_LAM TC_ALPHA TC_VAL_EPS TC_TAG
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
import td_bound as T                       # noqa: E402
import td_balance as L                     # noqa: E402

CKPT = Path(os.environ.get(
    "TC_CKPT", str(HERE.parent / "balance" / "results" / "dagger3" / "weights_best.npz")))
OUT = HERE / "results" / os.environ.get('TC_TAG', '').lstrip("_")
TICKS = int(os.environ.get("TC_TICKS", "160000"))
GAMMA = float(os.environ.get("TC_GAMMA", "0.98"))
LAM = float(os.environ.get("TC_LAM", "0.9"))
ALPHA = float(os.environ.get("TC_ALPHA", "0.1"))
VAL_EPS = int(os.environ.get("TC_VAL_EPS", "60"))
# exploration among near-ties during phase A: with prob EPS act on a
# random candidate within MARGIN of the best match, and give the TD
# credit to the EXECUTED candidate. Without this, each template's V is
# learned only in its own home states, so near-tie V differences
# measure state quality, not action advantage — measured: biasing
# selection by such a V degrades EVAL monotonically (0.72 -> 0.22).
EPS = float(os.environ.get("TC_EPS", "0.0"))
MARGIN = float(os.environ.get("TC_MARGIN", "0.05"))
HOVER_LV = (W.NLEV // 2, W.NLEV // 2)


def act_on(st, j):
    """Decode template j's own command cells."""
    import td3_stack as M
    row = st.l3.W[j]
    ds = st.dim_sensor
    pc = np.clip(row[ds:ds + T.NCMD], 0, None)
    pd = np.clip(row[ds + T.NCMD:], 0, None)
    c = B.sharpen(pc, M.COLL_C) if pc.max() > 0 else W.HOVER
    d = B.sharpen(pd, M.DIFF_C) if pd.max() > 0 else 0.0
    return M.cd_to_levels(c, d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(CKPT.with_suffix(".json")))
    st = T.load_stack(np.load(CKPT), cfg)
    sense = L.make_sense(ns=cfg["ns"], vwarp=cfg["vwarp"],
                         fovea=cfg.get("fovea", False))
    V = np.zeros(st.l3.k, np.float64)
    e = np.zeros_like(V)
    t0 = time.time()

    # ---- phase A: TD along the frozen pupil's own flights ---------------
    rng = np.random.default_rng(0)
    s = L.air_init(rng)
    st.reset()
    ep_tick, j_prev, r_prev = 0, -1, 0.0
    for t in range(TICKS):
        st.push(sense(s))
        win, lv = -1, HOVER_LV
        i, vv = st._sensor_cells()
        if len(i) and st.l3.n:
            sc = st.l3.scores(i, vv)
            top = np.argsort(-sc)[:16]
            cand = top[sc[top] >= sc[top[0]] - MARGIN]
            if EPS > 0 and len(cand) > 1 and rng.random() < EPS:
                win = int(rng.choice(cand))
            else:
                win = int(cand[0])
            lv = act_on(st, win)
        s = W.physics(s, lv)
        r = 1.0 if L.balanced(s) else 0.0
        if j_prev >= 0 and win >= 0:
            delta = r_prev + GAMMA * V[win] - V[j_prev]
            e *= GAMMA * LAM
            e[j_prev] = 1.0
            V += ALPHA * delta * e
        j_prev, r_prev = win, r
        ep_tick += 1
        if ep_tick >= L.EP_LEN or np.hypot(s[2], s[3]) > 20:
            s = L.air_init(rng)
            st.reset()
            ep_tick, j_prev = 0, -1     # cut traces, no bootstrap across
            e[:] = 0.0
    print(f"phase A done ({(time.time() - t0) / 60:.1f} min)  "
          f"V: mean {V.mean():.2f}  max {V.max():.2f}  "
          f"nonzero {(V > 0.01).sum()}/{st.l3.n}", flush=True)

    # ---- phase B: validation, V frozen ----------------------------------
    rng = np.random.default_rng(1234)
    v_win, rets, spreads, disagree, dv = [], [], [], [], []
    for _ in range(VAL_EPS):
        s = L.air_init(rng)
        st.reset()
        wins_ep, rs = [], []
        for t in range(L.EP_LEN):
            st.push(sense(s))
            i, vv = st._sensor_cells()
            win = -1
            if len(i) and st.l3.n:
                sc = st.l3.scores(i, vv)
                win = int(np.argmax(sc))
                top = np.argsort(-sc)[:8]
                vt = V[top]
                spreads.append(float(vt.std()))
                jbest = int(top[np.argmax(vt)])
                disagree.append(float(jbest != win))
                dv.append(float(vt.max() - V[win]))
            lv, _, _ = st.act()
            s = W.physics(s, lv)
            wins_ep.append(win)
            rs.append(1.0 if L.balanced(s) else 0.0)
        g = 0.0
        gs = np.zeros(len(rs))
        for t in range(len(rs) - 1, -1, -1):
            g = rs[t] + GAMMA * g
            gs[t] = g
        keep = min(len(rs) - 100, len(rs))   # drop truncated-return tail
        for t in range(keep):
            if wins_ep[t] >= 0:
                v_win.append(V[wins_ep[t]])
                rets.append(gs[t])

    honesty = float(np.corrcoef(v_win, rets)[0, 1])

    # gate A: decode each template's present cells, split basin/tumble
    ds_p = st.dim_pres
    nb = B.NB1
    grp_v = {"basin": [], "tumble": [], "mid": []}
    for j in range(st.l3.n):
        row = st.l3.W[j]
        dec = []
        for c in range(st.ns):
            p = np.clip(row[c * nb:(c + 1) * nb], 0, None)
            dec.append(B.sharpen(p, B.CEN1) if p.max() > 0 else np.nan)
        if np.isnan(dec).any():
            continue
        wv = max(abs(dec[0]), abs(dec[1]))
        wt = abs(dec[2])
        if wv < 0.26 and wt < 0.40:
            grp_v["basin"].append(V[j])
        elif wv > 0.6 or wt > 0.7:
            grp_v["tumble"].append(V[j])
        else:
            grp_v["mid"].append(V[j])

    rep = {
        "ckpt": str(CKPT), "ticks": TICKS, "gamma": GAMMA, "lam": LAM,
        "alpha": ALPHA,
        "gateA_meanV": {k: [round(float(np.mean(v)), 2), len(v)]
                        for k, v in grp_v.items() if v},
        "gateB_corr_V_return": round(honesty, 3),
        "gateC_topk_V_spread": round(float(np.mean(spreads)), 2),
        "gateC_disagree_frac": round(float(np.mean(disagree)), 3),
        "gateC_mean_dV_over_top1": round(float(np.mean(dv)), 2),
        "mins": round((time.time() - t0) / 60, 1),
    }
    np.savez(OUT / "critic.npz", V=V)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
