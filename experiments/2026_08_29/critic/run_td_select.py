"""Stage 3: value-biased selection, read-only.

No training. The frozen 0.72 policy plus the learned critic: at each
tick, candidates = templates within MARGIN of the best masked score
(genuine near-ties only), winner = argmax(similarity + beta * V).
beta = 0 reproduces the baseline exactly. Content untouched, teacher
off — if any beta beats the baseline on the strict EVAL, it is the
project's first gain from its own experience alone.

Run:  .venv/bin/python experiments/2026_08_29/critic/run_td_select.py
Env:  TS_CKPT TS_CRITIC TS_MARGIN TS_EPS TS_BETAS
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

CKPT = Path(os.environ.get(
    "TS_CKPT", str(HERE.parent / "balance" / "results" / "dagger3" / "weights_best.npz")))
CRITIC = Path(os.environ.get(
    "TS_CRITIC", str(HERE / "results" / "critic.npz")))
MARGIN = float(os.environ.get("TS_MARGIN", "0.05"))
N_EPS = int(os.environ.get("TS_EPS", "50"))
BETAS = [float(x) for x in os.environ.get(
    "TS_BETAS", "0,0.005,0.01,0.02,0.05,0.1").split(",")]
HOVER_LV = (W.NLEV // 2, W.NLEV // 2)


def act_biased(st, V, beta):
    if st._codes is None or st.l3.n == 0:
        return HOVER_LV
    i, v = st._sensor_cells()
    if len(i) == 0:
        return HOVER_LV
    sc = st.l3.scores(i, v)
    top = np.argsort(-sc)[:16]
    cand = top[sc[top] >= sc[top[0]] - MARGIN]
    j = int(cand[np.argmax(sc[cand] + beta * V[cand])])
    row = st.l3.W[j]
    ds = st.dim_sensor
    pc = np.clip(row[ds:ds + T.NCMD], 0, None)
    pd = np.clip(row[ds + T.NCMD:], 0, None)
    c = B.sharpen(pc, M.COLL_C) if pc.max() > 0 else W.HOVER
    d = B.sharpen(pd, M.DIFF_C) if pd.max() > 0 else 0.0
    return M.cd_to_levels(c, d)


def evaluate(st, sense, V, beta, n, seed=99):
    rng = np.random.default_rng(seed)
    ok, frac = [], []
    for _ in range(n):
        s = L.air_init(rng)
        st.reset()
        bal = []
        for t in range(L.EP_LEN):
            st.push(sense(s))
            s = W.physics(s, act_biased(st, V, beta))
            bal.append(L.balanced(s))
        ok.append(all(bal[-150:]))
        frac.append(float(np.mean(bal[100:])))
    return float(np.mean(ok)), float(np.mean(frac))


def main():
    cfg = json.load(open(CKPT.with_suffix(".json")))
    st = T.load_stack(np.load(CKPT), cfg)
    sense = L.make_sense(ns=cfg["ns"], vwarp=cfg["vwarp"],
                         fovea=cfg.get("fovea", False))
    V = np.load(CRITIC)["V"]
    t0 = time.time()
    print(f"margin {MARGIN}, {N_EPS} episodes per arm")
    rows = []
    for beta in BETAS:
        ev, bf = evaluate(st, sense, V, beta, N_EPS)
        rows.append({"beta": beta, "EVAL": round(ev, 3),
                     "bal_frac": round(bf, 3)})
        print(f"  beta {beta:<6} EVAL {ev:.2f}   bal_frac {bf:.3f}",
              flush=True)
    with open(HERE / "results" / "select_sweep.json", "w") as f:
        json.dump({"margin": MARGIN, "eps": N_EPS, "rows": rows}, f,
                  indent=2)
    print(f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
